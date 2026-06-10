from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import allocation_split_numpy_core as core
import allocation_v35_pruned_enhancements as enh35
import allocation_v36_context_features as enh36
import allocation_v37_competition_features as enh37
import allocation_v39_feature_pruned_ak as enh39
import allocation_site802_specialist as site802
import allocation_ak_specialist as ak_specialist

APP_DIR = Path(__file__).resolve().parent
ART = APP_DIR
AK_SITES = {"248", "159", "212", "145", "121"}

st.set_page_config(page_title="Allocation Multiple Model", layout="wide")

# -----------------------------------------------------------------------------
# JSON / artifact helpers
# -----------------------------------------------------------------------------

def _read_json(path: Path, default=None):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {} if default is None else default


def _load_npz_from_artifacts(name: str):
    """Load a model stored as a full NPZ, flat parts, or model_parts/name.part000."""
    part_dir = ART / "model_parts"
    part_paths = sorted(part_dir.glob(f"{name}.part*")) if part_dir.exists() else []
    if not part_paths:
        part_paths = sorted(ART.glob(f"{name}.part*"))
    if part_paths:
        data = b"".join(p.read_bytes() for p in part_paths)
        return np.load(io.BytesIO(data), allow_pickle=True)
    direct = ART / name
    if direct.exists():
        return np.load(direct, allow_pickle=True)
    raise FileNotFoundError(f"Could not find {name}, model_parts/{name}.part*, or {name}.part*.")


def _unpack_mlp_from_prefix(z, prefix: str, task: str):
    w_keys = sorted([k for k in z.files if k.startswith(prefix + "__W")], key=lambda x: int(x.split("__W")[-1]))
    b_keys = sorted([k for k in z.files if k.startswith(prefix + "__b")], key=lambda x: int(x.split("__b")[-1]))
    if not w_keys or len(w_keys) != len(b_keys):
        return None
    weights = [z[k].astype(np.float32) for k in w_keys]
    biases = [z[k].astype(np.float32) for k in b_keys]
    input_dim = weights[0].shape[0]
    output_dim = weights[-1].shape[1]
    hidden = tuple(w.shape[1] for w in weights[:-1])
    model = core.NumpyMLP(input_dim, output_dim, hidden, task=task, dropout=0.0)
    model.W = weights
    model.b = biases
    model.mW = [np.zeros_like(w) for w in model.W]
    model.vW = [np.zeros_like(w) for w in model.W]
    model.mb = [np.zeros_like(b) for b in model.b]
    model.vb = [np.zeros_like(b) for b in model.b]
    return model


def _load_segment_models(segment: str):
    z = _load_npz_from_artifacts(f"{segment}_model.npz")
    classifiers = []
    regressors = []
    # New continued-training format: classifier_0, classifier_1, classifier_2, etc.
    for i in range(20):
        clf = _unpack_mlp_from_prefix(z, f"classifier_{i}", "softmax")
        if clf is None:
            break
        classifiers.append(clf)
    for i in range(20):
        reg = _unpack_mlp_from_prefix(z, f"regressor_{i}", "regression")
        if reg is None:
            break
        regressors.append(reg)
    # Older compact format fallback: classifier / regressor.
    if not classifiers:
        clf = _unpack_mlp_from_prefix(z, "classifier", "softmax")
        if clf is not None:
            classifiers.append(clf)
    if not regressors:
        reg = _unpack_mlp_from_prefix(z, "regressor", "regression")
        if reg is not None:
            regressors.append(reg)
    if not classifiers or not regressors:
        raise ValueError(f"Could not load {segment} classifier/regressor ensemble from {segment}_model.npz.")
    return {
        "classifiers": classifiers,
        "regressors": regressors,
        "classifier": classifiers[0],
        "regressor": regressors[0],
        "residual": None,
    }


def _safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        out[c] = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return out


def build_enhanced_feature_matrix(df: pd.DataFrame, config: Dict[str, Any], fit: bool = False):
    """Build the full v3.9 1,076-feature matrix used by the continued models."""
    canon = core.canonicalize_columns(df, target_required=core.TARGET_COL in df.columns)
    config = dict(config or {})
    core_cfg_dict = config.get("core_feature_config", {})
    core_cfg = core.FeatureConfig(**core_cfg_dict) if core_cfg_dict else core.FeatureConfig()
    X_core, core_cfg = core.build_features(canon, config=core_cfg, fit=fit)

    extra = enh37.competition_feature_frame(canon, base_pred_units=None, base_confidence=None)
    extra = _safe_numeric_frame(extra)
    pruned = set(config.get("pruned_features", getattr(enh39, "PRUNED_FEATURES", [])))
    extra = extra[[c for c in extra.columns if c not in pruned]]
    extra = extra.loc[:, ~extra.columns.duplicated()].copy()

    if fit:
        mean = extra.mean(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        std = extra.std(axis=0).replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(float)
        std[std < 1e-6] = 1.0
        config["extra_feature_names"] = list(extra.columns)
        config["extra_feature_mean"] = mean.tolist()
        config["extra_feature_std"] = std.tolist()
        config["core_feature_config"] = asdict(core_cfg)
        config["pruned_features"] = sorted(pruned)
    else:
        names = list(config.get("extra_feature_names", []))
        for c in names:
            if c not in extra.columns:
                extra[c] = 0.0
        extra = extra[names]
        mean = np.asarray(config.get("extra_feature_mean", [0.0] * len(names)), dtype=float)
        std = np.asarray(config.get("extra_feature_std", [1.0] * len(names)), dtype=float)
        std[std < 1e-6] = 1.0

    if len(extra.columns):
        X_extra = ((extra.to_numpy(dtype=float) - mean) / std).astype(np.float32)
        X_extra = np.nan_to_num(X_extra, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.hstack([X_core.astype(np.float32), X_extra])
    else:
        X = X_core.astype(np.float32)

    config["feature_names"] = list(getattr(core_cfg, "feature_names", []) or []) + [f"ctx__{c}" for c in list(extra.columns)]
    config["feature_count"] = int(X.shape[1])
    config["core_feature_count"] = int(X_core.shape[1])
    config["extra_feature_count"] = int(len(extra.columns))
    return X.astype(np.float32), config


@st.cache_resource(show_spinner="Loading models...")
def load_bundle():
    meta = _read_json(ART / "model_config.json")
    models = {"allocate": _load_segment_models("allocate"), "review": _load_segment_models("review")}
    bundle = {
        "meta": meta,
        "feature_config": meta.get("feature_config", {}),
        "models": models,
        "site802_model": site802.load_site802_model(str(ART / "site802_specialist_model.npz")),
        "ak_specialist_model": ak_specialist.load_ak_specialist_model(str(ART / "ak_specialist_model.npz")),
    }
    params35 = enh35.load_v35_params(str(ART / "v35_pruned_params.json"))
    params36 = enh36.load_v36_params(str(ART / "v36_context_params.json"))
    params37 = enh37.load_v37_params(str(ART / "v37_competition_params.json"))
    params39 = enh39.load_v39_params(str(ART / "v39_feature_pruned_params.json"))
    return bundle, params35, params36, params37, params39


# -----------------------------------------------------------------------------
# Input parsing / normalization
# -----------------------------------------------------------------------------

def safe_cell_to_str(x):
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def clean_site_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(".0", "", regex=False).str.strip()


def find_header_row(df):
    wanted = set(core.ALLOWED_FEATURES + [core.TARGET_COL])
    best_i, best_hits = 0, -1
    for i in range(min(len(df), 90)):
        hits = sum(1 for v in df.iloc[i].tolist() if core.CANONICAL_ALIASES.get(core._norm_name(v)) in wanted)
        if hits > best_hits:
            best_i, best_hits = i, hits
    return best_i if best_hits >= 8 else 0


def read_upload(uploaded, sheet_name=None):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    if name.endswith(".xlsb"):
        xl = pd.ExcelFile(io.BytesIO(data), engine="pyxlsb")
        sheet = sheet_name or ("3.3 Working Table" if "3.3 Working Table" in xl.sheet_names else xl.sheet_names[0])
        preview = pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine="pyxlsb", header=None, nrows=90)
        hdr = find_header_row(preview)
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet, engine="pyxlsb", header=hdr)
    xl = pd.ExcelFile(io.BytesIO(data))
    sheet = sheet_name or ("3.3 Working Table" if "3.3 Working Table" in xl.sheet_names else xl.sheet_names[0])
    preview = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, nrows=90)
    hdr = find_header_row(preview)
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=hdr)


def drop_unnecessary_rows(df, drop_non_model=True):
    work = df.dropna(how="all").copy()
    row_text = work.apply(lambda r: "|".join(safe_cell_to_str(v) for v in r.to_numpy()), axis=1).str.upper()
    repeated = row_text.str.contains("FINAL ALLOC", na=False) & row_text.str.contains("ALLOC", na=False) & row_text.str.contains("FLAG", na=False)
    work = work.loc[~repeated].reset_index(drop=True)
    if drop_non_model:
        try:
            canon = core.canonicalize_columns(work, target_required=False)
            flags = canon["Flag"].astype(str).str.upper()
            keep = (flags.str.contains("ALLOC", na=False) & ~flags.str.contains("NO", na=False)) | flags.str.contains("REVIEW", na=False)
            work = work.loc[keep.to_numpy()].reset_index(drop=True)
        except Exception:
            pass
    return work


def find_target_column(df: pd.DataFrame):
    for c in df.columns:
        if core.CANONICAL_ALIASES.get(core._norm_name(c)) == core.TARGET_COL:
            return c
    return None


def int_or_blank_series(values) -> pd.Series:
    nums = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    nums = np.rint(nums.to_numpy(dtype=float)).astype(np.int64)
    return pd.Series(np.where(nums > 0, nums.astype(object), ""))


def numeric_units(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


# -----------------------------------------------------------------------------
# Prediction using the full v3.9 enhanced feature matrix
# -----------------------------------------------------------------------------

def _average_classifier_proba(models: List[Any], X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict_proba(X) for m in models], axis=0)


def _average_regression(models: List[Any], X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def _segment_predict(segment_models: Dict[str, Any], X: np.ndarray, work_segment: pd.DataFrame, cfg: Dict[str, Any]):
    p = _average_classifier_proba(segment_models["classifiers"], X)
    positive_prob = 1.0 - p[:, 0]
    reg_packs = _average_regression(segment_models["regressors"], X)
    flm = np.maximum(core.to_float_array(work_segment["FLM"], 1.0), 1.0)
    rec_packs = core.to_float_array(work_segment["Alloc. Rec."], 0.0) / flm
    raw_packs = core._blended_pack_prediction(p, reg_packs, rec_packs, cfg)
    g = np.argmax(p, axis=1)
    return p, positive_prob, raw_packs, g


def predict_core_enhanced(cleaned: pd.DataFrame, bundle: Dict[str, Any], target_required: bool = False):
    work = core.canonicalize_columns(cleaned, target_required=target_required)
    X, _ = build_enhanced_feature_matrix(work, bundle["feature_config"], fit=False)
    expected_dim = int(bundle["models"]["allocate"]["classifier"].input_dim)
    if X.shape[1] != expected_dim:
        raise ValueError(f"Feature matrix has {X.shape[1]} columns but model expects {expected_dim}. Check model_config.json and feature modules.")

    flags = core.clean_flag(work["Flag"].values)
    alloc_mask = core.flag_mask(flags, "ALLOCATE")
    review_mask = core.flag_mask(flags, "REVIEW")
    n = len(work)
    pred = np.full(n, np.nan, dtype=float)
    group_pred = np.full(n, "", dtype=object)
    conf = np.zeros(n, dtype=float)
    raw_packs_all = np.zeros(n, dtype=float)
    cfg = dict(bundle["meta"].get("train_config", {}))
    cfg["allocate_threshold"] = bundle["meta"].get("allocate_threshold", cfg.get("allocate_threshold", 0.40))
    cfg["review_threshold"] = bundle["meta"].get("review_threshold", cfg.get("review_threshold", 0.50))

    for segment, mask, threshold in [
        ("allocate", alloc_mask, cfg.get("allocate_threshold", 0.40)),
        ("review", review_mask, cfg.get("review_threshold", 0.50)),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        p, positive_prob, raw_packs, g = _segment_predict(bundle["models"][segment], X[idx], work.iloc[idx], cfg)
        conf[idx] = positive_prob
        raw_packs_all[idx] = raw_packs
        group_pred[idx] = np.array(core.GROUP_LABELS, dtype=object)[g]
        if segment == "allocate":
            flm = np.maximum(core.to_float_array(work.iloc[idx]["FLM"], 1.0), 1.0)
            dc = core.to_float_array(work.iloc[idx]["Dc Avail"], 0.0)
            units = raw_packs * flm
            rounded = np.where((positive_prob >= threshold) & (dc > 0), np.round(units / flm) * flm, np.nan)
            rounded = np.minimum(rounded, dc)
            rounded = np.where((positive_prob >= threshold) & (dc > 0) & (dc < flm), dc, rounded)
            rounded = np.where(rounded <= 0, np.nan, rounded)
            pred[idx] = rounded

    ridx = np.where(review_mask)[0]
    if len(ridx):
        rwork = work.iloc[ridx].copy()
        need = core.row_need_score(rwork, conf[ridx], raw_packs_all[ridx])
        rwork["__idx"] = ridx
        rwork["__need"] = need
        rwork["__pool"] = (
            rwork["Class Name"].astype(str).str.upper().str.strip() + "|" +
            rwork["Line Name"].astype(str).str.upper().str.strip() + "|" +
            rwork["Cost"].astype(str).str.upper().str.strip() + "|" +
            rwork["Dc Avail"].astype(str).str.upper().str.strip()
        )
        for _, sub in rwork.sort_values("__need", ascending=False).groupby("__pool", sort=False):
            remaining = float(np.nanmax(core.to_float_array(sub["Dc Avail"], 0.0)))
            for _, row in sub.sort_values("__need", ascending=False).iterrows():
                i = int(row["__idx"])
                if remaining <= 0:
                    break
                if conf[i] < cfg.get("review_threshold", 0.50):
                    continue
                flm = max(float(row["FLM"] if not pd.isna(row["FLM"]) else 1.0), 1.0)
                want = max(raw_packs_all[i] * flm, 0.0)
                if want <= 0:
                    continue
                rounded = round(want / flm) * flm
                if rounded <= 0 and want > 0:
                    rounded = flm
                if remaining < flm:
                    alloc = remaining
                else:
                    alloc = min(rounded, remaining)
                    alloc = math.floor(alloc / flm) * flm
                    if alloc <= 0 and remaining > 0:
                        alloc = min(flm, remaining)
                if alloc > 0:
                    pred[i] = alloc
                    remaining -= alloc

    out = work.copy()
    out["Predicted Final Alloc"] = pred
    out["Predicted Group"] = group_pred
    out["Allocation Confidence"] = conf
    out["Raw Predicted FLMs"] = raw_packs_all
    out["Predicted Final Alloc"] = out["Predicted Final Alloc"].where(~np.isnan(out["Predicted Final Alloc"]), "")
    return out


def predict_allocation(cleaned: pd.DataFrame):
    bundle, params35, params36, params37, params39 = load_bundle()
    base = predict_core_enhanced(cleaned, bundle, target_required=False)
    canon = core.canonicalize_columns(cleaned, target_required=False)
    audit = enh35.apply_v35_pruned_no_residual(canon, base, params35)
    audit = enh36.apply_v36_context_enhanced(canon, audit, params36)
    audit = enh37.apply_v37_competition_aware(canon, audit, params37)
    audit = site802.apply_site802_specialist(canon, audit, bundle.get("site802_model"))
    audit = enh39.apply_v39_feature_pruned_ak(canon, audit, bundle.get("ak_specialist_model"), params39)
    audit = audit.copy()
    audit["Predicted Final Alloc"] = int_or_blank_series(audit["Predicted Final Alloc"]).values
    return canon, audit


# -----------------------------------------------------------------------------
# Metrics / audit helpers
# -----------------------------------------------------------------------------

def build_segment_masks(canon: pd.DataFrame, audit: pd.DataFrame):
    flags = canon["Flag"].astype(str).str.upper()
    sites = clean_site_series(canon["Site"])
    pred = numeric_units(audit["Predicted Final Alloc"])
    masks = {
        "All rows": np.ones(len(canon), dtype=bool),
        "Allocate rows": (flags.str.contains("ALLOC", na=False) & ~flags.str.contains("NO", na=False)).to_numpy(),
        "Review rows": flags.str.contains("REVIEW", na=False).to_numpy(),
        "Site 802 rows": sites.eq("802").to_numpy(),
        "AK store rows": sites.isin(AK_SITES).to_numpy(),
        "Nonzero predictions": pred > 0,
        "Blank / zero predictions": pred <= 0,
    }
    if "Site 802 Specialist Applied" in audit.columns:
        masks["Site 802 specialist applied"] = pd.to_numeric(audit["Site 802 Specialist Applied"], errors="coerce").fillna(0).to_numpy() > 0
    if "AK Specialist Applied" in audit.columns:
        masks["AK specialist applied"] = pd.to_numeric(audit["AK Specialist Applied"], errors="coerce").fillna(0).to_numpy() > 0
    if "V39 Feature Pruning Applied" in audit.columns:
        masks["v3.9 pruning applied"] = pd.to_numeric(audit["V39 Feature Pruning Applied"], errors="coerce").fillna(0).to_numpy() > 0
    return masks


def metric_row(name: str, mask: np.ndarray, pred: np.ndarray, actual: np.ndarray, flm: np.ndarray, dc: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None
    p = pred[mask]
    a = actual[mask]
    f = np.maximum(flm[mask], 1.0)
    d = dc[mask]
    err = p - a
    abs_err = np.abs(err)
    return {
        "Segment / model path": name,
        "Rows": int(mask.sum()),
        "MAE Units": float(np.mean(abs_err)),
        "RMSE Units": float(np.sqrt(np.mean(err ** 2))),
        "Exact Rate": float(np.mean(p == a)),
        "Within 1 FLM": float(np.mean(abs_err <= f)),
        "False Positives": int(((p > 0) & (a <= 0)).sum()),
        "False Negatives": int(((p <= 0) & (a > 0)).sum()),
        "Pred Units": int(round(float(p.sum()))),
        "Actual Units": int(round(float(a.sum()))),
        "Unit Delta": int(round(float(p.sum() - a.sum()))),
        "Negative Violations": int((p < 0).sum()),
        "Over-DC Violations": int((p - d > 1e-9).sum()),
    }


def compute_audit_metrics(canon: pd.DataFrame, audit: pd.DataFrame, actual_values) -> pd.DataFrame:
    pred = numeric_units(audit["Predicted Final Alloc"])
    actual = numeric_units(actual_values)
    flm = numeric_units(canon["FLM"])
    dc = numeric_units(canon["Dc Avail"])
    rows = []
    for name, mask in build_segment_masks(canon, audit).items():
        row = metric_row(name, mask, pred, actual, flm, dc)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def build_row_audit(cleaned: pd.DataFrame, canon: pd.DataFrame, audit: pd.DataFrame, actual_values) -> pd.DataFrame:
    out = cleaned.copy()
    pred = numeric_units(audit["Predicted Final Alloc"])
    actual = numeric_units(actual_values)
    out["Actual Final Alloc"] = np.rint(actual).astype(int)
    out["Predicted Final Alloc Audit"] = np.rint(pred).astype(int)
    out["Absolute Error Units"] = np.abs(out["Predicted Final Alloc Audit"] - out["Actual Final Alloc"])
    out["Signed Error Units"] = out["Predicted Final Alloc Audit"] - out["Actual Final Alloc"]
    out["Exact Match"] = out["Absolute Error Units"].eq(0)
    out["False Positive"] = (out["Predicted Final Alloc Audit"] > 0) & (out["Actual Final Alloc"] <= 0)
    out["False Negative"] = (out["Predicted Final Alloc Audit"] <= 0) & (out["Actual Final Alloc"] > 0)
    out["Model Segment"] = np.where(canon["Flag"].astype(str).str.upper().str.contains("REVIEW", na=False), "Review", "Allocate")
    out["Site 802 Specialist Applied"] = audit.get("Site 802 Specialist Applied", pd.Series([0] * len(audit))).values
    out["AK Specialist Applied"] = audit.get("AK Specialist Applied", pd.Series([0] * len(audit))).values
    out["V39 Feature Pruning Applied"] = audit.get("V39 Feature Pruning Applied", pd.Series([0] * len(audit))).values
    out["Allocation Confidence"] = audit.get("Allocation Confidence", pd.Series([np.nan] * len(audit))).values
    out["Raw Predicted FLMs"] = audit.get("Raw Predicted FLMs", pd.Series([np.nan] * len(audit))).values
    return out


# -----------------------------------------------------------------------------
# Feature information
# -----------------------------------------------------------------------------

def _load_feature_catalog():
    p = ART / "feature_catalog.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame()


def _load_feature_matrix():
    p = ART / "feature_decision_matrix.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame()


def overview_stats(bundle):
    meta = bundle.get("meta", {})
    summary = _read_json(ART / "model_summary.json")
    train_rows = summary.get("rows") or meta.get("rows") or meta.get("training_rows")
    features = summary.get("features") or meta.get("feature_count") or meta.get("feature_config", {}).get("feature_count")
    if not features:
        try:
            features = int(bundle["models"]["allocate"]["classifier"].input_dim)
        except Exception:
            features = "—"
    return train_rows or "—", features


def show_feature_review(bundle):
    feature_df = _load_feature_matrix()
    catalog = _load_feature_catalog()
    meta = bundle.get("meta", {})
    fc = meta.get("feature_config", {})
    st.markdown("### Feature review")
    st.write(
        "The model uses the approved worksheet fields first, then adds engineered signals for demand agreement, supply shortage, pack-size risk, DC pressure, class-line competition, and specialist behavior."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total features", f"{int(fc.get('feature_count', bundle['models']['allocate']['classifier'].input_dim)):,}")
    c2.metric("Core features", f"{int(fc.get('core_feature_count', 0)):,}" if fc.get("core_feature_count") else "—")
    c3.metric("Context features", f"{int(fc.get('extra_feature_count', 0)):,}" if fc.get("extra_feature_count") else "—")
    c4.metric("Pruned features", f"{len(fc.get('pruned_features', [])):,}")

    st.markdown("#### Most important feature families")
    st.markdown(
        """
| Feature family | What it tells the model | Most useful for |
|---|---|---|
| Original sheet fields | Direct worksheet signals such as `Alloc. Rec.`, `Proj. Demand`, `Supply`, `Dc Avail`, `FLM`, `Rank`, and demand history | All models |
| Demand agreement | Whether multiple demand signals agree that supply is short | Reducing weak false positives |
| Shortage / pressure | How far demand, projection, or recommendation exceeds current supply | Allocate and Review base models |
| Pack-size risk | Whether one FLM would over-supply the store | Single-FLM decisions |
| DC buckets and scarcity | Whether inventory is tight, moderate, or abundant | Review ranking and DC-constrained allocation |
| Class-line competition | Whether a row is strong relative to peer rows in the same item family | Review rows |
| Site 802 and AK specialist signals | Site-specific cut/rescue behavior | Specialist models |
"""
    )

    if not feature_df.empty:
        st.markdown("#### Feature pruning summary")
        if "decision" in feature_df.columns:
            counts = feature_df["decision"].fillna("Unknown").astype(str).value_counts().reset_index()
            counts.columns = ["Decision", "Count"]
            st.dataframe(counts, use_container_width=True)
        show_cols = [c for c in ["feature", "family", "decision", "scope", "rationale", "evidence_note"] if c in feature_df.columns]
        if show_cols:
            with st.expander("Feature decision matrix", expanded=False):
                st.dataframe(feature_df[show_cols], use_container_width=True)
        st.download_button(
            "Download feature decision matrix",
            feature_df.to_csv(index=False).encode("utf-8"),
            file_name="feature_decision_matrix.csv",
            mime="text/csv",
            key="download_feature_decision_matrix",
        )

    if not catalog.empty:
        with st.expander("Feature catalog", expanded=False):
            st.dataframe(catalog.head(2000), use_container_width=True)
        st.download_button(
            "Download feature catalog",
            catalog.to_csv(index=False).encode("utf-8"),
            file_name="feature_catalog.csv",
            mime="text/csv",
            key="download_feature_catalog",
        )

    st.markdown("#### How to audit feature behavior")
    st.write(
        "Rows with high demand agreement, strong shortage, positive recommendation, and enough DC inventory should usually receive allocation. Rows where `Alloc. Rec.` is high but recent sales support is weak are cut candidates. Review rows should be judged relative to peer rows in the same class-line group. Site 802 and AK rows should be checked separately because their specialist models can apply different cut/rescue behavior."
    )


# -----------------------------------------------------------------------------
# App UI
# -----------------------------------------------------------------------------
try:
    bundle, v35_params, v36_params, v37_params, v39_params = load_bundle()
except Exception as e:
    st.error("Model bundle failed to load.")
    st.exception(e)
    st.stop()

st.title("Allocation Multiple Model")
with st.sidebar:
    st.header("Controls")
    drop_non_model = st.checkbox("Only process Allocate and Review rows", value=True)
    st.divider()
    st.write("**Specialists**")
    st.write("Site 802:", "Loaded" if bundle.get("site802_model") is not None else "Not loaded")
    st.write("AK stores:", "Loaded" if bundle.get("ak_specialist_model") is not None else "Not loaded")

predict_tab, audit_tab, overview_tab, features_tab, files_tab = st.tabs([
    "Predict",
    "Audit",
    "Model overview",
    "Features",
    "Files",
])

with predict_tab:
    st.subheader("Predict Final Alloc.")
    up = st.file_uploader("Upload allocation workbook or CSV", type=["xlsb", "xlsx", "xlsm", "xls", "csv"], key="predict_upload")
    if up:
        try:
            raw = read_upload(up)
            cleaned = drop_unnecessary_rows(raw, drop_non_model)
            canon, audit = predict_allocation(cleaned)
            output = cleaned.copy()
            target_col = find_target_column(output) or core.TARGET_COL
            output[target_col] = int_or_blank_series(audit["Predicted Final Alloc"]).values
            pred_num = numeric_units(audit["Predicted Final Alloc"])
            st.success(f"Processed {len(output):,} rows.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(output):,}")
            c2.metric("Nonzero allocations", f"{int((pred_num > 0).sum()):,}")
            c3.metric("Predicted units", f"{int(pred_num.sum()):,}")
            c4.metric("Specialist rows", f"{int(clean_site_series(canon['Site']).eq('802').sum() + clean_site_series(canon['Site']).isin(AK_SITES).sum()):,}")

            masks = build_segment_masks(canon, audit)
            filter_choice = st.selectbox("View rows", list(masks.keys()), key="predict_filter")
            mask = masks.get(filter_choice, np.ones(len(output), dtype=bool))
            st.dataframe(output.loc[mask].head(1000), use_container_width=True)
            st.download_button(
                "Download filled CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="allocation_multiple_model_filled_output.csv",
                mime="text/csv",
                key="download_filled_csv",
            )
            st.download_button(
                "Download model audit CSV",
                audit.to_csv(index=False).encode("utf-8"),
                file_name="allocation_multiple_model_prediction_audit.csv",
                mime="text/csv",
                key="download_prediction_audit_csv",
            )
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

with audit_tab:
    st.subheader("Audit uploaded file")
    st.write("Upload a file that already contains `Final Alloc.` values to compare the model against the existing allocation decisions.")
    audit_up = st.file_uploader("Upload file for audit", type=["xlsb", "xlsx", "xlsm", "xls", "csv"], key="audit_upload")
    if audit_up:
        try:
            raw = read_upload(audit_up)
            cleaned = drop_unnecessary_rows(raw, drop_non_model)
            target_col = find_target_column(cleaned)
            if target_col is None:
                st.warning("No `Final Alloc.` column was detected. This file can be predicted, but accuracy cannot be audited.")
            else:
                actual_raw = cleaned[target_col]
                has_actual = actual_raw.notna().any() and (actual_raw.astype(str).str.strip() != "").any()
                if not has_actual:
                    st.warning("The `Final Alloc.` column exists but appears blank. No accuracy audit can be calculated.")
                else:
                    canon, audit = predict_allocation(cleaned)
                    metrics = compute_audit_metrics(canon, audit, actual_raw)
                    row_audit = build_row_audit(cleaned, canon, audit, actual_raw)
                    all_row = metrics[metrics["Segment / model path"].eq("All rows")].iloc[0]
                    st.success("Audit completed.")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Rows audited", f"{int(all_row['Rows']):,}")
                    c2.metric("MAE units", f"{all_row['MAE Units']:.3f}")
                    c3.metric("Exact rate", f"{all_row['Exact Rate']:.2%}")
                    c4.metric("Unit delta", f"{int(all_row['Unit Delta']):,}")
                    display_metrics = metrics.copy()
                    for col in ["MAE Units", "RMSE Units"]:
                        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.3f}")
                    for col in ["Exact Rate", "Within 1 FLM"]:
                        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.2%}")
                    st.dataframe(display_metrics, use_container_width=True)

                    error_filter = st.selectbox(
                        "Review audit rows",
                        ["All rows", "Errors only", "False positives", "False negatives", "Site 802 rows", "AK store rows", "Specialist-applied rows"],
                        key="audit_filter",
                    )
                    mask = np.ones(len(row_audit), dtype=bool)
                    sites = clean_site_series(canon["Site"])
                    if error_filter == "Errors only":
                        mask = row_audit["Absolute Error Units"].to_numpy() > 0
                    elif error_filter == "False positives":
                        mask = row_audit["False Positive"].to_numpy()
                    elif error_filter == "False negatives":
                        mask = row_audit["False Negative"].to_numpy()
                    elif error_filter == "Site 802 rows":
                        mask = sites.eq("802").to_numpy()
                    elif error_filter == "AK store rows":
                        mask = sites.isin(AK_SITES).to_numpy()
                    elif error_filter == "Specialist-applied rows":
                        mask = (pd.to_numeric(row_audit["Site 802 Specialist Applied"], errors="coerce").fillna(0).to_numpy() > 0) | (pd.to_numeric(row_audit["AK Specialist Applied"], errors="coerce").fillna(0).to_numpy() > 0)
                    st.dataframe(row_audit.loc[mask].head(1000), use_container_width=True)
                    st.download_button("Download audit metrics CSV", metrics.to_csv(index=False).encode("utf-8"), "uploaded_file_accuracy_metrics.csv", "text/csv", key="download_uploaded_accuracy_metrics")
                    st.download_button("Download row-level audit CSV", row_audit.to_csv(index=False).encode("utf-8"), "uploaded_file_row_level_audit.csv", "text/csv", key="download_uploaded_row_audit")
        except Exception as e:
            st.error("Audit failed.")
            st.exception(e)

with overview_tab:
    meta = bundle.get("meta", {})
    cfg = meta.get("train_config", {})
    train_rows, feature_count = overview_stats(bundle)
    st.subheader("Model status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", f"{int(train_rows):,}" if isinstance(train_rows, (int, float)) else str(train_rows))
    c2.metric("Input features", f"{int(feature_count):,}" if isinstance(feature_count, (int, float)) else str(feature_count))
    c3.metric("Allocate threshold", meta.get("allocate_threshold", cfg.get("allocate_threshold", "—")))
    c4.metric("Review threshold", meta.get("review_threshold", cfg.get("review_threshold", "—")))
    st.markdown("#### Model paths")
    st.write("Allocate model, Review model, Site 802 specialist, and AK specialist are loaded from the packaged artifacts.")
    with st.expander("Training configuration", expanded=False):
        st.json(cfg)
    with st.expander("Approved input columns", expanded=False):
        st.write(core.ALLOWED_FEATURES)

with features_tab:
    show_feature_review(bundle)

with files_tab:
    st.subheader("Included files")
    rows = []
    for p in sorted(APP_DIR.rglob("*")):
        if p.is_file():
            rows.append({"file": str(p.relative_to(APP_DIR)), "size_mb": round(p.stat().st_size / (1024 * 1024), 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
