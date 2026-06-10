from __future__ import annotations
import os, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import allocation_split_numpy_core as core
import allocation_v35_pruned_enhancements as enh35
import allocation_v36_context_features as enh36
import allocation_v37_competition_features as enh37
import allocation_site802_specialist as site802
import allocation_ak_specialist as ak_specialist
import allocation_v39_feature_pruned_ak as enh39

APP_DIR = Path(__file__).resolve().parent
ART = APP_DIR
REPORTS = APP_DIR

st.set_page_config(
    page_title="Allocation Split Expert v3.9 Feature-Pruned AK + Site 802",
    layout="wide",
)
st.title("Allocation Split Expert v3.9 Feature-Pruned AK + Site 802")
st.caption(
    "Flat Streamlit app for the v3.9 feature-pruned allocation model: original v3 split models, "
    "Site 802 specialist, AK specialist, competition-aware/context features, and no residual correction."
)


def _read_json(path: Path, default=None):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {} if default is None else default


def _load_npz_from_flat_or_parts(name: str):
    """Load a model stored either as name.npz or flat split files name.npz.part000..."""
    direct = ART / name
    if direct.exists():
        return np.load(direct, allow_pickle=True)
    part_paths = sorted(ART.glob(f"{name}.part*"))
    if not part_paths:
        raise FileNotFoundError(f"Could not find {name} or split parts {name}.part000...")
    data = b"".join(p.read_bytes() for p in part_paths)
    return np.load(io.BytesIO(data), allow_pickle=True)


def _model_from_compact(z, role: str):
    meta = json.loads(str(z[f"{role}__meta"].item()))
    model = core.NumpyMLP(
        meta["input_dim"],
        meta["output_dim"],
        tuple(meta["hidden"]),
        meta["task"],
        dropout=meta.get("dropout", 0.0),
    )
    n = len(meta["hidden"]) + 1
    model.W = [z[f"{role}__W{i}"].astype(np.float32) for i in range(n)]
    model.b = [z[f"{role}__b{i}"].astype(np.float32) for i in range(n)]
    model.mW = [np.zeros_like(w) for w in model.W]
    model.vW = [np.zeros_like(w) for w in model.W]
    model.mb = [np.zeros_like(b) for b in model.b]
    model.vb = [np.zeros_like(b) for b in model.b]
    return model


@st.cache_resource(show_spinner="Loading v3.9 model bundle...")
def load_bundle():
    meta = _read_json(ART / "model_config.json")
    fc = meta.get("feature_config", {})
    feat_cfg = core.FeatureConfig(
        hash_dim_class=fc.get("hash_dim_class", 96),
        hash_dim_line=fc.get("hash_dim_line", 128),
        hash_dim_site=fc.get("hash_dim_site", 96),
        hash_dim_rank=fc.get("hash_dim_rank", 8),
        hash_dim_flag=fc.get("hash_dim_flag", 8),
        hash_dim_dc_bucket=fc.get("hash_dim_dc_bucket", 8),
        hash_dim_raw_dc_bucket=fc.get("hash_dim_raw_dc_bucket", 12),
        numeric_mean=fc.get("numeric_mean"),
        numeric_std=fc.get("numeric_std"),
        feature_names=fc.get("feature_names"),
    )
    models = {}
    for seg in ["allocate", "review"]:
        z = _load_npz_from_flat_or_parts(f"{seg}_model.npz")
        clf = _model_from_compact(z, "classifier")
        reg = _model_from_compact(z, "regressor")
        models[seg] = {
            "classifier": clf,
            "regressor": reg,
            "classifiers": [clf],
            "regressors": [reg],
            "residual": None,
        }
    params35 = enh35.load_v35_params(str(ART / "v35_pruned_params.json"))
    params36 = enh36.load_v36_params(str(ART / "v36_context_params.json"))
    params37 = enh37.load_v37_params(str(ART / "v37_competition_params.json"))
    params39 = enh39.load_v39_params(str(ART / "v39_feature_pruned_params.json"))
    site_model = site802.load_site802_model(str(ART / "site802_specialist_model.npz"))
    ak_model = ak_specialist.load_ak_specialist_model(str(ART / "ak_specialist_model.npz"))
    bundle = {
        "meta": meta,
        "feature_config": feat_cfg,
        "models": models,
        "site802_model": site_model,
        "ak_specialist_model": ak_model,
    }
    return bundle, params35, params36, params37, params39


def safe_cell_to_str(x):
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def find_header_row(df):
    wanted = set(core.ALLOWED_FEATURES + [core.TARGET_COL])
    best_i, best_hits = 0, -1
    for i in range(min(len(df), 90)):
        hits = sum(
            1
            for v in df.iloc[i].tolist()
            if core.CANONICAL_ALIASES.get(core._norm_name(v)) in wanted
        )
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


def predict_allocation(cleaned: pd.DataFrame):
    bundle, params35, params36, params37, params39 = load_bundle()
    base = core.predict_dataframe(cleaned, bundle, target_required=False)
    canon = core.canonicalize_columns(cleaned, target_required=False)
    audit = enh35.apply_v35_pruned_no_residual(canon, base, params35)
    audit = enh36.apply_v36_context_enhanced(canon, audit, params36)
    audit = enh37.apply_v37_competition_aware(canon, audit, params37)
    audit = site802.apply_site802_specialist(canon, audit, bundle.get("site802_model"))
    audit = enh39.apply_v39_feature_pruned_ak(canon, audit, bundle.get("ak_specialist_model"), params39)
    return canon, audit


try:
    bundle, v35_params, v36_params, v37_params, v39_params = load_bundle()
except Exception as e:
    st.error("Model bundle failed to load.")
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("v3.9 model")
    st.write("Flat artifact layout")
    st.code("model_config.json\nallocate_model.npz\nreview_model.npz\nsite802_specialist_model.npz\nak_specialist_model.npz")
    drop_non_model = st.checkbox("Only process Allocate and Review rows", value=True)
    st.write("Site 802 specialist loaded:", bundle.get("site802_model") is not None)
    st.write("AK specialist loaded:", bundle.get("ak_specialist_model") is not None)

model_tab, predict_tab, report_tab, files_tab = st.tabs([
    "Model overview",
    "Predict allocation",
    "Smoke-test report",
    "Artifact files",
])

with model_tab:
    meta = bundle.get("meta", {})
    cfg = meta.get("train_config", {})
    summary = _read_json(ART / "model_summary.json")
    st.subheader("What this model does")
    st.markdown(
        """
This is the **v3.9 Feature-Pruned AK + Site 802** app. It uses separate **Allocate** and **Review** two-stage models, then applies the feature-pruned business logic and specialist layers.

The model only uses the approved worksheet fields: `Class Name`, `Line Name`, `Site`, `MIL`, `FLM`, `Cost`, `L30`, `D30`, `D60`, `LW`, `TTM`, `Supply`, `Dc Avail`, `Rank`, `Proj. Demand`, `Alloc. Rec.`, and `Flag`.

The deployed stack includes:

- Original v3 split Allocate/Review classifiers and FLM regressors
- v3.3 raw `Dc Avail` buckets and `Proj. Demand` / `Alloc. Rec.` feature expansion
- v3.5 feature-pruned no-residual layer
- v3.6 context features
- v3.7 competition-aware workbook/class-line features
- Site 802 specialist model
- AK-store specialist model for sites `248`, `159`, `212`, `145`, and `121`
- v3.9 AK Review feature pruning
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", f"{int(summary.get('rows', 0)):,}" if summary else "—")
    c2.metric("Features", f"{int(summary.get('features', 0)):,}" if summary else "—")
    c3.metric("Allocate threshold", meta.get("allocate_threshold", cfg.get("allocate_threshold", "—")))
    c4.metric("Review threshold", meta.get("review_threshold", cfg.get("review_threshold", "—")))
    with st.expander("Training configuration", expanded=False):
        st.json(cfg)
    with st.expander("Approved input columns", expanded=False):
        st.write(core.ALLOWED_FEATURES)

with predict_tab:
    up = st.file_uploader("Upload allocation workbook or CSV", type=["xlsb", "xlsx", "xlsm", "xls", "csv"])
    if up:
        try:
            raw = read_upload(up)
            cleaned = drop_unnecessary_rows(raw, drop_non_model)
            canon, audit = predict_allocation(cleaned)
            output = cleaned.copy()
            target_col = None
            for c in output.columns:
                if core.CANONICAL_ALIASES.get(core._norm_name(c)) == core.TARGET_COL:
                    target_col = c
                    break
            if target_col is None:
                target_col = core.TARGET_COL
            output[target_col] = audit["Predicted Final Alloc"].values
            pred_num = pd.to_numeric(audit["Predicted Final Alloc"], errors="coerce").fillna(0)
            flags = canon["Flag"].astype(str).str.upper()
            st.success(f"Predicted {len(output):,} eligible rows.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(output):,}")
            c2.metric("Nonzero allocations", f"{int((pred_num > 0).sum()):,}")
            c3.metric("Predicted units", f"{int(pred_num.sum()):,}")
            c4.metric("AK rows", f"{int(canon['Site'].astype(str).str.replace('.0','',regex=False).isin(['248','159','212','145','121']).sum()):,}")

            filter_choice = st.selectbox(
                "Spot-check filter",
                [
                    "All rows",
                    "Allocate rows",
                    "Review rows",
                    "Nonzero predictions",
                    "Blank/zero predictions",
                    "Site 802 rows",
                    "AK store rows",
                    "v3.9 pruned rows",
                ],
            )
            mask = np.ones(len(output), dtype=bool)
            if filter_choice == "Allocate rows":
                mask = (flags.str.contains("ALLOC", na=False) & ~flags.str.contains("NO", na=False)).to_numpy()
            elif filter_choice == "Review rows":
                mask = flags.str.contains("REVIEW", na=False).to_numpy()
            elif filter_choice == "Nonzero predictions":
                mask = pred_num.to_numpy() > 0
            elif filter_choice == "Blank/zero predictions":
                mask = pred_num.to_numpy() <= 0
            elif filter_choice == "Site 802 rows":
                mask = canon["Site"].astype(str).str.replace(".0", "", regex=False).eq("802").to_numpy()
            elif filter_choice == "AK store rows":
                mask = canon["Site"].astype(str).str.replace(".0", "", regex=False).isin(["248", "159", "212", "145", "121"]).to_numpy()
            elif filter_choice == "v3.9 pruned rows":
                mask = audit.get("V39 Feature Pruning Applied", pd.Series([0] * len(audit))).astype(bool).to_numpy()

            st.dataframe(output.loc[mask].head(1000), use_container_width=True)
            st.download_button(
                "Download filled CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="allocation_v39_feature_pruned_ak_site802_filled_output.csv",
                mime="text/csv",
                key="download_filled_csv",
            )
            st.download_button(
                "Download audit CSV",
                audit.to_csv(index=False).encode("utf-8"),
                file_name="allocation_v39_feature_pruned_ak_site802_audit.csv",
                mime="text/csv",
                key="download_audit_csv",
            )
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

with report_tab:
    report_path = REPORTS / "SMOKE_TEST_REPORT.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("Smoke-test report not found in package.")

    downloadable_reports = [
        ("v39_summary_metrics_weighted.csv", "Download v3.9 weighted summary metrics"),
        ("v39_summary_metrics_file_avg.csv", "Download v3.9 file-average summary metrics"),
        ("v39_file_segment_metrics.csv", "Download v3.9 file/segment metrics"),
        ("v39_mae_delta_vs_v38.csv", "Download v3.9 vs v3.8 deltas"),
        ("v39_changed_rows_vs_v38.csv", "Download changed rows vs v3.8"),
        ("feature_decision_matrix.csv", "Download feature decision matrix"),
    ]
    for fname, label in downloadable_reports:
        p = REPORTS / fname
        if p.exists():
            st.download_button(label, p.read_bytes(), file_name=fname, mime="text/csv", key=f"dl_{fname}")

with files_tab:
    st.subheader("Flat package contents")
    rows = []
    for p in sorted(APP_DIR.iterdir()):
        if p.is_file():
            rows.append({"file": p.name, "size_mb": round(p.stat().st_size / (1024 * 1024), 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
