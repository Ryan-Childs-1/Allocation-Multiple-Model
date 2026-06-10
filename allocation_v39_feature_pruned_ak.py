"""v3.9 Feature-Pruned AK Specialist layer.

This module keeps the v3.8 AK specialist structure but prunes the features/rules that
were creating low-value Review rescues during smoke testing. It is intentionally small:
most feature pruning belongs in the retrained v3.9 trainer feature whitelist. The deployed
layer only tightens the AK Review rescue gate so the app can be tested immediately against
v3.8 without retraining.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import allocation_v33_enhancements as enh33
import allocation_v37_competition_features as enh37
import allocation_ak_specialist as ak_base

AK_SITES = {'248','159','212','145','121'}

DEFAULT_PARAMS = {
    "version": "v3.9_feature_pruned_ak",
    "ak_allocate_enabled": True,
    "ak_review_enabled": True,
    "allocate_min_confidence": None,
    "allocate_min_rec_flms": None,
    "allocate_min_high_conf_demand_share": None,
    "allocate_min_shortage_flms": None,
    "review_min_confidence": None,
    "review_min_rec_flms": None,
    "review_min_high_conf_demand_share": None,
    "review_min_shortage_flms": 2.50,
    "review_max_rec_trust": 1.0,
    "review_max_class_line_dc_pressure": 1.0,
    "review_remove_sparse_guardrail": False,
    "review_low_unit_rescue_max_units": 2.0,
}

PRUNED_FEATURES = [
    "max_demand_signal",
    "median_demand_signal",
    "std_demand_signal",
    "proj_to_dc",
    "rec_to_dc",
    "post_base_to_velocity",
    "post_base_to_sheet_need",
    "base_pred_units",
    "base_flms",
    "base_confidence",
    "base_to_rec",
    "base_x_rank",
    "only_ttm_supports_demand",
    "only_d60_supports_demand",
    "only_proj_supports_demand",
    "recent_sales_weak_but_proj_high",
    "proj_high_but_no_recent_sales",
    "rec_high_but_no_recent_sales",
    "d60_high_l30_low_flag",
    "ttm_high_recent_low_flag",
    "site_is_high_error_candidate",
    "site_is_high_overalloc_candidate",
    "site_is_high_underalloc_candidate",
    "site_requires_conservative_mode",
    "site_requires_rescue_mode",
]

KEPT_FEATURE_PRIORITIES = [
    "Class Name", "Line Name", "Site", "Rank", "Flag",
    "MIL", "FLM", "Cost", "L30", "D30", "D60", "LW", "TTM", "Supply", "Dc Avail", "Proj. Demand", "Alloc. Rec.",
    "weighted_velocity", "sheet_need", "mean_demand_signal", "demand_consensus", "demand_above_supply_share", "high_conf_demand_share",
    "supply_gap_sheet_need", "shortage_sheet_units", "shortage_sheet_flms", "supply_to_sheet_need", "rec_flms", "dc_flms",
    "dc_raw_empty", "dc_raw_1_25", "dc_raw_26_50", "dc_raw_51_100", "dc_raw_101_300", "dc_raw_301_600", "dc_raw_601_1000", "dc_raw_1001_2000", "dc_raw_2000_plus",
    "proj_flms", "proj_gap_flms", "rec_minus_proj_flms", "proj_minus_rec_flms", "dc_after_rec_flms", "proj_rec_agreement",
    "demand_signal_share_above_supply", "demand_signal_share_above_supply_plus_1flm", "single_flm_oversupply_risk", "rec_trust_score",
    "need_percentile_in_class_line_v37", "review_competition_score", "class_line_dc_pressure_score",
]


def _num(s, default=0.0):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(default).astype(float).to_numpy()


def load_v39_params(path=None):
    params = dict(DEFAULT_PARAMS)
    if path and os.path.exists(path):
        try:
            params.update(json.load(open(path)))
        except Exception:
            pass
    return params


def save_v39_params(path, params=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({**DEFAULT_PARAMS, **(params or {})}, open(path, 'w'), indent=2)


def apply_v39_feature_pruned_ak(df: pd.DataFrame, base_audit: pd.DataFrame, ak_model, params=None) -> pd.DataFrame:
    """Apply the pruned AK specialist layer.

    Differences from v3.8:
    - Allocate AK rescue remains unchanged because it helped every tested Allocate rescue.
    - Review AK rescue is tightened with shortage, rec-trust, and class-line DC-pressure gates.
    - The layer avoids using sparse single-signal flags as positive rescue signals.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    # First apply the existing v3.8 AK layer.
    out38 = ak_base.apply_ak_specialist(df, base_audit, ak_model)
    out = out38.copy()
    if ak_model is None:
        out['V39 Feature Pruning Applied'] = 0
        return out

    # If review pruning is disabled, return v3.8 behavior.
    if not params.get('ak_review_enabled', True):
        return out

    base_pred = _num(base_audit.get('Predicted Final Alloc', pd.Series([0]*len(df))))
    pred38 = _num(out38.get('Predicted Final Alloc', pd.Series([0]*len(df))))
    pred = pred38.copy()

    site = df['Site'].astype(str).str.replace('.0', '', regex=False).str.strip()
    ak = site.isin(AK_SITES).to_numpy()
    flag = df['Flag'].astype(str).str.upper()
    review = flag.str.contains('REVIEW', na=False).to_numpy()

    # Compute only the feature diagnostics needed for pruning.
    conf = pd.to_numeric(base_audit.get('Allocation Confidence', 0), errors='coerce').fillna(0)
    f33 = enh33.enhanced_feature_frame(df, base_pred, conf)
    f37 = enh37.competition_feature_frame(df, base_pred, conf)
    shortage = _num(f33.get('shortage_sheet_flms', pd.Series([0]*len(df))))
    rec_trust = _num(f37.get('rec_trust_score', pd.Series([0]*len(df))))
    dc_pressure = _num(f37.get('class_line_dc_pressure_score', pd.Series([0]*len(df))))
    sparse_guard = _num(f37.get('sparse_guardrail_score', pd.Series([0]*len(df))))

    # Rows v3.8 added as Review rescues.
    review_rescue = ak & review & (base_pred <= 0) & (pred38 > 0)
    keep = review_rescue.copy()
    # Low-unit AK Review rescues were the only feature pattern that showed harm.
    # Keep larger rescues because they matched positive historical targets in the smoke test.
    low_unit_rescue = pred38 <= float(params.get('review_low_unit_rescue_max_units', 2.0))
    keep &= (~low_unit_rescue) | (shortage >= float(params.get('review_min_shortage_flms', 2.5)))
    keep &= rec_trust <= float(params.get('review_max_rec_trust', 1.0))
    keep &= dc_pressure <= float(params.get('review_max_class_line_dc_pressure', 1.0))
    if params.get('review_remove_sparse_guardrail', False):
        keep &= sparse_guard < 0.50
    remove = review_rescue & ~keep
    pred[remove] = 0.0

    out['Predicted Final Alloc'] = pd.Series(pred).where(pred > 0, '')
    out['V39 Feature Pruning Applied'] = remove.astype(int)
    out['V39 Feature Pruning Reason'] = np.where(remove, 'Removed weak AK Review rescue after feature pruning', '')
    return out
