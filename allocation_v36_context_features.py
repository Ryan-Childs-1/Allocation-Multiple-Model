"""
Allocation Split Expert v3.6 Context-Enhanced Features.

Builds on v3.5 feature-pruned Site 802 branch and adds context-enhanced features:
- demand agreement/confidence and disagreement/volatility
- Rec/Proj trust and support features
- one-FLM/two-FLM pack outcome features
- DC scarcity and Class+Line group context
- site-relative and class-line-relative ranks
- rank/cost interactions
- sparse-demand and review-priority scores
- Site 802 cut/rescue helper flags

The deployed no-retrain layer uses the existing compact v3.5 models and applies a deliberately light deterministic adjustment. The trainer package includes these features directly so a future retrain can learn them inside the neural models.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import allocation_v35_pruned_enhancements as v35
EPS = 1e-6


def _num(df, col, default=0.0):
    return pd.to_numeric(df.get(col, default), errors='coerce').replace([np.inf,-np.inf], np.nan).fillna(default).to_numpy(float)


def _rank_score_series(df):
    return np.array([v35._rank_score(x) for x in df.get('Rank', pd.Series(['']*len(df))).values], float)


def _percentile_by_group(values: np.ndarray, groups: pd.Series) -> np.ndarray:
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    g = groups.astype(str).fillna('').replace('nan','')
    out = s.groupby(g, dropna=False).rank(pct=True, method='average').fillna(0.5).to_numpy(float)
    return out


def _safe_div(a, b, default=0.0):
    return np.divide(a, np.maximum(np.asarray(b, float), EPS), out=np.full_like(np.asarray(a, float), default, dtype=float), where=np.maximum(np.asarray(b, float), EPS)!=0)


def context_feature_frame(df: pd.DataFrame, base_pred_units=None, base_confidence=None) -> pd.DataFrame:
    """Return v3.5 features plus the v3.6 context features.

    Only approved worksheet columns are used, plus optional base prediction metadata for the runtime post-model layer.
    The trainer can call this with base_pred_units=None to avoid dependency on old-model outputs.
    """
    base = v35.enhanced_feature_frame(df, base_pred_units, base_confidence)
    n = len(df)
    flm = np.maximum(_num(df, 'FLM', 1.0), 1.0)
    l30 = _num(df, 'L30')
    d30 = _num(df, 'D30')
    d60 = _num(df, 'D60')
    lw = _num(df, 'LW')
    ttm = _num(df, 'TTM')
    supply = _num(df, 'Supply')
    dc = _num(df, 'Dc Avail')
    proj = _num(df, 'Proj. Demand')
    rec = _num(df, 'Alloc. Rec.')
    cost = _num(df, 'Cost')
    rank_score = _rank_score_series(df)
    lw_month = lw * 4.29
    d60_month = d60 / 2.0
    ttm_month = ttm / 12.0
    demand_stack = np.vstack([l30, d30, d60_month, lw_month, ttm_month, proj])
    weighted_velocity = base['weighted_velocity'].to_numpy(float)
    sheet_need = base['sheet_need'].to_numpy(float)
    shortage_sheet_flms = base['shortage_sheet_flms'].to_numpy(float)

    # 1. Demand agreement and confidence
    demand_signal_count_above_supply = (demand_stack > supply).sum(axis=0)
    demand_signal_share_above_supply = demand_signal_count_above_supply / demand_stack.shape[0]
    demand_signal_count_above_supply_plus_1flm = (demand_stack > (supply + flm)).sum(axis=0)
    demand_signal_share_above_supply_plus_1flm = demand_signal_count_above_supply_plus_1flm / demand_stack.shape[0]
    demand_positive_count = (demand_stack > 0).sum(axis=0)
    demand_zero_count = (demand_stack <= 0).sum(axis=0)

    # 2. Demand disagreement/volatility
    demand_signal_range = np.max(demand_stack, axis=0) - np.min(demand_stack, axis=0)
    demand_signal_std = np.std(demand_stack, axis=0)
    demand_signal_mean = np.mean(demand_stack, axis=0)
    demand_signal_cv = demand_signal_std / np.maximum(np.abs(demand_signal_mean), 1.0)
    recent_avg = (l30 + lw_month) / 2.0
    medium_avg = (d30 + d60_month) / 2.0
    long_avg = ttm_month
    recent_vs_long_term_disagreement = recent_avg - long_avg
    short_term_vs_medium_term_disagreement = recent_avg - medium_avg
    demand_disagreement_risk = demand_signal_cv * (1.0 - demand_signal_share_above_supply)

    # 3. Rec / projection trust
    rec_proj_abs_gap = np.abs(rec - proj)
    rec_proj_gap_flms = rec_proj_abs_gap / flm
    rec_proj_agreement_flag = (rec_proj_gap_flms <= 0.50).astype(float)
    rec_supported_by_recent_sales = (weighted_velocity + flm >= rec).astype(float)
    proj_supported_by_recent_sales = (weighted_velocity + flm >= proj).astype(float)
    rec_same_direction_as_velocity = ((rec > 0) & (weighted_velocity > supply)).astype(float)
    proj_same_direction_as_velocity = ((proj > supply) & (weighted_velocity > supply)).astype(float)

    # 4. Pack sensitivity / one-more-FLM outcomes
    flm_to_l30 = flm / np.maximum(l30, 1.0)
    flm_to_d30 = flm / np.maximum(d30, 1.0)
    flm_to_d60_monthly = flm / np.maximum(d60_month, 1.0)
    flm_to_proj = flm / np.maximum(proj, 1.0)
    flm_to_weighted_velocity = flm / np.maximum(weighted_velocity, 1.0)
    supply_after_1flm = supply + flm
    supply_after_2flm = supply + 2.0 * flm
    supply_after_rec = supply + rec
    need_after_1flm = sheet_need - supply_after_1flm
    need_after_2flm = sheet_need - supply_after_2flm
    oversupply_after_1flm = np.maximum(0, supply_after_1flm - sheet_need)
    oversupply_after_2flm = np.maximum(0, supply_after_2flm - sheet_need)
    single_flm_supply_to_velocity = supply_after_1flm / np.maximum(weighted_velocity, 1.0)
    single_flm_supply_to_sheet_need = supply_after_1flm / np.maximum(sheet_need, 1.0)
    single_flm_oversupply_risk = ((single_flm_supply_to_sheet_need > 1.35) & (demand_signal_share_above_supply_plus_1flm < 0.34)).astype(float)

    # 5/6/7. Context features: Class+Line, Site relative, and scarcity.
    class_name = df.get('Class Name', pd.Series(['']*n)).astype(str).fillna('')
    line_name = df.get('Line Name', pd.Series(['']*n)).astype(str).fillna('')
    site_name = df.get('Site', pd.Series(['']*n)).astype(str).fillna('')
    class_line = class_name + '|' + line_name
    class_line_total_dc_avail = pd.Series(dc).groupby(class_line, dropna=False).transform('max').to_numpy(float)
    class_line_total_proj_gap = pd.Series(np.maximum(proj - supply, 0)).groupby(class_line, dropna=False).transform('sum').to_numpy(float)
    class_line_total_rec = pd.Series(rec).groupby(class_line, dropna=False).transform('sum').to_numpy(float)
    class_line_avg_supply = pd.Series(supply).groupby(class_line, dropna=False).transform('mean').to_numpy(float)
    class_line_avg_velocity = pd.Series(weighted_velocity).groupby(class_line, dropna=False).transform('mean').to_numpy(float)
    class_line_row_count = pd.Series(np.ones(n)).groupby(class_line, dropna=False).transform('sum').to_numpy(float)
    class_line_positive_rec_rate = pd.Series((rec > 0).astype(float)).groupby(class_line, dropna=False).transform('mean').to_numpy(float)
    class_line_need_percentile = _percentile_by_group(shortage_sheet_flms, class_line)
    class_line_rank_percentile = _percentile_by_group(rank_score, class_line)
    class_line_velocity_percentile = _percentile_by_group(weighted_velocity, class_line)
    class_line_rec_percentile = _percentile_by_group(rec, class_line)
    dc_to_total_need_proxy = dc / np.maximum(class_line_total_proj_gap, 1.0)
    dc_to_proj_gap = dc / np.maximum(np.maximum(proj - supply, 0), 1.0)
    dc_to_rec_gap = dc / np.maximum(rec, 1.0)
    dc_scarcity_score = np.minimum(dc_to_total_need_proxy, 10.0)
    dc_can_cover_rec_flag = (dc >= rec).astype(float)
    dc_can_cover_shortage_flag = (dc >= np.maximum(sheet_need - supply, 0)).astype(float)

    site_need_percentile = _percentile_by_group(shortage_sheet_flms, site_name)
    site_velocity_percentile = _percentile_by_group(weighted_velocity, site_name)
    site_rec_percentile = _percentile_by_group(rec, site_name)
    site_supply_pressure_percentile = _percentile_by_group(sheet_need - supply, site_name)
    site_vs_group_velocity_rank = class_line_velocity_percentile - site_velocity_percentile
    site_vs_group_shortage_rank = class_line_need_percentile - site_need_percentile
    site_vs_group_rec_rank = class_line_rec_percentile - site_rec_percentile

    # 8. Rank interactions
    rank_inverse = 1.0 / np.maximum(rank_score, 0.05)
    rank_x_shortage_flms = rank_score * shortage_sheet_flms
    rank_x_weighted_velocity = rank_score * weighted_velocity
    rank_x_proj_gap = rank_score * np.maximum(proj - supply, 0)
    top_rank_flag = (rank_score >= 1.0).astype(float)
    bottom_rank_flag = (rank_score <= 0.30).astype(float)

    # 9. Cost-aware risk
    cost_log = np.log1p(np.maximum(cost, 0))
    allocation_cost_if_1flm = cost * flm
    allocation_cost_if_rec = cost * rec
    shortage_cost_proxy = cost * np.maximum(sheet_need - supply, 0)
    overalloc_cost_proxy = cost * np.maximum(supply_after_rec - sheet_need, 0)
    cost_x_shortage_flms = cost_log * shortage_sheet_flms
    cost_x_rec_excess = cost_log * np.maximum(rec - np.maximum(sheet_need - supply, 0), 0) / flm

    # 10. Sparse demand pattern
    zero_l30_flag = (l30 <= 0).astype(float)
    zero_d30_flag = (d30 <= 0).astype(float)
    zero_d60_flag = (d60 <= 0).astype(float)
    zero_lw_flag = (lw <= 0).astype(float)
    zero_ttm_flag = (ttm <= 0).astype(float)
    zero_recent_sales_flag = ((l30 <= 0) & (lw <= 0)).astype(float)
    zero_all_recent_demand_flag = ((l30 <= 0) & (d30 <= 0) & (d60 <= 0) & (lw <= 0)).astype(float)
    sparse_demand_count = zero_l30_flag + zero_d30_flag + zero_d60_flag + zero_lw_flag + zero_ttm_flag
    has_any_sales_flag = ((l30 > 0) | (d30 > 0) | (d60 > 0) | (lw > 0) | (ttm > 0)).astype(float)

    # 12. Allocation Rec decomposition
    rec_is_zero = (rec <= 0).astype(float)
    rec_is_one_flm = ((rec > 0) & (rec <= 1.25 * flm)).astype(float)
    rec_is_multi_flm = (rec > 1.25 * flm).astype(float)
    rec_exceeds_dc = (rec > dc).astype(float)
    rec_equals_dc = (np.abs(rec - dc) <= EPS).astype(float)
    rec_below_dc = (rec < dc).astype(float)
    rec_exceeds_proj = (rec > proj).astype(float)
    rec_exceeds_velocity = (rec > weighted_velocity).astype(float)
    rec_exceeds_shortage = (rec > np.maximum(sheet_need - supply, 0)).astype(float)
    rec_supported_by_shortage = (rec <= np.maximum(sheet_need - supply, 0) + flm).astype(float)

    # 13. Review priority scores
    review_priority_score = (
        0.35 * demand_signal_share_above_supply
        + 0.25 * np.clip(shortage_sheet_flms / 4.0, 0, 1)
        + 0.20 * np.clip(rank_score / 1.2, 0, 1)
        + 0.20 * base['proj_rec_agreement'].to_numpy(float)
    )
    review_top_need_in_group = (class_line_need_percentile >= 0.80).astype(float)
    review_top_rank_in_group = (class_line_rank_percentile >= 0.80).astype(float)
    review_high_confidence_need = ((review_priority_score >= 0.68) & (demand_signal_share_above_supply >= 0.50)).astype(float)
    review_low_confidence_need = ((review_priority_score <= 0.35) | (demand_signal_share_above_supply <= 0.17)).astype(float)

    # 14. Site 802 specialist helper flags
    site802 = site_name.str.replace('.0','',regex=False).str.strip().eq('802').to_numpy(float)
    site802_high_rec_low_velocity = site802 * ((rec > weighted_velocity + flm) & (demand_signal_share_above_supply < 0.34)).astype(float)
    site802_proj_rec_disagreement = site802 * (rec_proj_gap_flms > 1.0).astype(float)
    site802_single_flm_risk = site802 * single_flm_oversupply_risk
    site802_strong_demand_support = site802 * ((demand_signal_share_above_supply >= 0.50) & (shortage_sheet_flms >= 1.0)).astype(float)
    site802_cut_candidate = site802 * ((rec > sheet_need - supply + flm) & (demand_signal_share_above_supply < 0.50)).astype(float)
    site802_rescue_candidate = site802 * ((rec > 0) & (demand_signal_share_above_supply >= 0.50) & (dc > 0)).astype(float)

    # 15. Conservative / aggressive scores
    conservative_need_score = (
        0.35 * demand_signal_share_above_supply_plus_1flm
        + 0.25 * np.clip(shortage_sheet_flms / 3.0, 0, 1)
        + 0.20 * rec_supported_by_recent_sales
        + 0.20 * (1.0 - single_flm_oversupply_risk)
    )
    aggressive_need_score = (
        0.35 * np.clip(shortage_sheet_flms / 4.0, 0, 1)
        + 0.30 * (rec > 0).astype(float)
        + 0.20 * np.clip((proj - supply) / np.maximum(flm * 3, 1), 0, 1)
        + 0.15 * demand_signal_share_above_supply
    )

    extra = pd.DataFrame({
        'demand_signal_count_above_supply': demand_signal_count_above_supply,
        'demand_signal_share_above_supply': demand_signal_share_above_supply,
        'demand_signal_count_above_supply_plus_1flm': demand_signal_count_above_supply_plus_1flm,
        'demand_signal_share_above_supply_plus_1flm': demand_signal_share_above_supply_plus_1flm,
        'demand_positive_count': demand_positive_count,
        'demand_zero_count': demand_zero_count,
        'demand_signal_range': demand_signal_range,
        'demand_signal_std_v36': demand_signal_std,
        'demand_signal_cv': demand_signal_cv,
        'recent_vs_long_term_disagreement': recent_vs_long_term_disagreement,
        'short_term_vs_medium_term_disagreement': short_term_vs_medium_term_disagreement,
        'demand_disagreement_risk': demand_disagreement_risk,
        'rec_proj_abs_gap': rec_proj_abs_gap,
        'rec_proj_gap_flms': rec_proj_gap_flms,
        'rec_proj_agreement_flag': rec_proj_agreement_flag,
        'rec_supported_by_recent_sales': rec_supported_by_recent_sales,
        'proj_supported_by_recent_sales': proj_supported_by_recent_sales,
        'rec_same_direction_as_velocity': rec_same_direction_as_velocity,
        'proj_same_direction_as_velocity': proj_same_direction_as_velocity,
        'flm_to_l30': flm_to_l30,
        'flm_to_d30': flm_to_d30,
        'flm_to_d60_monthly': flm_to_d60_monthly,
        'flm_to_proj': flm_to_proj,
        'flm_to_weighted_velocity': flm_to_weighted_velocity,
        'supply_after_1flm': supply_after_1flm,
        'supply_after_2flm': supply_after_2flm,
        'supply_after_rec': supply_after_rec,
        'need_after_1flm': need_after_1flm,
        'need_after_2flm': need_after_2flm,
        'oversupply_after_1flm': oversupply_after_1flm,
        'oversupply_after_2flm': oversupply_after_2flm,
        'single_flm_supply_to_velocity': single_flm_supply_to_velocity,
        'single_flm_supply_to_sheet_need': single_flm_supply_to_sheet_need,
        'single_flm_oversupply_risk': single_flm_oversupply_risk,
        'class_line_total_dc_avail': class_line_total_dc_avail,
        'class_line_total_proj_gap': class_line_total_proj_gap,
        'class_line_total_rec': class_line_total_rec,
        'class_line_avg_supply': class_line_avg_supply,
        'class_line_avg_velocity': class_line_avg_velocity,
        'class_line_row_count': class_line_row_count,
        'class_line_positive_rec_rate': class_line_positive_rec_rate,
        'class_line_need_percentile': class_line_need_percentile,
        'class_line_rank_percentile': class_line_rank_percentile,
        'class_line_velocity_percentile': class_line_velocity_percentile,
        'class_line_rec_percentile': class_line_rec_percentile,
        'dc_to_total_need_proxy': dc_to_total_need_proxy,
        'dc_to_proj_gap': dc_to_proj_gap,
        'dc_to_rec_gap': dc_to_rec_gap,
        'dc_scarcity_score': dc_scarcity_score,
        'dc_can_cover_rec_flag': dc_can_cover_rec_flag,
        'dc_can_cover_shortage_flag': dc_can_cover_shortage_flag,
        'site_need_percentile': site_need_percentile,
        'site_velocity_percentile': site_velocity_percentile,
        'site_rec_percentile': site_rec_percentile,
        'site_supply_pressure_percentile': site_supply_pressure_percentile,
        'site_vs_group_velocity_rank': site_vs_group_velocity_rank,
        'site_vs_group_shortage_rank': site_vs_group_shortage_rank,
        'site_vs_group_rec_rank': site_vs_group_rec_rank,
        'rank_inverse': rank_inverse,
        'rank_x_shortage_flms': rank_x_shortage_flms,
        'rank_x_weighted_velocity': rank_x_weighted_velocity,
        'rank_x_proj_gap': rank_x_proj_gap,
        'top_rank_flag': top_rank_flag,
        'bottom_rank_flag': bottom_rank_flag,
        'allocation_cost_if_1flm': allocation_cost_if_1flm,
        'allocation_cost_if_rec': allocation_cost_if_rec,
        'shortage_cost_proxy': shortage_cost_proxy,
        'overalloc_cost_proxy': overalloc_cost_proxy,
        'cost_x_shortage_flms': cost_x_shortage_flms,
        'cost_x_rec_excess': cost_x_rec_excess,
        'zero_l30_flag': zero_l30_flag,
        'zero_d30_flag': zero_d30_flag,
        'zero_d60_flag': zero_d60_flag,
        'zero_lw_flag': zero_lw_flag,
        'zero_ttm_flag': zero_ttm_flag,
        'zero_recent_sales_flag': zero_recent_sales_flag,
        'zero_all_recent_demand_flag': zero_all_recent_demand_flag,
        'sparse_demand_count': sparse_demand_count,
        'has_any_sales_flag': has_any_sales_flag,
        'rec_is_zero': rec_is_zero,
        'rec_is_one_flm': rec_is_one_flm,
        'rec_is_multi_flm': rec_is_multi_flm,
        'rec_exceeds_dc': rec_exceeds_dc,
        'rec_equals_dc': rec_equals_dc,
        'rec_below_dc': rec_below_dc,
        'rec_exceeds_proj': rec_exceeds_proj,
        'rec_exceeds_velocity': rec_exceeds_velocity,
        'rec_exceeds_shortage': rec_exceeds_shortage,
        'rec_supported_by_shortage': rec_supported_by_shortage,
        'review_priority_score': review_priority_score,
        'review_top_need_in_group': review_top_need_in_group,
        'review_top_rank_in_group': review_top_rank_in_group,
        'review_high_confidence_need': review_high_confidence_need,
        'review_low_confidence_need': review_low_confidence_need,
        'site802_high_rec_low_velocity': site802_high_rec_low_velocity,
        'site802_proj_rec_disagreement': site802_proj_rec_disagreement,
        'site802_single_flm_risk': site802_single_flm_risk,
        'site802_strong_demand_support': site802_strong_demand_support,
        'site802_cut_candidate': site802_cut_candidate,
        'site802_rescue_candidate': site802_rescue_candidate,
        'conservative_need_score': conservative_need_score,
        'aggressive_need_score': aggressive_need_score,
    })
    return pd.concat([base, extra], axis=1).replace([np.inf, -np.inf], 0).fillna(0)


DEFAULT_PARAMS = {
    'allocate_extra_weak_single_flm_clamp': True,
    'weak_single_flm_confidence_max': 0.34,
    'weak_single_flm_need_share_max': 0.17,
    'weak_single_flm_over_ratio_min': 1.55,
    'weak_single_flm_rec_excess_min': 2.50,
    'allocate_strong_rescue_enabled': True,
    'allocate_rescue_need_share_min': 0.34,
    'allocate_rescue_shortage_flms_min': 2.00,
    'allocate_rescue_confidence_min': 0.34,
    'review_feature_rescue_enabled': False,
    'review_keep_original': True,
}


def load_v36_params(path=None):
    params = dict(DEFAULT_PARAMS)
    if path and os.path.exists(path):
        try:
            params.update(json.load(open(path)))
        except Exception:
            pass
    return params


def save_v36_params(path, params=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({**DEFAULT_PARAMS, **(params or {})}, open(path, 'w'), indent=2)


def apply_v36_context_enhanced(df: pd.DataFrame, base_audit: pd.DataFrame, params=None) -> pd.DataFrame:
    """Apply lightweight v3.6 post-model controls on top of v3.5 pruned behavior.

    The heavy lift is intended for retraining. This runtime layer is intentionally small so it cannot overpower the original v3/v3.5 model.
    """
    params = load_v36_params(None) if params is None else {**DEFAULT_PARAMS, **dict(params)}
    out = base_audit.copy()
    pred = pd.to_numeric(out.get('Predicted Final Alloc', 0), errors='coerce').fillna(0).to_numpy(float)
    conf = pd.to_numeric(out.get('Allocation Confidence', 0), errors='coerce').fillna(0).to_numpy(float)
    flm = np.maximum(_num(df, 'FLM', 1.0), 1.0)
    dc = _num(df, 'Dc Avail')
    flag = df['Flag'].astype(str).str.upper()
    alloc = (flag.str.contains('ALLOC', na=False) & ~flag.str.contains('NO', na=False)).to_numpy(bool)
    review = flag.str.contains('REVIEW', na=False).to_numpy(bool)
    feat = context_feature_frame(df, pred, conf)

    # Tiny extra clamp: only cut Allocate rows that are obvious weak single-FLM risks.
    weak_single = (
        params.get('allocate_extra_weak_single_flm_clamp', True)
        & alloc
        & (pred > 0)
        & (feat['demand_signal_share_above_supply_plus_1flm'].to_numpy(float) <= params['weak_single_flm_need_share_max'])
        & (feat['single_flm_supply_to_sheet_need'].to_numpy(float) >= params['weak_single_flm_over_ratio_min'])
        & (feat['rec_minus_need_flms'].to_numpy(float) >= params['weak_single_flm_rec_excess_min'])
    )
    pred = np.where(weak_single, np.maximum(pred - flm, 0), pred)

    # Tiny rescue: only rescue Allocate rows with multiple supporting signals and clear shortage.
    rescue = (
        params.get('allocate_strong_rescue_enabled', True)
        & alloc
        & (pred <= 0)
        & (dc > 0)
        & (feat['demand_signal_share_above_supply'].to_numpy(float) >= params['allocate_rescue_need_share_min'])
        & (feat['shortage_sheet_flms'].to_numpy(float) >= params['allocate_rescue_shortage_flms_min'])
        & (conf >= params['allocate_rescue_confidence_min'])
    )
    pred = np.where(rescue, np.minimum(flm, dc), pred)

    # Review rows remain unchanged by default; v3.5/v3 Review performance was already stable.
    if params.get('review_feature_rescue_enabled', False):
        rrescue = (
            review & (pred <= 0) & (dc > 0)
            & (feat['review_high_confidence_need'].to_numpy(float) > 0)
            & (feat['shortage_sheet_flms'].to_numpy(float) >= 2.50)
        )
        pred = np.where(rrescue, np.minimum(flm, dc), pred)

    rounded = np.where(pred > 0, np.round(pred / flm) * flm, 0.0)
    rounded = np.minimum(rounded, dc)
    rounded = np.where((rounded <= 0) & (pred > 0) & (dc > 0) & (dc < flm), dc, rounded)
    rounded = np.where(rounded <= 0, np.nan, rounded)
    out['Predicted Final Alloc'] = pd.Series(rounded).where(~np.isnan(rounded), '')
    out['v3_6_context_adjusted'] = (np.nan_to_num(rounded, nan=0.0) != pd.to_numeric(base_audit.get('Predicted Final Alloc', 0), errors='coerce').fillna(0).to_numpy(float)).astype(int)
    out['v3_6_adjustment_reason'] = np.where(weak_single, 'weak_single_flm_context_clamp', np.where(rescue, 'strong_context_rescue', 'v35_kept'))
    out['v3_6_review_priority_score'] = feat['review_priority_score'].to_numpy(float)
    out['v3_6_demand_agreement'] = feat['demand_signal_share_above_supply'].to_numpy(float)
    out['v3_6_single_flm_risk'] = feat['single_flm_oversupply_risk'].to_numpy(float)
    return out
