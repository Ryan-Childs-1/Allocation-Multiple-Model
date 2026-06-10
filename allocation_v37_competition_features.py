"""
Allocation Split Expert v3.7 Competition-Aware Features.

Adds the next layer of workbook-relative features on top of v3.6:
- workbook-level competition ranks within Class Name + Line Name
- item-family DC pressure features
- hypothetical 1/2/3 FLM final-supply risk features
- recommendation trust score and follow/cut/add flags
- sparse-demand guardrails
- allocation size classes
- site-level behavior features
- candidate specialist trigger flags
- review-only competition score
- allocate-only cut/rescue flags
- rank/demand alignment and cost-normalized risk

The trainer uses these features directly. The deployed immediate layer uses a small deterministic v3.7
adjustment on top of v3.6 so the branch can be smoke-tested before a full retrain.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import allocation_v36_context_features as v36
EPS = 1e-6


def _num(df, col, default=0.0):
    return pd.to_numeric(df.get(col, default), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(float)


def _safe_div(a, b, default=0.0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.divide(a, np.maximum(np.abs(b), EPS), out=np.full_like(a, default, dtype=float), where=np.maximum(np.abs(b), EPS)!=0)


def _pct(values, groups):
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    g = pd.Series(groups).astype(str).fillna('')
    return s.groupby(g, dropna=False).rank(pct=True, method='average').fillna(0.5).to_numpy(float)


def _group_transform(values, groups, how):
    return pd.Series(values).groupby(pd.Series(groups).astype(str).fillna(''), dropna=False).transform(how).fillna(0).to_numpy(float)


def _rank_score_series(df):
    return np.array([v36.v35._rank_score(x) for x in df.get('Rank', pd.Series(['']*len(df))).values], float)


def _bucket(values, bins):
    values=np.asarray(values,float)
    return np.digitize(values, bins, right=False).astype(float)


def competition_feature_frame(df: pd.DataFrame, base_pred_units=None, base_confidence=None) -> pd.DataFrame:
    base = v36.context_feature_frame(df, base_pred_units, base_confidence)
    n=len(df)
    flm=np.maximum(_num(df,'FLM',1.0),1.0)
    l30=_num(df,'L30'); d30=_num(df,'D30'); d60=_num(df,'D60'); lw=_num(df,'LW'); ttm=_num(df,'TTM')
    supply=_num(df,'Supply'); dc=_num(df,'Dc Avail'); proj=_num(df,'Proj. Demand'); rec=_num(df,'Alloc. Rec.'); cost=_num(df,'Cost')
    rank_score=_rank_score_series(df)
    lw_month=lw*4.29; d60_month=d60/2.0; ttm_month=ttm/12.0
    weighted_velocity=base.get('weighted_velocity', pd.Series(np.maximum.reduce([l30,d30,d60_month,lw_month,ttm_month,proj]))).to_numpy(float)
    sheet_need=base.get('sheet_need', pd.Series(np.maximum.reduce([weighted_velocity, proj, d30, d60_month]))).to_numpy(float)
    shortage=np.maximum(sheet_need-supply,0.0)
    shortage_flms=shortage/flm
    proj_gap=np.maximum(proj-supply,0.0)
    rec_flms=rec/flm
    rec_need_gap=rec-shortage
    demand_share=base.get('demand_signal_share_above_supply', pd.Series(np.zeros(n))).to_numpy(float)
    demand_share_plus=base.get('demand_signal_share_above_supply_plus_1flm', pd.Series(np.zeros(n))).to_numpy(float)
    single_risk=base.get('single_flm_oversupply_risk', pd.Series(np.zeros(n))).to_numpy(float)

    class_name=df.get('Class Name', pd.Series(['']*n)).astype(str).fillna('')
    line_name=df.get('Line Name', pd.Series(['']*n)).astype(str).fillna('')
    site=df.get('Site', pd.Series(['']*n)).astype(str).fillna('').str.replace('.0','',regex=False).str.strip()
    class_line=class_name+'|'+line_name

    # 1. Workbook-level competition features within Class+Line.
    need_rank_in_class_line=_pct(shortage_flms,class_line)
    need_percentile_in_class_line=need_rank_in_class_line
    rec_rank_in_class_line=_pct(rec_flms,class_line)
    velocity_rank_in_class_line=_pct(weighted_velocity,class_line)
    shortage_flm_rank_in_class_line=_pct(shortage_flms,class_line)
    rank_adjusted_need=shortage_flms*np.clip(rank_score,0,2)
    rank_adjusted_need_percentile=_pct(rank_adjusted_need,class_line)
    top5_need_in_class_line=(need_percentile_in_class_line>=0.95).astype(float)
    top10_need_in_class_line=(need_percentile_in_class_line>=0.90).astype(float)
    above_median_need_in_class_line=(need_percentile_in_class_line>=0.50).astype(float)

    # 2. Item-family DC pressure.
    total_shortage_flms=_group_transform(shortage_flms,class_line,'sum')
    total_alloc_rec_flms=_group_transform(rec_flms,class_line,'sum')
    total_proj_gap_flms=_group_transform(proj_gap/flm,class_line,'sum')
    rows_competing=_group_transform((shortage_flms>0).astype(float),class_line,'sum')
    group_dc=_group_transform(dc,class_line,'max')
    class_line_dc_to_total_shortage=_safe_div(group_dc, total_shortage_flms*flm, 10.0)
    class_line_dc_to_total_rec=_safe_div(group_dc, total_alloc_rec_flms*flm, 10.0)
    class_line_dc_pressure_score=1.0-np.clip(class_line_dc_to_total_shortage,0,1)
    class_line_rows_competing_for_dc=rows_competing
    class_line_tight_dc_flag=(class_line_dc_to_total_shortage<0.75).astype(float)
    class_line_abundant_dc_flag=(class_line_dc_to_total_shortage>=1.50).astype(float)

    # 3/11. Hypothetical final-supply features.
    supply_after_1=supply+flm; supply_after_2=supply+2*flm; supply_after_3=supply+3*flm; supply_after_rec=supply+rec
    gap_after_1flm_to_proj=proj-supply_after_1; gap_after_2flm_to_proj=proj-supply_after_2; gap_after_3flm_to_proj=proj-supply_after_3
    gap_after_1flm_to_velocity=weighted_velocity-supply_after_1; gap_after_2flm_to_velocity=weighted_velocity-supply_after_2; gap_after_3flm_to_velocity=weighted_velocity-supply_after_3
    oversupply_after_1flm=np.maximum(0,supply_after_1-sheet_need); oversupply_after_2flm=np.maximum(0,supply_after_2-sheet_need); oversupply_after_3flm=np.maximum(0,supply_after_3-sheet_need)
    hyp_final_supply_if_rec=supply_after_rec
    hyp_final_supply_if_1flm=supply_after_1; hyp_final_supply_if_2flm=supply_after_2; hyp_final_supply_if_3flm=supply_after_3
    hyp_final_supply_if_rec_to_proj_ratio=_safe_div(hyp_final_supply_if_rec, np.maximum(proj,1),0)
    hyp_final_supply_if_rec_to_velocity_ratio=_safe_div(hyp_final_supply_if_rec, np.maximum(weighted_velocity,1),0)
    hyp_final_supply_if_1flm_to_proj_ratio=_safe_div(hyp_final_supply_if_1flm, np.maximum(proj,1),0)

    # 4. Recommendation trust score.
    proj_rec_gap_flms=np.abs(rec-proj)/flm
    proj_rec_agreement=np.clip(1.0-(proj_rec_gap_flms/2.0),0,1)
    rec_supported_by_velocity=(weighted_velocity+flm>=rec).astype(float)
    dc_can_cover_rec=(dc>=rec).astype(float)
    rec_trust_score=np.clip(0.30*proj_rec_agreement+0.25*demand_share+0.20*rec_supported_by_velocity+0.15*dc_can_cover_rec-0.10*single_risk,0,1)
    rec_trust_high_flag=(rec_trust_score>=0.70).astype(float)
    rec_trust_low_flag=(rec_trust_score<=0.35).astype(float)
    rec_cut_candidate=((rec>shortage+flm)&(rec_trust_score<0.50)).astype(float)
    rec_follow_candidate=((rec>0)&(rec_trust_score>=0.55)&(dc>=np.minimum(rec,flm))).astype(float)
    rec_add_candidate=((proj>rec+flm)&(demand_share>=0.50)&(shortage_flms>=1.5)).astype(float)

    # 5. Sparse-demand guardrails.
    only_ttm_supports_demand=((ttm_month>supply)&(l30<=0)&(d30<=0)&(d60<=0)&(lw<=0)).astype(float)
    only_d60_supports_demand=((d60_month>supply)&(l30<=0)&(d30<=0)&(lw<=0)).astype(float)
    only_proj_supports_demand=((proj>supply)&(l30<=0)&(d30<=0)&(d60<=0)&(lw<=0)).astype(float)
    recent_sales_all_zero=((l30<=0)&(lw<=0)).astype(float)
    recent_sales_weak_but_proj_high=((l30+lw_month<=0.25*proj)&(proj>supply+flm)).astype(float)
    proj_high_but_no_recent_sales=((proj>supply+flm)&(l30<=0)&(lw<=0)).astype(float)
    rec_high_but_no_recent_sales=((rec>=flm)&(l30<=0)&(lw<=0)).astype(float)
    d60_high_l30_low_flag=((d60_month>=supply+flm)&(l30<=0.25*d60_month)).astype(float)
    ttm_high_recent_low_flag=((ttm_month>=supply+flm)&(l30+lw_month<=0.25*ttm_month)).astype(float)
    sparse_guardrail_score=np.clip((only_ttm_supports_demand+only_d60_supports_demand+only_proj_supports_demand+recent_sales_all_zero+rec_high_but_no_recent_sales)/3.0,0,1)

    # 6. Allocation size class features.
    rec_size_class=_bucket(rec_flms,[0.01,1.0,2.0,3.0,5.0,9.0])
    proj_need_size_class=_bucket(np.maximum(proj-supply,0)/flm,[0.01,1.0,2.0,3.0,5.0,9.0])
    shortage_size_class=_bucket(shortage_flms,[0.01,1.0,2.0,3.0,5.0,9.0])
    dc_size_class=_bucket(dc/flm,[0.01,1.0,2.0,3.0,5.0,9.0])
    one_flm_candidate=((shortage_flms>=0.75)&(shortage_flms<1.75)).astype(float)
    two_flm_candidate=((shortage_flms>=1.75)&(shortage_flms<2.75)).astype(float)
    large_alloc_candidate=(shortage_flms>=4).astype(float)

    # 7/14. Site behavior and peer features.
    site_total_shortage=_group_transform(shortage,site,'sum')
    site_total_alloc_rec=_group_transform(rec,site,'sum')
    site_avg_velocity=_group_transform(weighted_velocity,site,'mean')
    site_avg_supply_gap=_group_transform(shortage,site,'mean')
    site_rows_with_positive_rec=_group_transform((rec>0).astype(float),site,'sum')
    site_positive_rec_rate=_group_transform((rec>0).astype(float),site,'mean')
    class_line_total_shortage=_group_transform(shortage,class_line,'sum')
    site_classline_key=site+'|'+class_line
    site_classline_need=_group_transform(shortage,site_classline_key,'sum')
    site_share_of_class_line_need=_safe_div(site_classline_need,class_line_total_shortage,0)
    site_need_percentile_v37=_pct(site_total_shortage,site)
    peer_avg_l30=_group_transform(l30,class_line,'mean'); peer_avg_d30=_group_transform(d30,class_line,'mean'); peer_avg_d60=_group_transform(d60,class_line,'mean')
    peer_avg_supply=_group_transform(supply,class_line,'mean'); peer_avg_proj=_group_transform(proj,class_line,'mean'); peer_avg_rec=_group_transform(rec,class_line,'mean')
    row_l30_vs_peer_avg=_safe_div(l30,peer_avg_l30+1,0); row_proj_vs_peer_avg=_safe_div(proj,peer_avg_proj+1,0); row_rec_vs_peer_avg=_safe_div(rec,peer_avg_rec+1,0); row_supply_gap_vs_peer_avg=shortage-_group_transform(shortage,class_line,'mean')

    # 8/15 Specialist triggers.
    requires_cut_behavior=((rec_cut_candidate>0)|(sparse_guardrail_score>=0.67)|(hyp_final_supply_if_rec_to_velocity_ratio>1.75)).astype(float)
    requires_rescue_behavior=((rec_add_candidate>0)|((rec>0)&(demand_share>=0.67)&(shortage_flms>=1.0)&(dc>0))).astype(float)
    requires_review_rank_behavior=((need_percentile_in_class_line>=0.80)&(class_line_tight_dc_flag>0)).astype(float)
    requires_low_dc_behavior=((dc/flm>0)&(dc/flm<2)).astype(float)
    requires_sparse_demand_behavior=(sparse_guardrail_score>0).astype(float)
    requires_large_alloc_behavior=(large_alloc_candidate>0).astype(float)

    # 9. Review competition score.
    dc_scarcity_adjustment=1.0-class_line_dc_pressure_score
    review_competition_score=np.clip(0.25*need_percentile_in_class_line+0.20*demand_share+0.20*np.clip(shortage_flms/4,0,1)+0.15*np.clip(rank_score/1.2,0,1)+0.10*rec_trust_score+0.10*dc_scarcity_adjustment,0,1)
    review_top_5pct_need=(need_percentile_in_class_line>=0.95).astype(float)
    review_top_10pct_need=(need_percentile_in_class_line>=0.90).astype(float)
    review_above_group_median_need=(need_percentile_in_class_line>=0.50).astype(float)
    review_below_group_median_need=(need_percentile_in_class_line<0.50).astype(float)

    # 10. Allocate cut/rescue flags.
    allocate_cut_candidate=((rec>proj+flm)&(demand_share<0.50)).astype(float)
    allocate_rescue_candidate=((rec>0)&(demand_share>=0.67)&(shortage_flms>=1)&(dc>0)).astype(float)
    allocate_follow_rec_candidate=((rec>0)&(rec_trust_score>=0.60)).astype(float)
    allocate_reduce_to_1flm_candidate=((rec_flms>1.25)&(shortage_flms<=1.75)&(demand_share>=0.34)).astype(float)
    allocate_zero_out_candidate=((rec>0)&(demand_share<=0.17)&(sparse_guardrail_score>=0.34)&(single_risk>0)).astype(float)

    # 12. Rank/demand alignment.
    rank_demand_alignment=np.clip(rank_score,0,1.5)*np.clip(shortage_flms/4,0,1)
    high_rank_high_need=((rank_score>=0.75)&(shortage_flms>=1)).astype(float)
    high_rank_low_need=((rank_score>=0.75)&(shortage_flms<0.5)).astype(float)
    low_rank_high_need=((rank_score<0.35)&(shortage_flms>=1)).astype(float)
    low_rank_low_need=((rank_score<0.35)&(shortage_flms<0.5)).astype(float)

    # 13. Cost-normalized allocation risk.
    cost_percentile_in_class_line=_pct(cost,class_line)
    cost_x_one_flm=cost*flm
    cost_x_rec_flms=cost*np.maximum(rec_flms,0)
    cost_adjusted_shortage=cost_percentile_in_class_line*shortage_flms
    cost_adjusted_oversupply_risk=cost_percentile_in_class_line*np.maximum(hyp_final_supply_if_rec_to_velocity_ratio-1.25,0)

    extra=pd.DataFrame({
        'need_rank_in_class_line':need_rank_in_class_line,'need_percentile_in_class_line_v37':need_percentile_in_class_line,'rec_rank_in_class_line':rec_rank_in_class_line,'velocity_rank_in_class_line':velocity_rank_in_class_line,'shortage_flm_rank_in_class_line':shortage_flm_rank_in_class_line,'rank_adjusted_need_percentile':rank_adjusted_need_percentile,'top5_need_in_class_line':top5_need_in_class_line,'top10_need_in_class_line':top10_need_in_class_line,'above_median_need_in_class_line':above_median_need_in_class_line,
        'class_line_total_shortage_flms_v37':total_shortage_flms,'class_line_total_alloc_rec_flms':total_alloc_rec_flms,'class_line_total_proj_gap_flms':total_proj_gap_flms,'class_line_dc_to_total_shortage':class_line_dc_to_total_shortage,'class_line_dc_to_total_rec':class_line_dc_to_total_rec,'class_line_dc_pressure_score':class_line_dc_pressure_score,'class_line_rows_competing_for_dc':class_line_rows_competing_for_dc,'class_line_tight_dc_flag':class_line_tight_dc_flag,'class_line_abundant_dc_flag':class_line_abundant_dc_flag,
        'supply_after_3flm':supply_after_3,'gap_after_1flm_to_proj':gap_after_1flm_to_proj,'gap_after_2flm_to_proj':gap_after_2flm_to_proj,'gap_after_3flm_to_proj':gap_after_3flm_to_proj,'gap_after_1flm_to_velocity':gap_after_1flm_to_velocity,'gap_after_2flm_to_velocity':gap_after_2flm_to_velocity,'gap_after_3flm_to_velocity':gap_after_3flm_to_velocity,'oversupply_after_3flm':oversupply_after_3flm,'hyp_final_supply_if_rec':hyp_final_supply_if_rec,'hyp_final_supply_if_1flm':hyp_final_supply_if_1flm,'hyp_final_supply_if_2flm':hyp_final_supply_if_2flm,'hyp_final_supply_if_3flm':hyp_final_supply_if_3flm,'hyp_final_supply_if_rec_to_proj_ratio':hyp_final_supply_if_rec_to_proj_ratio,'hyp_final_supply_if_rec_to_velocity_ratio':hyp_final_supply_if_rec_to_velocity_ratio,'hyp_final_supply_if_1flm_to_proj_ratio':hyp_final_supply_if_1flm_to_proj_ratio,
        'rec_trust_score':rec_trust_score,'rec_trust_high_flag':rec_trust_high_flag,'rec_trust_low_flag':rec_trust_low_flag,'rec_cut_candidate':rec_cut_candidate,'rec_follow_candidate':rec_follow_candidate,'rec_add_candidate':rec_add_candidate,
        'only_ttm_supports_demand':only_ttm_supports_demand,'only_d60_supports_demand':only_d60_supports_demand,'only_proj_supports_demand':only_proj_supports_demand,'recent_sales_all_zero':recent_sales_all_zero,'recent_sales_weak_but_proj_high':recent_sales_weak_but_proj_high,'proj_high_but_no_recent_sales':proj_high_but_no_recent_sales,'rec_high_but_no_recent_sales':rec_high_but_no_recent_sales,'d60_high_l30_low_flag':d60_high_l30_low_flag,'ttm_high_recent_low_flag':ttm_high_recent_low_flag,'sparse_guardrail_score':sparse_guardrail_score,
        'rec_size_class':rec_size_class,'proj_need_size_class':proj_need_size_class,'shortage_size_class':shortage_size_class,'dc_size_class':dc_size_class,'one_flm_candidate':one_flm_candidate,'two_flm_candidate':two_flm_candidate,'large_alloc_candidate':large_alloc_candidate,
        'site_total_shortage':site_total_shortage,'site_total_alloc_rec':site_total_alloc_rec,'site_avg_velocity':site_avg_velocity,'site_avg_supply_gap':site_avg_supply_gap,'site_rows_with_positive_rec':site_rows_with_positive_rec,'site_positive_rec_rate':site_positive_rec_rate,'site_share_of_class_line_need':site_share_of_class_line_need,'site_need_percentile_v37':site_need_percentile_v37,'peer_avg_l30':peer_avg_l30,'peer_avg_d30':peer_avg_d30,'peer_avg_d60':peer_avg_d60,'peer_avg_supply':peer_avg_supply,'peer_avg_proj':peer_avg_proj,'peer_avg_rec':peer_avg_rec,'row_l30_vs_peer_avg':row_l30_vs_peer_avg,'row_proj_vs_peer_avg':row_proj_vs_peer_avg,'row_rec_vs_peer_avg':row_rec_vs_peer_avg,'row_supply_gap_vs_peer_avg':row_supply_gap_vs_peer_avg,
        'requires_cut_behavior':requires_cut_behavior,'requires_rescue_behavior':requires_rescue_behavior,'requires_review_rank_behavior':requires_review_rank_behavior,'requires_low_dc_behavior':requires_low_dc_behavior,'requires_sparse_demand_behavior':requires_sparse_demand_behavior,'requires_large_alloc_behavior':requires_large_alloc_behavior,
        'review_competition_score':review_competition_score,'review_top_5pct_need':review_top_5pct_need,'review_top_10pct_need':review_top_10pct_need,'review_above_group_median_need':review_above_group_median_need,'review_below_group_median_need':review_below_group_median_need,
        'allocate_cut_candidate':allocate_cut_candidate,'allocate_rescue_candidate':allocate_rescue_candidate,'allocate_follow_rec_candidate':allocate_follow_rec_candidate,'allocate_reduce_to_1flm_candidate':allocate_reduce_to_1flm_candidate,'allocate_zero_out_candidate':allocate_zero_out_candidate,
        'rank_demand_alignment':rank_demand_alignment,'high_rank_high_need':high_rank_high_need,'high_rank_low_need':high_rank_low_need,'low_rank_high_need':low_rank_high_need,'low_rank_low_need':low_rank_low_need,
        'cost_percentile_in_class_line':cost_percentile_in_class_line,'cost_x_one_flm':cost_x_one_flm,'cost_x_rec_flms':cost_x_rec_flms,'cost_adjusted_shortage':cost_adjusted_shortage,'cost_adjusted_oversupply_risk':cost_adjusted_oversupply_risk,
    })
    return pd.concat([base,extra],axis=1).replace([np.inf,-np.inf],0).fillna(0)


DEFAULT_PARAMS={
    'allocate_competition_clamp_enabled': True,
    'allocate_zero_sparse_enabled': True,
    'allocate_context_rescue_enabled': True,
    'review_competition_rescue_enabled': False,
    'clamp_sparse_score_min': 0.67,
    'clamp_rec_trust_max': 0.42,
    'clamp_over_ratio_min': 1.70,
    'rescue_rec_trust_min': 0.62,
    'rescue_need_percentile_min': 0.80,
    'rescue_demand_share_min': 0.67,
    'review_keep_original': True,
}


def load_v37_params(path=None):
    params=dict(DEFAULT_PARAMS)
    if path and os.path.exists(path):
        try: params.update(json.load(open(path)))
        except Exception: pass
    return params


def save_v37_params(path, params=None):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    json.dump({**DEFAULT_PARAMS, **(params or {})}, open(path,'w'), indent=2)


def apply_v37_competition_aware(df: pd.DataFrame, base_audit: pd.DataFrame, params=None) -> pd.DataFrame:
    """Light deterministic v3.7 layer for pre-retrain smoke testing.

    This deliberately changes only obvious rows; the full feature value is intended to be learned
    by the retrained v3.7 neural models.
    """
    params=load_v37_params(None) if params is None else {**DEFAULT_PARAMS, **dict(params)}
    out=base_audit.copy()
    pred=pd.to_numeric(out.get('Predicted Final Alloc',0),errors='coerce').fillna(0).to_numpy(float)
    conf=pd.to_numeric(out.get('Allocation Confidence',0),errors='coerce').fillna(0).to_numpy(float)
    flm=np.maximum(_num(df,'FLM',1.0),1.0); dc=_num(df,'Dc Avail')
    flag=df['Flag'].astype(str).str.upper()
    alloc=(flag.str.contains('ALLOC',na=False)&~flag.str.contains('NO',na=False)).to_numpy(bool)
    review=flag.str.contains('REVIEW',na=False).to_numpy(bool)
    feat=competition_feature_frame(df,pred,conf)

    # 1. Cut obvious weak/sparse over-allocation candidates.
    sparse_cut=(params['allocate_zero_sparse_enabled'] and alloc & (pred>0) &
        (feat['sparse_guardrail_score'].to_numpy(float)>=params['clamp_sparse_score_min']) &
        (feat['rec_trust_score'].to_numpy(float)<=params['clamp_rec_trust_max']) &
        (feat['hyp_final_supply_if_1flm_to_proj_ratio'].to_numpy(float)>=params['clamp_over_ratio_min']))
    competition_cut=(params['allocate_competition_clamp_enabled'] and alloc & (pred>0) &
        (feat['allocate_cut_candidate'].to_numpy(float)>0) &
        (feat['need_percentile_in_class_line_v37'].to_numpy(float)<0.50) &
        (feat['class_line_tight_dc_flag'].to_numpy(float)>0))
    pred=np.where(sparse_cut|competition_cut, np.maximum(pred-flm,0), pred)

    # 2. Rescue strong Allocate rows that the model blanked, but only when multiple signals agree.
    rescue=(params['allocate_context_rescue_enabled'] and alloc & (pred<=0) & (dc>0) &
        (feat['allocate_rescue_candidate'].to_numpy(float)>0) &
        (feat['rec_trust_score'].to_numpy(float)>=params['rescue_rec_trust_min']) &
        (feat['need_percentile_in_class_line_v37'].to_numpy(float)>=params['rescue_need_percentile_min']) &
        (feat['demand_signal_share_above_supply'].to_numpy(float)>=params['rescue_demand_share_min']))
    pred=np.where(rescue, np.minimum(flm,dc), pred)

    # Review untouched by default; optionally rescue only top competition rows.
    if params.get('review_competition_rescue_enabled', False):
        rrescue=review & (pred<=0) & (dc>0) & (feat['review_competition_score'].to_numpy(float)>=0.82) & (feat['review_top_10pct_need'].to_numpy(float)>0)
        pred=np.where(rrescue, np.minimum(flm,dc), pred)

    rounded=np.where(pred>0,np.round(pred/flm)*flm,0.0)
    rounded=np.minimum(rounded,dc)
    rounded=np.where((rounded<=0)&(pred>0)&(dc>0)&(dc<flm),dc,rounded)
    rounded=np.where(rounded<=0,np.nan,rounded)
    before=pd.to_numeric(base_audit.get('Predicted Final Alloc',0),errors='coerce').fillna(0).to_numpy(float)
    out['Predicted Final Alloc']=pd.Series(rounded).where(~np.isnan(rounded),'')
    out['v3_7_competition_adjusted']=(np.nan_to_num(rounded,nan=0.0)!=before).astype(int)
    out['v3_7_adjustment_reason']=np.where(sparse_cut,'sparse_competition_clamp',np.where(competition_cut,'low_rank_tight_dc_clamp',np.where(rescue,'competition_rescue','v36_kept')))
    out['v3_7_rec_trust_score']=feat['rec_trust_score'].to_numpy(float)
    out['v3_7_review_competition_score']=feat['review_competition_score'].to_numpy(float)
    out['v3_7_need_percentile_in_class_line']=feat['need_percentile_in_class_line_v37'].to_numpy(float)
    return out
