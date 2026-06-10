
"""
Allocation Split Expert v3.3 Enhanced-No-Residual.

Keeps the expanded demand/supply feature layer from v3.1, but removes residual
correction entirely. This module is intentionally deterministic and uses only the
approved worksheet columns plus the base v3 model output metadata.
"""
from __future__ import annotations
import json, os, math
from typing import Dict, Any
import numpy as np
import pandas as pd
EPS=1e-6

def _num(df, col, default=0.0):
    return pd.to_numeric(df.get(col, default), errors='coerce').replace([np.inf,-np.inf], np.nan).fillna(default).to_numpy(float)

def _rank_score(x):
    s=str(x).strip().upper()
    return {'A+':1.20,'A':1.00,'B':0.78,'C':0.55,'D':0.30,'E':0.14,'F':0.07}.get(s,0.55)

def flag_segments(df):
    f=df['Flag'].astype(str).str.upper()
    alloc=(f.str.contains('ALLOC',na=False)&~f.str.contains('NO',na=False)).to_numpy(bool)
    review=f.str.contains('REVIEW',na=False).to_numpy(bool)
    return alloc, review

def enhanced_feature_frame(df: pd.DataFrame, base_pred_units=None, base_confidence=None) -> pd.DataFrame:
    """Richer demand/supply features derived only from approved sheet columns."""
    n=len(df)
    flm=np.maximum(_num(df,'FLM',1.0),1.0)
    mil=_num(df,'MIL'); cost=_num(df,'Cost'); l30=_num(df,'L30'); d30=_num(df,'D30'); d60=_num(df,'D60')
    lw=_num(df,'LW'); ttm=_num(df,'TTM'); supply=_num(df,'Supply'); dc=_num(df,'Dc Avail'); proj=_num(df,'Proj. Demand'); rec=_num(df,'Alloc. Rec.')
    if base_pred_units is None: base_pred_units=np.zeros(n,float)
    else: base_pred_units=np.asarray(base_pred_units,float)
    if base_confidence is None: base_confidence=np.zeros(n,float)
    else: base_confidence=np.asarray(base_confidence,float)
    rank=np.array([_rank_score(x) for x in df.get('Rank', pd.Series(['']*n)).values],float)

    lw_month=lw*4.29
    ttm_month=ttm/12.0
    d60_month=d60/2.0
    demand_stack=np.vstack([l30,d30,d60_month,lw_month,ttm_month,proj])

    # Strongly weighted toward original sheet demand/sales features.
    weighted_velocity=(0.26*l30 + 0.23*d30 + 0.19*d60_month + 0.18*lw_month + 0.08*ttm_month + 0.06*proj)
    sheet_need=np.maximum.reduce([proj, d30, d60_month, l30, lw_month, ttm_month])
    max_demand=np.max(demand_stack,axis=0)
    mean_demand=np.mean(demand_stack,axis=0)
    median_demand=np.median(demand_stack,axis=0)
    std_demand=np.std(demand_stack,axis=0)
    demand_consensus=np.mean(demand_stack>0,axis=0)
    demand_above_supply=np.mean(demand_stack>supply,axis=0)
    high_conf_demand=np.mean(demand_stack>=np.maximum(supply,1.0),axis=0)

    trend_l30_d30=l30-d30
    trend_d30_d60=d30-d60_month
    trend_lw_l30=lw_month-l30
    short_accel=(l30+lw_month)/2.0 - (d30+d60_month)/2.0
    ttm_vs_recent=ttm_month - (l30+d30+d60_month)/3.0
    recent_support=((l30>0).astype(float)+(lw>0).astype(float)+(d30>0).astype(float))/3.0
    stable_support=((d60>0).astype(float)+(ttm>0).astype(float)+(proj>0).astype(float))/3.0

    supply_gap_velocity=weighted_velocity-supply
    supply_gap_sheet_need=sheet_need-supply
    supply_gap_d30=d30-supply
    supply_gap_d60=d60_month-supply
    supply_gap_proj=proj-supply
    supply_gap_l30=l30-supply
    shortage_units=np.maximum(supply_gap_velocity,0)
    shortage_sheet_units=np.maximum(supply_gap_sheet_need,0)
    shortage_flms=shortage_units/flm
    shortage_sheet_flms=shortage_sheet_units/flm
    overstock_units=np.maximum(supply-weighted_velocity,0)

    supply_to_weighted=supply/np.maximum(weighted_velocity,1.0)
    supply_to_sheet_need=supply/np.maximum(sheet_need,1.0)
    supply_to_d30=supply/np.maximum(d30,1.0)
    supply_to_d60=supply/np.maximum(d60_month,1.0)
    supply_to_proj=supply/np.maximum(proj,1.0)
    woc_proxy=supply/np.maximum(np.maximum(lw,l30/4.29),0.25)

    rec_flms=rec/flm
    dc_flms=dc/flm
    # Raw Dc Avail buckets requested from Project APE: 25, 50, 100, 300, 600, 1000, 2000, 2000+.
    dc_raw_empty=(dc<=0).astype(float)
    dc_raw_1_25=((dc>0)&(dc<=25)).astype(float)
    dc_raw_26_50=((dc>25)&(dc<=50)).astype(float)
    dc_raw_51_100=((dc>50)&(dc<=100)).astype(float)
    dc_raw_101_300=((dc>100)&(dc<=300)).astype(float)
    dc_raw_301_600=((dc>300)&(dc<=600)).astype(float)
    dc_raw_601_1000=((dc>600)&(dc<=1000)).astype(float)
    dc_raw_1001_2000=((dc>1000)&(dc<=2000)).astype(float)
    dc_raw_2000_plus=(dc>2000).astype(float)
    base_flms=base_pred_units/flm
    post_rec_supply=supply+rec
    post_base_supply=supply+base_pred_units
    post_rec_to_velocity=post_rec_supply/np.maximum(weighted_velocity,1.0)
    post_base_to_velocity=post_base_supply/np.maximum(weighted_velocity,1.0)
    post_rec_to_sheet_need=post_rec_supply/np.maximum(sheet_need,1.0)
    post_base_to_sheet_need=post_base_supply/np.maximum(sheet_need,1.0)
    rec_minus_need_flms=(rec-shortage_units)/flm
    rec_to_need=rec/np.maximum(shortage_units,1.0)
    base_to_rec=base_pred_units/np.maximum(rec,1.0)
    proj_flms=proj/flm
    proj_gap_units=proj-supply
    proj_gap_flms=proj_gap_units/flm
    rec_minus_proj=rec-proj
    rec_minus_proj_flms=rec_minus_proj/flm
    proj_minus_rec_flms=(proj-rec)/flm
    post_rec_to_proj=post_rec_supply/np.maximum(proj,1.0)
    proj_to_rec=proj/np.maximum(rec,1.0)
    rec_to_proj=rec/np.maximum(proj,1.0)
    proj_to_dc=proj/np.maximum(dc,1.0)
    rec_to_dc=rec/np.maximum(dc,1.0)
    dc_after_rec=dc-rec
    dc_after_rec_flms=dc_after_rec/flm
    proj_supported_by_rec = (rec >= np.maximum(proj - 0.25 * flm, 0.0)).astype(float)
    rec_exceeds_proj_1flm = (rec > proj + flm).astype(float)
    proj_exceeds_rec_1flm = (proj > rec + flm).astype(float)
    has_proj_demand = (proj > 0).astype(float)
    has_alloc_rec = (rec > 0).astype(float)
    proj_rec_agreement = 1.0 - np.minimum(np.abs(proj - rec) / np.maximum(np.maximum(proj, rec), 1.0), 1.0)
    rec_above_need=(rec > shortage_units + flm).astype(float)
    need_above_rec=(shortage_units > rec + flm).astype(float)
    weak_demand=((l30<=0)&(d30<=0)&(d60<=0)&(lw<=0)&(ttm<=0)&(proj<=0)).astype(float)
    positive_sheet_signal=((l30>0)|(d30>0)|(d60>0)|(lw>0)|(ttm>0)|(proj>0)).astype(float)

    out=pd.DataFrame({
        'weighted_velocity':weighted_velocity,'sheet_need':sheet_need,'max_demand_signal':max_demand,'mean_demand_signal':mean_demand,
        'median_demand_signal':median_demand,'std_demand_signal':std_demand,'demand_consensus':demand_consensus,
        'demand_above_supply_share':demand_above_supply,'high_conf_demand_share':high_conf_demand,
        'trend_l30_vs_d30':trend_l30_d30,'trend_d30_vs_d60_month':trend_d30_d60,'trend_lw_month_vs_l30':trend_lw_l30,
        'short_term_acceleration':short_accel,'ttm_vs_recent':ttm_vs_recent,'recent_support':recent_support,'stable_support':stable_support,
        'supply_gap_velocity':supply_gap_velocity,'supply_gap_sheet_need':supply_gap_sheet_need,'supply_gap_d30':supply_gap_d30,
        'supply_gap_d60_month':supply_gap_d60,'supply_gap_proj':supply_gap_proj,'supply_gap_l30':supply_gap_l30,
        'shortage_units':shortage_units,'shortage_sheet_units':shortage_sheet_units,'shortage_flms':shortage_flms,
        'shortage_sheet_flms':shortage_sheet_flms,'overstock_units':overstock_units,'supply_to_weighted_velocity':supply_to_weighted,
        'supply_to_sheet_need':supply_to_sheet_need,'supply_to_d30':supply_to_d30,'supply_to_d60_month':supply_to_d60,
        'supply_to_proj':supply_to_proj,'woc_proxy':woc_proxy,'rec_flms':rec_flms,'dc_flms':dc_flms,'base_flms':base_flms,
        'post_rec_to_velocity':post_rec_to_velocity,'post_base_to_velocity':post_base_to_velocity,
        'post_rec_to_sheet_need':post_rec_to_sheet_need,'post_base_to_sheet_need':post_base_to_sheet_need,
        'rec_minus_need_flms':rec_minus_need_flms,'rec_to_need':rec_to_need,'base_to_rec':base_to_rec,
        'rec_above_need':rec_above_need,'need_above_rec':need_above_rec,'weak_demand':weak_demand,
        'positive_sheet_signal':positive_sheet_signal,'rank_score':rank,'cost_log1p':np.log1p(np.maximum(cost,0)),
        'base_confidence':base_confidence,'base_pred_units':base_pred_units,'flm':flm,'dc_avail':dc,'supply':supply,'alloc_rec':rec,
        'pressure_x_rank':shortage_sheet_flms*rank,'pressure_x_cost':shortage_sheet_flms*np.log1p(np.maximum(cost,0)),
        'rec_x_rank':rec_flms*rank,'base_x_rank':base_flms*rank,'dc_x_rank':dc_flms*rank,
        'dc_raw_empty':dc_raw_empty,'dc_raw_1_25':dc_raw_1_25,'dc_raw_26_50':dc_raw_26_50,'dc_raw_51_100':dc_raw_51_100,
        'dc_raw_101_300':dc_raw_101_300,'dc_raw_301_600':dc_raw_301_600,'dc_raw_601_1000':dc_raw_601_1000,
        'dc_raw_1001_2000':dc_raw_1001_2000,'dc_raw_2000_plus':dc_raw_2000_plus,
        'proj_flms':proj_flms,'proj_gap_units':proj_gap_units,'proj_gap_flms':proj_gap_flms,
        'rec_minus_proj':rec_minus_proj,'rec_minus_proj_flms':rec_minus_proj_flms,'proj_minus_rec_flms':proj_minus_rec_flms,
        'post_rec_to_proj':post_rec_to_proj,'proj_to_rec':proj_to_rec,'rec_to_proj':rec_to_proj,
        'proj_to_dc':proj_to_dc,'rec_to_dc':rec_to_dc,'dc_after_rec':dc_after_rec,'dc_after_rec_flms':dc_after_rec_flms,
        'proj_supported_by_rec':proj_supported_by_rec,'rec_exceeds_proj_1flm':rec_exceeds_proj_1flm,
        'proj_exceeds_rec_1flm':proj_exceeds_rec_1flm,'has_proj_demand':has_proj_demand,'has_alloc_rec_signal':has_alloc_rec,
        'proj_rec_agreement':proj_rec_agreement,
    })
    return out.replace([np.inf,-np.inf],0).fillna(0)

DEFAULT_PARAMS={
    # Conservative clamp tuned around original sheet signals, not residuals.
    'allocate_cap_margin_flm': 1.75,
    'review_cap_margin_flm': 1.50,
    'cap_sheet_need_mult': 1.05,
    'cap_velocity_mult': 1.10,
    'cap_consensus_max': 0.84,
    'cap_min_base_over_rec_flms': 1.00,
    'cap_reduce_to_rec_plus_flms': 1.00,
    'rescue_enabled': False,
    'review_keep_original': True,
}

def load_v33_params(path=None):
    params=dict(DEFAULT_PARAMS)
    if path and os.path.exists(path):
        try: params.update(json.load(open(path)))
        except Exception: pass
    return params

def apply_v33_no_residual(df: pd.DataFrame, base_audit: pd.DataFrame, params: Dict[str,Any]|None=None) -> pd.DataFrame:
    """Apply demand/supply enhanced no-residual adjustments to base v3 predictions."""
    params=load_v33_params() if params is None else {**DEFAULT_PARAMS, **dict(params)}
    out=base_audit.copy()
    base=pd.to_numeric(out.get('Predicted Final Alloc',0), errors='coerce').fillna(0).to_numpy(float)
    conf=pd.to_numeric(out.get('Allocation Confidence',0), errors='coerce').fillna(0).to_numpy(float)
    flm=np.maximum(_num(df,'FLM',1.0),1.0); dc=_num(df,'Dc Avail'); supply=_num(df,'Supply'); rec=_num(df,'Alloc. Rec.'); proj=_num(df,'Proj. Demand')
    alloc_mask, review_mask=flag_segments(df)
    pred=base.copy()
    feat=enhanced_feature_frame(df, base, conf)
    sheet_need=feat['sheet_need'].to_numpy(float)
    weighted=feat['weighted_velocity'].to_numpy(float)
    consensus=feat['demand_consensus'].to_numpy(float)
    post_supply=supply+pred

    # Primary improvement: trim only when original base prediction looks above what the sheet's own demand/sales signals support.
    cap_alloc=np.maximum(sheet_need*params['cap_sheet_need_mult'], weighted*params['cap_velocity_mult']) + params['allocate_cap_margin_flm']*flm
    cap_review=np.maximum(sheet_need*params['cap_sheet_need_mult'], weighted*params['cap_velocity_mult']) + params['review_cap_margin_flm']*flm
    cap=np.where(alloc_mask, cap_alloc, cap_review)

    base_over_rec_flms=(pred-rec)/flm
    obvious_over=(pred>0)&(post_supply>cap)&(consensus<=params['cap_consensus_max'])&(base_over_rec_flms>=params['cap_min_base_over_rec_flms'])
    # Preserve Review unless explicitly disabled; v3 Review has historically been strong.
    if params.get('review_keep_original', True):
        obvious_over = obvious_over & (~review_mask)
    # Reduce toward original Alloc Rec + margin rather than toward model residual.
    target_cap=np.maximum(0, rec + params['cap_reduce_to_rec_plus_flms']*flm)
    pred=np.where(obvious_over, np.minimum(pred-flm, target_cap), pred)

    # Secondary clamp: if prediction creates extreme supply-to-need ratio and support is weak, reduce one FLM.
    post2=supply+pred
    extreme=(pred>0)&(post2>np.maximum(sheet_need,weighted)+2.5*flm)&(feat['high_conf_demand_share'].to_numpy(float)<0.50)
    if params.get('review_keep_original', True):
        extreme = extreme & (~review_mask)
    pred=np.where(extreme, np.maximum(pred-flm,0), pred)

    # Optional very strict rescue path; disabled by default because original v3 had low false negatives.
    if params.get('rescue_enabled', False):
        rescue=(pred<=0)&alloc_mask&(dc>0)&(rec>=flm)&(feat['shortage_sheet_flms'].to_numpy(float)>=1.25)&(feat['high_conf_demand_share'].to_numpy(float)>=0.67)&(conf>=0.34)
        pred=np.where(rescue, np.minimum(flm,dc), pred)

    # Safety: FLM rounding and DC cap. Below-FLM allocation is allowed only when it is all that remains in DC.
    rounded=np.where(pred>0, np.round(pred/flm)*flm, 0.0)
    rounded=np.minimum(rounded, dc)
    rounded=np.where((rounded<=0)&(pred>0)&(dc>0)&(dc<flm), dc, rounded)
    rounded=np.where(rounded<=0, np.nan, rounded)
    out['Predicted Final Alloc']=pd.Series(rounded).where(~np.isnan(rounded),'')
    out['v3_3_adjusted_from_base']=np.where(np.nan_to_num(rounded, nan=0.0)!=base, 1, 0)
    out['v3_3_sheet_need']=sheet_need
    out['v3_3_weighted_velocity']=weighted
    out['v3_3_adjustment_reason']=np.where(obvious_over,'demand_supply_oversupply_clamp',np.where(extreme,'extreme_weak_support_clamp','base_v3_kept'))
    return out

def save_params(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({**DEFAULT_PARAMS, **dict(params)}, open(path,'w'), indent=2)
