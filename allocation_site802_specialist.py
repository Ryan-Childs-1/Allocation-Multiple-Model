from __future__ import annotations
import os, json
import numpy as np, pandas as pd
import allocation_v33_enhancements as enh

def _num(s, default=0.0):
    return pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(default).astype(float).to_numpy()

FEATURES = ['intercept', 'base_units', 'base_confidence', 'base_flms', 'rec_units', 'rec_flms', 'proj_units', 'proj_flms', 'supply', 'dc_avail', 'dc_flms', 'flm', 'cost_log1p', 'l30', 'd30', 'd60_month', 'lw_month', 'ttm_month', 'weighted_velocity', 'sheet_need', 'max_demand_signal', 'demand_consensus', 'high_conf_demand_share', 'shortage_units', 'shortage_flms', 'supply_to_sheet_need', 'post_base_to_sheet_need', 'post_rec_to_sheet_need', 'rec_minus_proj_flms', 'proj_rec_agreement', 'rank_score', 'dc_raw_1_25', 'dc_raw_26_50', 'dc_raw_51_100', 'dc_raw_101_300', 'dc_raw_301_600', 'dc_raw_601_1000', 'dc_raw_1001_2000', 'dc_raw_2000_plus']

def _segments(df):
    f=df['Flag'].astype(str).str.upper()
    alloc=(f.str.contains('ALLOC',na=False)&~f.str.contains('NO',na=False)).to_numpy()
    review=f.str.contains('REVIEW',na=False).to_numpy()
    return alloc, review

def _site802(df):
    return df['Site'].astype(str).str.replace('.0','',regex=False).str.strip().eq('802').to_numpy()

def _feature_matrix(df, base_audit):
    base=_num(base_audit['Predicted Final Alloc'])
    conf=_num(base_audit.get('Allocation Confidence', pd.Series([0]*len(df))))
    feat=enh.enhanced_feature_frame(df, base, conf)
    flm=np.maximum(_num(df['FLM']),1.0)
    d={'intercept':np.ones(len(df)),'base_units':base,'base_confidence':conf,'base_flms':base/flm,
       'rec_units':_num(df['Alloc. Rec.']),'rec_flms':_num(df['Alloc. Rec.'])/flm,
       'proj_units':_num(df['Proj. Demand']),'proj_flms':_num(df['Proj. Demand'])/flm,
       'supply':_num(df['Supply']),'dc_avail':_num(df['Dc Avail']),'dc_flms':_num(df['Dc Avail'])/flm,
       'flm':flm,'cost_log1p':np.log1p(np.maximum(_num(df['Cost']),0)),
       'l30':_num(df['L30']),'d30':_num(df['D30']),'d60_month':_num(df['D60'])/2.0,
       'lw_month':_num(df['LW'])*4.29,'ttm_month':_num(df['TTM'])/12.0}
    X=np.column_stack([d[c] if c in d else _num(feat[c]) if c in feat.columns else np.zeros(len(df)) for c in FEATURES])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def load_site802_model(path):
    if not os.path.exists(path): return None
    z=np.load(path, allow_pickle=True); man=json.loads(str(z['manifest'].item()))
    m={'features':list(z['features']),'version':str(z['version'].item()),'segments':{}}
    for seg,meta in man['segments'].items():
        mm=dict(meta)
        if mm.get('enabled'):
            mm['clf']={k:z[f'{seg}__clf__{k}'] for k in ['coef','mean','std']}
            mm['reg']={k:z[f'{seg}__reg__{k}'] for k in ['coef','mean','std']}
        m['segments'][seg]=mm
    return m

def _pred_ridge(m,X):
    return ((X-m['mean'])/m['std'])@m['coef']

def apply_site802_specialist(df, base_audit, model):
    out=base_audit.copy()
    if model is None: 
        out['Site 802 Specialist Applied']=0
        return out
    pred=_num(out['Predicted Final Alloc']).copy()
    site=_site802(df); alloc,review=_segments(df)
    for seg,mask in [('allocate',site&alloc),('review',site&review)]:
        sm=model.get('segments',{}).get(seg,{})
        if not sm.get('enabled') or mask.sum()==0: continue
        sub=df.loc[mask].reset_index(drop=True); aud=base_audit.loc[mask].reset_index(drop=True)
        X=_feature_matrix(sub,aud)
        prob=np.clip(_pred_ridge(sm['clf'],X),0,1); packs=np.maximum(_pred_ridge(sm['reg'],X),0)
        flm=np.maximum(_num(sub['FLM']),1.0); dc=_num(sub['Dc Avail'])
        units=np.where(prob>=sm['threshold'], np.round(packs)*flm, 0.0)
        units=np.minimum(units,dc)
        units=np.where((units<=0)&(prob>=sm['threshold'])&(packs>0)&(dc>0)&(dc<flm),dc,units)
        idx=np.where(mask)[0]
        pred[idx]=units
        out.loc[idx,'Site 802 Specialist Probability']=prob
        out.loc[idx,'Site 802 Specialist Raw FLMs']=packs
        out.loc[idx,'Site 802 Specialist Applied']=1
    out['Predicted Final Alloc']=pd.Series(pred).where(pred>0,'')
    out['Site 802 Specialist Applied']=out.get('Site 802 Specialist Applied', pd.Series([0]*len(out))).fillna(0).astype(int)
    return out
