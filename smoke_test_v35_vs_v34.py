"""Run the v3.5 feature-pruned smoke test against v3.4.

Usage:
    python smoke_test_v35_vs_v34.py "/path/to/Daily Allocation*.xlsb" --nrows 3500 --output-dir reports
"""
from __future__ import annotations
import argparse, glob, os, sys, json
import numpy as np, pandas as pd
import allocation_split_numpy_core as core
import allocation_v33_enhancements as old_enh
import allocation_v35_pruned_enhancements as new_enh
import allocation_site802_specialist as site802

APP_DIR=os.path.dirname(__file__)
ART=os.path.join(APP_DIR,'compact_artifacts')

def _model_from_compact(z, role):
    meta=json.loads(str(z[f'{role}__meta'].item()))
    model=core.NumpyMLP(meta['input_dim'], meta['output_dim'], tuple(meta['hidden']), meta['task'], dropout=meta.get('dropout',0.0))
    n=len(meta['hidden'])+1
    model.W=[z[f'{role}__W{i}'].astype(np.float32) for i in range(n)]
    model.b=[z[f'{role}__b{i}'].astype(np.float32) for i in range(n)]
    model.mW=[np.zeros_like(w) for w in model.W]; model.vW=[np.zeros_like(w) for w in model.W]
    model.mb=[np.zeros_like(b) for b in model.b]; model.vb=[np.zeros_like(b) for b in model.b]
    return model

def load_bundle():
    meta=json.load(open(os.path.join(ART,'model_config.json')))
    fc=meta['feature_config']
    feat_cfg=core.FeatureConfig(hash_dim_class=fc.get('hash_dim_class',96),hash_dim_line=fc.get('hash_dim_line',128),hash_dim_site=fc.get('hash_dim_site',96),hash_dim_rank=fc.get('hash_dim_rank',8),hash_dim_flag=fc.get('hash_dim_flag',8),hash_dim_dc_bucket=fc.get('hash_dim_dc_bucket',8),hash_dim_raw_dc_bucket=fc.get('hash_dim_raw_dc_bucket',12),numeric_mean=fc.get('numeric_mean'),numeric_std=fc.get('numeric_std'),feature_names=fc.get('feature_names'))
    models={}
    for seg in ['allocate','review']:
        z=np.load(os.path.join(ART,f'{seg}_model.npz'), allow_pickle=True)
        clf=_model_from_compact(z,'classifier'); reg=_model_from_compact(z,'regressor')
        models[seg]={'classifier':clf,'regressor':reg,'classifiers':[clf],'regressors':[reg],'residual':None}
    return {'meta':meta,'feature_config':feat_cfg,'models':models,'site802_model':site802.load_site802_model(os.path.join(ART,'site802_specialist_model.npz'))}

def safe_str(x):
    try:
        if pd.isna(x): return ''
    except Exception: pass
    return str(x)

def find_header_row(df):
    wanted=set(core.ALLOWED_FEATURES+[core.TARGET_COL]); best_i,best_hits=0,-1
    for i in range(min(len(df),90)):
        hits=sum(1 for v in df.iloc[i].tolist() if core.CANONICAL_ALIASES.get(core._norm_name(v)) in wanted)
        if hits>best_hits: best_i,best_hits=i,hits
    return best_i if best_hits>=8 else 0

def read_file(path, nrows):
    xl=pd.ExcelFile(path, engine='pyxlsb')
    sheet='3.3 Working Table' if '3.3 Working Table' in xl.sheet_names else xl.sheet_names[0]
    preview=pd.read_excel(path, sheet_name=sheet, engine='pyxlsb', header=None, nrows=90)
    hdr=find_header_row(preview)
    return pd.read_excel(path, sheet_name=sheet, engine='pyxlsb', header=hdr, nrows=nrows)

def drop_rows(df):
    work=df.dropna(how='all').copy()
    row_text=work.apply(lambda r:'|'.join(safe_str(v) for v in r.to_numpy()), axis=1).str.upper()
    rep=row_text.str.contains('FINAL ALLOC',na=False)&row_text.str.contains('ALLOC',na=False)&row_text.str.contains('FLAG',na=False)
    work=work.loc[~rep].reset_index(drop=True)
    canon=core.canonicalize_columns(work,target_required=True)
    flags=canon['Flag'].astype(str).str.upper()
    keep=(flags.str.contains('ALLOC',na=False)&~flags.str.contains('NO',na=False))|flags.str.contains('REVIEW',na=False)
    return work.loc[keep.to_numpy()].reset_index(drop=True)

def metric_rows(canon,audit):
    actual=pd.to_numeric(canon[core.TARGET_COL],errors='coerce').fillna(0).to_numpy(float)
    pred=pd.to_numeric(audit['Predicted Final Alloc'],errors='coerce').fillna(0).to_numpy(float)
    flag=canon['Flag'].astype(str).str.upper()
    alloc=(flag.str.contains('ALLOC',na=False)&~flag.str.contains('NO',na=False)).to_numpy()
    review=flag.str.contains('REVIEW',na=False).to_numpy()
    site802=canon['Site'].astype(str).str.replace('.0','',regex=False).str.strip().eq('802').to_numpy()
    out=[]
    for name,mask in [('All',np.ones(len(canon),bool)),('Allocate',alloc),('Review',review),('Site 802',site802),('Site 802 Allocate',site802&alloc),('Site 802 Review',site802&review),('Non-802',~site802)]:
        if mask.sum()==0: continue
        a=actual[mask]; p=pred[mask]
        out.append({'segment':name,'rows':int(mask.sum()),'mae':float(np.mean(np.abs(p-a))),'exact':float(np.mean(p==a)),'false_pos':int(((p>0)&(a<=0)).sum()),'false_neg':int(((p<=0)&(a>0)).sum()),'pred_units':float(p.sum()),'actual_units':float(a.sum()),'unit_delta':float(p.sum()-a.sum())})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('pattern'); ap.add_argument('--nrows',type=int,default=3500); ap.add_argument('--output-dir',default='reports')
    args=ap.parse_args(); os.makedirs(args.output_dir,exist_ok=True)
    bundle=load_bundle(); old_params=old_enh.load_v33_params(os.path.join(ART,'v33_no_residual_params.json')); new_params=new_enh.load_v35_params(os.path.join(ART,'v35_pruned_params.json'))
    allm=[]
    for path in glob.glob(args.pattern):
        df=drop_rows(read_file(path,args.nrows)); canon=core.canonicalize_columns(df,target_required=True); base=core.predict_dataframe(df,bundle,target_required=False)
        for model,mod,params in [('v3_4_older',old_enh,old_params),('v3_5_pruned',new_enh,new_params)]:
            audit=mod.apply_v33_no_residual(canon,base,params) if model=='v3_4_older' else mod.apply_v35_pruned_no_residual(canon,base,params)
            audit=site802.apply_site802_specialist(canon,audit,bundle.get('site802_model'))
            for m in metric_rows(canon,audit): m['model']=model; m['file']=os.path.basename(path); allm.append(m)
    dfm=pd.DataFrame(allm)
    summary=dfm.groupby(['model','segment']).agg(rows=('rows','sum'),mae=('mae','mean'),exact=('exact','mean'),false_pos=('false_pos','sum'),false_neg=('false_neg','sum'),pred_units=('pred_units','sum'),actual_units=('actual_units','sum')).reset_index()
    summary['unit_delta']=summary['pred_units']-summary['actual_units']
    dfm.to_csv(os.path.join(args.output_dir,'v35_pruned_file_segment_metrics.csv'),index=False)
    summary.to_csv(os.path.join(args.output_dir,'v35_pruned_summary_metrics.csv'),index=False)
    print(summary.to_string())
if __name__=='__main__': main()
