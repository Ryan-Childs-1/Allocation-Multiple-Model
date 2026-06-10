"""Run v3.6 Context-Enhanced smoke test against v3.5 Feature-Pruned.

Usage:
    python smoke_test_v36_vs_v35.py "/path/to/Daily Allocation*.xlsb" --nrows 3500 --output-dir reports
"""
from __future__ import annotations
import argparse, glob, os, sys, json
import numpy as np, pandas as pd
import allocation_split_numpy_core as core
import allocation_v35_pruned_enhancements as v35
import allocation_v36_context_features as v36
import allocation_site802_specialist as site802
from smoke_test_v35_vs_v34 import load_bundle, read_file, drop_rows, metric_rows
APP_DIR=os.path.dirname(__file__)
ART=os.path.join(APP_DIR,'compact_artifacts')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('pattern'); ap.add_argument('--nrows',type=int,default=3500); ap.add_argument('--output-dir',default='reports')
    args=ap.parse_args(); os.makedirs(args.output_dir,exist_ok=True)
    bundle=load_bundle(); site_model=bundle.get('site802_model')
    p35=v35.load_v35_params(os.path.join(ART,'v35_pruned_params.json'))
    p36=v36.load_v36_params(os.path.join(ART,'v36_context_params.json'))
    allm=[]; detail=[]
    for path in glob.glob(args.pattern):
        df=drop_rows(read_file(path,args.nrows)); canon=core.canonicalize_columns(df,target_required=True); base=core.predict_dataframe(df,bundle,target_required=False)
        for model in ['v3_5_pruned','v3_6_context_enhanced']:
            if model=='v3_5_pruned':
                audit=v35.apply_v35_pruned_no_residual(canon,base,p35)
            else:
                prior=v35.apply_v35_pruned_no_residual(canon,base,p35)
                audit=v36.apply_v36_context_enhanced(canon,prior,p36)
            audit=site802.apply_site802_specialist(canon,audit,site_model)
            for m in metric_rows(canon,audit):
                m['model']=model; m['file']=os.path.basename(path); allm.append(m)
            actual=pd.to_numeric(canon[core.TARGET_COL],errors='coerce').fillna(0).to_numpy(float)
            pred=pd.to_numeric(audit['Predicted Final Alloc'],errors='coerce').fillna(0).to_numpy(float)
            flags=canon['Flag'].astype(str).values
            detail.append(pd.DataFrame({'file':os.path.basename(path),'model':model,'Flag':flags,'Site':canon['Site'].astype(str).values,'actual':actual,'pred':pred,'abs_error':np.abs(pred-actual)}))
    dfm=pd.DataFrame(allm)
    # Store both weighted and file-average metrics.
    summary=dfm.groupby(['model','segment']).agg(rows=('rows','sum'),mae_file_avg=('mae','mean'),exact_file_avg=('exact','mean'),false_pos=('false_pos','sum'),false_neg=('false_neg','sum'),pred_units=('pred_units','sum'),actual_units=('actual_units','sum')).reset_index()
    summary['unit_delta']=summary['pred_units']-summary['actual_units']
    dfd=pd.concat(detail,ignore_index=True)
    weighted=[]
    for (model,seg), g in dfd.assign(segment=lambda x: np.where(x.Flag.astype(str).str.upper().str.contains('ALLOC') & ~x.Flag.astype(str).str.upper().str.contains('NO'), 'Allocate', np.where(x.Flag.astype(str).str.upper().str.contains('REVIEW'),'Review','Other'))).groupby(['model','segment']):
        weighted.append({'model':model,'segment':seg,'rows':len(g),'mae_weighted':float(g.abs_error.mean()),'exact_weighted':float((g.pred==g.actual).mean()),'false_pos':int(((g.pred>0)&(g.actual<=0)).sum()),'false_neg':int(((g.pred<=0)&(g.actual>0)).sum()),'pred_units':float(g.pred.sum()),'actual_units':float(g.actual.sum()),'unit_delta':float(g.pred.sum()-g.actual.sum())})
    weighted=pd.DataFrame(weighted)
    dfm.to_csv(os.path.join(args.output_dir,'v36_file_segment_metrics.csv'),index=False)
    summary.to_csv(os.path.join(args.output_dir,'v36_summary_metrics_file_avg.csv'),index=False)
    weighted.to_csv(os.path.join(args.output_dir,'v36_summary_metrics_weighted.csv'),index=False)
    dfd.to_csv(os.path.join(args.output_dir,'v36_prediction_detail.csv'),index=False)
    md=[]
    md.append('# v3.6 Context-Enhanced Smoke Test Report\n')
    md.append('Compared **v3.6 Context-Enhanced** against **v3.5 Feature-Pruned Site 802** on the provided daily allocation workbooks.\n')
    md.append('## File-average summary\n')
    md.append(summary.to_markdown(index=False))
    md.append('\n\n## Weighted row-level summary\n')
    md.append(weighted.to_markdown(index=False))
    open(os.path.join(args.output_dir,'V36_CONTEXT_ENHANCED_SMOKE_TEST_REPORT.md'),'w').write('\n'.join(md))
    print(summary.to_string())

if __name__=='__main__': main()
