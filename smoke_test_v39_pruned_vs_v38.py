"""Run v3.9 Feature-Pruned AK smoke test against v3.8 AK Specialist.

Usage:
    python smoke_test_v39_pruned_vs_v38.py "/mnt/data/Daily Allocation*.xlsb" --nrows 3500 --output-dir reports_v39
"""
from __future__ import annotations
import argparse, glob, os, json
import numpy as np, pandas as pd
import allocation_split_numpy_core as core
import allocation_v35_pruned_enhancements as v35
import allocation_v36_context_features as v36
import allocation_v37_competition_features as v37
import allocation_site802_specialist as site802
import allocation_ak_specialist as ak_specialist
import allocation_v39_feature_pruned_ak as v39
from smoke_test_v35_vs_v34 import load_bundle, read_file, drop_rows, metric_rows
APP_DIR=os.path.dirname(__file__)
ART=os.path.join(APP_DIR,'compact_artifacts')
AK_SITES={'248','159','212','145','121'}

def segment_for_flags(flags):
    f=flags.astype(str).str.upper()
    return np.where(f.str.contains('ALLOC',na=False)&~f.str.contains('NO',na=False),'Allocate',np.where(f.str.contains('REVIEW',na=False),'Review','Other'))

def add_weighted_summaries(dfd):
    dfd=dfd.copy(); dfd['segment']=segment_for_flags(dfd['Flag'])
    sites=dfd['Site'].astype(str).str.replace('.0','',regex=False).str.strip()
    dfd['site802']=sites.eq('802')
    dfd['ak_site']=sites.isin(AK_SITES)
    weighted=[]
    for model, g0 in dfd.groupby('model'):
        groups=[('All',g0),('Allocate',g0[g0.segment=='Allocate']),('Review',g0[g0.segment=='Review']),('AK Stores',g0[g0.ak_site]),('AK Allocate',g0[g0.ak_site & (g0.segment=='Allocate')]),('AK Review',g0[g0.ak_site & (g0.segment=='Review')]),('Non-AK',g0[~g0.ak_site]),('Site 802',g0[g0.site802])]
        for seg,g in groups:
            if len(g)==0: continue
            weighted.append({'model':model,'segment':seg,'rows':len(g),'mae_weighted':float(g.abs_error.mean()),'exact_weighted':float((g.pred==g.actual).mean()),'false_pos':int(((g.pred>0)&(g.actual<=0)).sum()),'false_neg':int(((g.pred<=0)&(g.actual>0)).sum()),'pred_units':float(g.pred.sum()),'actual_units':float(g.actual.sum()),'unit_delta':float(g.pred.sum()-g.actual.sum())})
    return pd.DataFrame(weighted)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('pattern'); ap.add_argument('--nrows',type=int,default=3500); ap.add_argument('--output-dir',default='v39_results')
    args=ap.parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    bundle=load_bundle(); site_model=bundle.get('site802_model')
    ak_model=ak_specialist.load_ak_specialist_model(os.path.join(ART,'ak_specialist_model.npz'))
    p35=v35.load_v35_params(os.path.join(ART,'v35_pruned_params.json'))
    p36=v36.load_v36_params(os.path.join(ART,'v36_context_params.json'))
    p37=v37.load_v37_params(os.path.join(ART,'v37_competition_params.json'))
    p39=v39.load_v39_params(os.path.join(ART,'v39_feature_pruned_params.json'))
    allm=[]; details=[]
    files=sorted(glob.glob(args.pattern))
    for path in files:
        raw=read_file(path,args.nrows)
        df=drop_rows(raw)
        canon=core.canonicalize_columns(df, target_required=True)
        base=core.predict_dataframe(df,bundle,target_required=False)
        audit35=v35.apply_v35_pruned_no_residual(canon,base,p35)
        audit36=v36.apply_v36_context_enhanced(canon,audit35,p36)
        audit36=site802.apply_site802_specialist(canon,audit36,site_model)
        audit37=v37.apply_v37_competition_aware(canon,audit36,p37)
        audit37=site802.apply_site802_specialist(canon,audit37,site_model)
        audit38=ak_specialist.apply_ak_specialist(canon,audit37,ak_model)
        audit39=v39.apply_v39_feature_pruned_ak(canon,audit37,ak_model,p39)
        for model,audit in [('v3_8_ak_specialist',audit38),('v3_9_feature_pruned',audit39)]:
            for m in metric_rows(canon,audit):
                m['model']=model; m['file']=os.path.basename(path); allm.append(m)
            actual=pd.to_numeric(canon[core.TARGET_COL],errors='coerce').fillna(0).to_numpy(float)
            pred=pd.to_numeric(audit['Predicted Final Alloc'],errors='coerce').fillna(0).to_numpy(float)
            details.append(pd.DataFrame({'file':os.path.basename(path),'model':model,'Flag':canon['Flag'].astype(str).values,'Site':canon['Site'].astype(str).values,'actual':actual,'pred':pred,'abs_error':np.abs(pred-actual)}))
    dfm=pd.DataFrame(allm)
    dfd=pd.concat(details,ignore_index=True) if details else pd.DataFrame()
    summary=dfm.groupby(['model','segment']).agg(rows=('rows','sum'),mae_file_avg=('mae','mean'),exact_file_avg=('exact','mean'),false_pos=('false_pos','sum'),false_neg=('false_neg','sum'),pred_units=('pred_units','sum'),actual_units=('actual_units','sum')).reset_index()
    summary['unit_delta']=summary['pred_units']-summary['actual_units']
    weighted=add_weighted_summaries(dfd) if len(dfd) else pd.DataFrame()
    dfm.to_csv(os.path.join(args.output_dir,'v39_file_segment_metrics.csv'),index=False)
    summary.to_csv(os.path.join(args.output_dir,'v39_summary_metrics_file_avg.csv'),index=False)
    weighted.to_csv(os.path.join(args.output_dir,'v39_summary_metrics_weighted.csv'),index=False)
    dfd.to_csv(os.path.join(args.output_dir,'v39_prediction_detail.csv'),index=False)
    piv=weighted.pivot(index='segment', columns='model', values='mae_weighted').reset_index() if len(weighted) else pd.DataFrame()
    if len(piv) and 'v3_8_ak_specialist' in piv and 'v3_9_feature_pruned' in piv:
        piv['mae_delta_v39_minus_v38']=piv['v3_9_feature_pruned']-piv['v3_8_ak_specialist']
        piv.to_csv(os.path.join(args.output_dir,'v39_mae_delta_vs_v38.csv'), index=False)
    # detail of changed rows v3.8 -> v3.9
    if len(dfd):
        tmp=dfd.copy(); tmp['seq']=tmp.groupby(['file','model']).cumcount()
        wide=tmp.pivot_table(index=['file','seq','Flag','Site','actual'], columns='model', values='pred', aggfunc='first').reset_index(); wide.columns.name=None
        if 'v3_8_ak_specialist' in wide and 'v3_9_feature_pruned' in wide:
            wide['pred_delta_v39_minus_v38']=wide['v3_9_feature_pruned']-wide['v3_8_ak_specialist']
            wide['err38']=(wide['v3_8_ak_specialist']-wide['actual']).abs(); wide['err39']=(wide['v3_9_feature_pruned']-wide['actual']).abs(); wide['error_delta_v39_minus_v38']=wide['err39']-wide['err38']
            wide[wide['pred_delta_v39_minus_v38']!=0].to_csv(os.path.join(args.output_dir,'v39_changed_rows_vs_v38.csv'),index=False)
    md=[]
    md.append('# v3.9 Feature-Pruned AK Smoke Test Report\n')
    md.append('Compared **v3.9 Feature-Pruned AK + Site 802** against **v3.8 AK Specialist** on the provided allocation workbooks. v3.9 prunes weak AK Review rescue features/rules while preserving the successful AK Allocate rescue path and the Site 802 specialist.\n')
    md.append(f'Files tested: {len(files)}\n')
    md.append('## Pruning parameters\n')
    md.append('```json\n'+json.dumps(p39,indent=2)+'\n```\n')
    md.append('## Weighted row-level summary\n')
    md.append(weighted.to_markdown(index=False) if len(weighted) else 'No rows.')
    md.append('\n\n## MAE delta, v3.9 minus v3.8\n')
    md.append(piv.to_markdown(index=False) if len(piv) else 'No delta table.')
    md.append('\n\n## File-average summary\n')
    md.append(summary.to_markdown(index=False) if len(summary) else 'No rows.')
    open(os.path.join(args.output_dir,'V39_FEATURE_PRUNED_AK_SMOKE_TEST_REPORT.md'),'w').write('\n'.join(md))
    print(weighted.to_string(index=False) if len(weighted) else summary.to_string(index=False))

if __name__=='__main__': main()
