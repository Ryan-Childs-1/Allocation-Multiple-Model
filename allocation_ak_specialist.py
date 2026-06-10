from __future__ import annotations
import os, json
import numpy as np, pandas as pd
import allocation_v33_enhancements as enh

AK_SITES = {'248','159','212','145','121'}

def _num(s, default=0.0):
    return pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(default).astype(float).to_numpy()

def _segments(df):
    f=df['Flag'].astype(str).str.upper()
    alloc=(f.str.contains('ALLOC',na=False)&~f.str.contains('NO',na=False)).to_numpy()
    review=f.str.contains('REVIEW',na=False).to_numpy()
    return alloc, review

def _ak_site_mask(df):
    return df['Site'].astype(str).str.replace('.0','',regex=False).str.strip().isin(AK_SITES).to_numpy()

def load_ak_specialist_model(path):
    if not os.path.exists(path): return None
    z=np.load(path, allow_pickle=True)
    manifest=json.loads(str(z['manifest'].item()))
    return {'version':str(z['version'].item()), 'manifest':manifest, 'ak_sites':manifest.get('ak_sites', sorted(AK_SITES))}

def apply_ak_specialist(df, base_audit, model):
    """AK-store specialist for Sites 248, 159, 212, 145, and 121.

    The specialist is deliberately conservative: it does not replace the v3.7 model output.
    It only rescues likely false-negative AK rows when the base v3.7 model predicted blank but the
    original worksheet features still show enough support for one FLM. Separate rules are used for
    AK Allocate and AK Review rows, mirroring the separate-specialist pattern used for Site 802.
    """
    out=base_audit.copy()
    if model is None:
        out['AK Specialist Applied']=0
        return out
    man=model.get('manifest',{})
    segs=man.get('segments',{})
    pred=_num(out['Predicted Final Alloc'])
    conf=_num(out.get('Allocation Confidence', pd.Series([0]*len(df))))
    ak=_ak_site_mask(df); alloc, review=_segments(df)
    flm=np.maximum(_num(df['FLM']),1.0); dc=np.maximum(_num(df['Dc Avail']),0.0)
    rec_flms=_num(df['Alloc. Rec.'])/flm
    # Reuse the v3.3 demand/supply feature frame because it includes the strongest original-sheet shortage features.
    feat=enh.enhanced_feature_frame(df, pred, conf)
    hcd=_num(feat.get('high_conf_demand_share', pd.Series([0]*len(df))))
    shortage=_num(feat.get('shortage_flms', pd.Series([0]*len(df))))
    applied=np.zeros(len(out), dtype=int)
    reasons=np.array(['']*len(out), dtype=object)
    for seg_name, seg_mask in [('allocate', alloc), ('review', review)]:
        prm=segs.get(seg_name,{})
        if not prm.get('enabled', False):
            continue
        mask=ak & seg_mask & (pred<=0) & (dc>0)
        mask &= conf >= float(prm.get('rescue_conf_min', 1.0))
        mask &= rec_flms >= float(prm.get('rescue_rec_flms_min', 0.0))
        mask &= hcd >= float(prm.get('rescue_high_conf_demand_share_min', 0.0))
        mask &= shortage >= float(prm.get('rescue_shortage_flms_min', 0.0))
        if mask.sum()==0:
            continue
        pred[mask]=np.minimum(flm[mask], dc[mask])
        applied[mask]=1
        reasons[mask]=f'AK {seg_name} rescue to one FLM/remainder'
    out['Predicted Final Alloc']=pd.Series(pred).where(pred>0,'')
    out['AK Specialist Applied']=applied
    out['AK Specialist Reason']=reasons
    return out
