"""Reproducible blind-test exports; no blind-test labels or paid inference."""
import sys, json, hashlib, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import DataStructs
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import lightgbm as lgb
from features import featurize

OUT=ROOT/'submissions/blind_test'
SOURCE=ROOT.parent/'Val-Test-set/PGK2_Test_split.csv'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def similarities(x,refs):
    right=np.asarray(refs.sum(1)).ravel()
    out=np.empty((x.shape[0],refs.shape[0]),dtype=np.float32)
    for start in range(0,x.shape[0],10000):
        q=x[start:start+10000]; dot=(q@refs.T).toarray()
        out[start:start+len(dot)]=dot/np.maximum(np.asarray(q.sum(1))+right[None,:]-dot,1e-12)
    return out

def choose(meta,x,score,threshold=None):
    selected=[]; keys=set(); scaffolds={}
    for row in np.lexsort((meta.CatalogID.to_numpy(),-score)):
        if not meta.valid.iat[row] or meta.canonical.iat[row] in keys: continue
        scaffold=meta.scaffold.iat[row]
        if threshold is not None:
            if scaffolds.get(scaffold,0)>=2: continue
            if selected and similarities(x[row],x[selected]).max()>=threshold: continue
        selected.append(int(row));keys.add(meta.canonical.iat[row]);scaffolds[scaffold]=scaffolds.get(scaffold,0)+1
        if len(selected)==50:return selected
    raise RuntimeError('Could not select 50')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(ROOT/'features/test_meta.parquet'); original=pd.read_csv(SOURCE)
    assert len(meta)==184632 and meta.CatalogID.equals(original.CatalogID)
    assert meta.SMILES.equals(original.SMILES) and meta.CatalogID.is_unique
    x=sparse.load_npz(ROOT/'features/test_ecfp.npz').tocsr()
    anchors=np.load(ROOT/'reports/ood/test_anchor_similarity.npy')
    assert anchors.shape==(len(meta),2,4)
    # Verify cache orientation and fingerprint convention against RDKit directly.
    refs=json.loads((ROOT/'models/frozen_round03/verified_ligands.json').read_text())
    bits=np.load(ROOT/'features/test_bits.npy',mmap_mode='r')
    for row in [0,12345,len(meta)-1]:
        for j,ref in enumerate(refs):
            packed=featurize(ref['SMILES'])[2]
            for k in [0,1]:
                q=DataStructs.CreateFromBinaryText(packed[k].tobytes())
                f=DataStructs.CreateFromBinaryText(bits[row,k].tobytes())
                assert np.isclose(DataStructs.TanimotoSimilarity(q,f),anchors[row,j,k])
                assert np.isclose(DataStructs.TverskySimilarity(q,f,.2,1),anchors[row,j,k+2])
    predictions=[]
    for seed in [2035,2036]:
        model=lgb.Booster(model_file=str(ROOT/'models/frozen_round03'/('ecfp_%d.txt'%seed)))
        predictions.append(model.predict(x,num_threads=8))
    ml=np.mean(predictions,axis=0)
    ligand=(.65*anchors[:,:,0]+.20*anchors[:,:,1]+.15*anchors[:,:,2]).max(1)
    frozen=.95*ligand+.05*ml
    # Validate exact reproduction against the preexisting unlabelled test audit.
    previous=pd.read_parquet(ROOT/'reports/ood/test_top50_audit_only.parquet')
    lookup=dict(zip(meta.CatalogID,frozen))
    assert np.allclose(previous.CatalogID.map(lookup),previous.score,rtol=1e-6,atol=1e-8)
    vm=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    forced=pd.read_parquet(ROOT/'reports/validation_forced_labels.parquet')
    positive=forced[forced.forced_label.eq(1)]
    assert len(positive)==8
    vr=vm.index[vm.CatalogID.isin(positive.CatalogID)].to_numpy()
    vx=sparse.load_npz(ROOT/'features/validation_ecfp.npz').tocsr()
    exact=similarities(x,vx[vr]).max(1)
    # Small raw-similarity contribution avoids promoting weak tail scores by percentiles.
    portfolio=.80*frozen+.20*exact
    selections={'T1_FROZEN_G':choose(meta,x,frozen),
                'T2_DIVERSE_TRANSFER':choose(meta,x,portfolio,.60)}
    all_scores={'T1_FROZEN_G':frozen,'T2_DIVERSE_TRANSFER':portfolio}
    report={'source_rows':len(meta),'source_sha256':sha(SOURCE),'test_labels_accessed':False,
            'test_submissions_made':0,'boltz_spend_usd':0,'files':{},
            'methods':{'T1':'0.95 * original two-ligand similarity + 0.05 * frozen DEL ensemble; top 50 unique parent structures',
                       'T2':'0.80 * T1 score + 0.20 * max ECFP similarity to eight forced positives; selected pairwise ECFP <0.60 and at most two per Murcko scaffold'},
            'limitations':['Scores are ranking confidences, not calibrated binding probabilities.',
                           'T2 weights and diversity constraints are heuristic, not independently validated.',
                           'Murcko scaffold diversity does not equal organizer chemical-series diversity.',
                           'Validation feedback was adaptively reused; it cannot estimate blind-test performance.']}
    files=[]
    for label,rows in selections.items():
        scores=all_scores[label]; assert np.isfinite(scores).all()
        flags=np.zeros(len(meta),dtype=int);flags[rows]=1
        frame=pd.DataFrame({'CatalogID':meta.CatalogID,'Sel_50':flags,'Score':scores})
        p=OUT/('Team_MMELON_'+label+'.csv');frame.to_csv(p,index=False,float_format='%.12g');files.append(p)
        saved=pd.read_csv(p)
        assert list(saved.columns)==['CatalogID','Sel_50','Score']
        assert saved.CatalogID.equals(original.CatalogID) and saved.Sel_50.sum()==50
        assert saved.Sel_50.isin([0,1]).all() and np.isfinite(saved.Score).all()
        assert meta.iloc[rows].canonical.nunique()==50
        detail=meta.iloc[rows].copy();detail['Score']=scores[rows];detail['ligand_similarity']=ligand[rows]
        detail['DEL_score']=ml[rows];detail['exact_active_similarity']=exact[rows]
        detail.to_csv(OUT/(label+'_selected_details.csv'),index=False)
        report['files'][p.name]={'rows':len(saved),'selected':50,'unique_Murcko_scaffolds':int(detail.scaffold.nunique()),'sha256':sha(p)}
    report['selected_overlap']=len(set(selections['T1_FROZEN_G'])&set(selections['T2_DIVERSE_TRANSFER']))
    report['confirmed_positive_references']=vm.CatalogID.iloc[vr].tolist()
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'START_HERE.txt').write_text('Submit the two Team_MMELON_*.csv files separately to the BLIND TEST queue.\nEach contains all 184,632 test IDs, scores for every row, and exactly 50 Sel_50=1 flags.\nT1 preserves frozen G. T2 adds modest exact-active transfer and structural diversity.\nDo not submit detail CSVs or the ZIP itself. Both files are prepared locally; neither has been uploaded or submitted.\nNo Boltz credits were spent. Scores are uncalibrated confidence rankings.\n')
    with zipfile.ZipFile(OUT/'blind_test_submissions.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.write(OUT/'START_HERE.txt','START_HERE.txt')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
