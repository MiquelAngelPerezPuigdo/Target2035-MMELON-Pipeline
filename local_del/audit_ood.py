"""Unlabelled chemistry coverage audit of the frozen successful round03 score."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import lightgbm as lgb
from features import featurize
from train import matrix


def score_anchors(name,ligands):
    b=np.load(ROOT/'features'/f'{name}_bits.npy',mmap_mode='r')
    scores=np.zeros((len(b),len(ligands),4),dtype=np.float32)
    for k in (0,1):
        fps=[DataStructs.CreateFromBinaryText(row[k].tobytes()) for row in b]
        for j,ref in enumerate(ligands):
            q=DataStructs.CreateFromBinaryText(featurize(ref['SMILES'])[2][k].tobytes())
            scores[:,j,k]=DataStructs.BulkTanimotoSimilarity(q,fps)
            scores[:,j,k+2]=DataStructs.BulkTverskySimilarity(q,fps,.2,1)
    return scores


def summary(df,s,score):
    top=df.assign(score=score).loc[df.valid].sort_values(['score','CatalogID'],ascending=[False,True]).drop_duplicates('canonical').head(50)
    idx=top.index.to_numpy()
    by_anchor=.65*s[:,:,0]+.20*s[:,:,1]+.15*s[:,:,2]
    max_t=s[:,:,0].max(axis=1)
    core=Chem.MolFromSmiles('c1ccc2ncncc2c1')
    core_count=sum(Chem.MolFromSmiles(sm).HasSubstructMatch(core) for sm in top.SMILES)
    return {'rows':len(df),'top50':{
       'anchor_preference_counts':np.bincount(by_anchor[idx].argmax(axis=1),minlength=2).tolist(),
       'max_ECFP_similarity_quantiles':dict(zip(['min','q25','median','q75','max'],map(float,np.quantile(max_t[idx],[0,.25,.5,.75,1])))),
       'quinazoline_core_count':core_count,'unique_Murcko_scaffolds':top.scaffold.nunique(),
       'score_range':[float(top.score.min()),float(top.score.max())]},
       'full_library_similarity_counts':{str(x):int((max_t>=x).sum()) for x in [.3,.35,.4,.45,.5]},
       'full_library_similarity_fractions':{str(x):float((max_t>=x).mean()) for x in [.3,.35,.4,.45,.5]}},top


def main():
    out=ROOT/'reports/ood';out.mkdir(exist_ok=True)
    refs=json.loads((ROOT/'models/frozen_round03/verified_ligands.json').read_text())
    result={'method_defined_before_result':True,'method_archived_after_result':True,'test_activity_labels_accessed':False,'test_submissions_made':0,
      'meaning':'Unlabelled chemistry coverage only; not an estimate of test hits or successful scaffold transfer.',
      'method':'Frozen round03 G: 95% verified-ligand similarity, 5% DEL-trained ECFP ensemble',
      'reference_order':[r['name'] for r in refs]}
    selected={}
    for name in ['validation','test']:
        d=pd.read_parquet(ROOT/'features'/f'{name}_meta.parquet')
        if name=='validation':
            s=np.load(ROOT/'models/deep_audit/known_ligand_similarity.npy')
            ml=np.load(ROOT/'models/validation_ecfp.npy')
        else:
            s=score_anchors(name,refs);np.save(out/'test_anchor_similarity.npy',s)
            x=matrix(name,'ecfp')
            predictions=[]
            for seed in [2035,2036]:
                m=lgb.Booster(model_file=str(ROOT/'models/frozen_round03'/f'ecfp_{seed}.txt'))
                predictions.append(m.predict(x,num_threads=8))
            ml=np.mean(predictions,axis=0)
        signal=np.max(.65*s[:,:,0]+.20*s[:,:,1]+.15*s[:,:,2],axis=1)
        score=.95*signal+.05*ml
        result[name],selected[name]=summary(d,s,score)
        selected[name].to_parquet(out/f'{name}_top50_audit_only.parquet',index=False)
        print(name,json.dumps(result[name]),flush=True)
    result['exact_Murcko_overlap_between_top50']=len(set(selected['validation'].scaffold)&set(selected['test'].scaffold))
    result['note']='Our Murcko scaffold counts are not the organizers chemical-series assignments. Test top50 is an audit snapshot, not a submission recommendation.'
    (out/'coverage.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
