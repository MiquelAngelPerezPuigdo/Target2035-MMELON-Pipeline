"""DEL-labelled nearest-neighbour evidence, with explicit control-reference contrast."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from rdkit import DataStructs

ROOT=Path(__file__).resolve().parent
_REF=None


def init_worker(pos,neg):
    global _REF
    _REF=([ [DataStructs.CreateFromBinaryText(row[k].tobytes()) for row in pos] for k in (0,1)],
          [ [DataStructs.CreateFromBinaryText(row[k].tobytes()) for row in neg] for k in (0,1)])


def score_chunk(bits):
    out=np.zeros((len(bits),4),dtype=np.float32)
    for j,b in enumerate(bits):
        for k in (0,1):
            q=DataStructs.CreateFromBinaryText(b[k].tobytes())
            pos=np.array(DataStructs.BulkTanimotoSimilarity(q,_REF[0][k]),dtype=np.float32)
            neg=np.array(DataStructs.BulkTanimotoSimilarity(q,_REF[1][k]),dtype=np.float32)
            # Mean of top 3 neighbours reduces reliance on a single noisy DEL observation.
            out[j,k]=np.partition(pos,-3)[-3:].mean()
            out[j,k+2]=neg.max()
    return out


def choose_refs(d,mask):
    pool=d.loc[mask].copy()
    pool['strength']=np.log1p(pool.count_PGK2.fillna(0))/(1+np.maximum(pool.count_NTC.fillna(0),pool.count_PGK2_with_inhibitor.fillna(0)))
    pos=pool[pool.strict.eq(1)&~pool.ntc_present.fillna(False)].sort_values(['strength','canonical'],ascending=[False,True])
    pos=pos.groupby('scaffold',sort=False).head(2).head(3000)
    neg=pool[(pool.strict.eq(0)) & ((pool.count_NTC>=pool.count_PGK2)&(pool.count_NTC>=3) |
        (pool.count_PGK2_with_inhibitor>=pool.count_PGK2)&(pool.count_PGK2_with_inhibitor>=3))]
    neg=neg.sort_values(['historic_hits','canonical'],ascending=[False,True]).groupby('scaffold',sort=False).head(1)
    neg=neg.sample(n=min(1000,len(neg)),random_state=2035)
    assert len(pos)>=3 and len(neg)>0
    return pos.index.to_numpy(),neg.index.to_numpy()


def score(name,bits,refbits,pos,neg):
    path=ROOT/'models'/f'{name}_similarity.npy'
    fingerprint=hashlib.sha256()
    fingerprint.update(np.ascontiguousarray(bits).tobytes())
    fingerprint.update(np.ascontiguousarray(refbits[pos]).tobytes())
    fingerprint.update(np.ascontiguousarray(refbits[neg]).tobytes())
    signature=fingerprint.hexdigest()
    signature_path=path.with_suffix('.sha256')
    if path.exists() and signature_path.exists() and signature_path.read_text().strip()==signature:return np.load(path)
    started=time.time();parts=[]
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(refbits[pos],refbits[neg])) as pool:
        chunks=(bits[i:i+512] for i in range(0,len(bits),512))
        for j,result in enumerate(pool.map(score_chunk,chunks,chunksize=1)):
            parts.append(result)
            if (j+1)%40==0:print(name,min((j+1)*512,len(bits)),len(bits),'seconds',round(time.time()-started),flush=True)
    out=np.concatenate(parts);np.save(path,out);signature_path.write_text(signature+'\n')
    print('Completed',name,'seconds',round(time.time()-started),flush=True);return out


def main():
    d=pd.read_parquet(ROOT/'features/training_meta.parquet')
    good=np.load(ROOT/'data/eligible.npy');held=np.load(ROOT/'data/heldout.npy')
    b=np.load(ROOT/'features/training_bits.npy',mmap_mode='r')
    v=np.load(ROOT/'features/validation_bits.npy',mmap_mode='r')
    pos,neg=choose_refs(d,good)
    d.loc[pos].to_parquet(ROOT/'models/similarity_positive_refs.parquet',index=True)
    d.loc[neg].to_parquet(ROOT/'models/similarity_negative_refs.parquet',index=True)
    score('validation',v,b,pos,neg)
    hp,hn=choose_refs(d,good&~held)
    eval_ids=np.flatnonzero(good&held&~((d.broad.to_numpy()==1)&(d.strict.to_numpy()==0)))
    score('heldout',b[eval_ids],b,hp,hn)
    np.save(ROOT/'models/similarity_heldout_ids.npy',eval_ids)
    (ROOT/'reports/similarity_method.json').write_text(json.dumps({
      'positive_references':len(pos),'negative_references':len(neg),
      'positive_selection':'Strongest strict DEL positives, max 2 per non-chiral Murcko scaffold, exclude NTC supplement overlap.',
      'negative_selection':'Sample of inhibitor/NTC-associated DEL compounds with control count >= target count and >= 3.',
      'features':'Top-3 mean Tanimoto to positives and maximum Tanimoto to controls, ECFP4 and FCFP4.',
      'holdout':'Separate reference set excluding every held-out scaffold.'},indent=2)+'\n')


if __name__=='__main__':main()
