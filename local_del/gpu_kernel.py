"""Exact dense fingerprint kernels on the local Apple GPU, using unified memory."""
from pathlib import Path
import os,json,time,gc
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'models/deep_audit'


def main():
    assert torch.backends.mps.is_available(),'Apple GPU is required for this explicit GPU run'
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.75)
    v=np.load(ROOT/'features/validation_bits.npy',mmap_mode='r')
    r=np.load(OUT/'reference_bits.npy',mmap_mode='r')
    meta=pd.read_parquet(OUT/'reference_metadata.parquet')
    controls=pd.read_parquet(OUT/'control_metadata.parquet')
    t=np.load(ROOT/'features/training_bits.npy',mmap_mode='r');c=t[controls.index]
    reliability=np.clip(np.log1p(meta.count_PGK2.fillna(3).to_numpy())/np.log(4),1,4)
    libcounts=meta.library.fillna('unknown').value_counts()
    balance=np.clip(np.sqrt(float(libcounts.median())/meta.library.fillna('unknown').map(libcounts).to_numpy()),.4,3)
    w=reliability*balance;w=(w/w.sum()).astype('float32')
    public=meta.public_selective.to_numpy().astype('float32');public/=public.sum()
    weights=torch.tensor(np.stack([w,public],axis=1),device='mps')
    # Hold both complete candidate feature arrays on the GPU; no validation prescreen.
    qgpu=[];rgpu=[];cgpu=[]
    for k in (0,1):
        qgpu.append(torch.from_numpy(np.unpackbits(v[:,k],axis=1)).to(device='mps',dtype=torch.float32))
        rgpu.append(torch.from_numpy(np.unpackbits(r[:,k],axis=1)).to(device='mps',dtype=torch.float32))
        cgpu.append(torch.from_numpy(np.unpackbits(c[:,k],axis=1)).to(device='mps',dtype=torch.float32))
    sums_r=[x.sum(1)[None,:] for x in rgpu];sums_c=[x.sum(1)[None,:] for x in cgpu]
    result=np.zeros((len(v),2,7),dtype=np.float32)
    # Channels: weighted positive tan^4, public tan^4, weighted positive TV^8,
    # public TV^8, control tan^4, control TV^8, maximum control TV.
    start=time.time();peak=0;batch=8192
    for k in (0,1):
        for begin in range(0,len(v),batch):
            end=min(begin+batch,len(v));q=qgpu[k][begin:end];qc=q.sum(1)[:,None]
            inter=q@rgpu[k].T
            tan=inter/(qc+sums_r[k]-inter).clamp_min(1)
            den=tan.pow(4)@weights
            result[begin:end,k,:2]=den.cpu().numpy()
            del tan,den
            tv=inter/(qc+.2*sums_r[k]-.2*inter).clamp_min(1)
            den=tv.pow(8)@weights
            result[begin:end,k,2:4]=den.cpu().numpy()
            del tv,den,inter
            inter=q@cgpu[k].T
            tan=inter/(qc+sums_c[k]-inter).clamp_min(1)
            tv=inter/(qc+.2*sums_c[k]-.2*inter).clamp_min(1)
            result[begin:end,k,4]=tan.pow(4).mean(1).cpu().numpy()
            result[begin:end,k,5]=tv.pow(8).mean(1).cpu().numpy()
            result[begin:end,k,6]=tv.max(1).values.cpu().numpy()
            torch.mps.synchronize()
            peak=max(peak,torch.mps.driver_allocated_memory())
            del inter,tan,tv
            if begin%(batch*5)==0:print('GPU fingerprint',k,'rows',end,'/',len(v),'seconds',round(time.time()-start),'driver GB',round(peak/1e9,2),flush=True)
    assert np.isfinite(result).all() and (result>=0).all()
    # Independently verify GPU kernel values against native RDKit, for both fingerprints.
    for k in (0,1):
        refs=[DataStructs.CreateFromBinaryText(x[k].tobytes()) for x in r]
        ctl=[DataStructs.CreateFromBinaryText(x[k].tobytes()) for x in c]
        for row in np.random.default_rng(2040).choice(len(v),4,replace=False):
            q=DataStructs.CreateFromBinaryText(v[row,k].tobytes())
            a=np.array(DataStructs.BulkTanimotoSimilarity(q,refs));b=np.array(DataStructs.BulkTverskySimilarity(q,refs,1,.2))
            ca=np.array(DataStructs.BulkTanimotoSimilarity(q,ctl));cb=np.array(DataStructs.BulkTverskySimilarity(q,ctl,1,.2))
            expected=[np.sum(a**4*w),np.sum(a**4*public),np.sum(b**8*w),np.sum(b**8*public),np.mean(ca**4),np.mean(cb**8),cb.max()]
            assert np.allclose(expected,result[row,k],rtol=2e-4,atol=2e-7),(row,k,expected,result[row,k])
    np.save(OUT/'gpu_kernel_scores.npy',result)
    record={'device':'Apple MPS GPU','validation_rows':len(v),'reference_rows':len(r),'control_rows':len(c),
      'batch_rows':batch,'peak_driver_allocated_bytes':peak,'gpu_memory_limit_bytes':int(torch.mps.recommended_max_memory()*.75),
      'seconds':time.time()-start,'full_candidate_arrays_resident_on_gpu':True,
      'verification':'8 native RDKit kernel checks (4 random molecules x 2 fingerprints) passed',
      'positive_weighting':'Read-count reliability capped at 4; inverse-square-root library balance capped 0.4..3; normalized over full bank',
      'channels':['positive_Tanimoto_power4','public_Tanimoto_power4','positive_Tversky_power8','public_Tversky_power8','control_Tanimoto_power4','control_Tversky_power8','control_Tversky_max']}
    (OUT/'gpu_complete.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
