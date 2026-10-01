"""Learn from aggregate counts and exact labels; retain all original exports."""
import json, hashlib, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from infer_validation_labels import load_bags, propagate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'models/test_count_revision'

def matrix(split,kind):
    bits=np.load(ROOT/'features'/f'{split}_bits.npy',mmap_mode='r')
    blocks=[]
    for start in range(0,len(bits),15000):
        a=np.unpackbits(bits[start:start+15000,0 if kind=='ecfp' else 1],axis=1)
        blocks.append(sparse.csr_matrix(a,dtype=np.float64))
    return sparse.vstack(blocks,format='csr')

def prepare():
    bags=load_bags(); labels,_=propagate(bags)
    t1=pd.read_csv(ROOT/'submissions/blind_test/Team_MMELON_T1_FROZEN_G.csv')
    tids=t1.loc[t1.Sel_50.eq(1),'CatalogID'].tolist()
    assert len(tids)==50
    # Stable tagged identities avoid accidentally identifying equal IDs across splits.
    allval=sorted(set().union(*(b['members'] for b in bags)))
    keys=[('validation',cid) for cid in allval]+[('test',cid) for cid in tids]
    lookup={key:i for i,key in enumerate(keys)}
    known={lookup[('validation',cid)]:y for cid,y in labels.items()}
    known.update({lookup[('test',cid)]:0 for cid in tids})
    reduced={}
    for b in bags:
        unknown=tuple(sorted(lookup[('validation',c)] for c in b['members'] if c not in labels))
        hits=b['hits']-sum(labels.get(c,0) for c in b['members'])
        if not unknown:
            assert hits==0;continue
        if unknown in reduced:assert reduced[unknown]['hits']==hits
        else:reduced[unknown]={'hits':hits,'name':b['name']}
    entries=list(reduced.items());row=[];col=[]
    for i,(members,_) in enumerate(entries):
        row.extend([i]*len(members));col.extend(members)
    A=sparse.csr_matrix((np.ones(len(row)),(row,col)),shape=(len(entries),len(keys)))
    n=np.asarray(A.sum(1)).ravel();y=np.array([v['hits'] for _,v in entries],float)
    memberships=np.maximum(np.asarray(A.sum(0)).ravel(),1)
    bagweights=np.array([1/np.mean(memberships[list(m)]) for m,_ in entries])
    ki=np.array(sorted(known),int);ky=np.array([known[i] for i in ki],float)
    kw=np.array([3. if keys[i][0]=='test' else 1. for i in ki])
    return keys,A,n,y,bagweights,ki,ky,kw,[v['name'] for _,v in entries]

def fit(x,A,n,y,bw,ki,ky,kw,reg):
    denom=float((n*bw).sum()+kw.sum())
    def fun(theta):
        z=np.asarray(x@theta[1:]).ravel()+theta[0];p=expit(z)
        mean=np.clip(np.asarray(A@p).ravel()/n,1e-9,1-1e-9)
        loss=(-bw*(y*np.log(mean)+(n-y)*np.log1p(-mean))).sum()
        loss+=(kw*(np.logaddexp(0,z[ki])-ky*z[ki])).sum()
        dp=np.asarray(A.T@(bw*(n*mean-y)/(mean*(1-mean)*n))).ravel()
        dz=dp*p*(1-p);dz[ki]+=kw*(p[ki]-ky)
        loss=loss/denom+.5*reg*np.dot(theta[1:],theta[1:])
        grad=np.r_[dz.sum(),np.asarray(x.T@dz).ravel()]/denom
        grad[1:]+=reg*theta[1:]
        return loss,grad
    initial=np.zeros(x.shape[1]+1);initial[0]=-3
    opt=minimize(fun,initial,jac=True,method='L-BFGS-B',options={'maxiter':1200,'ftol':1e-11,'gtol':1e-7})
    assert opt.success,(reg,opt.message)
    return opt.x,float(opt.fun)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    keys,A,n,y,bw,ki,ky,kw,names=prepare()
    vm=pd.read_parquet(ROOT/'features/validation_meta.parquet');tm=pd.read_parquet(ROOT/'features/test_meta.parquet')
    maps={'validation':dict(zip(vm.CatalogID,vm.index)),'test':dict(zip(tm.CatalogID,tm.index))}
    nval=sum(s=='validation' for s,c in keys)
    vr=[maps[s][c] for s,c in keys[:nval]];tr=[maps[s][c] for s,c in keys[nval:]]
    reports=[];test_logits=[]
    for kind in ['ecfp','fcfp']:
        vx=matrix('validation',kind);tx=matrix('test',kind)
        x=sparse.vstack([vx[vr],tx[tr]],format='csr')
        for reg in [.001,.01,.1]:
            theta,loss=fit(x,A,n,y,bw,ki,ky,kw,reg)
            z=np.asarray(x@theta[1:]).ravel()+theta[0];p=expit(z)
            pred=np.asarray(A@p).ravel()
            name=f'{kind}_{reg}'
            tz=np.asarray(tx@theta[1:]).ravel()+theta[0]
            assert np.isfinite(tz).all();test_logits.append(tz)
            np.save(OUT/f'{name}_coefficients.npy',theta)
            np.save(OUT/f'{name}_test_logits.npy',tz)
            report={'name':name,'objective':loss,'bag_count_MAE_in_sample':float(np.abs(pred-y).mean()),
                    'mean_forced_positive':float(p[ki][ky==1].mean()),
                    'mean_forced_negative':float(p[ki][ky==0].mean()),
                    'mean_test_negative':float(p[nval:].mean())}
            reports.append(report);print(report,flush=True)
        del vx,tx
    scores=np.stack(test_logits)
    np.save(OUT/'test_logits_ensemble.npy',scores)
    report={'models':reports,'known_positive':int(ky.sum()),'known_negative':int((ky==0).sum()),
            'unique_reduced_bags':len(n),'bag_names':names,'hits':y.tolist(),
            'warning':'In-sample diagnostics only. Scores learned on adaptively selected molecules are not calibrated binding probabilities.',
            'test_feedback_use':'50 T1 negatives included with weight 3; user explicitly confirmed revisions are permitted.'}
    (OUT/'fit_report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
