"""Remove all labels/constraints touching a held-out bag for a cautious diagnostic."""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from revise_test_count_model import ROOT, OUT, prepare, matrix, fit
from infer_validation_labels import load_bags

keys,A,n,y,bw,ki,ky,kw,names=prepare();keymap={k:i for i,k in enumerate(keys)}
vm=pd.read_parquet(ROOT/'features/validation_meta.parquet');tm=pd.read_parquet(ROOT/'features/test_meta.parquet')
maps={'validation':dict(zip(vm.CatalogID,vm.index)),'test':dict(zip(tm.CatalogID,tm.index))}
nv=sum(s=='validation' for s,c in keys);vr=[maps[s][c] for s,c in keys[:nv]];tr=[maps[s][c] for s,c in keys[nv:]]
bags={b['name']:b for b in load_bags()}
rows=[]
for kind in ['ecfp','fcfp']:
    vx=matrix('validation',kind);tx=matrix('test',kind);x=sparse.vstack([vx[vr],tx[tr]],format='csr');del vx,tx
    for name in ['G_KNOWN_LIGANDS','K_NEW_CORES','O_DIVERSE_BAG','AI_CHEMBERTA_OOD','AP_EXACT_HIT_ANALOGS','AQ_EXACT_HIT_SCAFFOLD_HOPS']:
        b=bags[name];held=np.array([keymap[('validation',c)] for c in sorted(b['members'])])
        keep=np.asarray(A[:,held].sum(1)).ravel()==0;kk=~np.isin(ki,held)
        # Bag multiplicity weights remain fixed; no held-out activity outcomes enter fitting.
        coef,loss=fit(x,A[keep],n[keep],y[keep],bw[keep],ki[kk],ky[kk],kw[kk],.01)
        expected=float(expit(np.asarray(x[held]@coef[1:]).ravel()+coef[0]).sum())
        row={'kind':kind,'heldout_bag':name,'observed_hits':b['hits'],'sum_scores':expected,
             'remaining_constraints':int(keep.sum()),'remaining_exact_positives':int(ky[kk].sum())}
        rows.append(row);print(row,flush=True)
(OUT/'heldout_bag_audit.json').write_text(json.dumps({'diagnostics':rows,'limitation':'Bag-disjoint labels, but adaptively collected data and related chemistry remain. Sum scores is not a calibrated count forecast.'},indent=2)+'\n')
