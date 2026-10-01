"""Two complementary DEL-supervised models and honest scaffold holdout evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,hashlib,gc
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score,roc_auc_score
import lightgbm as lgb


def matrix(name,kind):
    cache=ROOT/'features'/f'{name}_{kind}.npz'
    if cache.exists():return sparse.load_npz(cache)
    b=np.load(ROOT/'features'/f'{name}_bits.npy',mmap_mode='r')
    fp_index=0 if kind=='ecfp' else 1
    chunks=[]
    for begin in range(0,len(b),10000):
        x=sparse.csr_matrix(np.unpackbits(b[begin:begin+10000,fp_index],axis=1),dtype=np.float32)
        if kind=='fcfp':
            d=np.load(ROOT/'features'/f'{name}_desc.npy',mmap_mode='r')[begin:begin+10000]
            x=sparse.hstack([x,sparse.csr_matrix(d)],format='csr')
        chunks.append(x)
    x=sparse.vstack(chunks,format='csr');sparse.save_npz(cache,x);return x


def eligible(d):
    good=d.valid & d.canonical.ne('')
    # Two enumerated libraries have no rows in the supplied target selection.
    # Their absence cannot establish a negative target label.
    observed_libraries=set(d.loc[d.source.str.startswith('selection'),'library'])
    good &= ~d.source.eq('library_unlabeled') | d.library.isin(observed_libraries)
    # Resolve label disagreements conservatively: remove the entire constitutional structure.
    conflicts=d[good].groupby('canonical').broad.nunique()
    conflict_keys=set(conflicts[conflicts>1].index)
    good &= ~d.canonical.isin(conflict_keys)
    # Collapse stereoisomers/salt duplicates because the model uses non-chiral parent features.
    good &= ~d.canonical.duplicated(keep='first')
    return good.to_numpy(),len(conflict_keys)


def split_groups(d):
    return np.array([int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%5==0 for s in d.scaffold])


def metrics(y,p):
    order=np.argsort(-p,kind='stable')
    return {'n':len(y),'positive_n':int(y.sum()),'prevalence':float(y.mean()),
        'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
        'precision_at_50':float(y[order[:50]].mean()),'precision_at_500':float(y[order[:500]].mean())}


def weights(d,kind):
    w=np.ones(len(d),dtype=np.float32)
    w[d.source.eq('library_unlabeled')]=.5
    w[d.source.eq('ntc_presence')]=.75
    weak=d.source.eq('selection_negative') & d.count_PGK2.le(1) & d.count_NTC.eq(0) & d.count_PGK2_with_inhibitor.eq(0)
    w[weak]=.5
    w[d.ntc_present.fillna(False)&d.broad.eq(1)]*=.3
    if kind=='fcfp':
        # Balance positive evidence across represented libraries, gently, with capped weights.
        pos=d.broad.eq(1)
        counts=d[pos].library.value_counts()
        if len(counts):
            norm=float(counts.median())
            w[pos]*=np.clip(np.sqrt(norm/d.loc[pos,'library'].map(counts).to_numpy()),.4,3)
        strength=np.clip(np.log1p(d.count_PGK2.fillna(0).to_numpy())/np.log(6),.7,2)
        w[pos]*=strength[pos]
    return w


def select_training(d,mask,kind,seed):
    label='strict' if kind=='ecfp' else 'broad'
    good=mask.copy()
    if kind=='ecfp':good &= ~((d.broad.to_numpy()==1)&(d.strict.to_numpy()==0))
    ids=np.flatnonzero(good)
    y=d[label].to_numpy(dtype=np.int8)
    pos=ids[y[ids]==1];neg=ids[y[ids]==0]
    rng=np.random.default_rng(seed)
    # Two negatives per positive for ECFP; four for the complementary FCFP model.
    take=min(len(neg),len(pos)*(2 if kind=='ecfp' else 4))
    neg=rng.choice(neg,size=take,replace=False)
    out=np.concatenate([pos,neg]);rng.shuffle(out)
    return out,y


def main():
    d=pd.read_parquet(ROOT/'features/training_meta.parquet')
    val=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    good,conflicts=eligible(d)
    held=split_groups(d)
    assert not set(d.loc[good&held,'scaffold']) & set(d.loc[good&~held,'scaffold'])
    reports={'structural_label_conflicts_removed':conflicts,'eligible_rows':int(good.sum()),
      'unobserved_library_policy':'Exclude enumerated background from libraries absent from supplied target selection; retain directly observed NTC controls.',
      'heldout_rule':'SHA256(non-chiral Bemis-Murcko scaffold) modulo 5 == 0; no held-out scaffolds in training',
      'warning':'DEL-derived labels are noisy proxies; these metrics do not estimate ASMS hit rates.',
      'validation_labels_used':False,'models':{}}
    np.save(ROOT/'data/eligible.npy',good);np.save(ROOT/'data/heldout.npy',held)
    for kind in ['ecfp','fcfp']:
        print('Loading features',kind,flush=True)
        x=matrix('training',kind);xv=matrix('validation',kind)
        train_ids,y=select_training(d,good&~held,kind,2035)
        evalmask=good&held
        if kind=='ecfp':evalmask &= ~((d.broad.to_numpy()==1)&(d.strict.to_numpy()==0))
        eval_ids=np.flatnonzero(evalmask)
        params=dict(objective='binary',n_estimators=900,learning_rate=.035,num_leaves=31,
          max_depth=-1,min_child_samples=50,colsample_bytree=.8,subsample=.8,subsample_freq=1,
          reg_lambda=10,reg_alpha=.1,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
        model=lgb.LGBMClassifier(**params,random_state=2035)
        print('Fitting scaffold holdout',kind,len(train_ids),len(eval_ids),flush=True)
        model.fit(x[train_ids],y[train_ids],sample_weight=weights(d.iloc[train_ids],kind),
          eval_set=[(x[eval_ids],y[eval_ids])],eval_metric='average_precision',
          callbacks=[lgb.early_stopping(60,first_metric_only=False,verbose=False)])
        hp=model.predict_proba(x[eval_ids])[:,1]
        result=metrics(y[eval_ids],hp);result['iterations']=int(model.best_iteration_)
        result['training_rows']=len(train_ids);result['training_positives']=int(y[train_ids].sum())
        print(kind,json.dumps(result),flush=True)
        pd.DataFrame({'row':eval_ids,'label':y[eval_ids],'score':hp}).to_parquet(ROOT/'reports'/f'{kind}_heldout.parquet',index=False)
        model.booster_.save_model(str(ROOT/'models'/f'{kind}_holdout.txt'))
        # Fit independent negative samples / seeds; averaging reduces single-run instability.
        predictions=[]
        iterations=max(100,min(900,int(model.best_iteration_)))
        for seed in [2035,2036]:
            ids,y=select_training(d,good,kind,seed)
            final=lgb.LGBMClassifier(**dict(params,n_estimators=iterations),random_state=seed)
            print('Full training',kind,seed,len(ids),flush=True)
            final.fit(x[ids],y[ids],sample_weight=weights(d.iloc[ids],kind))
            final.booster_.save_model(str(ROOT/'models'/f'{kind}_{seed}.txt'))
            pred=final.predict_proba(xv)[:,1]
            pred[~val.valid.to_numpy()]=0
            np.save(ROOT/'models'/f'validation_{kind}_{seed}.npy',pred)
            predictions.append(pred)
        np.save(ROOT/'models'/f'validation_{kind}.npy',np.mean(predictions,axis=0))
        reports['models'][kind]=result
        (ROOT/'reports/model_metrics.json').write_text(json.dumps(reports,indent=2)+'\n')
        del x,xv,model,final;gc.collect()
    print('All model fits completed',flush=True)


if __name__=='__main__':main()
