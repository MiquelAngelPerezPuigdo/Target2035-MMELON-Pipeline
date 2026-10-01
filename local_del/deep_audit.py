"""Full validation/reference comparisons and full selection-table diagnostics on this Mac."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import polars as pl
from rdkit import DataStructs
from features import featurize

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'models/deep_audit'
REPORT=ROOT/'reports/deep_audit'
REF=None;CONTROL=None;REFCOUNT=None;PUB=None
K=32


def init(ref,control,public):
    global REF,CONTROL,REFCOUNT,PUB
    REF=[[DataStructs.CreateFromBinaryText(row[k].tobytes()) for row in ref] for k in (0,1)]
    CONTROL=[[DataStructs.CreateFromBinaryText(row[k].tobytes()) for row in control] for k in (0,1)]
    REFCOUNT=[np.array([x.GetNumOnBits() for x in f],dtype=np.float32) for f in REF]
    PUB=public


def nearest_chunk(bits):
    ids=np.zeros((len(bits),4,K),dtype=np.int32)
    values=np.zeros((len(bits),4,K),dtype=np.float32)
    # Public PGK2-over-PGK1 DEL reference similarity for Tanimoto and Tversky, both fingerprints.
    aux=np.zeros((len(bits),8),dtype=np.float32)
    for j,row in enumerate(bits):
        for k in (0,1):
            q=DataStructs.CreateFromBinaryText(row[k].tobytes())
            tan=np.asarray(DataStructs.BulkTanimotoSimilarity(q,REF[k]),dtype=np.float32)
            intersection=tan*(q.GetNumOnBits()+REFCOUNT[k])/(1+tan)
            tv=intersection/(q.GetNumOnBits()+.2*REFCOUNT[k]-.2*intersection)
            for kind,s in [(k,tan),(k+2,tv)]:
                top=np.argpartition(s,-K)[-K:];top=top[np.argsort(-s[top],kind='stable')]
                ids[j,kind]=top;values[j,kind]=s[top]
                aux[j,kind]=np.partition(s[PUB],-3)[-3:].mean()
            ct=np.asarray(DataStructs.BulkTanimotoSimilarity(q,CONTROL[k]),dtype=np.float32)
            aux[j,k+4]=ct.max()
            aux[j,k+6]=q.GetNumOnBits()
    return ids,values,aux


def table_audit():
    print('Auditing complete deduplicated selection table',flush=True)
    d=pl.read_parquet(ROOT/'data/selection_dedup.parquet').with_columns(
        pl.col('compound').str.split('-').list.first().alias('library'))
    p=pl.col('count_PGK2');i=pl.col('count_PGK2_with_inhibitor');n=pl.col('count_NTC')
    d=d.with_columns(((p>=3)&(i<.1*p)&(n==0)&(pl.col('historic_hits')<5)).alias('strict'))
    stats=d.group_by('library').agg(pl.len().alias('rows'),p.sum().alias('target_reads'),
        p.median().alias('median_target_reads'),pl.col('strict').sum().alias('strict_positive_rows'),
        (p==1).sum().alias('singletons'),(i>0).sum().alias('with_inhibitor_observed'),(n>0).sum().alias('ntc_observed'))
    stats=stats.with_columns((pl.col('strict_positive_rows')/pl.col('rows')).alias('positive_fraction')).sort('library')
    stats.write_csv(REPORT/'full_library_summary.csv')
    d.select('count_PGK2','count_PGK2_with_inhibitor','count_NTC','historic_hits').describe().write_csv(REPORT/'full_count_summary.csv')
    # Complete observed-library BB marginals expose concentration of positives by reaction position.
    for cycle in [1,2,3]:
        x=d.with_columns(pl.col('compound').str.split('-').list.get(cycle,null_on_oob=True).alias('building_block'))
        x.group_by('library','building_block').agg(pl.len().alias('observed_compounds'),
            pl.col('strict').sum().alias('strict_positive_compounds'),p.sum().alias('target_reads'),
            i.sum().alias('inhibitor_reads'),n.sum().alias('ntc_reads')).sort('library','building_block').write_parquet(REPORT/f'cycle{cycle}_marginals.parquet')
    audit={'selection_rows_audited':len(d),'strict_positive_rows':int(d['strict'].sum()),
        'note':'Building-block marginals describe the supplied target-selected export, not enrichment against a sequenced naive library.'}
    del d
    t=pd.read_parquet(ROOT/'features/training_meta.parquet');v=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    td=np.load(ROOT/'features/training_desc.npy');vd=np.load(ROOT/'features/validation_desc.npy')
    summaries={}
    for name,x in [('DEL_positive',td[t.strict.eq(1)]),('validation',vd)]:
        summaries[name]={label:{'q10':float(np.quantile(x[:,j],.1)),'median':float(np.median(x[:,j])),'q90':float(np.quantile(x[:,j],.9))}
            for j,label in [(0,'MW'),(1,'LogP'),(2,'TPSA'),(5,'RotatableBonds'),(9,'HeavyAtoms')]}
    audit['chemical_space']=summaries
    (REPORT/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


def main():
    OUT.mkdir(exist_ok=True,parents=True);REPORT.mkdir(exist_ok=True,parents=True)
    table_audit()
    t=pd.read_parquet(ROOT/'features/training_meta.parquet');v=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    good=np.load(ROOT/'data/eligible.npy');tb=np.load(ROOT/'features/training_bits.npy',mmap_mode='r')
    vb=np.load(ROOT/'features/validation_bits.npy',mmap_mode='r')
    p=t.loc[good&t.strict.eq(1)&~t.ntc_present.fillna(False)].sort_values('canonical').copy()
    bits=[tb[p.index]];p['public_selective']=False
    ext=pd.read_csv(ROOT/'external/pgk2/DEL_hit_candidates_1.csv')
    extra=[];eb=[];keys=set()
    for row in ext.itertuples():
        f=featurize(row.SMILES)
        if f is None:continue
        keys.add(f[0])
        if f[0] not in set(p.canonical):
            extra.append({'SMILES':row.SMILES,'canonical':f[0],'scaffold':f[1],
                'count_PGK2':row.count_PGK2,'public_selective':True,'source':'public_CACHE7_DEL','library':'public_CACHE7'})
            eb.append(f[2])
    p['public_selective']=p.canonical.isin(keys)
    if extra:p=pd.concat([p,pd.DataFrame(extra)],ignore_index=True);bits.append(np.stack(eb))
    else:p=p.reset_index(drop=True)
    rb=np.concatenate(bits)
    keep=~p.canonical.duplicated();p=p.loc[keep].reset_index(drop=True);rb=rb[keep]
    pub=np.flatnonzero(p.public_selective.to_numpy());assert len(pub)>=3
    p.to_parquet(OUT/'reference_metadata.parquet',index=False);np.save(OUT/'reference_bits.npy',rb)
    n=t.loc[good&t.source.eq('ntc_presence')].sort_values('canonical').drop_duplicates('scaffold').sample(n=2000,random_state=2038)
    observed=pd.read_parquet(ROOT/'models/similarity_negative_refs.parquet')
    n=pd.concat([n,observed]).drop_duplicates('canonical')
    n=n.loc[~n.canonical.isin(p.canonical)]
    cb=tb[n.index];n.to_parquet(OUT/'control_metadata.parquet')
    print('Full validation:',len(v),'reference bank:',len(p),'public selective:',len(pub),'controls:',len(n),'workers:8',flush=True)
    signature=hashlib.sha256()
    for arr in [vb,rb,cb,pub]:signature.update(np.ascontiguousarray(arr).tobytes())
    signature.update(b'deep_audit_v1_top32_tanimoto_tversky_1_0.2')
    sig=signature.hexdigest();done=OUT/'complete.json'
    if done.exists() and json.loads(done.read_text())['signature']==sig:
        print('Verified completed full-comparison cache',flush=True);return
    ids=np.lib.format.open_memmap(OUT/'neighbor_ids.npy',mode='w+',dtype=np.int32,shape=(len(v),4,K))
    vals=np.lib.format.open_memmap(OUT/'neighbor_scores.npy',mode='w+',dtype=np.float32,shape=(len(v),4,K))
    aux=np.lib.format.open_memmap(OUT/'auxiliary_scores.npy',mode='w+',dtype=np.float32,shape=(len(v),8))
    start=time.time();batch=256
    with ProcessPoolExecutor(max_workers=8,initializer=init,initargs=(rb,cb,pub)) as workers:
        chunks=(vb[i:i+batch] for i in range(0,len(v),batch))
        for j,(ix,s,a) in enumerate(workers.map(nearest_chunk,chunks,chunksize=1)):
            begin=j*batch;end=begin+len(ix);ids[begin:end]=ix;vals[begin:end]=s;aux[begin:end]=a
            if (j+1)%40==0:print('Full comparison',end,'/',len(v),'seconds',round(time.time()-start),flush=True)
    ids.flush();vals.flush();aux.flush()
    # Recompute random rows and verify Tversky calculation against the RDKit native implementation.
    sample=np.random.default_rng(2038).choice(len(v),8,replace=False)
    init(rb,cb,pub)
    ti,ts,ta=nearest_chunk(vb[sample])
    assert np.array_equal(ti,ids[sample]) and np.allclose(ts,vals[sample]) and np.allclose(ta,aux[sample])
    for row in sample:
        q=DataStructs.CreateFromBinaryText(vb[row,0].tobytes());r=int(ids[row,2,0])
        exact=DataStructs.TverskySimilarity(q,REF[0][r],1.0,.2)
        assert abs(exact-float(vals[row,2,0]))<1e-6
    record={'signature':sig,'validation_rows':len(v),'references':len(p),'public_selective_references':len(pub),
        'control_references':len(n),'workers':8,'seconds':time.time()-start,
        'metrics':['ECFP Tanimoto','FCFP Tanimoto','ECFP Tversky alpha=1 beta=.2','FCFP Tversky alpha=1 beta=.2'],
        'verification':'Full row coverage; deterministic replay on 8 random rows; Tversky checked against native RDKit',
        'new_submissions_created':False}
    done.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
