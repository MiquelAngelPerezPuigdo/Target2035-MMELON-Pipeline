"""Broader DEL-neighbour comparisons after round 1 feedback. Local computation only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,time,zipfile,itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from rdkit import DataStructs

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'submissions/round02'
CACHE=ROOT/'models/round02'
REFS=None
CONFIG={'seed':2037,'prescreen_top_per_signal':20000,'positive_references':'All eligible strict DEL positives, excluding NTC supplement overlap',
 'ntc_reference_max':1000,'workers':4,'chunk_size':256,
 'D':'0.85 percentile(mean top-3 ECFP positive similarity) + 0.15 percentile(round01 ECFP model score)',
 'E':'0.85 percentile(mean top-3 FCFP positive similarity) + 0.15 percentile(round01 FCFP model score)',
 'F':'0.50 percentile(ECFP positive similarity - 0.15 control similarity) + 0.35 percentile(FCFP positive similarity - 0.15 control similarity) + 0.15 percentile(round01 ECFP model score)',
 'selection':'Top 50 distinct non-chiral parent structures; no imposed scaffold diversity quota; no exclusions or forced retention based on round01 memberships',
 'reference_labels':'DEL only; no candidate-level activity labels inferred from aggregate feedback'}


def pct(x):return rankdata(x,method='average')/len(x)


def initialize(pos,neg):
    global REFS
    REFS=([[DataStructs.CreateFromBinaryText(r[k].tobytes()) for r in pos] for k in (0,1)],
          [[DataStructs.CreateFromBinaryText(r[k].tobytes()) for r in neg] for k in (0,1)])


def chunk_scores(bits):
    out=np.zeros((len(bits),6),dtype=np.float32)
    for j,row in enumerate(bits):
        for k in (0,1):
            fp=DataStructs.CreateFromBinaryText(row[k].tobytes())
            p=np.asarray(DataStructs.BulkTanimotoSimilarity(fp,REFS[0][k]),dtype=np.float32)
            n=np.asarray(DataStructs.BulkTanimotoSimilarity(fp,REFS[1][k]),dtype=np.float32)
            out[j,k]=np.partition(p,-3)[-3:].mean()
            out[j,k+2]=n.max()
            out[j,k+4]=p.max()
    return out


def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    t=pd.read_parquet(ROOT/'features/training_meta.parquet')
    v=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    good=np.load(ROOT/'data/eligible.npy')
    tb=np.load(ROOT/'features/training_bits.npy',mmap_mode='r')
    vb=np.load(ROOT/'features/validation_bits.npy',mmap_mode='r')
    a=np.load(ROOT/'models/validation_ecfp.npy');b=np.load(ROOT/'models/validation_fcfp.npy')
    old=np.load(ROOT/'models/validation_similarity.npy')
    prior_signal=.5*old[:,0]+.5*old[:,1]-.15*old[:,2]-.15*old[:,3]
    # Prescreen all 244,328 candidates using complementary existing signals.
    # This union is fixed before expensive all-positive-reference neighbour scoring.
    pool=np.unique(np.concatenate([np.argsort(-s,kind='stable')[:CONFIG['prescreen_top_per_signal']]
        for s in [a,b,old[:,0],old[:,1],prior_signal]]))
    pool=pool[v.valid.to_numpy()[pool]]
    pos=t.loc[good&t.strict.eq(1)&~t.ntc_present.fillna(False)].sort_values('canonical')
    control=t.loc[good&t.source.eq('ntc_presence')].sort_values('canonical').drop_duplicates('scaffold')
    control=control.sample(n=min(CONFIG['ntc_reference_max'],len(control)),random_state=CONFIG['seed'])
    observed=pd.read_parquet(ROOT/'models/similarity_negative_refs.parquet')
    neg=pd.concat([control,observed]).drop_duplicates('canonical').sort_values('canonical')
    pi=pos.index.to_numpy();ni=neg.index.to_numpy()
    assert pos.canonical.is_unique and not set(pos.canonical)&set(neg.canonical)
    assert np.all(good[pi]) and np.all(good[ni])
    pos.to_parquet(CACHE/'positive_references.parquet');neg.to_parquet(CACHE/'control_references.parquet')
    np.save(CACHE/'candidate_rows.npy',pool)
    print('Prescreened',len(pool),'of',len(v),'candidates; positive references',len(pi),'control references',len(ni),flush=True)
    signature=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode())
    for arr in [pool,vb[pool],tb[pi],tb[ni]]:signature.update(np.ascontiguousarray(arr).tobytes())
    sig=signature.hexdigest();sigfile=CACHE/'scores.sha256';scorefile=CACHE/'similarities.npy'
    if scorefile.exists() and sigfile.exists() and sigfile.read_text().strip()==sig:
        s=np.load(scorefile)
    else:
        pieces=[];start=time.time()
        with ProcessPoolExecutor(max_workers=CONFIG['workers'],initializer=initialize,initargs=(tb[pi],tb[ni])) as workers:
            chunks=(vb[pool[i:i+CONFIG['chunk_size']]] for i in range(0,len(pool),CONFIG['chunk_size']))
            for j,result in enumerate(workers.map(chunk_scores,chunks,chunksize=1)):
                pieces.append(result)
                if (j+1)%20==0:print('Scored',min((j+1)*CONFIG['chunk_size'],len(pool)),'/',len(pool),'seconds',round(time.time()-start),flush=True)
        s=np.concatenate(pieces);np.save(scorefile,s);sigfile.write_text(sig+'\n')
    assert s.shape==(len(pool),6) and np.isfinite(s).all() and np.all((s>=0)&(s<=1))
    scores={
      'D_STRUCTURAL':.85*pct(s[:,0])+.15*pct(a[pool]),
      'E_CHEMICAL_FEATURES':.85*pct(s[:,1])+.15*pct(b[pool]),
      'F_CONTROL_CONTRAST':.50*pct(s[:,0]-.15*s[:,2])+.35*pct(s[:,1]-.15*s[:,3])+.15*pct(a[pool])}
    candidates=v.iloc[pool].copy()
    candidates['ECFP_positive_top3']=s[:,0];candidates['FCFP_positive_top3']=s[:,1]
    candidates['ECFP_control_max']=s[:,2];candidates['FCFP_control_max']=s[:,3]
    candidates['ECFP_positive_max']=s[:,4];candidates['FCFP_positive_max']=s[:,5]
    candidates['ECFP_model_score']=a[pool];candidates['FCFP_model_score']=b[pool]
    oldsets={}
    with zipfile.ZipFile(ROOT/'submissions/round01_validation_batch.zip') as z:
        for n in z.namelist():
            if n.startswith('Team_'):oldsets[n]=set(z.read(n).decode().splitlines())
    manifest={'round':2,'config':CONFIG,'prescreen_n':len(pool),'total_validation_n':len(v),
      'positive_references':len(pi),'control_references':len(ni),'sets':{},'between_set_overlap':{},
      'feedback_used':'A=0, B=0, C=1 hit / 1 cluster, p=0.05201186181800377. Used only to choose three neighbour-heavy workflow variants.',
      'limitation':'All-reference neighbour computation is limited to a prescreened union; candidates outside that union could be missed.',
      'no_new_internal_accuracy_claim':'These are exploratory ranking variants; performance must be judged on official feedback, not the previous DEL holdout metrics.'}
    sets={}
    for name,score in scores.items():
        ranked=candidates.assign(Score=score).sort_values(['Score','CatalogID'],ascending=[False,True]).drop_duplicates('canonical')
        top=ranked.head(50);assert len(top)==50 and top.CatalogID.nunique()==50
        file=OUT/f'Team_MMELON_R02_{name}.txt'
        file.write_text('\n'.join(top.CatalogID.astype(str))+'\n')
        top.to_csv(OUT/f'{name}_details.csv',index=False)
        sets[name]=set(top.CatalogID)
        manifest['sets'][name]={'file':file.name,'count':50,'sha256':sha_file(file),
          'unique_scaffolds':top.scaffold.nunique(),'round01_overlap':{n:len(sets[name]&ids) for n,ids in oldsets.items()},
          'median_ECFP_positive_top3':float(top.ECFP_positive_top3.median()),
          'median_FCFP_positive_top3':float(top.FCFP_positive_top3.median())}
        candidates[name+'_score']=score
    for i,j in itertools.combinations(sets,2):manifest['between_set_overlap'][i+'__'+j]=len(sets[i]&sets[j])
    allids=set.union(*sets.values());oldall=set.union(*oldsets.values())
    manifest['distinct_candidates']=len(allids);manifest['new_candidates_vs_round01']=len(allids-oldall)
    candidates.to_parquet(CACHE/'rankings.parquet',index=False)
    # Independently recompute a small random sample in the parent process.
    check=np.random.default_rng(CONFIG['seed']).choice(len(pool),12,replace=False)
    initialize(tb[pi],tb[ni]);assert np.allclose(chunk_scores(vb[pool[check]]),s[check],atol=1e-7)
    manifest['verification']=['Exact neighbour-score replay on 12 randomly selected candidates',
      'Reference labels, eligibility and positive/control disjointness checked',
      'Three serialized files each have 50 unique valid CatalogIDs and distinct parent structures']
    for name,item in manifest['sets'].items():
        ids=(OUT/item['file']).read_text().splitlines()
        assert len(ids)==50 and len(set(ids))==50 and set(ids)<=set(v.CatalogID)
        assert v.set_index('CatalogID').loc[ids,'canonical'].nunique()==50
    manifest['code_sha256']=sha_file(Path(__file__))
    manifest['round01_zip_sha256']=sha_file(ROOT/'submissions/round01_validation_batch.zip')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=ROOT/'reports/official_results.json';r=json.loads(results.read_text())
    if 'round02' not in r:r['round02']={name:{'submission_id':None,'hits':None,'chemical_series':None,'p_value':None} for name in scores}
    results.write_text(json.dumps(r,indent=2)+'\n')
    (OUT/'START_HERE.txt').write_text('Round 2: upload the THREE Team_MMELON_R02_*.txt files to VALIDATION, one per submission.\n\nD: Structural neighbours, expanded DEL hit reference set.\nE: Chemical-feature neighbours, expanded DEL hit reference set.\nF: Structural and chemical-feature neighbours with control-binding contrast.\n\nEach has 50 checked CatalogIDs. Report D/E/F hit counts and cluster counts (p-values and submission IDs helpful). No files were submitted automatically. These are exploratory variants, not proven improvements. The same Team_MMELON filename prefix is retained from your repository.\n')
    archive=ROOT/'submissions/round02_validation_batch.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.glob('Team_*.txt')):z.write(f,f.name)
        z.write(OUT/'START_HERE.txt','START_HERE.txt')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len([n for n in z.namelist() if n.startswith('Team_')])==3
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
