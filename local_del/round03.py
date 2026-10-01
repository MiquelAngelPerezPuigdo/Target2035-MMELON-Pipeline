"""Three evidence-based follow-ups from the full-data and Apple-GPU analysis."""
from pathlib import Path
import numpy as np
import pandas as pd
import json,zipfile,hashlib,itertools
from rdkit import DataStructs

ROOT=Path(__file__).resolve().parent
DEEP=ROOT/'models/deep_audit'
OUT=ROOT/'submissions/round03'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_gpu(k):
    rb=np.load(DEEP/'reference_bits.npy');vb=np.load(ROOT/'features/validation_bits.npy',mmap_mode='r')
    metadata=pd.read_parquet(DEEP/'reference_metadata.parquet')
    controls=pd.read_parquet(DEEP/'control_metadata.parquet')
    tb=np.load(ROOT/'features/training_bits.npy',mmap_mode='r');cb=tb[controls.index]
    reliability=np.clip(np.log1p(metadata.count_PGK2.fillna(3).to_numpy())/np.log(4),1,4)
    lib=metadata.library.fillna('unknown');n=lib.value_counts()
    w=reliability*np.clip(np.sqrt(float(n.median())/lib.map(n).to_numpy()),.4,3);w=w/w.sum()
    public=metadata.public_selective.to_numpy().astype(float);public/=public.sum()
    for fp in [0,1]:
        refs=[DataStructs.CreateFromBinaryText(x[fp].tobytes()) for x in rb]
        ctls=[DataStructs.CreateFromBinaryText(x[fp].tobytes()) for x in cb]
        for row in np.random.default_rng(2041).choice(len(vb),8,replace=False):
            q=DataStructs.CreateFromBinaryText(vb[row,fp].tobytes())
            t=np.array(DataStructs.BulkTanimotoSimilarity(q,refs));v=np.array(DataStructs.BulkTverskySimilarity(q,refs,1,.2))
            a=np.array(DataStructs.BulkTanimotoSimilarity(q,ctls));b=np.array(DataStructs.BulkTverskySimilarity(q,ctls,1,.2))
            expected=[np.sum(t**4*w),np.sum(t**4*public),np.sum(v**8*w),np.sum(v**8*public),np.mean(a**4),np.mean(b**8),b.max()]
            assert np.isfinite(expected).all() and np.allclose(expected,k[row,fp],rtol=2e-4,atol=2e-7)


def main():
    assert (DEEP/'complete.json').exists() and (DEEP/'gpu_complete.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    v=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    k=np.load(DEEP/'gpu_kernel_scores.npy');anchors=np.load(DEEP/'known_ligand_similarity.npy')
    ml=np.load(ROOT/'models/validation_ecfp.npy')
    verify_gpu(k)
    eps=1e-8
    # Keep meaningful similarity distances; do not compress the anchor tail into percentile ranks.
    anchor_signal=np.max(.65*anchors[:,:,0]+.20*anchors[:,:,1]+.15*anchors[:,:,2],axis=1)
    selective=.60*np.log((k[:,:,3]+eps)/(k[:,:,5]+eps))+.40*np.log((k[:,:,1]+eps)/(k[:,:,4]+eps))
    general=.65*np.log((k[:,:,2]+eps)/(k[:,:,5]+eps))+.35*np.log((k[:,:,0]+eps)/(k[:,:,4]+eps))
    scores={'G_KNOWN_LIGANDS':.95*anchor_signal+.05*ml,
        'H_SELECTIVE_DEL':.70*selective[:,0]+.30*selective[:,1],
        'I_SIZE_AWARE_KERNEL':.70*general[:,0]+.30*general[:,1]}
    oldsets={}
    for rnd in [1,2]:
        with zipfile.ZipFile(ROOT/f'submissions/round0{rnd}_validation_batch.zip') as z:
            for f in z.namelist():
                if f.startswith('Team_'):oldsets[f]=set(z.read(f).decode().splitlines())
    negatives=set.union(*[s for name,s in oldsets.items() if 'C_NEIGHBORS' not in name])
    oldall=set.union(*oldsets.values())
    # Feedback is used only to avoid repeating previously evaluated zero-hit candidates.
    # Preserve raw full-library rankings for transfer to the blind test split.
    allowed=v.valid & ~v.CatalogID.isin(negatives)
    d=v.copy();d['known_ligand_signal']=anchor_signal
    d['known_ligand_ECFP_max']=anchors[:,:,0].max(1)
    d['previously_evaluated_zero_hit']=d.CatalogID.isin(negatives)
    d['public_Tversky_kernel']=k[:,0,3];d['DEL_Tversky_kernel']=k[:,0,2];d['control_Tversky_kernel']=k[:,0,5]
    manifest={'round':3,'validation_rows_scored':len(v),'candidate_prescreen':False,
        'GPU':json.loads((DEEP/'gpu_complete.json').read_text()),'sets':{},'overlap':{},
        'feedback_policy':f'Exclude {len(negatives)} IDs in five user-reported zero-hit submissions from validation selection only. No candidate is labelled positive from the one-hit aggregate result. Raw model rankings are preserved.',
        'external_data_used':True,'external_sources':['https://cache-challenge.org/challenges/finding-selective-inhibitors-of-phosphoglycerate-kinase-2'],
        'warning':'Adaptive validation experiments, not independent evidence of blind-test improvement. External public DEL candidates are not off-DNA confirmed hits. No quinazoline exclusion is imported from the separate CACHE #7 challenge.',
        'quota_before_submission':42,'quota_after_three_submissions':39,
        'score_formulas':{
          'G':'0.95 * max over crystal ligands [0.65 ECFP Tanimoto + 0.20 FCFP Tanimoto + 0.15 ECFP Tversky(1,0.2)] + 0.05 DEL-trained ECFP ensemble',
          'H':'70% ECFP + 30% FCFP; each is 60% log(public-selective TV^8 density / control TV^8 density) + 40% log(public-selective Tanimoto^4 density / control Tanimoto^4 density)',
          'I':'70% ECFP + 30% FCFP; each is 65% log(weighted-positive TV^8 density / control TV^8 density) + 35% log(weighted-positive Tanimoto^4 density / control Tanimoto^4 density)',
          'density_floor':'1e-8 added to numerator and denominator; scores are not calibrated probabilities'}}
    selectedsets={}
    for name,score in scores.items():
        assert len(score)==len(v) and np.isfinite(score).all()
        d[name+'_score']=score
        ranked=d.assign(Score=score).sort_values(['Score','CatalogID'],ascending=[False,True])
        rawtop=ranked.loc[v.valid].drop_duplicates('canonical').head(50)
        selected=ranked.loc[allowed].drop_duplicates('canonical').head(50)
        file=OUT/f'Team_MMELON_R03_{name}.txt';file.write_text('\n'.join(selected.CatalogID.astype(str))+'\n')
        selected.to_csv(OUT/f'{name}_details.csv',index=False)
        ids=file.read_text().splitlines();assert len(ids)==50 and len(set(ids))==50
        assert set(ids)<=set(v.CatalogID) and not set(ids)&negatives
        assert selected.canonical.nunique()==50
        selectedsets[name]=set(ids)
        manifest['sets'][name]={'file':file.name,'sha256':sha(file),'unique_scaffolds':selected.scaffold.nunique(),
            'replaced_known_zero_ids_from_raw_top50':int(rawtop.previously_evaluated_zero_hit.sum()),
            'new_candidates':len(set(ids)-oldall),
            'overlap_with_previous_sets':{n:len(set(ids)&s) for n,s in oldsets.items()}}
    for a,b in itertools.combinations(selectedsets,2):manifest['overlap'][a+'__'+b]=len(selectedsets[a]&selectedsets[b])
    manifest['distinct_candidates']=len(set.union(*selectedsets.values()))
    manifest['new_candidates']=len(set.union(*selectedsets.values())-oldall)
    manifest['verification']=['Full candidate coverage; finite GPU-derived scores','GPU densities checked against CPU RDKit on 16 fingerprint/molecule pairs',
        'Exactly three files of 50 valid unique CatalogIDs and distinct parent structures','All 196 previously zero-hit IDs excluded from selected validation sets']
    manifest['provenance']={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'gpu_kernel.py',ROOT/'deep_audit.py',ROOT/'external/pgk2/verified_ligands.json',ROOT/'external/pgk2/DEL_hit_candidates_1.csv']}
    d.to_parquet(DEEP/'round03_full_rankings.parquet',index=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=ROOT/'reports/official_results.json';r=json.loads(results.read_text())
    r.setdefault('round03',{name:{'hits':None,'chemical_series':None,'p_value':None,'submission_id':None} for name in scores})
    r['quota_estimate']['remaining_after_planned_round03']=39
    results.write_text(json.dumps(r,indent=2)+'\n')
    text='Round 3: submit the three Team_MMELON_R03_*.txt files to VALIDATION.\n\nG: public crystal-bound PGK2 ligand similarity plus DEL ML.\nH: public PGK2-selective DEL signal versus NTC controls.\nI: size-aware, read- and library-weighted DEL signal versus controls.\n\nAll 244,328 validation candidates were scored. Exactly 50 IDs per submission; 196 IDs from prior zero-hit submissions are excluded. The one earlier hit is still unidentified. These are adaptive validation experiments, not proof of improved blind-test performance. External public data are used in this round.\n\nSend G/H/I hit counts and cluster counts, plus p-values if available. 42 validations remain before these submissions, 39 afterward.\n'
    (OUT/'START_HERE.txt').write_text(text)
    dest=ROOT/'submissions/round03_validation_batch.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob('Team_*.txt')):z.write(p,p.name)
        z.write(OUT/'START_HERE.txt','START_HERE.txt')
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None and len([x for x in z.namelist() if x.startswith('Team_')])==3
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
