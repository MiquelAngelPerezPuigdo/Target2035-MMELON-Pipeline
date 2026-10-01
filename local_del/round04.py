"""Prospective validation stress tests for transfer beyond the successful ligand series."""
from pathlib import Path
import json,zipfile,itertools,hashlib
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'submissions/round04'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    v=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    bits=np.load(ROOT/'features/validation_bits.npy',mmap_mode='r')
    anchors=np.load(ROOT/'models/deep_audit/known_ligand_similarity.npy')
    ml=np.load(ROOT/'models/validation_ecfp.npy')
    oldsets={}
    for rnd in [1,2,3]:
        with zipfile.ZipFile(ROOT/f'submissions/round0{rnd}_validation_batch.zip') as z:
            for n in z.namelist():
                if n.startswith('Team_'):oldsets[n]=set(z.read(n).decode().splitlines())
    seen=set.union(*oldsets.values())
    g=next(ids for n,ids in oldsets.items() if 'G_KNOWN_LIGANDS' in n)
    base=v.valid & ~v.CatalogID.isin(seen)
    # Distances to the entire old selection are used; hit identities remain unknown.
    fps=[DataStructs.CreateFromBinaryText(row[0].tobytes()) for row in bits]
    similarity_to_g=np.zeros(len(v),dtype=np.float32)
    for row in np.flatnonzero(v.CatalogID.isin(g)):
        similarity_to_g=np.maximum(similarity_to_g,np.asarray(DataStructs.BulkTanimotoSimilarity(fps[row],fps),dtype=np.float32))
    np.save(ROOT/'reports/ood/validation_similarity_to_G.npy',similarity_to_g)
    frozen=.95*np.max(.65*anchors[:,:,0]+.20*anchors[:,:,1]+.15*anchors[:,:,2],axis=1)+.05*ml
    ligand47=.95*(.65*anchors[:,1,0]+.20*anchors[:,1,1]+.15*anchors[:,1,2])+.05*ml
    corehop=.95*np.max(.30*anchors[:,:,0]+.45*anchors[:,:,1]+.25*anchors[:,:,3],axis=1)+.05*ml
    # Structural-core check only needs the high-ranked candidate pool for core-hopping.
    core=Chem.MolFromSmiles('c1ccc2ncncc2c1')
    noncore=np.zeros(len(v),dtype=bool)
    order=np.argsort(-corehop,kind='stable')
    checked=0;kept=0
    for row in order:
        if not base.iloc[row]:continue
        m=Chem.MolFromSmiles(v.SMILES.iloc[row]);checked+=1
        if not m.HasSubstructMatch(core):noncore[row]=True;kept+=1
        if kept>=300:break
    # 0.36 matches the observed nearest-anchor scale in the unlabelled test chemistry audit.
    # The additional 0.40 gate avoids close analogues of any of the original 50 predictions.
    distant=base & (anchors[:,:,0].max(axis=1)<=.36) & (similarity_to_g<=.40)
    specs={'J_LIGAND47':(ligand47,base),
           'K_NEW_CORES':(corehop,base&noncore),
           'L_DISTANT_TRANSFER':(frozen,distant)}
    manifest={'round':4,'purpose':'Transfer stress tests, not optimization for repeating the 25 validation hits.',
      'previously_submitted_IDs_excluded':len(seen),'hit_identities_known':False,
      'test_activity_labels_accessed':False,'test_submissions_made':0,
      'unlabelled_test_usage':'The structural distance audit motivated the <=0.36 anchor-similarity stress gate. No activity outcomes were used.',
      'definitions':{
        'J':'Fresh candidates ranked against crystal ligand 47 only, using frozen 65/20/15 ligand metrics and 5% DEL model contribution.',
        'K':'Fresh candidates without a quinazoline core; 30% structural Tanimoto, 45% feature Tanimoto, 25% feature Tversky, across the two ligands; 5% DEL contribution.',
        'L':'Frozen winning score on fresh candidates with maximum reference ECFP similarity <=0.36 and maximum ECFP similarity to any prior G candidate <=0.40.'},
      'caveat':'These gates create harder validation experiments, but do not reproduce the organizers hidden test series or prove OOD generalization. Expected validation yield may be lower.',
      'sets':{},'overlap':{},'remaining_before_submission':39,'remaining_after_three_submissions':36}
    groups={}
    for name,(score,mask) in specs.items():
        d=v.assign(Score=score,nearest_reference_ECFP=anchors[:,:,0].max(axis=1),nearest_previous_G_ECFP=similarity_to_g)
        top=d.loc[mask].sort_values(['Score','CatalogID'],ascending=[False,True]).drop_duplicates('canonical').head(50)
        assert len(top)==50 and top.CatalogID.nunique()==50 and top.canonical.nunique()==50
        assert np.isfinite(top.Score).all() and not set(top.CatalogID)&seen
        if name.startswith('K'):assert all(not Chem.MolFromSmiles(s).HasSubstructMatch(core) for s in top.SMILES)
        if name.startswith('L'):assert (top.nearest_reference_ECFP<=.36).all() and (top.nearest_previous_G_ECFP<=.40).all()
        f=OUT/f'Team_MMELON_R04_{name}.txt';f.write_text('\n'.join(top.CatalogID.astype(str))+'\n')
        assert len(f.read_text().splitlines())==50
        top.to_csv(OUT/f'{name}_details.csv',index=False)
        groups[name]=set(top.CatalogID)
        manifest['sets'][name]={'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
          'unique_Murcko_scaffolds':top.scaffold.nunique(),'quinazoline_core_count':sum(Chem.MolFromSmiles(s).HasSubstructMatch(core) for s in top.SMILES),
          'median_nearest_reference_ECFP':float(top.nearest_reference_ECFP.median()),
          'median_nearest_G_ECFP':float(top.nearest_previous_G_ECFP.median()),'previously_submitted_candidates':0}
    for a,b in itertools.combinations(groups,2):manifest['overlap'][a+'__'+b]=len(groups[a]&groups[b])
    manifest['unique_new_candidates']=len(set.union(*groups.values()))
    manifest['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rpath=ROOT/'reports/official_results.json';r=json.loads(rpath.read_text())
    r.setdefault('round04',{name:{'hits':None,'chemical_series':None,'p_value':None,'submission_id':None} for name in groups})
    r['quota_estimate']['remaining_after_planned_round04']=36;rpath.write_text(json.dumps(r,indent=2)+'\n')
    (OUT/'START_HERE.txt').write_text('Round 4 — VALIDATION only. Submit all three Team_MMELON_R04_*.txt files separately.\n\nJ: fresh predictions from ligand 47.\nK: predictions with different cores (no quinazoline).\nL: predictions distant from the original successful set and reference ligands.\n\nThese are deliberately harder transfer tests. None of the selected IDs occurred in our earlier submissions. Lower hit counts are possible; report hits AND clusters for J/K/L. We are preserving the successful G method unchanged. No test submission has been made. 39 validations before this batch; 36 after.\n')
    archive=ROOT/'submissions/round04_validation_batch.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.glob('Team_*.txt')):z.write(f,f.name)
        z.write(OUT/'START_HERE.txt','START_HERE.txt')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len([n for n in z.namelist() if n.startswith('Team_')])==3
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
