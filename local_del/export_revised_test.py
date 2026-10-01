"""Export one replacement T2, incorporating authorized first-test feedback."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from rdkit import Chem

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'submissions/blind_test_revised'

def aromatic_core(smiles):
    mol=Chem.MolFromSmiles(smiles)
    # Aromatic connected components join directly linked aromatic rings. Remove
    # saturated appendages; cap the largest aromatic component, not full scaffold.
    aromatic={a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    components=[]
    while aromatic:
        seed=min(aromatic);component={seed};todo=[seed];aromatic.remove(seed)
        while todo:
            current=todo.pop()
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                j=neighbor.GetIdx()
                if j in aromatic:aromatic.remove(j);component.add(j);todo.append(j)
        components.append(component)
    if not components:return '__NO_AROMATIC__'
    fragments=[(len(c),Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(c),isomericSmiles=False)) for c in components]
    return sorted(fragments,key=lambda v:(-v[0],v[1]))[0][1]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(ROOT/'features/test_meta.parquet')
    source=pd.read_csv(ROOT.parent/'Val-Test-set/PGK2_Test_split.csv')
    t1=pd.read_csv(ROOT/'submissions/blind_test/Team_MMELON_T1_FROZEN_G.csv')
    old_t2=pd.read_csv(ROOT/'submissions/blind_test/Team_MMELON_T2_DIVERSE_TRANSFER.csv')
    assert meta.CatalogID.equals(source.CatalogID) and t1.CatalogID.equals(source.CatalogID)
    anchors=np.load(ROOT/'reports/ood/test_anchor_similarity.npy')
    a=.65*anchors[:,:,0]+.20*anchors[:,:,1]+.15*anchors[:,:,2]
    ml=(t1.Score.to_numpy()-.95*a.max(1))/.05
    assert ml.min()>-1e-8 and ml.max()<1.00000001
    ml=np.clip(ml,0,1)
    logits=np.load(ROOT/'models/test_count_revision/test_logits_ensemble.npy')
    mean=logits.mean(0);disagreement=logits.std(0)
    count_score=expit(mean-.25*disagreement)
    score=.85*count_score+.10*a[:,0]+.05*ml
    negatives=t1.Sel_50.eq(1).to_numpy()
    negative_parents=set(meta.loc[negatives,'canonical'])
    allowed=meta.valid.to_numpy() & ~meta.canonical.isin(negative_parents).to_numpy()
    score[negatives]=0
    x=sparse.load_npz(ROOT/'features/test_ecfp.npz').tocsr()
    selected=[];parents=set();scaffolds=Counter();cores=Counter();coremap={}
    for row in np.lexsort((meta.CatalogID.to_numpy(),-score)):
        if not allowed[row] or meta.canonical.iat[row] in parents:continue
        scaffold=meta.scaffold.iat[row]
        if scaffolds[scaffold]>=2:continue
        core=aromatic_core(meta.SMILES.iat[row]);coremap[int(row)]=core
        if cores[core]>=8:continue
        if selected:
            query=x[row];refs=x[selected];dot=(refs@query.T).toarray().ravel()
            sim=dot/(np.asarray(refs.sum(1)).ravel()+query.sum()-dot)
            if sim.max()>=.60:continue
        selected.append(int(row));parents.add(meta.canonical.iat[row]);scaffolds[scaffold]+=1;cores[core]+=1
        if len(selected)==50:break
    assert len(selected)==50 and not negatives[selected].any()
    flags=np.zeros(len(meta),dtype=int);flags[selected]=1
    csv=OUT/'Team_MMELON_T2_REVISED_COUNTS.csv'
    pd.DataFrame({'CatalogID':meta.CatalogID,'Sel_50':flags,'Score':score}).to_csv(csv,index=False,float_format='%.12g')
    detail=meta.iloc[selected].copy();detail['Score']=score[selected]
    detail['count_model_score']=count_score[selected];detail['model_disagreement']=disagreement[selected]
    detail['ligand21_similarity']=a[selected,0];detail['DEL_score']=ml[selected]
    detail['aromatic_core']=[coremap[r] for r in selected]
    detail.to_csv(OUT/'selected_candidates_details.csv',index=False)
    reread=pd.read_csv(csv)
    assert list(reread)==['CatalogID','Sel_50','Score'] and len(reread)==184632
    assert reread.CatalogID.equals(source.CatalogID) and reread.CatalogID.is_unique
    assert reread.Sel_50.isin([0,1]).all() and reread.Sel_50.sum()==50
    assert np.isfinite(reread.Score).all() and (reread.loc[negatives,'Score']==0).all()
    report={'file':csv.name,'sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),
            'rows':len(reread),'selected':50,'overlap_with_failed_T1':int(flags[negatives].sum()),
            'overlap_with_old_unsubmitted_T2':int(flags[old_t2.Sel_50.eq(1)].sum()),
            'distinct_Murcko_scaffolds':len(scaffolds),'aromatic_core_counts':dict(cores),
            'formula':'85% sigmoid(mean six-model logit minus 0.25 logit standard deviation) + 10% ligand21 raw similarity + 5% frozen DEL ensemble',
            'selection':'Exclude all T1 parent structures; unique parent structures; pairwise ECFP <0.60; at most two per Murcko and eight per largest connected aromatic core',
            'feedback_permission':'User explicitly confirmed first-test feedback may be used to revise remaining test submission.',
            'calibration':'Score is uncalibrated ranking confidence; no numerical forecast of test hits.',
            'status':'Prepared locally, not submitted','boltz_spent_usd':0}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'START_HERE.txt').write_text('REPLACEMENT FOR THE UNSUBMITTED T2\nSubmit only Team_MMELON_T2_REVISED_COUNTS.csv to TEST.\nDo not submit the old T2_DIVERSE_TRANSFER.csv.\nOne test submission remains before this file, zero afterward.\nAll 184,632 original test IDs are retained; exactly 50 new candidates selected; no T1-selected molecule is selected again.\nThis revision uses authorized T1 feedback. No Boltz credit was spent.\n')
    ledger=ROOT/'reports/official_results.json';history=json.loads(ledger.read_text())
    history['blind_test_revision']={'file':str(csv.relative_to(ROOT)),**report}
    history['blind_test_preparation']['status']='T1 evaluated; original T2 unsubmitted and superseded by T2_REVISED_COUNTS'
    ledger.write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
