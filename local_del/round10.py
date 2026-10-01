"""Identify Y's exact hit with two coded tests and probe fresh Y analogues."""
from pathlib import Path
import hashlib, json, zipfile
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"submissions/round10"
GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

def archive_ids(rnd,marker):
    with zipfile.ZipFile(ROOT/f"submissions/round{rnd:02d}_validation_batch.zip") as z:
        n=next(x for x in z.namelist() if x.startswith("Team_") and marker in x)
        return z.read(n).decode().splitlines()

def all_seen():
    seen=set()
    for rnd in range(1,10):
        with zipfile.ZipFile(ROOT/f"submissions/round{rnd:02d}_validation_batch.zip") as z:
            for n in z.namelist():
                if n.startswith("Team_"):seen.update(z.read(n).decode().splitlines())
    return seen

def support(query,refs):
    out=np.zeros((len(query),6),dtype=np.float32)
    for kind in [0,1]:
        targets=[DataStructs.CreateFromBinaryText(x[kind].tobytes()) for x in refs]
        for i,p in enumerate(query[:,kind]):
            fp=DataStructs.CreateFromBinaryText(p.tobytes())
            sims=np.sort(np.asarray(DataStructs.BulkTanimotoSimilarity(fp,targets),dtype=np.float32))
            out[i,3*kind:3*kind+3]=[sims[-1],sims[-min(3,len(sims)):].mean(),sims.mean()]
    return out

def diverse_pick(frame,n=50,cutoff=.55):
    chosen=[];fps=[]
    for idx,row in frame.iterrows():
        fp=GEN.GetFingerprint(Chem.MolFromSmiles(row.SMILES))
        if fps and max(DataStructs.BulkTanimotoSimilarity(fp,fps))>=cutoff:continue
        chosen.append(idx);fps.append(fp)
        if len(chosen)==n:break
    assert len(chosen)==n
    return frame.loc[chosen]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for old in OUT.glob("Team_*.txt"):old.unlink()
    v=pd.read_parquet(ROOT/"features/validation_meta.parquet")
    bits=np.load(ROOT/"features/validation_bits.npy",mmap_mode="r")
    y=archive_ids(9,"Y_X_THIENO4")[:4]
    z=archive_ids(9,"Z_X_ISOXAZOLE3")[:3]
    aa=archive_ids(9,"AA_X_SINGLETONS3")[:3]
    j=archive_ids(4,"J_LIGAND47")
    n_ids=archive_ids(5,"N_GROUP_A_B")
    o_ids=archive_ids(5,"O_DIVERSE_BAG")
    s15=archive_ids(8,"V_S_MAIN15")[:15]
    w7=archive_ids(8,"W_T_MAIN7")[:7]
    assert len(y)==4 and len(set(y))==4
    yref=bits[v.index[v.CatalogID.isin(y)]]
    nref=bits[v.index[v.CatalogID.isin(set(z+aa))]]
    ys=support(bits,yref); ns=support(bits,nref)
    positive=.45*ys[:,0]+.20*ys[:,1]+.25*ys[:,3]+.10*ys[:,4]
    negative=.60*ns[:,0]+.40*ns[:,3]
    score=positive-.12*negative
    seen=all_seen()
    expansion=(v.assign(Score=score,Y_ECFP_max=ys[:,0],Y_FCFP_max=ys[:,3],negative_similarity=negative)
        .loc[v.valid & ~v.CatalogID.isin(seen)]
        .sort_values(["Score","CatalogID"],ascending=[False,True])
        .drop_duplicates("canonical").head(50))
    assert len(expansion)==50
    # A separate discovery arm uses the older multi-cluster N/O bags, while
    # suppressing the now-exhausted S/W core neighborhood and enforcing breadth.
    broad_ref=bits[v.index[v.CatalogID.isin(set(n_ids+o_ids))]]
    dominant_ref=bits[v.index[v.CatalogID.isin(set(s15+w7))]]
    broad=support(bits,broad_ref); dominant=support(bits,dominant_ref)
    broad_score=.40*broad[:,0]+.20*broad[:,1]+.25*broad[:,3]+.15*broad[:,4]-.18*(.6*dominant[:,0]+.4*dominant[:,3])
    exploration=(v.assign(Score=broad_score,broad_ECFP_max=broad[:,0],broad_FCFP_max=broad[:,3],dominant_similarity=.6*dominant[:,0]+.4*dominant[:,3])
        .loc[v.valid & ~v.CatalogID.isin(seen|set(expansion.CatalogID))]
        .sort_values(["Score","CatalogID"],ascending=[False,True]).drop_duplicates("canonical"))
    exploration=diverse_pick(exploration,50,.55)
    candidates={
        "AB_Y_PAIR_SPLIT":[y[0],y[1]],
        "AC_Y4_EXPANSION":expansion.CatalogID.tolist(),
        "AD_SERIES_EXPLORATION":exploration.CatalogID.tolist(),
    }
    manifest={
        "round":10,"known_fact":"Exactly one of the four ordered Y candidates is active.",
        "revision":"Exploration-weighted replacement; do not submit the earlier AB/AC diagnostic version.",
        "AB_decoding":"AB=1 means the hit is Y candidate 1 or 2; AB=0 means it is Y candidate 3 or 4.",
        "AC_strategy":"50 new molecules ranked by ECFP/FCFP support from Y4, penalizing similarity to confirmed-negative Z/AA.",
        "AD_strategy":"Scaffold-diverse search supported by multi-cluster N/O bags, penalizing the exhausted S/W dominant series.",
        "sets":{},"remaining_before":21,"remaining_after":18,
    }
    for name,cand in candidates.items():
        fillers=[] if len(cand)==50 else j[:50-len(cand)]
        ids=cand+fillers
        assert len(ids)==len(set(ids))==50
        file=OUT/f"Team_MMELON_R10_{name}.txt";file.write_text("\n".join(ids)+"\n")
        detail=v.set_index("CatalogID",drop=False).loc[ids].copy()
        detail["role"]=["candidate"]*len(cand)+["proven_zero_J_filler"]*len(fillers)
        if name=="AC_Y4_EXPANSION":
            lookup=expansion.set_index("CatalogID")
            for col in ["Score","Y_ECFP_max","Y_FCFP_max","negative_similarity"]:
                detail[col]=detail["CatalogID"].map(lookup[col])
        if name=="AD_SERIES_EXPLORATION":
            lookup=exploration.set_index("CatalogID")
            for col in ["Score","broad_ECFP_max","broad_FCFP_max","dominant_similarity"]:
                detail[col]=detail["CatalogID"].map(lookup[col])
        detail.to_csv(OUT/f"{name}_details.csv",index=False)
        manifest["sets"][name]={"file":file.name,"real_candidates":len(cand),"zero_fillers":len(fillers),"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
    manifest["verification"]=["AB narrows the Y hit to one of two molecules","AC and AD contain 100 mutually disjoint new valid structures","AD uses a fingerprint diversity gate","All diagnostic padding comes from zero-hit J"]
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"START_HERE.txt").write_text(
        "Round 10 REVISED — submit AB, AC and AD separately to VALIDATION. Do not use an older round10 ZIP.\n\n"
        "AB splits Y's four-member one-hit bag into two halves. AC tests 50 new Y-supported analogues. AD searches broadly for additional series using the multi-cluster N/O evidence while avoiding the exhausted S/W core. Report hits, clusters and p-values in AB/AC/AD order. 21 validations remain before this batch; 18 afterward.\n")
    archive=ROOT/"submissions/round10_validation_batch.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.glob("Team_*.txt")):z.write(f,f.name)
        z.write(OUT/"START_HERE.txt","START_HERE.txt")
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len([n for n in z.namelist() if n.startswith("Team_")])==3
    p=ROOT/"reports/official_results.json";r=json.loads(p.read_text())
    r["round10"]={k:{"hits":None,"chemical_series":None,"p_value":None,"submission_id":None} for k in candidates}
    r["quota_estimate"]["remaining_now"]=21;r["quota_estimate"]["last_confirmed_after_round"]=9;r["quota_estimate"]["remaining_after_planned_round10"]=18
    p.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
