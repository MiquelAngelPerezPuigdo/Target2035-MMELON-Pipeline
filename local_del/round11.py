"""Resolve the two dense hit series in the original 21-of-39 K-minus-J bag."""
from pathlib import Path
import hashlib,json,zipfile
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"submissions/round11"
GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

def ids(rnd,marker):
    with zipfile.ZipFile(ROOT/f"submissions/round{rnd:02d}_validation_batch.zip") as z:
        n=next(x for x in z.namelist() if x.startswith("Team_") and marker in x)
        return z.read(n).decode().splitlines()

def groups(frame):
    fps=[GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in frame.SMILES];dist=[]
    for i in range(1,len(fps)):dist.extend(1-x for x in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]))
    return sorted(Butina.ClusterData(dist,len(fps),.5,isDistData=True),key=len,reverse=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for f in OUT.glob("Team_*.txt"):f.unlink()
    v=pd.read_parquet(ROOT/"features/validation_meta.parquet").set_index("CatalogID",drop=False)
    k=set(ids(4,"K_NEW_CORES"));j=ids(4,"J_LIGAND47")
    bag=v.loc[sorted(k-set(j))];g=groups(bag)
    assert len(bag)==39 and [len(x) for x in g[:2]]==[10,7]
    selections={
        "AE_K_GROUP10":bag.iloc[list(g[0])].CatalogID.tolist(),
        "AF_K_GROUP7":bag.iloc[list(g[1])].CatalogID.tolist(),
        "AG_K_REST22":bag.iloc[[i for x in g[2:] for i in x]].CatalogID.tolist(),
    }
    assert set().union(*map(set,selections.values()))==set(bag.CatalogID)
    manifest={"round":11,"known_fact":"K minus zero-hit J contains exactly 21 hits in two assay clusters among 39 molecules.",
      "strategy":"Test its two largest fingerprint groups separately and place all remaining groups in a third set.",
      "expected_check":"AE + AF + AG must total 21 hits; their assay-cluster union must explain K's two clusters.",
      "sets":{},"remaining_before":18,"remaining_after":15}
    for name,cand in selections.items():
        fillers=j[:50-len(cand)];out=cand+fillers
        assert len(out)==len(set(out))==50 and not set(cand)&set(fillers)
        f=OUT/f"Team_MMELON_R11_{name}.txt";f.write_text("\n".join(out)+"\n")
        d=v.loc[out].copy();d["role"]=["K39_candidate"]*len(cand)+["proven_zero_J_filler"]*len(fillers);d.to_csv(OUT/f"{name}_details.csv",index=False)
        manifest["sets"][name]={"file":f.name,"K39_candidates":len(cand),"zero_fillers":len(fillers),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()}
    manifest["verification"]=["The three sets exactly partition all 39 informative K molecules","Every filler is from independently zero-hit J","Each submission has 50 unique valid IDs"]
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"START_HERE.txt").write_text("Round 11 — submit AE, AF and AG separately to VALIDATION.\n\nThese sets partition all 39 informative K-minus-J molecules. K-minus-J has exactly 21 hits in two assay clusters, so AE+AF+AG must equal 21 hits. Report results in AE/AF/AG order. 18 validations remain before this batch; 15 afterward.\n")
    archive=ROOT/"submissions/round11_validation_batch.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.glob("Team_*.txt")):z.write(f,f.name)
        z.write(OUT/"START_HERE.txt","START_HERE.txt")
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len([n for n in z.namelist() if n.startswith("Team_")])==3
    p=ROOT/"reports/official_results.json";r=json.loads(p.read_text());r["round11"]={x:{"hits":None,"chemical_series":None,"p_value":None,"submission_id":None} for x in selections}
    r["quota_estimate"]["remaining_now"]=18;r["quota_estimate"]["last_confirmed_after_round"]=10;r["quota_estimate"]["remaining_after_planned_round11"]=15;p.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
