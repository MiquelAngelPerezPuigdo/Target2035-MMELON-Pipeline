"""Locate X's single, relatively OOD-transferable hit among three chemistry buckets."""
from pathlib import Path
import hashlib, json, zipfile
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"submissions/round09"
GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

def ids_from(archive,marker):
    with zipfile.ZipFile(archive) as z:
        n=next(x for x in z.namelist() if x.startswith("Team_") and marker in x)
        return z.read(n).decode().splitlines()

def cluster(frame):
    fps=[GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in frame.SMILES]
    dist=[]
    for i in range(1,len(fps)):
        dist.extend(1-x for x in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]))
    return sorted(Butina.ClusterData(dist,len(fps),.5,isDistData=True),key=len,reverse=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    v=pd.read_parquet(ROOT/"features/validation_meta.parquet").set_index("CatalogID",drop=False)
    x_ids=ids_from(ROOT/"submissions/round08_validation_batch.zip","X_T_REST10")[:10]
    j=ids_from(ROOT/"submissions/round04_validation_batch.zip","J_LIGAND47")
    x=v.loc[x_ids]
    gs=cluster(x)
    assert [len(g) for g in gs]==[4,3,1,1,1]
    buckets={
        "Y_X_THIENO4":x.iloc[list(gs[0])].CatalogID.tolist(),
        "Z_X_ISOXAZOLE3":x.iloc[list(gs[1])].CatalogID.tolist(),
        "AA_X_SINGLETONS3":x.iloc[[i for g in gs[2:] for i in g]].CatalogID.tolist(),
    }
    assert set().union(*map(set,buckets.values()))==set(x_ids)
    manifest={
        "round":9,"type":"diagnostic partition of X's one-hit bag",
        "known_fact":"X contains exactly one hit in one organizer cluster among ten molecules.",
        "strategy":"Partition X into its 4-member thienopyrimidine group, 3-member isoxazole group, and three remaining singletons.",
        "expected_check":"Y + Z + AA must equal exactly one hit; exactly one set should report one cluster.",
        "sets":{},"remaining_before":24,"remaining_after":21,
    }
    for name,candidates in buckets.items():
        fillers=j[:50-len(candidates)]; ids=candidates+fillers
        assert len(ids)==len(set(ids))==50 and not set(candidates)&set(fillers)
        file=OUT/f"Team_MMELON_R09_{name}.txt";file.write_text("\n".join(ids)+"\n")
        d=v.loc[ids].copy();d["role"]=["X_candidate"]*len(candidates)+["proven_zero_J_filler"]*len(fillers)
        d.to_csv(OUT/f"{name}_details.csv",index=False)
        manifest["sets"][name]={"file":file.name,"X_candidates":len(candidates),"zero_fillers":len(fillers),"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
    manifest["verification"]=["The three candidate buckets exactly partition X","All fillers are from independently zero-hit J","Every submission has 50 unique valid IDs"]
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"START_HERE.txt").write_text(
        "Round 9 — submit Y, Z and AA separately to VALIDATION.\n\n"
        "These three files exactly partition the ten real candidates in X, which had one hit. The other rows are proven-zero J fillers. Therefore Y+Z+AA must equal one hit, and exactly one set should contain one cluster. Report results in Y/Z/AA order. 24 validations remain before this batch; 21 afterward.\n")
    archive=ROOT/"submissions/round09_validation_batch.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.glob("Team_*.txt")):z.write(f,f.name)
        z.write(OUT/"START_HERE.txt","START_HERE.txt")
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len([n for n in z.namelist() if n.startswith("Team_")])==3
    p=ROOT/"reports/official_results.json";r=json.loads(p.read_text())
    r["round09"]={k:{"hits":None,"chemical_series":None,"p_value":None,"submission_id":None} for k in buckets}
    r["quota_estimate"]["remaining_now"]=24;r["quota_estimate"]["last_confirmed_after_round"]=8;r["quota_estimate"]["remaining_after_planned_round09"]=21
    p.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
