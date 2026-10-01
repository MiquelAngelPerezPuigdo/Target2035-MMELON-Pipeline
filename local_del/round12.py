"""One-query scaffold-hop validation using reduced pharmacophore graphs."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator,rdReducedGraphs

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"submissions/round12";MODEL=ROOT/"models/round12"
GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

def ids(rnd,marker):
    with zipfile.ZipFile(ROOT/f"submissions/round{rnd:02d}_validation_batch.zip") as z:
        n=next(x for x in z.namelist() if x.startswith("Team_") and marker in x)
        return z.read(n).decode().splitlines()

def seen_ids():
    s=set()
    for r in range(1,12):
        with zipfile.ZipFile(ROOT/f"submissions/round{r:02d}_validation_batch.zip") as z:
            for n in z.namelist():
                if n.startswith("Team_"):s.update(z.read(n).decode().splitlines())
    return s

def tanimoto_matrix(x,r):
    # Chunked float64 avoids Accelerate/BLAS overflow warnings observed for a
    # large float32 memmap matrix multiplication on Apple Silicon.
    r=np.asarray(r,dtype=np.float64);rr=(r*r).sum(1)[None,:]
    out=np.empty((len(x),len(r)),dtype=np.float32)
    for begin in range(0,len(x),20000):
        q=np.asarray(x[begin:begin+20000],dtype=np.float64)
        dot=np.einsum("ij,kj->ik",q,r,optimize=False);xx=np.einsum("ij,ij->i",q,q)[:,None]
        out[begin:begin+len(q)]=(dot/np.maximum(xx+rr-dot,1e-12)).astype(np.float32)
    assert np.isfinite(out).all()
    return out

def group_score(x,r):
    sims=tanimoto_matrix(x,r); sims.sort(axis=1)
    return .65*sims[:,-1]+.35*sims[:,-min(3,sims.shape[1]):].mean(1)

def pct(x):
    order=np.argsort(x,kind="mergesort");out=np.empty(len(x),dtype=np.float32);out[order]=np.linspace(0,1,len(x),dtype=np.float32);return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True)
    for f in OUT.glob("Team_*.txt"):f.unlink()
    vm=pd.read_parquet(ROOT/"features/validation_meta.parquet")
    ve=np.load(ROOT/"features/validation_erg.npy",mmap_mode="r")
    y=ids(9,"Y_X_THIENO4")[:4]
    anchors={
      "ligand21":None,
      "AF7_exact":ids(11,"AF_K_GROUP7")[:7],
      "AE10_dense":ids(11,"AE_K_GROUP10")[:10],
      "Y34_onehit":y[2:4],
      "AG22_two_series":ids(11,"AG_K_REST22")[:22],
      "AD50_two_series":ids(10,"AD_SERIES_EXPLORATION")[:50],
      "O50_three_series":ids(5,"O_DIVERSE_BAG")[:50],
    }
    ligand=Chem.SDMolSupplier(str(ROOT/"external/pgk2/compound21_verified.sdf"),removeHs=True)[0]
    ligand_fp=np.asarray(rdReducedGraphs.GetErGFingerprint(ligand),dtype=np.float32)[None,:]
    refs={"ligand21":ligand_fp}
    for k,v in anchors.items():
        if v is not None: refs[k]=ve[vm.index[vm.CatalogID.isin(v)]]
    scores={k:group_score(ve,r) for k,r in refs.items()}
    # Percentiles put different anchor groups on comparable scales. The combined
    # score is retained for the future full-library ranking; selection uses quotas.
    percentiles={k:pct(v) for k,v in scores.items()}
    combined=np.maximum.reduce([percentiles[k] for k in percentiles])
    seen=seen_ids();allowed=vm.valid & ~vm.CatalogID.isin(seen)
    quotas={"ligand21":9,"AF7_exact":8,"AE10_dense":7,"Y34_onehit":7,"AG22_two_series":7,"AD50_two_series":6,"O50_three_series":6}
    selected=[];selected_ids=set();selected_fps=[];origin={}
    for anchor,q in quotas.items():
        ranking=np.lexsort((vm.CatalogID.to_numpy(),-scores[anchor]))
        got=0
        for idx in ranking:
            cid=vm.CatalogID.iat[idx]
            if not allowed.iat[idx] or cid in selected_ids:continue
            fp=GEN.GetFingerprint(Chem.MolFromSmiles(vm.SMILES.iat[idx]))
            if selected_fps and max(DataStructs.BulkTanimotoSimilarity(fp,selected_fps))>=.55:continue
            selected.append(idx);selected_ids.add(cid);selected_fps.append(fp);origin[cid]=anchor;got+=1
            if got==q:break
        assert got==q,(anchor,got)
    assert len(selected)==50
    out=vm.iloc[selected].copy();out["anchor"]=out.CatalogID.map(origin);out["combined_percentile"]=combined[selected]
    for k in scores:out[k+"_erg_score"]=scores[k][selected]
    file=OUT/"Team_MMELON_R12_AH_ERG_SCAFFOLD_HOPS.txt";file.write_text("\n".join(out.CatalogID)+"\n");out.to_csv(OUT/"AH_ERG_SCAFFOLD_HOPS_details.csv",index=False)
    np.save(MODEL/"validation_combined_percentile.npy",combined)
    # Apply the same frozen anchors to the whole test library now; validation
    # feedback may later calibrate weights but must not change these features.
    tm=pd.read_parquet(ROOT/"features/test_meta.parquet");te=np.load(ROOT/"features/test_erg.npy",mmap_mode="r")
    test_raw={k:group_score(te,r) for k,r in refs.items()};test_combined=np.maximum.reduce([pct(v) for v in test_raw.values()]);np.save(MODEL/"test_combined_percentile.npy",test_combined)
    manifest={"round":12,"submissions":1,"method":"ErG reduced-graph pharmacophore scaffold hopping",
      "anchor_quotas":quotas,"selection_diversity":"ECFP4 pairwise Tanimoto < 0.55",
      "file":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"remaining_before":15,"remaining_after":14,
      "test_preparation":"Frozen per-anchor pharmacophore features and a provisional full-library test score were saved; no test submission made."}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"START_HERE.txt").write_text("Round 12 — submit AH only to VALIDATION.\n\nAH contains 50 new scaffold-diverse pharmacophore matches allocated across seven evidence sources. Report hits, clusters and p-value. 15 validations remain before this submission; 14 afterward.\n")
    archive=ROOT/"submissions/round12_validation_batch.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:z.write(file,file.name);z.write(OUT/"START_HERE.txt","START_HERE.txt")
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    p=ROOT/"reports/official_results.json";r=json.loads(p.read_text());r["round12"]={"AH_ERG_SCAFFOLD_HOPS":{"hits":None,"chemical_series":None,"p_value":None,"submission_id":None}}
    r["quota_estimate"]["remaining_now"]=15;r["quota_estimate"]["last_confirmed_after_round"]=11;r["quota_estimate"]["remaining_after_planned_round12"]=14;p.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
