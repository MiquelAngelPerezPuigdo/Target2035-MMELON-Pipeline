"""Audit remaining validation and OOD test coverage around round-8 chemistry groups."""
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
from rdkit import DataStructs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports/round08_audit"


def submitted_sets(max_round=8):
    out = {}
    for rnd in range(1, max_round+1):
        with zipfile.ZipFile(ROOT/f"submissions/round0{rnd}_validation_batch.zip") as z:
            for n in z.namelist():
                if n.startswith("Team_"): out[n] = set(z.read(n).decode().splitlines())
    return out


def support(query, refs):
    result=np.zeros((len(query),6),dtype=np.float32)
    for kind in [0,1]:
        targets=[DataStructs.CreateFromBinaryText(x[kind].tobytes()) for x in refs]
        for i,p in enumerate(query[:,kind]):
            fp=DataStructs.CreateFromBinaryText(p.tobytes())
            sims=np.sort(np.asarray(DataStructs.BulkTanimotoSimilarity(fp,targets),dtype=np.float32))
            result[i,3*kind:3*kind+3]=[sims[-1],sims[-min(3,len(sims)):].mean(),sims[-min(7,len(sims)):].mean()]
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sets=submitted_sets(); seen=set.union(*sets.values())
    group_files={
        "S15":ROOT/"submissions/round08/V_S_MAIN15_details.csv",
        "T7":ROOT/"submissions/round08/W_T_MAIN7_details.csv",
        "T10":ROOT/"submissions/round08/X_T_REST10_details.csv",
    }
    groups={k:set(pd.read_csv(p).query("role == 'diagnostic_candidate'").CatalogID) for k,p in group_files.items()}
    vm=pd.read_parquet(ROOT/"features/validation_meta.parquet")
    vb=np.load(ROOT/"features/validation_bits.npy",mmap_mode="r")
    refs={k:vb[vm.index[vm.CatalogID.isin(ids)]] for k,ids in groups.items()}
    summary={"seen_distinct_validation_ids":len(seen),"groups":{},"splits":{}}
    prior_names={key:next(ids for name,ids in sets.items() if key in name) for key in ["P_M_EXPANSION","Q_K_M_CONSENSUS","R_SERIES_SEARCH"]}
    for split in ["validation","test"]:
        meta=pd.read_parquet(ROOT/f"features/{split}_meta.parquet")
        bits=np.load(ROOT/f"features/{split}_bits.npy",mmap_mode="r")
        frame=meta.copy()
        for name,ref in refs.items():
            f=support(bits,ref)
            for j,col in enumerate(["ecfp_max","ecfp_top3","ecfp_top7","fcfp_max","fcfp_top3","fcfp_top7"]):
                frame[f"{name}_{col}"]=f[:,j]
            frame[f"{name}_score"]=.45*f[:,0]+.20*f[:,1]+.25*f[:,3]+.10*f[:,4]
        frame["best_group"]=frame[[f"{x}_score" for x in refs]].idxmax(axis=1).str.replace("_score","")
        frame["best_score"]=frame[[f"{x}_score" for x in refs]].max(axis=1)
        if split=="validation": frame=frame[~frame.CatalogID.isin(seen)].copy()
        frame.sort_values(["best_score","CatalogID"],ascending=[False,True]).head(3000).to_parquet(OUT/f"{split}_top3000.parquet",index=False)
        counts={}
        for name in refs:
            counts[name]={str(t):int((frame[f"{name}_ecfp_max"]>=t).sum()) for t in [.9,.8,.7,.6,.5]}
        summary["splits"][split]={"eligible_rows":len(frame),"ecfp_max_counts":counts,"top50_group_counts":frame.nlargest(50,"best_score").best_group.value_counts().to_dict()}
        if split=="validation":
            summary["prior_round06_support"]={}
            full=meta.join(pd.DataFrame({c: support(bits,ref)[:,j] for name,ref in refs.items() for j,c in enumerate([f"{name}_ecfp_max",f"{name}_ecfp_top3",f"{name}_ecfp_top7",f"{name}_fcfp_max",f"{name}_fcfp_top3",f"{name}_fcfp_top7"])}))
            for pname,ids in prior_names.items():
                z=full[full.CatalogID.isin(ids)]
                summary["prior_round06_support"][pname]={name:{"median_ecfp_max":float(z[f"{name}_ecfp_max"].median()),"max_ecfp_max":float(z[f"{name}_ecfp_max"].max())} for name in refs}
    for name,ids in groups.items(): summary["groups"][name]={"size":len(ids)}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
