"""Exploit round05 yield while preserving a chemical-series discovery arm."""
from pathlib import Path
import hashlib, itertools, json, zipfile
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions/round06"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def prior_sets():
    result = {}
    for rnd in range(1, 6):
        with zipfile.ZipFile(ROOT / f"submissions/round0{rnd}_validation_batch.zip") as z:
            for name in z.namelist():
                if name.startswith("Team_"):
                    result[name] = set(z.read(name).decode().splitlines())
    return result


def support(query, references):
    out = np.zeros((len(query), 2, 3), dtype=np.float32)
    for kind in [0, 1]:
        refs = [DataStructs.CreateFromBinaryText(x[kind].tobytes()) for x in references]
        for row, packed in enumerate(query[:, kind]):
            fp = DataStructs.CreateFromBinaryText(packed.tobytes())
            sims = np.sort(np.asarray(DataStructs.BulkTanimotoSimilarity(fp, refs), dtype=np.float32))
            out[row, kind] = [sims[-1], sims[-min(3, len(sims)):].mean(), sims[-min(8, len(sims)):].mean()]
    return out


def combined(x):
    return .45*x[:,0,0] + .20*x[:,0,1] + .25*x[:,1,0] + .10*x[:,1,1]


def choose(frame, score, permitted, used, diverse=None):
    ranked = frame.assign(Score=score).loc[permitted & ~frame.CatalogID.isin(used)]
    ranked = ranked.sort_values(["Score", "CatalogID"], ascending=[False, True]).drop_duplicates("canonical")
    if diverse is None:
        return ranked.head(50)
    rows, fps = [], []
    for idx, row in ranked.iterrows():
        fp = GEN.GetFingerprint(Chem.MolFromSmiles(row.SMILES))
        if fps and max(DataStructs.BulkTanimotoSimilarity(fp, fps)) >= diverse:
            continue
        rows.append(idx); fps.append(fp)
        if len(rows) == 50:
            break
    assert len(rows) == 50
    return ranked.loc[rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sets = prior_sets(); seen = set.union(*sets.values())
    v = pd.read_parquet(ROOT / "features/validation_meta.parquet")
    bits = np.load(ROOT / "features/validation_bits.npy", mmap_mode="r")
    ids = {
        "K": next(x for n,x in sets.items() if "K_NEW_CORES" in n),
        "J": next(x for n,x in sets.items() if "J_LIGAND47" in n),
        "M": next(x for n,x in sets.items() if "M_K_BAG" in n),
        "N": next(x for n,x in sets.items() if "N_GROUP_A_B" in n),
        "O": next(x for n,x in sets.items() if "O_DIVERSE_BAG" in n),
    }
    ids["K39"] = ids["K"] - ids["J"]
    assert len(ids["K39"]) == 39
    features = {name: support(bits, bits[v.index[v.CatalogID.isin(group)]]) for name,group in ids.items() if name in ["K39","M","N","O"]}
    scores = {name: combined(value) for name,value in features.items()}
    # P maximizes expected hit yield from the newly validated 21/50 M bag.
    p_score = scores["M"]
    # Q rewards agreement between the original 21/39 bag and its successful 21/50 expansion.
    q_score = np.sqrt(np.clip(scores["K39"],0,None) * np.clip(scores["M"],0,None))
    # R emphasizes breadth evidence: N yielded 2 clusters; O yielded 3 clusters.
    r_score = .30*scores["M"] + .25*scores["N"] + .45*scores["O"]
    permitted = v.valid & ~v.CatalogID.isin(seen)
    selected = {}
    selected["P_M_EXPANSION"] = choose(v,p_score,permitted,set())
    used = set(selected["P_M_EXPANSION"].CatalogID)
    selected["Q_K_M_CONSENSUS"] = choose(v,q_score,permitted,used)
    used |= set(selected["Q_K_M_CONSENSUS"].CatalogID)
    selected["R_SERIES_SEARCH"] = choose(v,r_score,permitted,used,diverse=.50)
    manifest = {
        "round":6,
        "feedback": {"M":{"hits":21,"clusters":2},"N":{"hits":8,"clusters":2},"O":{"hits":4,"clusters":3}},
        "individual_labels_inferred":False,
        "strategy": {
            "P":"Fresh nearest-supported analogues of M, which yielded 21/50.",
            "Q":"Fresh candidates supported by both K-minus-J (21/39) and M (21/50), using geometric-mean support.",
            "R":"Fresh scaffold-diverse candidates supported by M/N/O, weighted 30/25/45 toward observed series breadth."
        },
        "warning":"Aggregate hit bags remain ambiguously labelled. RDKit diversity groups are not organizer assay clusters.",
        "sets":{}, "overlap":{}, "previously_submitted_excluded":len(seen),
        "remaining_before":33,"remaining_after":30,
    }
    outsets={}
    for name,top in selected.items():
        assert len(top)==50 and top.CatalogID.nunique()==50 and top.canonical.nunique()==50
        assert not set(top.CatalogID)&seen
        file=OUT/f"Team_MMELON_R06_{name}.txt"
        file.write_text("\n".join(top.CatalogID.astype(str))+"\n")
        top.to_csv(OUT/f"{name}_details.csv",index=False)
        outsets[name]=set(top.CatalogID)
        manifest["sets"][name]={"file":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
            "unique_Murcko_scaffolds":int(top.scaffold.nunique()),"median_score":float(top.Score.median())}
    for a,b in itertools.combinations(outsets,2):manifest["overlap"][a+"__"+b]=len(outsets[a]&outsets[b])
    assert all(x==0 for x in manifest["overlap"].values())
    manifest["distinct_new_candidates"]=len(set.union(*outsets.values()))
    manifest["verification"]=["150 new valid mutually-disjoint CatalogIDs","50 unique parent structures per file","No individual hit identities assumed"]
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"START_HERE.txt").write_text(
        "Round 6 — submit P, Q and R separately to VALIDATION.\n\nP: M analogue expansion for hit yield.\nQ: K/M consensus chemistry.\nR: diverse series-search arm weighted toward O's three clusters.\n\nAll 150 candidates are new and mutually disjoint. Report hits, clusters and p-values in P/Q/R order. 33 validations remain before this batch; 30 afterward.\n")
    archive=ROOT/"submissions/round06_validation_batch.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for file in sorted(OUT.glob("Team_*.txt")):z.write(file,file.name)
        z.write(OUT/"START_HERE.txt","START_HERE.txt")
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len([n for n in z.namelist() if n.startswith("Team_")])==3
    results_path=ROOT/"reports/official_results.json";results=json.loads(results_path.read_text())
    results["round06"]={name:{"hits":None,"chemical_series":None,"p_value":None,"submission_id":None} for name in selected}
    results["quota_estimate"]["remaining_after_planned_round06"]=30
    results_path.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
