"""Resolve which observable chemistry groups carry the 13-hit S and 6-hit T bags."""
from pathlib import Path
import hashlib, json, zipfile
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions/round08"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def read_ids(archive, marker):
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if marker in n and n.startswith("Team_"))
        return z.read(name).decode().splitlines()


def groups(frame, cutoff=.5):
    fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in frame.SMILES]
    distances = []
    for i in range(1, len(fps)):
        distances.extend(1-x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    return sorted(Butina.ClusterData(distances, len(fps), cutoff, isDistData=True), key=len, reverse=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v = pd.read_parquet(ROOT / "features/validation_meta.parquet").set_index("CatalogID", drop=False)
    r7 = ROOT / "submissions/round07_validation_batch.zip"
    r4 = ROOT / "submissions/round04_validation_batch.zip"
    s_all = read_ids(r7, "S_M_PART1")
    t_all = read_ids(r7, "T_M_PART2")
    j_zero = read_ids(r4, "J_LIGAND47")
    # Round 7 details identify which entries are M candidates rather than J padding.
    s = pd.read_csv(ROOT / "submissions/round07/S_M_PART1_details.csv")
    t = pd.read_csv(ROOT / "submissions/round07/T_M_PART2_details.csv")
    s = s[s.role == "M_candidate"].set_index("CatalogID", drop=False)
    t = t[t.role == "M_candidate"].set_index("CatalogID", drop=False)
    sg, tg = groups(s), groups(t)
    assert [len(x) for x in sg] == [15, 1, 1]
    assert sum(map(len, tg)) == 17 and len(tg[0]) == 7
    selections = {
        "V_S_MAIN15": s.iloc[list(sg[0])].CatalogID.tolist(),
        "W_T_MAIN7": t.iloc[list(tg[0])].CatalogID.tolist(),
        "X_T_REST10": t.iloc[[i for g in tg[1:] for i in g]].CatalogID.tolist(),
    }
    assert len(selections["X_T_REST10"]) == 10
    manifest = {
        "round": 8,
        "type": "diagnostic chemistry-group resolution",
        "known_feedback": {
            "S": {"candidates":17,"hits":13,"clusters":1},
            "T": {"candidates":17,"hits":6,"clusters":2},
            "U": {"candidates":16,"hits":2,"clusters":1},
        },
        "logic": {
            "V": "Tests S's 15-member common-core group; the two S singletons account for exactly 13 minus V hits.",
            "W_X": "Partition all T candidates; W plus X must reproduce exactly 6 hits and two assay clusters.",
        },
        "sets": {}, "remaining_before":27, "remaining_after":24,
    }
    for label, candidates in selections.items():
        fillers = j_zero[:50-len(candidates)]
        ids = candidates + fillers
        assert len(ids)==50 and len(set(ids))==50 and not set(candidates)&set(fillers)
        file = OUT / f"Team_MMELON_R08_{label}.txt"
        file.write_text("\n".join(ids)+"\n")
        detail = v.loc[ids].copy()
        detail["role"] = ["diagnostic_candidate"]*len(candidates)+["proven_zero_J_filler"]*len(fillers)
        detail.to_csv(OUT/f"{label}_details.csv", index=False)
        manifest["sets"][label] = {
            "file":file.name, "diagnostic_candidates":len(candidates),
            "proven_zero_fillers":len(fillers),
            "sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
        }
    manifest["verification"] = [
        "Each file contains 50 unique valid CatalogIDs",
        "Every filler comes from J, which independently returned zero hits",
        "V is the complete largest S fingerprint group; W and X exactly partition T",
    ]
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (OUT/"START_HERE.txt").write_text(
        "Round 8 — submit V, W and X separately to VALIDATION.\n\n"
        "V isolates the 15-member common-core group inside S (13 hits/17, one cluster).\n"
        "W isolates T's seven-member main group. X contains T's other ten candidates.\n"
        "All unused rows are proven-zero J fillers. W+X must equal 6 hits and reproduce T's two clusters. "
        "The two S singletons contain exactly 13 minus V hits. Report hits, clusters and p-values in V/W/X order.\n"
        "27 validations remain before this batch; 24 afterward.\n")
    archive = ROOT/"submissions/round08_validation_batch.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.glob("Team_*.txt")): z.write(f,f.name)
        z.write(OUT/"START_HERE.txt","START_HERE.txt")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith("Team_")])==3
    results_path=ROOT/"reports/official_results.json"
    results=json.loads(results_path.read_text())
    results["round08"]={k:{"hits":None,"chemical_series":None,"p_value":None,"submission_id":None} for k in selections}
    results["quota_estimate"]["remaining_now"]=27
    results["quota_estimate"]["last_confirmed_after_round"]=7
    results["quota_estimate"]["remaining_after_planned_round08"]=24
    results_path.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__": main()
