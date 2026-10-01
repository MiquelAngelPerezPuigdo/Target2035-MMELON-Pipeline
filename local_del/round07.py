"""Chemistry-preserving partition of the 21-hit M bag using proven-zero fillers."""
from pathlib import Path
import hashlib, json, zipfile
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions/round07"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def sets_through(round_number):
    result = {}
    for rnd in range(1, round_number + 1):
        with zipfile.ZipFile(ROOT / f"submissions/round0{rnd}_validation_batch.zip") as z:
            for name in z.namelist():
                if name.startswith("Team_"):
                    result[name] = z.read(name).decode().splitlines()
    return result


def clusters(frame, cutoff=.5):
    fps = [GEN.GetFingerprint(Chem.MolFromSmiles(x)) for x in frame.SMILES]
    distances = []
    for i in range(1, len(fps)):
        distances.extend(1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    return sorted(Butina.ClusterData(distances, len(fps), cutoff, isDistData=True), key=len, reverse=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = sets_through(6)
    m_ids = next(ids for name, ids in prior.items() if "M_K_BAG" in name)
    j_ids = next(ids for name, ids in prior.items() if "J_LIGAND47" in name)
    assert len(m_ids) == 50 and len(j_ids) == 50 and not set(m_ids) & set(j_ids)
    v = pd.read_parquet(ROOT / "features/validation_meta.parquet").set_index("CatalogID", drop=False)
    m = v.loc[m_ids].copy()
    groups = clusters(m, .5)
    # Keep observable chemistry groups intact, greedily balancing three partitions.
    bins = [[], [], []]
    for group in groups:
        target = min(range(3), key=lambda x: len(bins[x]))
        bins[target].extend(group)
    assert sorted(x for group in bins for x in group) == list(range(50))
    manifest = {
        "round": 7,
        "type": "diagnostic partition",
        "known_fact": "M contains exactly 21 hits among 50 molecules and two organizer-defined hit clusters.",
        "method": "Partition all M molecules exactly once while keeping Butina ECFP4 groups together; pad each file to 50 with candidates from zero-hit J.",
        "filler_validity": "J returned zero hits, so fillers cannot change hit or hit-cluster counts if evaluation is consistent.",
        "observable_cluster_cutoff": .5,
        "observable_cluster_sizes": [len(x) for x in groups],
        "individual_hit_labels_assumed": False,
        "sets": {},
        "expected_consistency_check": "S + T + U hit counts must equal 21.",
        "remaining_before": 30,
        "remaining_after": 27,
    }
    union = set()
    labels = ["S_M_PART1", "T_M_PART2", "U_M_PART3"]
    for label, members in zip(labels, bins):
        candidates = m.iloc[members]
        fillers = j_ids[:50-len(candidates)]
        ids = candidates.CatalogID.tolist() + fillers
        assert len(ids) == 50 and len(set(ids)) == 50
        assert not set(candidates.CatalogID) & set(fillers)
        union |= set(candidates.CatalogID)
        file = OUT / f"Team_MMELON_R07_{label}.txt"
        file.write_text("\n".join(ids) + "\n")
        details = v.loc[ids].copy()
        details["role"] = ["M_candidate"]*len(candidates) + ["proven_zero_J_filler"]*len(fillers)
        details.to_csv(OUT / f"{label}_details.csv", index=False)
        manifest["sets"][label] = {
            "file": file.name,
            "M_candidates": len(candidates),
            "proven_zero_fillers": len(fillers),
            "observable_groups": sum(set(group).issubset(set(members)) for group in groups),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
    assert union == set(m_ids)
    manifest["verification"] = [
        "Every M candidate occurs in exactly one diagnostic set",
        "Every filler comes from the independently evaluated zero-hit J set",
        "Each serialized file contains exactly 50 unique valid CatalogIDs",
    ]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 7 — diagnostic VALIDATION batch. Submit S, T and U separately.\n\n"
        "Together they partition all 50 molecules from M, which contained 21 hits. Each is padded to 50 using molecules from J, which had zero hits. Therefore S+T+U must equal 21 hits. Chemistry groups were kept together where possible.\n\n"
        "This batch identifies which portions of M carry the active series; it does not test new molecules. Report hits, clusters and p-values in S/T/U order. 30 validations remain before this batch; 27 afterward.\n"
    )
    archive = ROOT / "submissions/round07_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(OUT.glob("Team_*.txt")):
            z.write(file, file.name)
        z.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len([n for n in z.namelist() if n.startswith("Team_")]) == 3
    results_path = ROOT / "reports/official_results.json"
    results = json.loads(results_path.read_text())
    results["round07"] = {name: {"hits":None,"chemical_series":None,"p_value":None,"submission_id":None} for name in labels}
    results["quota_estimate"]["remaining_after_planned_round07"] = 27
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
