"""Expand the 21-of-39 K-minus-J positive bag into three disjoint validation sets."""
from pathlib import Path
import hashlib, itertools, json, zipfile
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions/round05"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def read_sets():
    sets = {}
    for rnd in [1, 2, 3, 4]:
        with zipfile.ZipFile(ROOT / f"submissions/round0{rnd}_validation_batch.zip") as z:
            for name in z.namelist():
                if name.startswith("Team_"):
                    sets[name] = set(z.read(name).decode().splitlines())
    return sets


def butina_groups(frame, cutoff=.5):
    fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in frame.SMILES]
    distances = []
    for i in range(1, len(fps)):
        distances.extend(1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    return sorted(Butina.ClusterData(distances, len(fps), cutoff, isDistData=True), key=len, reverse=True)


def similarity_features(query_bits, ref_bits):
    out = np.zeros((len(query_bits), 8), dtype=np.float32)
    for fp_kind in [0, 1]:
        refs = [DataStructs.CreateFromBinaryText(x[fp_kind].tobytes()) for x in ref_bits]
        for row, packed in enumerate(query_bits[:, fp_kind]):
            query = DataStructs.CreateFromBinaryText(packed.tobytes())
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, refs), dtype=np.float32)
            ordered = np.sort(sims)
            out[row, fp_kind * 4 + 0] = ordered[-1]
            out[row, fp_kind * 4 + 1] = ordered[-min(3, len(ordered)):].mean()
            out[row, fp_kind * 4 + 2] = ordered[-min(8, len(ordered)):].mean()
            out[row, fp_kind * 4 + 3] = (sims >= .35).sum()
    return out


def rank_select(frame, score, allowed, excluded, n=50, diversity=None):
    ranked = frame.assign(Score=score).loc[allowed & ~frame.CatalogID.isin(excluded)]
    ranked = ranked.sort_values(["Score", "CatalogID"], ascending=[False, True]).drop_duplicates("canonical")
    if diversity is None:
        return ranked.head(n)
    selected = []
    selected_fps = []
    for idx, row in ranked.iterrows():
        fp = GEN.GetFingerprint(Chem.MolFromSmiles(row.SMILES))
        if selected_fps and max(DataStructs.BulkTanimotoSimilarity(fp, selected_fps)) >= diversity:
            continue
        selected.append(idx); selected_fps.append(fp)
        if len(selected) == n:
            break
    assert len(selected) == n
    return ranked.loc[selected]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = read_sets()
    seen = set.union(*prior.values())
    j = next(ids for name, ids in prior.items() if "J_LIGAND47" in name)
    k = next(ids for name, ids in prior.items() if "K_NEW_CORES" in name)
    positive_bag_ids = k - j
    assert len(positive_bag_ids) == 39
    v = pd.read_parquet(ROOT / "features/validation_meta.parquet")
    bits = np.load(ROOT / "features/validation_bits.npy", mmap_mode="r")
    bag = v.loc[v.CatalogID.isin(positive_bag_ids)].copy()
    assert len(bag) == 39 and bag.CatalogID.nunique() == 39
    groups = butina_groups(bag, .5)
    # The two largest observable chemistry groups contain 10 and 7 members.
    # Hidden assay clusters are unavailable; these are hypotheses, not hit labels.
    group_a = bag.iloc[list(groups[0])]
    group_b = bag.iloc[list(groups[1])]
    all_features = similarity_features(bits, bits[bag.index])
    a_features = similarity_features(bits, bits[group_a.index])
    b_features = similarity_features(bits, bits[group_b.index])
    # Max similarity captures close analogues; top-k support resists one noisy bag member.
    bag_score = .45 * all_features[:, 0] + .20 * all_features[:, 1] + .25 * all_features[:, 4] + .10 * all_features[:, 5]
    a_score = .50 * a_features[:, 0] + .20 * a_features[:, 1] + .20 * a_features[:, 4] + .10 * a_features[:, 5]
    b_score = .50 * b_features[:, 0] + .20 * b_features[:, 1] + .20 * b_features[:, 4] + .10 * b_features[:, 5]
    allowed = v.valid & ~v.CatalogID.isin(seen)
    selections = {}
    selections["M_K_BAG"] = rank_select(v, bag_score, allowed, set())
    used = set(selections["M_K_BAG"].CatalogID)
    selections["N_GROUP_A_B"] = rank_select(v, np.maximum(a_score, b_score), allowed, used)
    used |= set(selections["N_GROUP_A_B"].CatalogID)
    # A diversity gate probes additional series while still using the successful bag signal.
    selections["O_DIVERSE_BAG"] = rank_select(v, bag_score, allowed, used, diversity=.55)
    manifest = {
        "round": 5,
        "feedback_fact": "K has 21 hits; J has zero; K intersects J in 11 IDs, therefore K minus J is a 39-molecule bag containing exactly 21 hits.",
        "individual_hit_identities_known": False,
        "positive_bag_size": len(bag),
        "positive_bag_hits": 21,
        "observable_Butina_cutoff": .5,
        "largest_observable_group_sizes": [len(x) for x in groups[:8]],
        "warning": "Observable fingerprint groups are not organizer assay clusters. All 39 bag members remain ambiguously labelled.",
        "sets": {},
        "overlap": {},
        "previously_submitted_excluded": len(seen),
        "remaining_before": 36,
        "remaining_after": 33,
    }
    selected_sets = {}
    for name, top in selections.items():
        assert len(top) == 50 and top.CatalogID.nunique() == 50 and top.canonical.nunique() == 50
        assert not set(top.CatalogID) & seen
        file = OUT / f"Team_MMELON_R05_{name}.txt"
        file.write_text("\n".join(top.CatalogID.astype(str)) + "\n")
        top.to_csv(OUT / f"{name}_details.csv", index=False)
        ids = set(file.read_text().splitlines())
        assert len(ids) == 50
        selected_sets[name] = ids
        manifest["sets"][name] = {
            "file": file.name,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "unique_Murcko_scaffolds": int(top.scaffold.nunique()),
            "median_score": float(top.Score.median()),
            "prior_submission_overlap": 0,
        }
    for a, b in itertools.combinations(selected_sets, 2):
        manifest["overlap"][f"{a}__{b}"] = len(selected_sets[a] & selected_sets[b])
    assert all(value == 0 for value in manifest["overlap"].values())
    manifest["distinct_new_candidates"] = len(set.union(*selected_sets.values()))
    manifest["verification"] = [
        "39-member K-minus-J bag and exact aggregate count checked",
        "All 150 selected CatalogIDs are new, valid, and mutually disjoint",
        "Each file has exactly 50 unique IDs and parent structures",
    ]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 5 — submit M, N and O separately to VALIDATION.\n\n"
        "M: nearest supported analogues of the 39 K-minus-J molecules containing all 21 K hits.\n"
        "N: fresh analogues around the two largest observable chemistry groups in that bag.\n"
        "O: scaffold-diverse expansion of the whole bag.\n\n"
        "The sets contain 150 distinct, previously unsubmitted molecules. Observable groups are not the organizers' hidden hit clusters. Send hits, clusters and p-values for M/N/O. 36 validations remain before this batch; 33 afterward.\n"
    )
    archive = ROOT / "submissions/round05_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(OUT.glob("Team_*.txt")):
            z.write(file, file.name)
        z.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len([n for n in z.namelist() if n.startswith("Team_")]) == 3
    results_path = ROOT / "reports/official_results.json"
    results = json.loads(results_path.read_text())
    results["round05"] = {name: {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None} for name in selections}
    results["quota_estimate"]["remaining_after_planned_round05"] = 33
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
