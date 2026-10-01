"""ChemBERTa committee probe with weak validation supervision."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import sparse
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round13"


def submission_ids(round_number: int, marker: str) -> list[str]:
    archive = ROOT / "submissions" / f"round{round_number:02d}_validation_batch.zip"
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if name.startswith("Team_") and marker in name]
        if len(names) != 1:
            raise ValueError((round_number, marker, names))
        return zf.read(names[0]).decode().splitlines()


def previously_submitted() -> set[str]:
    result: set[str] = set()
    for archive in sorted((ROOT / "submissions").glob("round*_validation_batch.zip")):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.startswith("Team_"):
                    result.update(zf.read(name).decode().splitlines())
    return result


def normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def prototypes(matrix: np.ndarray, count: int) -> np.ndarray:
    matrix = normalized(matrix)
    if len(matrix) <= count:
        return matrix
    labels = KMeans(n_clusters=count, random_state=2035, n_init=20).fit_predict(matrix)
    centers = []
    for label in range(count):
        members = matrix[labels == label]
        center = normalized(members.mean(0, keepdims=True))[0]
        centers.append(members[np.argmax(members @ center)])
    return np.stack(centers)


def group_similarity(features: np.ndarray, refs: np.ndarray) -> np.ndarray:
    refs = normalized(refs)
    result = np.empty(len(features), dtype=np.float32)
    for begin in range(0, len(features), 20_000):
        query = normalized(features[begin : begin + 20_000])
        similarities = query @ refs.T
        similarities.sort(axis=1)
        result[begin : begin + len(query)] = 0.7 * similarities[:, -1] + 0.3 * similarities[:, -min(2, len(refs)) :].mean(1)
    return result


def percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.linspace(0, 1, len(values), dtype=np.float32)
    return result


def tanimoto_to_selected(matrix: sparse.csr_matrix, row: int, selected: list[int]) -> float:
    if not selected:
        return 0.0
    query = matrix[row]
    references = matrix[selected]
    intersection = (references @ query.T).toarray().ravel()
    query_count = query.multiply(query).sum()
    reference_count = np.asarray(references.multiply(references).sum(1)).ravel()
    return float(np.max(intersection / np.maximum(query_count + reference_count - intersection, 1e-12)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(ROOT / "features" / "validation_meta.parquet")
    language = np.load(ROOT / "features" / "validation_chemberta.npy", mmap_mode="r")
    ecfp = sparse.load_npz(ROOT / "features" / "validation_ecfp.npz").tocsr()
    descriptors = np.load(ROOT / "features" / "validation_desc.npy", mmap_mode="r")
    row_for_id = pd.Series(metadata.index.to_numpy(), index=metadata.CatalogID).to_dict()

    anchor_ids = {
        "G_25hits_3series": submission_ids(3, "G_KNOWN_LIGANDS"),
        "K_21hits_2series": submission_ids(4, "K_NEW_CORES"),
        "AF_7confirmed": submission_ids(11, "AF_K_GROUP7")[:7],
        "O_4hits_3series": submission_ids(5, "O_DIVERSE_BAG"),
        "AD_2hits_2series": submission_ids(10, "AD_SERIES_EXPLORATION"),
    }
    prototype_counts = {"G_25hits_3series": 10, "K_21hits_2series": 10, "AF_7confirmed": 7,
                        "O_4hits_3series": 10, "AD_2hits_2series": 10}
    anchor_features = {
        name: prototypes(language[[row_for_id[cid] for cid in ids]], prototype_counts[name])
        for name, ids in anchor_ids.items()
    }
    forced = pd.read_parquet(ROOT / "reports" / "validation_forced_labels.parquet")
    negative_ids = forced.loc[forced.forced_label == 0, "CatalogID"].tolist()
    negative_refs = prototypes(language[[row_for_id[cid] for cid in negative_ids]], 32)

    raw = {name: group_similarity(language, refs) for name, refs in anchor_features.items()}
    negative_similarity = group_similarity(language, negative_refs)
    ranks = {name: percentile(score) for name, score in raw.items()}
    negative_rank = percentile(negative_similarity)
    consensus = np.maximum.reduce(list(ranks.values()))
    # Prefer ordinary drug-like ranges while preserving unusual chemistry when its
    # language score is strong. Descriptor order is documented in features.py.
    mw, logp, tpsa, charge = descriptors[:, 0], descriptors[:, 1], descriptors[:, 2], descriptors[:, 8]
    druglike = ((mw >= 260) & (mw <= 560) & (logp >= 0.5) & (logp <= 5.5) &
                (tpsa >= 25) & (tpsa <= 140) & (np.abs(charge) <= 1)).astype(np.float32)
    anchor_rankings = {
        name: 0.72 * rank + 0.18 * consensus + 0.10 * druglike - 0.18 * negative_rank
        for name, rank in ranks.items()
    }

    quotas = {"G_25hits_3series": 12, "K_21hits_2series": 12, "AF_7confirmed": 10,
              "O_4hits_3series": 9, "AD_2hits_2series": 7}
    seen = previously_submitted()
    allowed = metadata.valid & ~metadata.CatalogID.isin(seen)
    selected: list[int] = []
    selected_ids: set[str] = set()
    origin: dict[str, str] = {}
    for name, quota in quotas.items():
        ranking = np.lexsort((metadata.CatalogID.to_numpy(), -anchor_rankings[name]))
        got = 0
        for row in ranking:
            cid = metadata.CatalogID.iat[row]
            if not allowed.iat[row] or cid in selected_ids:
                continue
            if tanimoto_to_selected(ecfp, row, selected) >= 0.58:
                continue
            selected.append(int(row))
            selected_ids.add(cid)
            origin[cid] = name
            got += 1
            if got == quota:
                break
        if got != quota:
            raise RuntimeError((name, got, quota))
    if len(selected) != 50:
        raise RuntimeError(len(selected))

    details = metadata.iloc[selected].copy()
    details["anchor"] = details.CatalogID.map(origin)
    details["consensus_percentile"] = consensus[selected]
    details["negative_percentile"] = negative_rank[selected]
    details["max_pairwise_ecfp"] = [tanimoto_to_selected(ecfp, row, selected[:i]) for i, row in enumerate(selected)]
    for name in raw:
        details[name + "_cosine"] = raw[name][selected]
    filename = "Team_MMELON_R13_AI_CHEMBERTA_OOD.txt"
    submission = OUT / filename
    submission.write_text("\n".join(details.CatalogID) + "\n")
    details.to_csv(OUT / "AI_CHEMBERTA_OOD_details.csv", index=False)
    manifest = {
        "round": 13,
        "submission": "AI_CHEMBERTA_OOD",
        "method": "ChemBERTa committee over five validation evidence groups, negative-prototype penalty, ECFP diversity",
        "quotas": quotas,
        "pairwise_ecfp_maximum": 0.58,
        "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "remaining_before": 14,
        "remaining_after": 13,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 13 - submit AI_CHEMBERTA_OOD only to VALIDATION.\n\n"
        "This is one 50-compound ChemBERTa OOD probe. Report hits, clusters, and p-value.\n"
        "Fourteen validation submissions remain before this submission; thirteen afterward.\n"
    )
    archive = ROOT / "submissions" / "round13_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission, submission.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")
    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_now"] = 14
    results["quota_estimate"]["last_confirmed_after_round"] = 12
    results["quota_estimate"]["remaining_after_planned_round13"] = 13
    results["round13"] = {"AI_CHEMBERTA_OOD": {"hits": None, "chemical_series": None,
                                                  "p_value": None, "submission_id": None}}
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(details[["CatalogID", "anchor", "consensus_percentile", "negative_percentile",
                   "max_pairwise_ecfp"]].to_string(index=False))


if __name__ == "__main__":
    main()
