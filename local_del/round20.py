"""Two blind-test-style validation rehearsals using leaderboard-calibrated evidence."""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round20"


def archive_ids(round_number: int, marker: str) -> list[str]:
    with zipfile.ZipFile(ROOT / "submissions" / f"round{round_number:02d}_validation_batch.zip") as zf:
        name = next(name for name in zf.namelist() if name.startswith("Team_") and marker in name)
        return zf.read(name).decode().splitlines()


def previously_submitted() -> set[str]:
    result: set[str] = set()
    for archive in sorted((ROOT / "submissions").glob("round*_validation_batch.zip")):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.startswith("Team_"):
                    result.update(zf.read(name).decode().splitlines())
    return result


def percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.linspace(0, 1, len(values), dtype=np.float32)
    return result


def tanimoto_block(matrix: sparse.csr_matrix, refs: sparse.csr_matrix, begin: int, end: int) -> np.ndarray:
    query = matrix[begin:end]
    intersection = (query @ refs.T).toarray()
    left = np.asarray(query.multiply(query).sum(1))
    right = np.asarray(refs.multiply(refs).sum(1)).T
    return intersection / np.maximum(left + right - intersection, 1e-12)


def group_score(matrix: sparse.csr_matrix, refs: sparse.csr_matrix, weights: np.ndarray | None = None) -> np.ndarray:
    result = np.empty(matrix.shape[0], dtype=np.float32)
    for begin in range(0, matrix.shape[0], 15_000):
        similarities = tanimoto_block(matrix, refs, begin, min(begin + 15_000, matrix.shape[0]))
        if weights is not None:
            similarities *= weights[None, :]
        similarities.sort(axis=1)
        result[begin : begin + len(similarities)] = 0.7 * similarities[:, -1] + 0.3 * similarities[:, -min(3, similarities.shape[1]) :].mean(1)
    return result


def weak_reference_probabilities() -> dict[str, float]:
    results = json.loads((ROOT / "reports" / "official_results.json").read_text())
    evidence: defaultdict[str, list[float]] = defaultdict(list)
    # Exploration rounds contain genuine 50-molecule rankings. Later diagnostic
    # splits contain deliberate negative fillers and are represented instead by
    # the forced-label table.
    for round_number in range(1, 14):
        round_name = f"round{round_number:02d}"
        archive = ROOT / "submissions" / f"{round_name}_validation_batch.zip"
        with zipfile.ZipFile(archive) as zf:
            entries = [name for name in zf.namelist() if name.startswith("Team_")]
            for key, outcome in results[round_name].items():
                name = next(name for name in entries if f"_{key}.txt" in name)
                rate = float(outcome["hits"]) / 50.0
                for cid in zf.read(name).decode().splitlines():
                    evidence[cid].append(rate)
    probabilities = {cid: max(rates) for cid, rates in evidence.items()}
    forced = pd.read_parquet(ROOT / "reports" / "validation_forced_labels.parquet")
    for row in forced.itertuples():
        probabilities[row.CatalogID] = float(row.forced_label)
    return probabilities


def pairwise_max(matrix: sparse.csr_matrix, row: int, selected: list[int]) -> float:
    if not selected:
        return 0.0
    return float(tanimoto_block(matrix, matrix[selected], row, row + 1).max())


def take(ranking: np.ndarray, allowed: np.ndarray, matrix: sparse.csr_matrix,
         selected: list[int], selected_set: set[int], count: int, threshold: float) -> None:
    got = 0
    for row in ranking:
        row = int(row)
        if not allowed[row] or row in selected_set:
            continue
        if pairwise_max(matrix, row, selected) >= threshold:
            continue
        selected.append(row)
        selected_set.add(row)
        got += 1
        if got == count:
            return
    raise RuntimeError((got, count))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(ROOT / "features" / "validation_meta.parquet")
    ecfp = sparse.load_npz(ROOT / "features" / "validation_ecfp.npz").tocsr()
    row_for_id = pd.Series(metadata.index.to_numpy(), index=metadata.CatalogID).to_dict()
    g = pd.read_parquet(ROOT / "models" / "deep_audit" / "round03_full_rankings.parquet",
                        columns=["CatalogID", "G_KNOWN_LIGANDS_score"])
    g = g.set_index("CatalogID").loc[metadata.CatalogID].G_KNOWN_LIGANDS_score.to_numpy()
    del_ml = np.load(ROOT / "models" / "validation_ecfp.npy")

    weak = weak_reference_probabilities()
    weak_ids = [cid for cid, probability in weak.items() if probability >= 0.10 and cid in row_for_id]
    weak_ids.sort(key=lambda cid: (-weak[cid], cid))
    weak_rows = [row_for_id[cid] for cid in weak_ids]
    weak_weights = np.sqrt(np.asarray([weak[cid] for cid in weak_ids], dtype=np.float32))
    weak_score = group_score(ecfp, ecfp[weak_rows], weak_weights)
    k_rows = [row_for_id[cid] for cid in archive_ids(4, "K_NEW_CORES")]
    o_rows = [row_for_id[cid] for cid in archive_ids(5, "O_DIVERSE_BAG")]
    ad_rows = [row_for_id[cid] for cid in archive_ids(10, "AD_SERIES_EXPLORATION")]
    k_score = group_score(ecfp, ecfp[k_rows])
    o_score = group_score(ecfp, ecfp[o_rows])
    ad_score = group_score(ecfp, ecfp[ad_rows])
    scores = {
        "G": percentile(g),
        "weak": percentile(weak_score),
        "K": percentile(k_score),
        "O": percentile(o_score),
        "AD": percentile(ad_score),
        "DEL": percentile(del_ml),
    }
    consensus = 0.40 * scores["G"] + 0.25 * scores["weak"] + 0.20 * scores["K"] + 0.15 * scores["DEL"]
    seen = previously_submitted()
    allowed = (metadata.valid & ~metadata.CatalogID.isin(seen)).to_numpy()
    tie = metadata.CatalogID.to_numpy()

    at: list[int] = []
    take(np.lexsort((tie, -consensus)), allowed, ecfp, at, set(), 50, 0.72)
    au: list[int] = []
    au_set: set[int] = set(at)
    quotas = {"G": 14, "weak": 12, "K": 10, "O": 6, "AD": 5, "DEL": 3}
    for name, count in quotas.items():
        take(np.lexsort((tie, -scores[name])), allowed, ecfp, au, au_set, count, 0.46)

    submissions = {"AT_TESTSTYLE_CONSENSUS": at, "AU_TESTSTYLE_SERIES_PORTFOLIO": au}
    files = []
    for label, rows in submissions.items():
        path = OUT / f"Team_MMELON_R20_{label}.txt"
        path.write_text("\n".join(metadata.CatalogID.iloc[rows]) + "\n")
        files.append(path)
        details = metadata.iloc[rows].copy()
        for name, values in scores.items():
            details[name + "_percentile"] = values[rows]
        details["consensus"] = consensus[rows]
        details.to_csv(OUT / f"{label}_details.csv", index=False)

    manifest = {
        "round": 20,
        "submissions": [path.stem for path in files],
        "AT_method": "Hit-focused consensus of G, weak validation labels, K and DEL ML",
        "AU_method": "Series-focused quotas across G, weak labels, K, O, AD and DEL ML",
        "AU_quotas": quotas,
        "weak_references": len(weak_ids),
        "overlap": len(set(metadata.CatalogID.iloc[at]) & set(metadata.CatalogID.iloc[au])),
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "remaining_before": 3,
        "remaining_after": 1,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 20 - submit AT and AU to VALIDATION in filename order.\n\n"
        "AT rehearses the hit-focused blind-test model. AU rehearses the series-diverse model. Send both results back in AT/AU order.\n"
        "Three validations remain before these files; one afterward.\n"
    )
    archive = ROOT / "submissions" / "round20_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")
    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_after_planned_round20"] = 1
    results["round20"] = {
        "AT_TESTSTYLE_CONSENSUS": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
        "AU_TESTSTYLE_SERIES_PORTFOLIO": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    for label, rows in submissions.items():
        print(label, {"scaffolds": int(metadata.scaffold.iloc[rows].nunique()),
                      "median_G_percentile": float(np.median(scores["G"][rows])),
                      "median_weak_percentile": float(np.median(scores["weak"][rows])),
                      "median_K_percentile": float(np.median(scores["K"][rows]))})


if __name__ == "__main__":
    main()
