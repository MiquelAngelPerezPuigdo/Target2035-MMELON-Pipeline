"""Exploit and scaffold-hop around the exact ChemBERTa-discovered active."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round18"
MODEL = ROOT / "models" / "round18"
ACTIVE_ID = "Z1514105447"


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


def sparse_tanimoto_to_query(matrix: sparse.csr_matrix, query: sparse.csr_matrix) -> np.ndarray:
    intersection = (matrix @ query.T).toarray().ravel()
    left = np.asarray(matrix.multiply(matrix).sum(1)).ravel()
    right = float(query.multiply(query).sum())
    return (intersection / np.maximum(left + right - intersection, 1e-12)).astype(np.float32)


def generalized_tanimoto_to_query(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    result = np.empty(len(matrix), dtype=np.float32)
    right = float(query @ query)
    for begin in range(0, len(matrix), 20_000):
        chunk = np.asarray(matrix[begin : begin + 20_000], dtype=np.float32)
        intersection = chunk @ query
        left = np.square(chunk).sum(1)
        result[begin : begin + len(chunk)] = intersection / np.maximum(left + right - intersection, 1e-12)
    return result


def cosine_to_query(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    result = np.empty(len(matrix), dtype=np.float32)
    for begin in range(0, len(matrix), 20_000):
        chunk = np.asarray(matrix[begin : begin + 20_000], dtype=np.float32)
        chunk /= np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12)
        result[begin : begin + len(chunk)] = chunk @ query
    return result


def tanimoto_to_selected(matrix: sparse.csr_matrix, row: int, selected: list[int]) -> float:
    if not selected:
        return 0.0
    query = matrix[row]
    references = matrix[selected]
    intersection = (references @ query.T).toarray().ravel()
    left = np.asarray(references.multiply(references).sum(1)).ravel()
    right = float(query.multiply(query).sum())
    return float(np.max(intersection / np.maximum(left + right - intersection, 1e-12)))


def select(ranking: np.ndarray, allowed: np.ndarray, ecfp: sparse.csr_matrix,
           selected_elsewhere: set[int], diversity: float, count: int) -> list[int]:
    chosen: list[int] = []
    for row in ranking:
        row = int(row)
        if not allowed[row] or row in selected_elsewhere:
            continue
        if tanimoto_to_selected(ecfp, row, chosen) >= diversity:
            continue
        chosen.append(row)
        if len(chosen) == count:
            return chosen
    raise RuntimeError(f"Selected only {len(chosen)} of {count}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(ROOT / "features" / "validation_meta.parquet")
    query_row = int(metadata.index[metadata.CatalogID == ACTIVE_ID][0])
    ecfp = sparse.load_npz(ROOT / "features" / "validation_ecfp.npz").tocsr()
    erg = np.load(ROOT / "features" / "validation_erg.npy", mmap_mode="r")
    language = np.load(ROOT / "features" / "validation_chemberta.npy", mmap_mode="r")
    descriptors = np.load(ROOT / "features" / "validation_desc.npy", mmap_mode="r")
    ecfp_similarity = sparse_tanimoto_to_query(ecfp, ecfp[query_row])
    erg_similarity = generalized_tanimoto_to_query(erg, erg[query_row])
    language_similarity = cosine_to_query(language, language[query_row])

    scales = np.asarray([85.0, 1.4, 32.0, 2.0, 1.0, 3.0, 1.5, 0.25, 1.0, 7.0, 2.0], dtype=np.float32)
    descriptor_distance = np.sqrt(np.square((np.asarray(descriptors) - descriptors[query_row]) / scales).mean(1))
    descriptor_similarity = np.exp(-descriptor_distance).astype(np.float32)
    ranks = {
        "ecfp": percentile(ecfp_similarity),
        "erg": percentile(erg_similarity),
        "language": percentile(language_similarity),
        "descriptor": percentile(descriptor_similarity),
    }
    exploit_score = 0.48 * ranks["ecfp"] + 0.22 * ranks["erg"] + 0.20 * ranks["language"] + 0.10 * ranks["descriptor"]
    hop_score = 0.48 * ranks["language"] + 0.34 * ranks["erg"] + 0.18 * ranks["descriptor"]

    seen = previously_submitted()
    base_allowed = (metadata.valid & ~metadata.CatalogID.isin(seen)).to_numpy()
    ap_ranking = np.lexsort((metadata.CatalogID.to_numpy(), -exploit_score))
    ap = select(ap_ranking, base_allowed, ecfp, set(), diversity=0.82, count=50)
    hop_allowed = base_allowed & (ecfp_similarity < 0.38)
    aq_ranking = np.lexsort((metadata.CatalogID.to_numpy(), -hop_score))
    aq = select(aq_ranking, hop_allowed, ecfp, set(ap), diversity=0.52, count=50)

    submissions = {
        "AP_EXACT_HIT_ANALOGS": ap,
        "AQ_EXACT_HIT_SCAFFOLD_HOPS": aq,
    }
    files = []
    for label, rows in submissions.items():
        path = OUT / f"Team_MMELON_R18_{label}.txt"
        path.write_text("\n".join(metadata.CatalogID.iloc[rows]) + "\n")
        files.append(path)
        details = metadata.iloc[rows].copy()
        details["ecfp_to_active"] = ecfp_similarity[rows]
        details["erg_to_active"] = erg_similarity[rows]
        details["chemberta_to_active"] = language_similarity[rows]
        details["descriptor_to_active"] = descriptor_similarity[rows]
        details["score"] = exploit_score[rows] if label.startswith("AP_") else hop_score[rows]
        details.to_csv(OUT / f"{label}_details.csv", index=False)

    # Freeze the same four exact-active views for the complete blind-test library.
    test_ecfp = sparse.load_npz(ROOT / "features" / "test_ecfp.npz").tocsr()
    test_erg = np.load(ROOT / "features" / "test_erg.npy", mmap_mode="r")
    test_language = np.load(ROOT / "features" / "test_chemberta.npy", mmap_mode="r")
    test_descriptors = np.load(ROOT / "features" / "test_desc.npy", mmap_mode="r")
    test_ecfp_similarity = sparse_tanimoto_to_query(test_ecfp, ecfp[query_row])
    test_erg_similarity = generalized_tanimoto_to_query(test_erg, erg[query_row])
    test_language_similarity = cosine_to_query(test_language, language[query_row])
    test_descriptor_distance = np.sqrt(np.square((np.asarray(test_descriptors) - descriptors[query_row]) / scales).mean(1))
    test_descriptor_similarity = np.exp(-test_descriptor_distance).astype(np.float32)
    np.savez_compressed(MODEL / "test_exact_active_views.npz", ecfp=test_ecfp_similarity,
                        erg=test_erg_similarity, chemberta=test_language_similarity,
                        descriptor=test_descriptor_similarity)

    manifest = {
        "round": 18,
        "exact_active": ACTIVE_ID,
        "submissions": [path.stem for path in files],
        "AP_method": "Local analog exploitation using ECFP, ErG, ChemBERTa and descriptors",
        "AQ_method": "Scaffold hopping using ChemBERTa, ErG and descriptors with ECFP-to-active below 0.38",
        "overlap": len(set(metadata.CatalogID.iloc[ap]) & set(metadata.CatalogID.iloc[aq])),
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "remaining_before": 7,
        "remaining_after": 5,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 18 - submit AP and AQ to VALIDATION in filename order.\n\n"
        "AP exploits close analogs of the exact active. AQ tests scaffold hops. Send both full results back in AP/AQ order.\n"
        "Seven validations remain before these files; five afterward.\n"
    )
    archive = ROOT / "submissions" / "round18_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")

    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_after_planned_round18"] = 5
    results["round18"] = {
        "AP_EXACT_HIT_ANALOGS": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
        "AQ_EXACT_HIT_SCAFFOLD_HOPS": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    for label, rows in submissions.items():
        print(label, {
            "scaffolds": int(metadata.scaffold.iloc[rows].nunique()),
            "ecfp_min": float(ecfp_similarity[rows].min()),
            "ecfp_median": float(np.median(ecfp_similarity[rows])),
            "ecfp_max": float(ecfp_similarity[rows].max()),
            "erg_median": float(np.median(erg_similarity[rows])),
            "chemberta_median": float(np.median(language_similarity[rows])),
        })


if __name__ == "__main__":
    main()
