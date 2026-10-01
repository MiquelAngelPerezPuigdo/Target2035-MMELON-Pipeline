"""Crystal-shape reranking of compound-21 and G/K-derived candidates."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round19"


def ids(round_number: int, marker: str) -> list[str]:
    with zipfile.ZipFile(ROOT / "submissions" / f"round{round_number:02d}_validation_batch.zip") as zf:
        name = next(name for name in zf.namelist() if name.startswith("Team_") and marker in name)
        return zf.read(name).decode().splitlines()


def percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.linspace(0, 1, len(values), dtype=np.float32)
    return result


def normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def tanimoto_to_selected(matrix: sparse.csr_matrix, row: int, selected: list[int]) -> float:
    if not selected:
        return 0.0
    query = matrix[row]
    references = matrix[selected]
    intersection = (references @ query.T).toarray().ravel()
    left = np.asarray(references.multiply(references).sum(1)).ravel()
    right = float(query.multiply(query).sum())
    return float(np.max(intersection / np.maximum(left + right - intersection, 1e-12)))


def choose(ranking: np.ndarray, rows: np.ndarray, ecfp: sparse.csr_matrix,
           excluded: set[int], threshold: float) -> list[int]:
    chosen: list[int] = []
    for local_row in ranking:
        global_row = int(rows[local_row])
        if global_row in excluded:
            continue
        if tanimoto_to_selected(ecfp, global_row, chosen) >= threshold:
            continue
        chosen.append(global_row)
        if len(chosen) == 50:
            return chosen
    raise RuntimeError(f"Only selected {len(chosen)} candidates")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(ROOT / "features" / "validation_meta.parquet")
    shape = pd.read_parquet(ROOT / "models" / "crystal_shape_ligand21.parquet")
    shape = shape[shape.usrcat.notna()].reset_index(drop=True)
    rows = shape.row.to_numpy(dtype=int)
    language = np.load(ROOT / "features" / "validation_chemberta.npy", mmap_mode="r")
    row_for_id = pd.Series(metadata.index.to_numpy(), index=metadata.CatalogID).to_dict()
    k_rows = [row_for_id[cid] for cid in ids(4, "K_NEW_CORES")]
    k_refs = normalized(language[k_rows])
    k_similarity = np.empty(len(rows), dtype=np.float32)
    for begin in range(0, len(rows), 2000):
        query = normalized(language[rows[begin : begin + 2000]])
        similarities = query @ k_refs.T
        similarities.sort(axis=1)
        k_similarity[begin : begin + len(query)] = 0.7 * similarities[:, -1] + 0.3 * similarities[:, -3:].mean(1)

    g_rank = percentile(shape.known_ligand_score.to_numpy())
    erg_rank = percentile(shape.erg_to_ligand21.to_numpy())
    shape_rank = percentile(shape.usrcat.to_numpy())
    k_rank = percentile(k_similarity)
    crystal_score = 0.42 * g_rank + 0.38 * shape_rank + 0.20 * erg_rank
    portfolio_score = 0.32 * shape_rank + 0.27 * erg_rank + 0.26 * k_rank + 0.15 * g_rank
    tie = shape.CatalogID.to_numpy()
    ecfp = sparse.load_npz(ROOT / "features" / "validation_ecfp.npz").tocsr()
    ar = choose(np.lexsort((tie, -crystal_score)), rows, ecfp, set(), threshold=0.68)
    ass = choose(np.lexsort((tie, -portfolio_score)), rows, ecfp, set(ar), threshold=0.48)

    row_to_local = {int(row): i for i, row in enumerate(rows)}
    submissions = {"AR_LIGAND21_CRYSTAL_3D": ar, "AS_GK_3D_SERIES_PORTFOLIO": ass}
    files = []
    for label, selected in submissions.items():
        path = OUT / f"Team_MMELON_R19_{label}.txt"
        path.write_text("\n".join(metadata.CatalogID.iloc[selected]) + "\n")
        files.append(path)
        local = np.asarray([row_to_local[row] for row in selected])
        details = metadata.iloc[selected].copy()
        details["usrcat_to_crystal21"] = shape.usrcat.iloc[local].to_numpy()
        details["erg_to_ligand21"] = shape.erg_to_ligand21.iloc[local].to_numpy()
        details["known_ligand_score"] = shape.known_ligand_score.iloc[local].to_numpy()
        details["K_chemberta_score"] = k_similarity[local]
        details["combined_score"] = crystal_score[local] if label.startswith("AR_") else portfolio_score[local]
        details.to_csv(OUT / f"{label}_details.csv", index=False)

    manifest = {
        "round": 19,
        "submissions": [path.stem for path in files],
        "AR_method": "Compound-21 crystal USRCAT shape plus original known-ligand and ErG scores",
        "AS_method": "Scaffold-diverse 3D/ErG portfolio with K-bag ChemBERTa evidence",
        "shortlist_conformers": int(len(shape)),
        "overlap": len(set(metadata.CatalogID.iloc[ar]) & set(metadata.CatalogID.iloc[ass])),
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "remaining_before": 5,
        "remaining_after": 3,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 19 - submit AR and AS to VALIDATION in filename order.\n\n"
        "AR uses crystal-ligand 3D shape. AS is a diverse G/K 3D portfolio. Send both results back in AR/AS order.\n"
        "Five validations remain before these files; three afterward.\n"
    )
    archive = ROOT / "submissions" / "round19_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")

    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_after_planned_round19"] = 3
    results["round19"] = {
        "AR_LIGAND21_CRYSTAL_3D": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
        "AS_GK_3D_SERIES_PORTFOLIO": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    for label, selected in submissions.items():
        local = np.asarray([row_to_local[row] for row in selected])
        print(label, {"scaffolds": int(metadata.scaffold.iloc[selected].nunique()),
                      "median_usrcat": float(shape.usrcat.iloc[local].median()),
                      "median_erg": float(shape.erg_to_ligand21.iloc[local].median()),
                      "median_G": float(shape.known_ligand_score.iloc[local].median())})


if __name__ == "__main__":
    main()
