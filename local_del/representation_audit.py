"""Compare molecular representations on labels forced by validation feedback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def generalized_tanimoto(matrix: np.ndarray, references: np.ndarray) -> np.ndarray:
    dot = matrix @ references.T
    left = np.square(matrix).sum(1, keepdims=True)
    right = np.square(references).sum(1)[None, :]
    return dot / np.maximum(left + right - dot, 1e-12)


def sparse_tanimoto(matrix: sparse.csr_matrix, references: sparse.csr_matrix) -> np.ndarray:
    dot = (matrix @ references.T).toarray()
    left = np.asarray(matrix.multiply(matrix).sum(1))
    right = np.asarray(references.multiply(references).sum(1)).T
    return dot / np.maximum(left + right - dot, 1e-12)


def cosine(matrix: np.ndarray, references: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    references = references.astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    references /= np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-12)
    return matrix @ references.T


def score_from_similarity(similarity: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positive_columns = np.flatnonzero(labels == 1)
    score = np.empty(len(labels), dtype=np.float32)
    for row in range(len(labels)):
        columns = positive_columns[positive_columns != row]
        score[row] = similarity[row, columns].max()
    return score


def main() -> None:
    metadata = pd.read_parquet(ROOT / "features" / "validation_meta.parquet")
    forced = pd.read_parquet(ROOT / "reports" / "validation_forced_labels.parquet")
    rows = metadata.index[metadata.CatalogID.isin(forced.CatalogID)].to_numpy()
    label_map = dict(zip(forced.CatalogID, forced.forced_label))
    labels = metadata.CatalogID.iloc[rows].map(label_map).to_numpy(dtype=np.int8)

    ecfp = sparse.load_npz(ROOT / "features" / "validation_ecfp.npz")[rows].tocsr()
    fcfp = sparse.load_npz(ROOT / "features" / "validation_fcfp.npz")[rows].tocsr()
    erg = np.asarray(np.load(ROOT / "features" / "validation_erg.npy", mmap_mode="r")[rows])
    language = np.asarray(np.load(ROOT / "features" / "validation_chemberta.npy", mmap_mode="r")[rows])
    similarities = {
        "ECFP4": sparse_tanimoto(ecfp, ecfp),
        "FCFP4_plus_descriptors": sparse_tanimoto(fcfp, fcfp),
        "ErG_pharmacophore": generalized_tanimoto(erg, erg),
        "ChemBERTa_mean_pool": cosine(language, language),
    }
    report = {"forced_examples": len(labels), "positives": int(labels.sum()), "representations": {}}
    for name, similarity in similarities.items():
        score = score_from_similarity(similarity, labels)
        report["representations"][name] = {
            "average_precision": float(average_precision_score(labels, score)),
            "roc_auc": float(roc_auc_score(labels, score)),
            "positive_score_mean": float(score[labels == 1].mean()),
            "negative_score_mean": float(score[labels == 0].mean()),
        }
    (ROOT / "reports" / "representation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
