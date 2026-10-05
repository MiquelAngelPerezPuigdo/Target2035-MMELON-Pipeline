"""Measure DEL/experimental-neighborhood blends on chemistry-grouped OOF scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def main() -> None:
    path = ROOT / "data" / "release_oof_scores.parquet"
    frame = pd.read_parquet(path)
    y = frame.Label.to_numpy(dtype=np.int8)
    groups = frame.ECFP4_cluster.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2605)
    fold_for_row = np.full(len(frame), -1, dtype=np.int8)
    for fold, (_, heldout) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        fold_for_row[heldout] = fold
    if np.any(fold_for_row < 0):
        raise ValueError("Some release rows were not assigned to an OOF fold")

    signals = {
        "frozen_DEL_ECFP": frame.frozen_DEL_ECFP.to_numpy(),
        "frozen_DEL_FCFP": frame.frozen_DEL_FCFP.to_numpy(),
        "mean_frozen_DEL_rank": None,
        "historical_G": frame.historical_G.to_numpy(),
        "DEL_plus_release_adaptation_OOF": frame.DEL_adaptation_oof.to_numpy(),
    }
    neighborhood = frame.release_neighbor_oof.to_numpy()
    weights = [round(value, 1) for value in np.linspace(0.0, 1.0, 11)]
    rows = []
    for name, signal in signals.items():
        for del_weight in weights:
            fold_hits, fold_ap, fold_auc = [], [], []
            for fold in range(5):
                idx = np.flatnonzero(fold_for_row == fold)
                neighbor_rank = percentile(neighborhood[idx])
                if name == "mean_frozen_DEL_rank":
                    del_rank = (
                        percentile(frame.frozen_DEL_ECFP.to_numpy()[idx])
                        + percentile(frame.frozen_DEL_FCFP.to_numpy()[idx])
                    ) / 2
                else:
                    del_rank = percentile(signal[idx])
                blended = (1 - del_weight) * neighbor_rank + del_weight * del_rank
                order = np.argsort(-blended, kind="stable")
                fold_hits.append(int(y[idx[order[:50]]].sum()))
                fold_ap.append(float(average_precision_score(y[idx], blended)))
                fold_auc.append(float(roc_auc_score(y[idx], blended)))
            rows.append({
                "DEL_signal": name,
                "DEL_rank_weight": del_weight,
                "top50_hits_across_five_heldout_folds": int(sum(fold_hits)),
                "fold_top50_hits": fold_hits,
                "mean_fold_average_precision": float(np.mean(fold_ap)),
                "mean_fold_roc_auc": float(np.mean(fold_auc)),
            })

    summaries = {}
    for name in signals:
        baseline = next(row for row in rows if row["DEL_signal"] == name and row["DEL_rank_weight"] == 0)
        best_top50 = max(row["top50_hits_across_five_heldout_folds"] for row in rows if row["DEL_signal"] == name)
        summaries[name] = {
            "neighbor_only_top50_hits": baseline["top50_hits_across_five_heldout_folds"],
            "best_blend_top50_hits": best_top50,
            "max_average_precision": max(row["mean_fold_average_precision"] for row in rows if row["DEL_signal"] == name),
            "top50_improved_over_neighbor_only": best_top50 > baseline["top50_hits_across_five_heldout_folds"],
        }

    report = {
        "source": "phase2/data/release_oof_scores.parquet",
        "method": "Combine within-fold percentile ranks of the release-label neighborhood and each DEL-containing score at fixed weights; evaluate only held-out rows from the original five chemistry-group folds.",
        "folds": 5,
        "weight_grid": weights,
        "positive_count": int(y.sum()),
        "grouping": "Butina ECFP4 clusters at Tanimoto >= 0.65, StratifiedGroupKFold seed 2605",
        "summary": summaries,
        "results": rows,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "limitation": "The 39 binders were selected by participant workflows. These OOF comparisons measure transfer within that selected release and do not estimate hits in the remaining test pool. Repeated weight comparisons also create model-selection risk; no test submission is justified by this sweep alone.",
    }
    out = ROOT / "reports" / "release_del_blend_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
