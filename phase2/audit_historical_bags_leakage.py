"""Leakage-safe leave-one-panel-out check for aggregate validation hit counts.

Unlike the earlier exploratory audit in local_del/audit_count_model.py, this
rebuilds exact molecule labels from training panels only after removing every
training panel that touches a molecule in the held-out panel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEL = ROOT / "local_del"
sys.path.insert(0, str(LOCAL_DEL))
from infer_validation_labels import load_bags, propagate  # noqa: E402
from revise_test_count_model import fit  # noqa: E402


def selected_feature_rows(split: str, catalog_ids: list[str], kind: str) -> sparse.csr_matrix:
    """Unpack only requested rows from the cached phase-one fingerprints."""
    metadata = pd.read_parquet(LOCAL_DEL / "features" / f"{split}_meta.parquet")
    row_for_id = dict(zip(metadata.CatalogID.astype(str), metadata.index))
    rows = np.asarray([row_for_id[catalog_id] for catalog_id in catalog_ids], dtype=np.int64)
    bits = np.load(LOCAL_DEL / "features" / f"{split}_bits.npy", mmap_mode="r")
    bit_column = 0 if kind == "ecfp" else 1
    unpacked = np.unpackbits(bits[rows, bit_column], axis=1)
    return sparse.csr_matrix(unpacked, dtype=np.float64)


def training_problem(train_bags: list[dict], test_negatives: list[str]):
    labels, _ = propagate(train_bags)
    val_ids = sorted(set().union(*(bag["members"] for bag in train_bags))) if train_bags else []
    keys = [("validation", cid) for cid in val_ids] + [("test", cid) for cid in test_negatives]
    col = {key: i for i, key in enumerate(keys)}

    reduced: dict[tuple[int, ...], dict] = {}
    for bag in train_bags:
        unknown = tuple(sorted(col[("validation", cid)] for cid in bag["members"] if cid not in labels))
        remaining_hits = int(bag["hits"] - sum(labels.get(cid) == 1 for cid in bag["members"]))
        if not unknown:
            if remaining_hits != 0:
                raise ValueError(f"Contradictory training bag after propagation: {bag['name']}")
            continue
        if remaining_hits < 0 or remaining_hits > len(unknown):
            raise ValueError(f"Impossible residual count in training bag: {bag['name']}")
        if unknown in reduced and reduced[unknown]["hits"] != remaining_hits:
            raise ValueError("Duplicate residual panels imply different hit counts")
        reduced.setdefault(unknown, {"hits": remaining_hits, "name": bag["name"]})

    entries = list(reduced.items())
    row_idx, col_idx = [], []
    for row, (members, _) in enumerate(entries):
        row_idx.extend([row] * len(members))
        col_idx.extend(members)
    incidence = sparse.csr_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)), shape=(len(entries), len(keys))
    )
    sizes = np.asarray(incidence.sum(axis=1)).ravel()
    counts = np.asarray([value["hits"] for _, value in entries], dtype=float)
    memberships = np.maximum(np.asarray(incidence.sum(axis=0)).ravel(), 1)
    bag_weights = np.asarray([
        1.0 / np.mean(memberships[list(members)]) for members, _ in entries
    ])

    known = {col[("validation", cid)]: label for cid, label in labels.items() if cid in set(val_ids)}
    known.update({col[("test", cid)]: 0 for cid in test_negatives})
    known_idx = np.asarray(sorted(known), dtype=int)
    known_y = np.asarray([known[i] for i in known_idx], dtype=float)
    known_weights = np.asarray([3.0 if keys[i][0] == "test" else 1.0 for i in known_idx])
    return keys, incidence, sizes, counts, bag_weights, known_idx, known_y, known_weights, labels


def main() -> None:
    bags = load_bags()
    by_name = {bag["name"]: bag for bag in bags}
    test_submission = pd.read_csv(LOCAL_DEL / "submissions" / "blind_test" / "Team_MMELON_T1_FROZEN_G.csv")
    test_negatives = test_submission.loc[test_submission.Sel_50.eq(1), "CatalogID"].astype(str).tolist()
    targets = [
        "G_KNOWN_LIGANDS", "K_NEW_CORES", "O_DIVERSE_BAG",
        "AI_CHEMBERTA_OOD", "AP_EXACT_HIT_ANALOGS", "AQ_EXACT_HIT_SCAFFOLD_HOPS",
    ]
    results = []
    for kind in ("ecfp", "fcfp"):
        for target_name in targets:
            target = by_name[target_name]
            held = set(target["members"])
            train_bags = [bag for bag in bags if bag["members"].isdisjoint(held)]
            problem = training_problem(train_bags, test_negatives)
            keys, incidence, sizes, counts, bag_weights, known_idx, known_y, known_weights, labels = problem
            validation_ids = [cid for split, cid in keys if split == "validation"]
            test_ids = [cid for split, cid in keys if split == "test"]
            x = sparse.vstack([
                selected_feature_rows("validation", validation_ids, kind),
                selected_feature_rows("test", test_ids, kind),
            ], format="csr")
            coef, objective = fit(
                x, incidence, sizes, counts, bag_weights,
                known_idx, known_y, known_weights, 0.01,
            )
            val_meta = pd.read_parquet(LOCAL_DEL / "features" / "validation_meta.parquet")
            row_for_id = dict(zip(val_meta.CatalogID.astype(str), val_meta.index))
            held_rows = [row_for_id[cid] for cid in sorted(held)]
            held_x = selected_feature_rows("validation", sorted(held), kind)
            held_prob = expit(np.asarray(held_x @ coef[1:]).ravel() + coef[0])
            # Compare the expected count to the observed panel count. This is a
            # diagnostic of panel-count fit, not a calibrated hit forecast.
            result = {
                "representation": kind,
                "heldout_panel": target_name,
                "heldout_ids": len(held),
                "observed_hits": int(target["hits"]),
                "predicted_hit_mass": float(held_prob.sum()),
                "absolute_count_error": float(abs(held_prob.sum() - target["hits"])),
                "training_panels_after_overlap_exclusion": len(train_bags),
                "training_residual_constraints": len(counts),
                "training_forced_positives": int((known_y == 1).sum()),
                "training_forced_negatives_including_T1": int((known_y == 0).sum()),
                "heldout_panel_used_to_infer_labels": False,
                "objective": objective,
            }
            # Keep row_for_id lookup exercised so an ID/index integrity mistake
            # fails loudly if the source metadata changes.
            assert len(held_rows) == len(held_prob)
            results.append(result)
            print(json.dumps(result), flush=True)

    grouped = {}
    for result in results:
        grouped.setdefault(result["representation"], []).append(result["absolute_count_error"])
    report = {
        "source": "48 archived round validation lists and exact aggregate hit counts",
        "method": "For each held-out panel, remove every other panel sharing any CatalogID, then recompute forced labels from only the remaining panels before fitting.",
        "regularization": 0.01,
        "diagnostics": results,
        "mean_absolute_count_error": {kind: float(np.mean(errors)) for kind, errors in grouped.items()},
        "limitation": "The panels were adaptively selected. Even overlap-excluded held-out panels are not an independent sample of the remaining library; predicted probability sums are count-fit diagnostics, not calibrated forecasts.",
    }
    out = ROOT / "phase2" / "reports" / "historical_bag_transfer_leakage_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
