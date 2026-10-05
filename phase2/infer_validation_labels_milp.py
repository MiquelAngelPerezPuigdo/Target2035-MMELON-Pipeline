"""Find extra molecule labels logically implied by exact validation bag counts.

The existing propagation audit recognizes only bags with zero or all remaining
hits. This script checks each unresolved CatalogID against the complete binary
integer system and records a label only when one of the two assignments is
provably infeasible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEL = ROOT / "local_del"
sys.path.insert(0, str(LOCAL_DEL))
from infer_validation_labels import load_bags, propagate  # noqa: E402


def main() -> None:
    bags = load_bags()
    labels, _ = propagate(bags)
    catalog_ids = sorted(set().union(*(bag["members"] for bag in bags)))
    index = {catalog_id: i for i, catalog_id in enumerate(catalog_ids)}
    incidence = lil_matrix((len(bags), len(catalog_ids)), dtype=np.float64)
    hit_counts = np.asarray([bag["hits"] for bag in bags], dtype=np.float64)
    for row, bag in enumerate(bags):
        for catalog_id in bag["members"]:
            incidence[row, index[catalog_id]] = 1.0

    incidence = incidence.tocsr()
    known = np.asarray([labels.get(catalog_id, -1) for catalog_id in catalog_ids])
    unknown_indices = np.flatnonzero(known < 0)
    integrality = np.ones(len(catalog_ids), dtype=np.int8)
    constraint = LinearConstraint(incidence, hit_counts, hit_counts)
    extra_labels: dict[str, int] = {}
    checked_alternatives = 0

    for catalog_index in unknown_indices:
        outcomes: dict[int, int] = {}
        for proposed_label in (0, 1):
            lower = np.zeros(len(catalog_ids))
            upper = np.ones(len(catalog_ids))
            fixed_indices = np.flatnonzero(known >= 0)
            lower[fixed_indices] = known[fixed_indices]
            upper[fixed_indices] = known[fixed_indices]
            lower[catalog_index] = upper[catalog_index] = proposed_label
            result = milp(
                np.zeros(len(catalog_ids)),
                integrality=integrality,
                bounds=Bounds(lower, upper),
                constraints=constraint,
            )
            checked_alternatives += 1
            if result.status == 0:
                outcomes[proposed_label] = 1
            elif result.status == 2:
                outcomes[proposed_label] = 0
            else:
                raise RuntimeError(
                    f"Solver could not prove feasibility for {catalog_ids[catalog_index]}="
                    f"{proposed_label}: {result.message}"
                )
        if outcomes == {0: 1, 1: 0}:
            extra_labels[catalog_ids[catalog_index]] = 0
        elif outcomes == {0: 0, 1: 1}:
            extra_labels[catalog_ids[catalog_index]] = 1
        elif outcomes != {0: 1, 1: 1}:
            raise ValueError(f"No feasible assignment for {catalog_ids[catalog_index]}")

    report = {
        "source": "local_del archived validation batch lists and exact aggregate hit counts",
        "method": "For every unresolved ID, test label 0 and label 1 against the full binary MILP; add a label only if the opposite assignment is infeasible.",
        "solver": {"name": "scipy.optimize.milp (HiGHS)", "scipy_version": scipy.__version__},
        "bags": len(bags),
        "unique_submitted_ids": len(catalog_ids),
        "propagation_labels": {
            "positive": sum(value == 1 for value in labels.values()),
            "negative": sum(value == 0 for value in labels.values()),
        },
        "unresolved_checked": len(unknown_indices),
        "binary_feasibility_checks": checked_alternatives,
        "additional_forced_labels": [
            {"CatalogID": catalog_id, "label": label}
            for catalog_id, label in sorted(extra_labels.items())
        ],
        "combined_labels": {
            "positive": sum(value == 1 for value in labels.values())
            + sum(value == 1 for value in extra_labels.values()),
            "negative": sum(value == 0 for value in labels.values())
            + sum(value == 0 for value in extra_labels.values()),
            "unresolved": len(catalog_ids) - len(labels) - len(extra_labels),
        },
        "limitation": "These are adaptive validation feedback labels inferred from participant-selected panels, not an independent estimate of generalization.",
    }
    out = ROOT / "phase2" / "reports" / "validation_label_milp.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
