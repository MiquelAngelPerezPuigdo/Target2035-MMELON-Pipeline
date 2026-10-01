"""Infer molecule-level labels forced by aggregate validation hit counts."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_bags() -> list[dict]:
    results = json.loads((ROOT / "reports" / "official_results.json").read_text())
    bags = []
    for round_name, round_results in results.items():
        if not round_name.startswith("round"):
            continue
        archive = ROOT / "submissions" / f"{round_name}_validation_batch.zip"
        with zipfile.ZipFile(archive) as zf:
            entries = [name for name in zf.namelist() if name.startswith("Team_")]
            for key, result in round_results.items():
                matches = [name for name in entries if f"_{key}.txt" in name]
                if len(matches) != 1:
                    raise ValueError((round_name, key, matches))
                members = set(zf.read(matches[0]).decode().splitlines())
                if len(members) != 50:
                    raise ValueError((matches[0], len(members)))
                bags.append({"name": key, "members": members, "hits": int(result["hits"])})
    return bags


def propagate(bags: list[dict]) -> tuple[dict[str, int], list[dict]]:
    labels: dict[str, int] = {}
    audit = []
    changed = True
    while changed:
        changed = False
        for bag in bags:
            known_positive = sum(labels.get(member) == 1 for member in bag["members"])
            unknown = [member for member in bag["members"] if member not in labels]
            remaining_hits = bag["hits"] - known_positive
            if remaining_hits < 0 or remaining_hits > len(unknown):
                raise ValueError(f"Contradictory bag constraints: {bag['name']}")
            inferred = None
            if remaining_hits == 0:
                inferred = 0
            elif remaining_hits == len(unknown):
                inferred = 1
            if inferred is not None and unknown:
                for member in unknown:
                    labels[member] = inferred
                audit.append({"bag": bag["name"], "label": inferred, "count": len(unknown)})
                changed = True
    return labels, audit


def main() -> None:
    bags = load_bags()
    labels, audit = propagate(bags)
    metadata = pd.read_parquet(ROOT / "features" / "validation_meta.parquet")
    output = metadata[metadata.CatalogID.isin(labels)].copy()
    output["forced_label"] = output.CatalogID.map(labels).astype(int)
    output.to_parquet(ROOT / "reports" / "validation_forced_labels.parquet", index=False)
    report = {
        "bags": len(bags),
        "unique_submitted": len(set().union(*(bag["members"] for bag in bags))),
        "forced_positive": sum(value == 1 for value in labels.values()),
        "forced_negative": sum(value == 0 for value in labels.values()),
        "unresolved_submitted": len(set().union(*(bag["members"] for bag in bags))) - len(labels),
        "propagation": audit,
    }
    (ROOT / "reports" / "validation_label_inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
