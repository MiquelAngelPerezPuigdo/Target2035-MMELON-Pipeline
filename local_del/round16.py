"""Two-test code that reduces 12 candidates to one group of three."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round16"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r13 = pd.read_csv(ROOT / "submissions" / "round13" / "AI_CHEMBERTA_OOD_details.csv")
    r15 = pd.read_csv(ROOT / "submissions" / "round15" / "AK_AI_HIT_SPLIT2_details.csv")
    pool_ids = r15.loc[r15.role == "round14_candidate", "CatalogID"].tolist()
    pool = r13.set_index("CatalogID").loc[pool_ids].reset_index()
    if len(pool) != 12:
        raise RuntimeError(f"Expected 12 candidates, found {len(pool)}")

    # Round-robin each anchor across groups so every code group has a mixture of
    # evidence sources rather than accidentally testing one anchor against another.
    groups = {name: [] for name in "ABCD"}
    cursor = 0
    for _, anchor_group in pool.sort_values(["anchor", "consensus_percentile"], ascending=[True, False]).groupby("anchor"):
        for cid in anchor_group.CatalogID:
            groups["ABCD"[cursor % 4]].append(cid)
            cursor += 1
    # Rebalance deterministically if the anchor round-robin did not make 3/3/3/3.
    while max(map(len, groups.values())) > 3:
        source = next(name for name in "ABCD" if len(groups[name]) > 3)
        target = next(name for name in "ABCD" if len(groups[name]) < 3)
        groups[target].append(groups[source].pop())
    if sorted(map(len, groups.values())) != [3, 3, 3, 3]:
        raise RuntimeError(groups)

    forced = pd.read_parquet(ROOT / "reports" / "validation_forced_labels.parquet")
    fillers = forced.loc[forced.forced_label == 0, "CatalogID"].tolist()[:44]
    tests = {
        "AL_AI_CODE_BIT1": groups["A"] + groups["B"],
        "AM_AI_CODE_BIT2": groups["A"] + groups["C"],
    }
    files = []
    for label, diagnostic in tests.items():
        ids = diagnostic + fillers
        if len(ids) != 50 or len(set(ids)) != 50:
            raise RuntimeError((label, len(ids), len(set(ids))))
        filename = f"Team_MMELON_R16_{label}.txt"
        path = OUT / filename
        path.write_text("\n".join(ids) + "\n")
        files.append(path)

    group_rows = []
    for name, ids in groups.items():
        for cid in ids:
            group_rows.append({"code_group": name, "CatalogID": cid})
    pd.DataFrame(group_rows).merge(
        pool[["CatalogID", "anchor", "SMILES", "consensus_percentile"]], on="CatalogID", how="left"
    ).to_csv(OUT / "AI_CODE_GROUPS_private.csv", index=False)
    decoding = {
        "AL=1, AM=1": "group A",
        "AL=1, AM=0": "group B",
        "AL=0, AM=1": "group C",
        "AL=0, AM=0": "group D",
    }
    manifest = {
        "round": 16,
        "submissions": [path.stem for path in files],
        "purpose": "Two-bit group test reducing 12 candidates to 3",
        "candidate_groups": {name: len(ids) for name, ids in groups.items()},
        "diagnostic_candidates_per_file": 6,
        "confirmed_negative_fillers_per_file": 44,
        "decoding": decoding,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "remaining_before": 11,
        "remaining_after": 9,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 16 - submit AL and AM to VALIDATION in filename order.\n\n"
        "Send both hit counts back in the same AL/AM order. The two results locate the active molecule within one group of three.\n"
        "Eleven validations remain before these files; nine afterward.\n"
    )
    archive = ROOT / "submissions" / "round16_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")

    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_after_planned_round16"] = 9
    results["round16"] = {
        "AL_AI_CODE_BIT1": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
        "AM_AI_CODE_BIT2": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
