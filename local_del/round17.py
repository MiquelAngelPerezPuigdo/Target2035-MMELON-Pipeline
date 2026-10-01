"""Two singleton tests that identify the exact round-13 active molecule."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round17"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(ROOT / "submissions" / "round16" / "AI_CODE_GROUPS_private.csv")
    candidates = groups.loc[groups.code_group == "A"].sort_values("CatalogID").reset_index(drop=True)
    if len(candidates) != 3:
        raise RuntimeError(f"Expected three group-A candidates, found {len(candidates)}")
    forced = pd.read_parquet(ROOT / "reports" / "validation_forced_labels.parquet")
    fillers = forced.loc[forced.forced_label == 0, "CatalogID"].tolist()[:49]

    labels = ["AN_AI_SINGLETON_1", "AO_AI_SINGLETON_2"]
    files = []
    detail_rows = []
    for index, label in enumerate(labels):
        candidate = candidates.CatalogID.iat[index]
        ids = [candidate] + fillers
        if len(ids) != 50 or len(set(ids)) != 50:
            raise RuntimeError((label, len(ids), len(set(ids))))
        path = OUT / f"Team_MMELON_R17_{label}.txt"
        path.write_text("\n".join(ids) + "\n")
        files.append(path)
        detail_rows.append({"submission": label, "candidate": candidate,
                            "anchor": candidates.anchor.iat[index], "SMILES": candidates.SMILES.iat[index]})
    # Candidate 3 is inferred active if both submitted singleton tests score zero.
    detail_rows.append({"submission": "inferred_if_AN0_AO0", "candidate": candidates.CatalogID.iat[2],
                        "anchor": candidates.anchor.iat[2], "SMILES": candidates.SMILES.iat[2]})
    pd.DataFrame(detail_rows).to_csv(OUT / "AI_SINGLETON_DECODING_private.csv", index=False)
    decoding = {
        "AN=1, AO=0": candidates.CatalogID.iat[0],
        "AN=0, AO=1": candidates.CatalogID.iat[1],
        "AN=0, AO=0": candidates.CatalogID.iat[2],
        "AN=1, AO=1": "Contradiction: recheck submission order/results",
    }
    manifest = {
        "round": 17,
        "submissions": [path.stem for path in files],
        "purpose": "Identify the exact active molecule among code group A's three candidates",
        "diagnostic_candidates_per_file": 1,
        "confirmed_negative_fillers_per_file": 49,
        "decoding": decoding,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "remaining_before": 9,
        "remaining_after": 7,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 17 - submit AN and AO to VALIDATION in filename order.\n\n"
        "Send both hit counts back in the same AN/AO order. The pair identifies the exact active molecule.\n"
        "Nine validations remain before these files; seven afterward.\n"
    )
    archive = ROOT / "submissions" / "round17_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")

    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_after_planned_round17"] = 7
    results["round17"] = {
        "AN_AI_SINGLETON_1": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
        "AO_AI_SINGLETON_2": {"hits": None, "chemical_series": None, "p_value": None, "submission_id": None},
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
