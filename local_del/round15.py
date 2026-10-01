"""Second adaptive split to locate the single round-13 ChemBERTa hit."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round15"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r13 = pd.read_csv(ROOT / "submissions" / "round13" / "AI_CHEMBERTA_OOD_details.csv")
    r14 = pd.read_csv(ROOT / "submissions" / "round14" / "AJ_AI_HIT_SPLIT1_details.csv")
    pool_ids = r14.loc[r14.role == "round13_candidate", "CatalogID"]
    pool = r13[r13.CatalogID.isin(pool_ids)].copy()
    if len(pool) != 25:
        raise RuntimeError(f"Expected 25 candidates, found {len(pool)}")

    quotas = {
        "G_25hits_3series": 3,
        "K_21hits_2series": 3,
        "AF_7confirmed": 3,
        "O_4hits_3series": 2,
        "AD_2hits_2series": 1,
    }
    diagnostic = []
    for anchor, count in quotas.items():
        group = pool[pool.anchor == anchor].sort_values(
            ["consensus_percentile", "CatalogID"], ascending=[False, True]
        )
        diagnostic.extend(group.iloc[::2].head(count).CatalogID.tolist())
    if len(diagnostic) != 12 or len(set(diagnostic)) != 12:
        raise RuntimeError("Diagnostic subset must contain 12 unique candidates")

    forced = pd.read_parquet(ROOT / "reports" / "validation_forced_labels.parquet")
    fillers = forced.loc[forced.forced_label == 0, "CatalogID"].tolist()[:38]
    ids = diagnostic + fillers
    if len(ids) != 50 or len(set(ids)) != 50:
        raise RuntimeError("Submission must contain 50 unique IDs")
    filename = "Team_MMELON_R15_AK_AI_HIT_SPLIT2.txt"
    submission = OUT / filename
    submission.write_text("\n".join(ids) + "\n")
    pd.DataFrame({
        "CatalogID": ids,
        "role": ["round14_candidate"] * 12 + ["confirmed_negative_filler"] * 38,
    }).to_csv(OUT / "AK_AI_HIT_SPLIT2_details.csv", index=False)
    manifest = {
        "round": 15,
        "submission": "AK_AI_HIT_SPLIT2",
        "purpose": "Second adaptive split to locate the one round-13 hit",
        "diagnostic_candidates": 12,
        "confirmed_negative_fillers": 38,
        "interpretation": {
            "0 hits": "The active molecule is among the other 13 round-14 candidates",
            "1 hit": "The active molecule is among these 12 diagnostic candidates",
        },
        "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "remaining_before": 12,
        "remaining_after": 11,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 15 - submit AK_AI_HIT_SPLIT2 only to VALIDATION.\n\n"
        "Send back the result before submitting anything else.\n"
        "Twelve validations remain before this submission; eleven afterward.\n"
    )
    archive = ROOT / "submissions" / "round15_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission, submission.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")

    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_after_planned_round15"] = 11
    results["round15"] = {"AK_AI_HIT_SPLIT2": {"hits": None, "chemical_series": None,
                                                 "p_value": None, "submission_id": None}}
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
