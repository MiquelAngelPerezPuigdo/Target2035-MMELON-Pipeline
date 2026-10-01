"""First adaptive split to locate the single round-13 hit."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round14"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    details = pd.read_csv(ROOT / "submissions" / "round13" / "AI_CHEMBERTA_OOD_details.csv")
    quotas = {
        "G_25hits_3series": 6,
        "K_21hits_2series": 6,
        "AF_7confirmed": 5,
        "O_4hits_3series": 5,
        "AD_2hits_2series": 3,
    }
    diagnostic = []
    for anchor, count in quotas.items():
        group = details[details.anchor == anchor].sort_values(
            ["consensus_percentile", "CatalogID"], ascending=[False, True]
        )
        # Alternating ranks makes this half similar in model confidence to the
        # complement, so a zero or one result remains equally informative.
        chosen = pd.concat([group.iloc[::2], group.iloc[1::2]]).head(count)
        diagnostic.extend(chosen.CatalogID.tolist())
    if len(diagnostic) != 25 or len(set(diagnostic)) != 25:
        raise RuntimeError("Diagnostic half must contain 25 unique round-13 candidates")

    forced = pd.read_parquet(ROOT / "reports" / "validation_forced_labels.parquet")
    fillers = forced.loc[forced.forced_label == 0, "CatalogID"].tolist()[:25]
    if set(diagnostic) & set(fillers):
        raise RuntimeError("A diagnostic candidate was used as a negative filler")
    ids = diagnostic + fillers
    filename = "Team_MMELON_R14_AJ_AI_HIT_SPLIT1.txt"
    submission = OUT / filename
    submission.write_text("\n".join(ids) + "\n")
    pd.DataFrame({
        "CatalogID": ids,
        "role": ["round13_candidate"] * 25 + ["confirmed_negative_filler"] * 25,
    }).to_csv(OUT / "AJ_AI_HIT_SPLIT1_details.csv", index=False)
    manifest = {
        "round": 14,
        "submission": "AJ_AI_HIT_SPLIT1",
        "purpose": "Adaptive binary split to locate the one round-13 hit",
        "diagnostic_candidates": 25,
        "confirmed_negative_fillers": 25,
        "interpretation": {
            "0 hits": "The active molecule is among the other 25 round-13 candidates",
            "1 hit": "The active molecule is among these 25 diagnostic candidates",
        },
        "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "remaining_before": 13,
        "remaining_after": 12,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Round 14 - submit AJ_AI_HIT_SPLIT1 only to VALIDATION.\n\n"
        "This is an adaptive split, so send back the result before submitting anything else.\n"
        "Thirteen validations remain before this submission; twelve afterward.\n"
    )
    archive = ROOT / "submissions" / "round14_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission, submission.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")
    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_after_planned_round14"] = 12
    results["round14"] = {"AJ_AI_HIT_SPLIT1": {"hits": None, "chemical_series": None,
                                                 "p_value": None, "submission_id": None}}
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
