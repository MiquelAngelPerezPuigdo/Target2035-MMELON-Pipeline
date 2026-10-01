"""Final validation: distinguish K-like from ligand/shape evidence in AS."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submissions" / "round21"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(ROOT / "submissions" / "round19" / "AS_GK_3D_SERIES_PORTFOLIO_details.csv")
    if len(source) != 50 or source.CatalogID.nunique() != 50:
        raise RuntimeError("AS source must contain 50 unique candidates")
    for column in ["K_chemberta_score", "known_ligand_score", "erg_to_ligand21", "usrcat_to_crystal21"]:
        source[column + "_rank"] = source[column].rank(method="average", pct=True)
    source["K_vs_ligand_axis"] = source.K_chemberta_score_rank - (
        source.known_ligand_score_rank + source.erg_to_ligand21_rank + source.usrcat_to_crystal21_rank
    ) / 3.0
    diagnostic = source.sort_values(["K_vs_ligand_axis", "CatalogID"], ascending=[False, True]).head(25)

    forced = pd.read_parquet(ROOT / "reports" / "validation_forced_labels.parquet")
    fillers = forced.loc[forced.forced_label == 0, "CatalogID"].tolist()
    fillers = [cid for cid in fillers if cid not in set(source.CatalogID)][:25]
    submission_ids = diagnostic.CatalogID.tolist() + fillers
    if len(submission_ids) != 50 or len(set(submission_ids)) != 50:
        raise RuntimeError("Final validation must contain 50 unique IDs")
    filename = "Team_MMELON_R21_AV_AS_KLIKE_HALF.txt"
    submission = OUT / filename
    submission.write_text("\n".join(submission_ids) + "\n")
    source.assign(selected_K_like=source.CatalogID.isin(diagnostic.CatalogID)).sort_values(
        ["selected_K_like", "K_vs_ligand_axis"], ascending=[False, False]
    ).to_csv(OUT / "AV_AS_EVIDENCE_SPLIT_private.csv", index=False)
    manifest = {
        "round": 21,
        "submission": "AV_AS_KLIKE_HALF",
        "purpose": "Use the final validation to locate AS's one hit in the K-like or ligand/shape half",
        "diagnostic_candidates": 25,
        "confirmed_negative_fillers": 25,
        "interpretation": {
            "1 hit": "Favor the K/learned branch in blind-test scoring",
            "0 hits": "Favor the compound-21 ligand/ErG branch; the AS hit is in the complementary half",
        },
        "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "remaining_before": 1,
        "remaining_after": 0,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "START_HERE.txt").write_text(
        "Final validation - submit AV_AS_KLIKE_HALF to VALIDATION.\n\n"
        "A hit favors K/learned evidence for the blind test. Zero favors ligand/ErG evidence.\n"
        "This uses the final validation submission.\n"
    )
    archive = ROOT / "submissions" / "round21_validation_batch.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission, submission.name)
        zf.write(OUT / "START_HERE.txt", "START_HERE.txt")
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Corrupt archive")
    results_path = ROOT / "reports" / "official_results.json"
    results = json.loads(results_path.read_text())
    results["quota_estimate"]["remaining_after_planned_round21"] = 0
    results["round21"] = {"AV_AS_KLIKE_HALF": {"hits": None, "chemical_series": None,
                                                 "p_value": None, "submission_id": None}}
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print({"selected_axis_median": float(diagnostic.K_vs_ligand_axis.median()),
           "complement_axis_median": float(source.loc[~source.CatalogID.isin(diagnostic.CatalogID), "K_vs_ligand_axis"].median())})


if __name__ == "__main__":
    main()
