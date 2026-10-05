"""Prepare a fresh validation probe using the exact successful R03-G score."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LOCAL = REPO / "local_del"
OUT = ROOT / "submissions" / "panels" / "DEL_ligand_replay"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    meta = pd.read_parquet(LOCAL / "features" / "validation_meta.parquet")
    rankings = pd.read_parquet(LOCAL / "models/deep_audit/round03_full_rankings.parquet")
    bits = np.load(LOCAL / "features/validation_bits.npy", mmap_mode="r")
    require(meta.CatalogID.astype(str).tolist() == rankings.CatalogID.astype(str).tolist(),
            "Validation metadata and frozen R03 rankings no longer share exact row order.")
    score = rankings.G_KNOWN_LIGANDS_score.to_numpy(dtype=np.float64)
    require(len(score) == len(meta) == len(bits) == 244328 and np.isfinite(score).all(),
            "Unexpected rows or non-finite frozen-G predictions.")

    previous: set[str] = set()
    for round_name, outcomes in json.loads((LOCAL / "reports/official_results.json").read_text()).items():
        if not round_name.startswith("round"):
            continue
        archive = LOCAL / "submissions" / f"{round_name}_validation_batch.zip"
        with zipfile.ZipFile(archive) as zf:
            entries = [entry for entry in zf.namelist() if entry.startswith("Team_")]
            for panel_name in outcomes:
                matches = [entry for entry in entries if entry.endswith(f"_{panel_name}.txt")]
                require(len(matches) == 1, f"Missing/duplicate archived panel: {round_name}/{panel_name}")
                previous.update(zf.read(matches[0]).decode().splitlines())
    for panel in (ROOT / "submissions/panels").glob("*/*.txt"):
        if panel.parent != OUT:
            previous.update(panel.read_text().splitlines())

    ids = meta.CatalogID.astype(str)
    valid = meta.valid.to_numpy() & meta.canonical.ne("").to_numpy()
    eligible = valid & ~ids.isin(previous).to_numpy()
    order = np.lexsort((ids.to_numpy(), -score))
    selected: list[int] = []
    parents: set[str] = set()
    scaffolds: dict[str, int] = {}
    fps = {}
    for row in order:
        if not eligible[row]:
            continue
        parent = meta.canonical.iat[row]
        scaffold = meta.scaffold.iat[row]
        if parent in parents or scaffolds.get(scaffold, 0) >= 2:
            continue
        fp = DataStructs.CreateFromBinaryText(bits[row, 0].tobytes())
        if selected:
            similarities = DataStructs.BulkTanimotoSimilarity(fp, [fps[i] for i in selected])
            if max(similarities) >= 0.65:
                continue
        selected.append(int(row))
        parents.add(parent)
        scaffolds[scaffold] = scaffolds.get(scaffold, 0) + 1
        fps[row] = fp
        if len(selected) == 50:
            break
    require(len(selected) == 50, f"Could select only {len(selected)} diverse fresh IDs.")

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "DEL_ligand_replay.txt"
    selected_ids = ids.iloc[selected].tolist()
    output.write_text("\n".join(selected_ids) + "\n")
    details = meta.iloc[selected][["CatalogID", "canonical", "scaffold", "SMILES"]].copy()
    details["exact_R03_G_score"] = score[selected]
    details["R03_known_ligand_signal"] = rankings.known_ligand_signal.to_numpy()[selected]
    details["R03_known_ligand_ECFP_max"] = rankings.known_ligand_ECFP_max.to_numpy()[selected]
    details["frozen_DEL_G_component"] = (score[selected] - .95 * details.R03_known_ligand_signal.to_numpy()) / .05
    details.to_csv(OUT / "selected_details.csv", index=False, lineterminator="\n", float_format="%.8g")

    report = {
        "status": "prepared_for_validation_submission",
        "method": "Exact replay of successful phase-one R03 G score: 95% max verified PGK2 crystal-ligand similarity + 5% frozen ECFP DEL score. Only selection filters differ: every previously selected CatalogID is excluded, with <=2 per Murcko scaffold and pairwise ECFP4 Tanimoto <0.65.",
        "candidate_universe_rows": len(meta),
        "previously_selected_unique_ids_excluded": len(previous),
        "selection_rules": {"valid_structure": True, "unique_nonchiral_parent": True,
                            "maximum_per_Murcko_scaffold": 2, "maximum_pairwise_ECFP4_Tanimoto": 0.65,
                            "rows": len(selected), "unique_CatalogIDs": len(set(selected_ids))},
        "selected_score_range": [float(np.min(score[selected])), float(np.max(score[selected]))],
        "mean_ligand_similarity_signal": float(details.R03_known_ligand_signal.mean()),
        "mean_DEL_ECFP_score": float(details.frozen_DEL_G_component.mean()),
        "file": output.relative_to(ROOT).as_posix(),
        "sha256": sha(output),
        "detail_file": (OUT / "selected_details.csv").relative_to(ROOT).as_posix(),
        "source_hashes": {
            "round03_full_rankings": sha(LOCAL / "models/deep_audit/round03_full_rankings.parquet"),
            "validation_metadata": sha(LOCAL / "features/validation_meta.parquet"),
            "validation_fingerprints": sha(LOCAL / "features/validation_bits.npy"),
            "builder": sha(Path(__file__)),
        },
        "limitation": "This probe tests whether the exact previously successful ligand-led validation score transfers to fresh validation chemistry. It does not demonstrate that a 5% DEL contribution is sufficient for the final challenge workflow, and validation feedback is adaptive.",
    }
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
