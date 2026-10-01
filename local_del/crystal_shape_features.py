"""Generate alignment-free 3D shape/pharmacophore scores to crystal ligand 21."""
from __future__ import annotations

import hashlib
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdReducedGraphs


ROOT = Path(__file__).resolve().parent
QUERY = Chem.RemoveHs(Chem.SDMolSupplier(str(ROOT / "external" / "pgk2" / "compound21_verified.sdf"), removeHs=False)[0])
QUERY_USRCAT = [float(value) for value in rdMolDescriptors.GetUSRCAT(QUERY)]


def previously_submitted() -> set[str]:
    result: set[str] = set()
    for archive in sorted((ROOT / "submissions").glob("round*_validation_batch.zip")):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.startswith("Team_"):
                    result.update(zf.read(name).decode().splitlines())
    return result


def score_one(item: tuple[str, str]) -> dict:
    catalog_id, smiles = item
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"CatalogID": catalog_id, "usrcat": np.nan, "conformers": 0}
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(hashlib.sha256(catalog_id.encode()).hexdigest()[:7], 16)
    params.numThreads = 1
    params.pruneRmsThresh = 0.35
    conformers = list(AllChem.EmbedMultipleConfs(mol, numConfs=4, params=params))
    if not conformers:
        return {"CatalogID": catalog_id, "usrcat": np.nan, "conformers": 0}
    heavy = Chem.RemoveHs(mol)
    scores = []
    for conformer in range(heavy.GetNumConformers()):
        descriptor = rdMolDescriptors.GetUSRCAT(heavy, confId=conformer)
        scores.append(rdMolDescriptors.GetUSRScore(QUERY_USRCAT, descriptor))
    return {"CatalogID": catalog_id, "usrcat": float(max(scores)), "conformers": len(scores)}


def generalized_tanimoto(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    right = float(query @ query)
    result = np.empty(len(matrix), dtype=np.float32)
    for begin in range(0, len(matrix), 20_000):
        chunk = np.asarray(matrix[begin : begin + 20_000], dtype=np.float32)
        intersection = chunk @ query
        left = np.square(chunk).sum(1)
        result[begin : begin + len(chunk)] = intersection / np.maximum(left + right - intersection, 1e-12)
    return result


def main() -> None:
    metadata = pd.read_parquet(ROOT / "features" / "validation_meta.parquet")
    ranking = pd.read_parquet(ROOT / "models" / "deep_audit" / "round03_full_rankings.parquet",
                              columns=["CatalogID", "G_KNOWN_LIGANDS_score"])
    ranking = ranking.set_index("CatalogID").loc[metadata.CatalogID].reset_index()
    erg = np.load(ROOT / "features" / "validation_erg.npy", mmap_mode="r")
    query_erg = np.asarray(rdReducedGraphs.GetErGFingerprint(QUERY), dtype=np.float32)
    erg_similarity = generalized_tanimoto(erg, query_erg)
    seen = previously_submitted()
    allowed = metadata.valid & ~metadata.CatalogID.isin(seen)
    by_g = np.flatnonzero(allowed.to_numpy())[np.argsort(-ranking.G_KNOWN_LIGANDS_score.to_numpy()[allowed])[:3500]]
    by_erg = np.flatnonzero(allowed.to_numpy())[np.argsort(-erg_similarity[allowed])[:3500]]
    rows = np.unique(np.concatenate([by_g, by_erg]))
    items = list(zip(metadata.CatalogID.iloc[rows], metadata.SMILES.iloc[rows]))
    with ProcessPoolExecutor(max_workers=10) as executor:
        scores = list(executor.map(score_one, items, chunksize=16))
    result = pd.DataFrame(scores).set_index("CatalogID").loc[metadata.CatalogID.iloc[rows]].reset_index()
    result["row"] = rows
    result["known_ligand_score"] = ranking.G_KNOWN_LIGANDS_score.iloc[rows].to_numpy()
    result["erg_to_ligand21"] = erg_similarity[rows]
    result["SMILES"] = metadata.SMILES.iloc[rows].to_numpy()
    result["scaffold"] = metadata.scaffold.iloc[rows].to_numpy()
    output = ROOT / "models" / "crystal_shape_ligand21.parquet"
    result.to_parquet(output, index=False)
    report = {
        "crystal_ligand": 21,
        "shortlist": len(result),
        "successful": int(result.usrcat.notna().sum()),
        "median_conformers": float(result.conformers.median()),
        "usrcat_max": float(result.usrcat.max()),
        "usrcat_median": float(result.usrcat.median()),
        "workers": 10,
        "conformers_requested": 4,
    }
    (ROOT / "reports" / "crystal_shape_ligand21.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
