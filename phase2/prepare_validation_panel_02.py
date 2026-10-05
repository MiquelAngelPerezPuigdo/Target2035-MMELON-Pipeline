"""Prepare a fresh, ligand/DEL consensus validation panel for review."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LOCAL = REPO / 'local_del'
OUT = ROOT / 'submissions/panels/DEL_ligand_consensus'


def main() -> None:
    meta = pd.read_parquet(LOCAL / 'features/validation_meta.parquet')
    bits = np.load(LOCAL / 'features/validation_bits.npy', mmap_mode='r')
    ecfp = np.load(LOCAL / 'models/validation_ecfp.npy')
    fcfp = np.load(LOCAL / 'models/validation_fcfp.npy')
    anchors = np.load(LOCAL / 'models/deep_audit/known_ligand_similarity.npy')
    require(len(meta) == len(ecfp) == len(fcfp) == len(anchors) == 244328,
            'Unexpected validation prediction universe.')
    require(np.isfinite(ecfp).all() and np.isfinite(fcfp).all() and np.isfinite(anchors).all(),
            'Non-finite model or ligand scores.')

    # Reproduce the successful R03 G ligand prior, then combine ranks so the
    # frozen DEL component has an explicit, substantial 40% contribution.
    anchor = np.max(.65 * anchors[:, :, 0] + .20 * anchors[:, :, 1]
                    + .15 * anchors[:, :, 2], axis=1)
    g_score = .95 * anchor + .05 * ecfp
    g_rank = 1 - (rankdata(-g_score, method='average') - 1) / len(meta)
    ecfp_rank = 1 - (rankdata(-ecfp, method='average') - 1) / len(meta)
    fcfp_rank = 1 - (rankdata(-fcfp, method='average') - 1) / len(meta)
    del_rank = (ecfp_rank + fcfp_rank) / 2
    score = .60 * g_rank + .40 * del_rank

    previous: set[str] = set()
    for round_name, outcomes in json.loads((LOCAL / 'reports/official_results.json').read_text()).items():
        if not round_name.startswith('round'):
            continue
        archive = LOCAL / 'submissions' / f'{round_name}_validation_batch.zip'
        with zipfile.ZipFile(archive) as zf:
            entries = [entry for entry in zf.namelist() if entry.startswith('Team_')]
            for panel_name in outcomes:
                matches = [entry for entry in entries if entry.endswith(f'_{panel_name}.txt')]
                require(len(matches) == 1, f'Missing or duplicate archived panel: {round_name}/{panel_name}')
                previous.update(zf.read(matches[0]).decode().splitlines())
    for panel_file in (ROOT / 'submissions/panels').glob('*/*.txt'):
        if panel_file.parent != OUT:
            previous.update(panel_file.read_text().splitlines())

    ids = meta.CatalogID.astype(str)
    eligible = meta.valid.to_numpy() & meta.canonical.ne('').to_numpy() & ~ids.isin(previous).to_numpy()
    order = np.argsort(-score, kind='stable')
    fps = [DataStructs.CreateFromBinaryText(bits[i, 0].tobytes()) for i in range(len(meta))]
    selected: list[int] = []
    scaffolds: dict[str, int] = {}
    seen_parents: set[str] = set()
    for i in order:
        if not eligible[i]:
            continue
        canonical = meta.canonical.iat[i]
        scaffold = meta.scaffold.iat[i]
        if canonical in seen_parents or scaffolds.get(scaffold, 0) >= 2:
            continue
        if selected and max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in selected])) >= .65:
            continue
        selected.append(int(i))
        seen_parents.add(canonical)
        scaffolds[scaffold] = scaffolds.get(scaffold, 0) + 1
        if len(selected) == 50:
            break
    require(len(selected) == 50, f'Only {len(selected)} eligible diverse candidates found.')

    OUT.mkdir(parents=True, exist_ok=True)
    submission = OUT / 'DEL_ligand_consensus.txt'
    selected_ids = ids.iloc[selected].tolist()
    submission.write_text('\n'.join(selected_ids) + '\n')
    details = meta.iloc[selected][['CatalogID', 'canonical', 'scaffold', 'SMILES']].copy()
    details['consensus_rank_score'] = score[selected]
    details['historical_G_rank_fraction'] = g_rank[selected]
    details['DEL_ECFP_rank_fraction'] = ecfp_rank[selected]
    details['DEL_FCFP_rank_fraction'] = fcfp_rank[selected]
    details.to_csv(OUT / 'selected_details.csv', index=False, lineterminator='\n', float_format='%.8g')

    forced = pd.read_parquet(LOCAL / 'reports/validation_forced_labels.parquet')
    forced_ix = pd.Index(ids).get_indexer(forced.CatalogID.astype(str))
    require((forced_ix >= 0).all(), 'A forced label is outside the validation universe.')
    forced_y = forced.forced_label.to_numpy(dtype=np.int8)
    forced_scores = score[forced_ix]
    report = {
        'status': 'prepared_for_review_not_submitted',
        'method': '60% historical G rank (95% public-ligand prior + 5% ECFP-DEL raw score) and 40% mean ECFP/FCFP DEL rank; rank fusion avoids mixing score scales.',
        'candidate_universe_rows': len(meta),
        'previously_evaluated_unique_ids_excluded': len(previous),
        'selection_rules': {'valid_structure': True, 'unique_nonchiral_parent': True,
                            'maximum_per_Murcko_scaffold': 2, 'maximum_pairwise_ECFP4_Tanimoto': .65,
                            'rows': len(selected), 'unique_CatalogIDs': len(set(selected_ids))},
        'historical_forced_label_diagnostic': {
            'warning': 'The forced labels are adaptive and not an independent evaluation set.',
            'forced_positives_in_selected_panel': int(np.sum((forced_y == 1) & np.isin(forced_ix, selected))),
            'forced_negatives_in_selected_panel': int(np.sum((forced_y == 0) & np.isin(forced_ix, selected))),
            'forced_positives_in_global_top_50': int(np.sum((forced_y == 1) & (forced_scores >= 1 - 49 / len(meta)))),
            'forced_positives_in_global_top_500': int(np.sum((forced_y == 1) & (forced_scores >= 1 - 499 / len(meta)))),
        },
        'file': submission.relative_to(ROOT).as_posix(),
        'submission_sha256': hashlib.sha256(submission.read_bytes()).hexdigest(),
        'detail_file': (OUT / 'selected_details.csv').relative_to(ROOT).as_posix(),
        'source_hashes': {
            'validation_catalog_ids': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            'validation_ECFP_scores': hashlib.sha256((LOCAL / 'models/validation_ecfp.npy').read_bytes()).hexdigest(),
            'validation_FCFP_scores': hashlib.sha256((LOCAL / 'models/validation_fcfp.npy').read_bytes()).hexdigest(),
            'known_ligand_similarity': hashlib.sha256((LOCAL / 'models/deep_audit/known_ligand_similarity.npy').read_bytes()).hexdigest(),
            'builder': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


if __name__ == '__main__':
    main()
