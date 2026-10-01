"""Check recovered scientific evidence without retraining or changing old reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_bags():
    results = json.loads((ROOT / 'reports/official_results.json').read_text())
    bags = []
    for round_name, outcomes in results.items():
        if not round_name.startswith('round'):
            continue
        archive = ROOT / 'submissions' / f'{round_name}_validation_batch.zip'
        with zipfile.ZipFile(archive) as handle:
            assert handle.testzip() is None, archive
            entries = [name for name in handle.namelist() if name.startswith('Team_')]
            assert len(entries) == len(outcomes), round_name
            for name, outcome in outcomes.items():
                matching = [p for p in entries if p.endswith(f'_{name}.txt')]
                assert len(matching) == 1, (round_name, name)
                lines = handle.read(matching[0]).decode().splitlines()
                assert len(lines) == len(set(lines)) == 50, matching[0]
                assert all(line and line.strip() == line for line in lines), matching[0]
                assert 0 <= outcome['hits'] <= 50, name
                extracted = ROOT / 'submissions' / round_name / matching[0]
                if extracted.exists():
                    assert extracted.read_bytes() == handle.read(matching[0]), extracted
                bags.append({'round': round_name, 'name': name,
                             'members': set(lines), 'hits': int(outcome['hits'])})
    return bags


def forced_labels(bags):
    labels = {}
    changed = True
    while changed:
        changed = False
        for bag in bags:
            positive = sum(labels.get(key) == 1 for key in bag['members'])
            unknown = sorted(bag['members'] - labels.keys())
            remaining = bag['hits'] - positive
            assert 0 <= remaining <= len(unknown), bag['name']
            if unknown and remaining in (0, len(unknown)):
                value = int(remaining == len(unknown))
                labels.update(dict.fromkeys(unknown, value))
                changed = True
    return labels


def check_test(path, expected_ids):
    seen = set()
    selected = set()
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['CatalogID', 'Sel_50', 'Score'], path
        for row in reader:
            key = row['CatalogID']
            assert key not in seen and key in expected_ids, (path, key)
            seen.add(key)
            assert row['Sel_50'] in ('0', '1'), path
            assert math.isfinite(float(row['Score'])), path
            if row['Sel_50'] == '1':
                selected.add(key)
    assert seen == expected_ids and len(selected) == 50, path
    return selected


def check_models():
    # Use the preserved local runtime, or an installed LightGBM on another machine.
    sys.path.insert(0, str(ROOT / 'vendor'))
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from rdkit import Chem, DataStructs
    from features import ECFP, FCFP, featurize
    from train import eligible, split_groups

    training = pd.read_parquet(ROOT / 'features/training_meta.parquet')
    validation = pd.read_parquet(ROOT / 'features/validation_meta.parquet')
    good, _ = eligible(training)
    held = split_groups(training)
    assert np.array_equal(good, np.load(ROOT / 'data/eligible.npy'))
    assert np.array_equal(held, np.load(ROOT / 'data/heldout.npy'))
    assert not set(training.loc[good & held, 'scaffold']) & set(training.loc[good & ~held, 'scaffold'])
    assert training.loc[good, 'canonical'].is_unique
    observed = set(training.loc[training.source.str.startswith('selection'), 'library'])
    assert set(training.loc[good & training.source.eq('library_unlabeled'), 'library']) <= observed

    assert featurize('invalid molecule') is None
    first, same, different = (featurize(s) for s in ('CCO', 'OCC', 'CCN'))
    assert first[0] == same[0] and np.array_equal(first[2], same[2])
    for index, generator in enumerate((ECFP, FCFP)):
        packed = DataStructs.TanimotoSimilarity(
            DataStructs.CreateFromBinaryText(first[2][index].tobytes()),
            DataStructs.CreateFromBinaryText(different[2][index].tobytes()))
        native = DataStructs.TanimotoSimilarity(generator.GetFingerprint(Chem.MolFromSmiles('CCO')),
                                              generator.GetFingerprint(Chem.MolFromSmiles('CCN')))
        assert abs(packed - native) < 1e-12

    sample = np.random.default_rng(42).choice(len(validation), 100, replace=False)
    for kind in ('ecfp', 'fcfp'):
        matrix = sparse.load_npz(ROOT / f'features/validation_{kind}.npz')[sample]
        for seed in (2035, 2036):
            model = lgb.Booster(model_file=str(ROOT / f'models/{kind}_{seed}.txt'))
            actual = model.predict(matrix, num_threads=2)
            expected = np.load(ROOT / f'models/validation_{kind}_{seed}.npy')[sample]
            assert np.allclose(actual, expected, rtol=1e-8, atol=1e-10), (kind, seed)
    return {'models_replayed': 4, 'sample_per_model': 100,
            'eligible_training_rows': int(good.sum()), 'scaffold_overlap': 0,
            'fingerprint_parity': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact', action='store_true', help='Check only evidence included in Git.')
    args = parser.parse_args()
    bags = read_bags()
    labels = forced_labels(bags)
    expected = json.loads((ROOT / 'reports/validation_label_inference.json').read_text())
    union = set().union(*(b['members'] for b in bags))
    summary = {'bags': len(bags), 'unique_submitted': len(union),
               'forced_positive': sum(v == 1 for v in labels.values()),
               'forced_negative': sum(v == 0 for v in labels.values()),
               'unresolved_submitted': len(union) - len(labels)}
    assert all(expected[key] == value for key, value in summary.items()), summary
    with (REPO / 'docs/recovery/validation_forced_labels.csv').open(newline='') as handle:
        exported = {row['CatalogID']: int(row['Label']) for row in csv.DictReader(handle)}
    assert exported == labels
    result = {'status': 'passed', 'validation': summary,
              'checks': ['21 original ZIP archives and 48 lists of 50 unique IDs',
                         'No contradictory aggregate counts; exported individual deductions match'],
              'mode': 'compact' if args.compact else 'full'}
    if not args.compact:
        with (REPO / 'Val-Test-set/PGK2_Validation_split.csv').open(newline='') as handle:
            val_ids = {row['CatalogID'] for row in csv.DictReader(handle)}
        assert len(val_ids) == 244328 and union <= val_ids
        with (REPO / 'Val-Test-set/PGK2_Test_split.csv').open(newline='') as handle:
            test_ids = {row['CatalogID'] for row in csv.DictReader(handle)}
        assert len(test_ids) == 184632
        t1 = ROOT / 'submissions/blind_test/Team_MMELON_T1_FROZEN_G.csv'
        t2 = ROOT / 'submissions/blind_test_revised/Team_MMELON_T2_REVISED_COUNTS.csv'
        first, second = check_test(t1, test_ids), check_test(t2, test_ids)
        assert not first & second
        for path in (t1, t2):
            verification = json.loads((path.parent / 'verification.json').read_text())
            if verification.get('file') == path.name:
                assert sha256(path) == verification['sha256']
        with (REPO / 'docs/recovery/test_selected_negatives.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 100 and {r['CatalogID'] for r in rows} == first | second
        assert all(r['Label'] == '0' for r in rows)
        frozen = ROOT / 'models/frozen_round03'
        hashes = json.loads((frozen / 'hashes.json').read_text())
        assert all(sha256(frozen / name) == digest for name, digest in hashes.items())
        inventory = json.loads((REPO / 'docs/recovery/asset_inventory.json').read_text())
        for item in inventory['files']:
            assert sha256(REPO / item['path']) == item['recovered_sha256'], item['path']
        result['checks'].extend(['Original validation and test ID universes',
                                 'Two full test CSVs, finite scores, 50 selections each, zero overlap',
                                 '100 user-reported selected test negatives',
                                 'Seven frozen round-three hashes and all copied asset hashes'])
        result['models'] = check_models()
        result['checks'].append('Scaffold separation, fingerprint parity, four saved-model prediction replays')
    output = REPO / '.local/recovery_verification.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
