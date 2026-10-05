"""Prepare three diverse validation lists for human review and possible upload."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from scipy import sparse
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LOCAL = REPO / 'local_del'
OUT = ROOT / 'submissions/panels'
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(REPO))


def tanimoto(q, r):
    overlap = (q @ r.T).toarray().astype(np.float32, copy=False)
    qn = np.asarray(q.sum(axis=1), dtype=np.float32).reshape(-1, 1)
    rn = np.asarray(r.sum(axis=1), dtype=np.float32).reshape(1, -1)
    return overlap / np.maximum(qn + rn - overlap, 1.0)


def linear_score(model, x):
    """Evaluate the fitted logistic linear predictor with bounded row-wise sums."""
    return np.sum(x * model.coef_.reshape(1, -1), axis=1) + model.intercept_[0]


def similarity_features(train_x, train_y, query_x, leave_one_out=False):
    pos_ix = np.flatnonzero(train_y == 1)
    neg_ix = np.flatnonzero(train_y == 0)
    pos, neg = train_x[pos_ix], train_x[neg_ix]
    pos_sim = tanimoto(train_x, pos)
    neg_sim = tanimoto(train_x, neg)
    if leave_one_out:
        for row in pos_ix:
            pos_sim[row, np.flatnonzero(pos_ix == row)[0]] = 0
        for row in neg_ix:
            neg_sim[row, np.flatnonzero(neg_ix == row)[0]] = 0

    def summarise(a, b):
        a_max, b_max = a.max(axis=1), b.max(axis=1)
        a_mean = np.exp(8 * (a - 1)).sum(axis=1) / max(a.shape[1], 1)
        b_mean = np.exp(8 * (b - 1)).sum(axis=1) / max(b.shape[1], 1)
        return np.column_stack([a_max, b_max, a_max - b_max, a_mean, b_mean])

    qpos = tanimoto(query_x, pos)
    qneg = tanimoto(query_x, neg)
    return summarise(pos_sim, neg_sim), summarise(qpos, qneg)


def diverse_top(frame, score, fps, threshold=.65, max_per_scaffold=2, n=50):
    order = np.argsort(-score, kind='stable')
    selected = []
    scaffolds = {}
    seen = set()
    for i in order:
        key = frame.CatalogID.iat[i]
        scaffold = frame.scaffold.iat[i]
        if not frame.valid.iat[i] or frame.canonical.iat[i] in seen:
            continue
        if scaffolds.get(scaffold, 0) >= max_per_scaffold:
            continue
        if selected and max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in selected])) >= threshold:
            continue
        selected.append(int(i))
        seen.add(frame.canonical.iat[i])
        scaffolds[scaffold] = scaffolds.get(scaffold, 0) + 1
        if len(selected) == n:
            return selected
    raise ValueError(f'Only selected {len(selected)} of {n} diverse candidates.')


def main():
    labels = pd.read_parquet(ROOT / 'data/labeled_compounds.parquet')
    lb = np.load(ROOT / 'data/labeled_bits.npy')
    labeled_x = sparse.csr_matrix(np.unpackbits(lb[:, 0, :], axis=1).astype(np.float32))
    labeled_y = labels.Label.to_numpy(dtype=np.int8)
    train_nn, _ = similarity_features(labeled_x, labeled_y, labeled_x, leave_one_out=True)
    del_ecfp = np.load(LOCAL / 'models/validation_ecfp.npy')
    del_fcfp = np.load(LOCAL / 'models/validation_fcfp.npy')
    meta = pd.read_parquet(LOCAL / 'features/validation_meta.parquet')
    bits = np.load(LOCAL / 'features/validation_bits.npy', mmap_mode='r')
    x_val = sparse.load_npz(LOCAL / 'features/validation_ecfp.npz').tocsr()
    require(len(meta) == x_val.shape[0] == len(bits) == 244328, 'Unexpected validation universe.')
    require(np.isfinite(del_ecfp).all() and np.isfinite(del_fcfp).all(), 'Invalid DEL predictions.')
    y = labeled_y

    # DEL priors for released compounds are predictions from frozen models.
    sys.path.insert(0, str(LOCAL / 'vendor'))
    import lightgbm as lgb
    from scipy.sparse import hstack
    del_inputs = []
    for kind in ['ecfp', 'fcfp']:
        ch = 0 if kind == 'ecfp' else 1
        xx = sparse.csr_matrix(np.unpackbits(lb[:, ch, :], axis=1).astype(np.float32))
        if kind == 'fcfp':
            xx = hstack([xx, sparse.csr_matrix(np.load(ROOT / 'data/labeled_descriptors.npy'))], format='csr')
        preds = []
        modeldir = REPO / 'local_del/models'
        for seed in [2035, 2036]:
            model = lgb.Booster(model_file=str(modeldir / f'{kind}_{seed}.txt'))
            preds.append(model.predict(xx, num_threads=4))
        del_inputs.append(np.mean(preds, axis=0))
    ligand_scores = []
    ligands = json.loads((REPO / 'local_del/models/frozen_round03/verified_ligands.json').read_text())
    from local_del.features import featurize
    label_fps = [[DataStructs.CreateFromBinaryText(row[ch].tobytes()) for row in lb]
                 for ch in (0, 1)]
    for ligand in ligands:
        packed = featurize(ligand['SMILES'])[2]
        e = DataStructs.CreateFromBinaryText(packed[0].tobytes())
        f = DataStructs.CreateFromBinaryText(packed[1].tobytes())
        et = np.asarray(DataStructs.BulkTanimotoSimilarity(e, label_fps[0]), dtype=np.float32)
        ft = np.asarray(DataStructs.BulkTanimotoSimilarity(f, label_fps[1]), dtype=np.float32)
        ev = np.asarray(DataStructs.BulkTverskySimilarity(e, label_fps[0], .2, 1), dtype=np.float32)
        ligand_scores.append(.65*et + .20*ft + .15*ev)
    ligand_prior = np.max(np.stack(ligand_scores, axis=1), axis=1)

    final_train = np.column_stack([
        logit(np.clip(del_inputs[0], 1e-6, 1-1e-6)),
        logit(np.clip(del_inputs[1], 1e-6, 1-1e-6)), ligand_prior, train_nn])
    require(np.isfinite(final_train).all(), 'Non-finite adaptation training features.')
    scaler = StandardScaler().fit(final_train)
    adapted = LogisticRegression(C=.03, class_weight='balanced', max_iter=2000,
                                 solver='liblinear', random_state=2605)
    adapted.fit(scaler.transform(final_train), y)

    # Score validation rows in memory-bounded blocks against the released label set.
    pos_ix, neg_ix = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    pos_x, neg_x = labeled_x[pos_ix], labeled_x[neg_ix]
    nn_val = np.empty((len(meta), 5), dtype=np.float32)
    batch = 512
    for begin in range(0, len(meta), batch):
        q = x_val[begin:begin+batch]
        a, b = tanimoto(q, pos_x), tanimoto(q, neg_x)
        aa, bb = a.max(axis=1), b.max(axis=1)
        nn_val[begin:begin+q.shape[0]] = np.column_stack([
            aa, bb, aa-bb, np.exp(8*(a-1)).mean(axis=1), np.exp(8*(b-1)).mean(axis=1)])
    val_ligand = []
    vfps = [[DataStructs.CreateFromBinaryText(row[ch].tobytes()) for row in bits]
            for ch in (0, 1)]
    for ligand in ligands:
        packed = featurize(ligand['SMILES'])[2]
        e = DataStructs.CreateFromBinaryText(packed[0].tobytes())
        f = DataStructs.CreateFromBinaryText(packed[1].tobytes())
        et = np.asarray(DataStructs.BulkTanimotoSimilarity(e, vfps[0]), dtype=np.float32)
        ft = np.asarray(DataStructs.BulkTanimotoSimilarity(f, vfps[1]), dtype=np.float32)
        ev = np.asarray(DataStructs.BulkTverskySimilarity(e, vfps[0], .2, 1), dtype=np.float32)
        val_ligand.append(.65*et + .20*ft + .15*ev)
    val_ligand = np.max(np.stack(val_ligand, axis=1), axis=1)

    val_stack = np.empty((len(meta), 8), dtype=np.float32)
    val_stack[:, 0] = logit(np.clip(del_ecfp, 1e-6, 1-1e-6))
    val_stack[:, 1] = logit(np.clip(del_fcfp, 1e-6, 1-1e-6))
    val_stack[:, 2] = val_ligand
    val_stack[:, 3:] = nn_val
    require(np.isfinite(val_stack).all(), 'Non-finite validation adaptation features.')
    val_scaled = scaler.transform(val_stack)
    require(np.isfinite(val_scaled).all() and np.max(np.abs(val_scaled)) < 1e6,
            'Adaptation standardization produced extreme or invalid values.')
    adapted_score = linear_score(adapted, val_scaled)
    labels_model = LogisticRegression(C=.03, class_weight='balanced', max_iter=2000,
                                      solver='liblinear', random_state=2605)
    labels_scaler = StandardScaler().fit(train_nn)
    labels_model.fit(labels_scaler.transform(train_nn), y)
    labels_only_score = linear_score(labels_model, labels_scaler.transform(nn_val))
    require(np.isfinite(adapted_score).all() and np.isfinite(labels_only_score).all(),
            'Non-finite validation ranking scores.')
    ecfp_rank = 1 - (rankdata(-del_ecfp, method='average') - 1) / len(meta)
    fcfp_rank = 1 - (rankdata(-del_fcfp, method='average') - 1) / len(meta)
    del_rank = (ecfp_rank + fcfp_rank) / 2
    adapted_rank = 1 - (rankdata(-adapted_score, method='average') - 1) / len(meta)
    neighbor_rank = 1 - (rankdata(-labels_only_score, method='average') - 1) / len(meta)
    panels = {
        'DEL_support': del_rank,
        'Experimental_adaptation': adapted_rank,
        'Model_disagreement': .55*adapted_rank + .45*(adapted_rank-del_rank+1)/2,
    }
    require(all(np.isfinite(score).all() for score in panels.values()),
            'A panel ranking contains non-finite values.')
    fps = [DataStructs.CreateFromBinaryText(row[0].tobytes()) for row in bits]
    OUT.mkdir(parents=True, exist_ok=True)
    selected_sets = {}
    report = {'status': 'prepared_for_review', 'validation_rows': len(meta),
              'label_rows_used_for_adaptation': len(labels), 'positive_labels': int(y.sum()),
              'validation_universe_sha256': hashlib.sha256('\n'.join(meta.CatalogID.astype(str)).encode()).hexdigest(),
              'rules': {'scaffold_cap': 2, 'max_pairwise_ECFP4_Tanimoto': .65,
                        'one_per_nonchiral_parent': True, 'file_format': '50 CatalogIDs, one per line; no header'},
              'source_hashes': {
                  'active_learning_labels': hashlib.sha256((ROOT/'data/raw/active_learning_data.csv').read_bytes()).hexdigest(),
                  'validation_catalog_ids': hashlib.sha256('\n'.join(meta.CatalogID.astype(str)).encode()).hexdigest(),
                  'validation_ECFP_model_scores': hashlib.sha256((LOCAL/'models/validation_ecfp.npy').read_bytes()).hexdigest(),
                  'validation_FCFP_model_scores': hashlib.sha256((LOCAL/'models/validation_fcfp.npy').read_bytes()).hexdigest(),
                  'labeled_feature_receipt': hashlib.sha256((ROOT/'reports/labeled_features.json').read_bytes()).hexdigest(),
                  'diagnosis_receipt': hashlib.sha256((ROOT/'reports/release_diagnosis.json').read_bytes()).hexdigest(),
                  'panel_builder': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'ranking_methods': {
                  'DEL_support': 'Mean of descending ECFP-DEL and FCFP-DEL rank fractions.',
                  'Experimental_adaptation': 'Full-release regularized logistic model using ECFP/FCFP DEL logits, known-ligand similarity, and label-neighborhood features.',
                  'Model_disagreement': '55% adaptation rank plus 45% rank-disagreement term; favors adaptation-high/DEL-low compounds.'},
              'DEL_key_component_check': 'Offline grouped CV did not improve held-out PR area or average top-50 recovery over the label-neighborhood-only comparator; do not present this adapter alone as satisfying the significant-DEL requirement.',
              'panel_model_numerics': {
                  'adaptation_max_abs_standardized_validation_feature': float(np.max(np.abs(val_scaled))),
                  'adaptation_max_abs_coefficient': float(np.max(np.abs(adapted.coef_))),
                  'adaptation_scores_finite': bool(np.isfinite(adapted_score).all()),
                  'labels_only_scores_finite': True},
              'panels': {}, 'fold_cv_context': json.loads((ROOT/'reports/release_diagnosis.json').read_text())['grouped_cv_fold_mean'],
              'limitations': ['Validation feedback is adaptive and will be declared in the writeup.',
                              'The validation sample is not independent of historical phase-one tuning.',
                              'This generated report records offline panel preparation; live Synapse submissions and scores are tracked separately in status.json.']}
    # Compare rankings with only the validation labels logically forced by the
    # previous aggregate hit-count feedback. These eight positives are an
    # adaptive, highly selected diagnostic, not an independent performance set.
    forced = pd.read_parquet(LOCAL / 'reports/validation_forced_labels.parquet')
    meta_index = pd.Index(meta.CatalogID.astype(str))
    forced_ix = meta_index.get_indexer(forced.CatalogID.astype(str))
    require((forced_ix >= 0).all(), 'A forced validation label is missing from the scoring universe.')
    forced_y = forced.forced_label.to_numpy(dtype=np.int8)
    forced_pos = forced_ix[forced_y == 1]
    forced_neg = forced_ix[forced_y == 0]
    forced_rankings = {
        'DEL_ECFP': ecfp_rank,
        'DEL_FCFP': fcfp_rank,
        'DEL_mean': del_rank,
        'released_label_adaptation': adapted_rank,
        'released_label_neighborhood_only': neighbor_rank,
    }
    report['historically_forced_validation_labels'] = {
        'rows': int(len(forced)), 'positives': int(len(forced_pos)), 'negatives': int(len(forced_neg)),
        'warning': 'Adaptive labels logically inferred from previous aggregate submission feedback; diagnostic only, not an independent estimate.',
        'rank_diagnostics': {
            name: {
                'positive_rank_fractions_best_to_worst': sorted(map(float, scores[forced_pos]), reverse=True),
                'known_positives_in_top_50': int(np.sum(scores[forced_pos] >= 1 - 49 / len(meta))),
                'known_positives_in_top_500': int(np.sum(scores[forced_pos] >= 1 - 499 / len(meta))),
                'best_negative_rank_fraction': float(np.max(scores[forced_neg])),
                'median_negative_rank_fraction': float(np.median(scores[forced_neg])),
                'known_negatives_in_top_50': int(np.sum(scores[forced_neg] >= 1 - 49 / len(meta))),
            } for name, scores in forced_rankings.items()
        }
    }
    # Re-score every archived phase-one validation list under the current
    # rankings. This tests whether phase-two rankings recover earlier selected
    # chemistries; historical bag hit counts remain aggregate outcomes.
    phase1 = json.loads((LOCAL / 'reports/official_results.json').read_text())
    historical_rank_audit = {}
    meta_ids = pd.Index(meta.CatalogID.astype(str))
    for round_name, outcomes in phase1.items():
        if not round_name.startswith('round'):
            continue
        archive = LOCAL / 'submissions' / f'{round_name}_validation_batch.zip'
        with zipfile.ZipFile(archive) as zf:
            entries = [entry for entry in zf.namelist() if entry.startswith('Team_')]
            for panel_name, outcome in outcomes.items():
                matches = [entry for entry in entries if entry.endswith(f'_{panel_name}.txt')]
                require(len(matches) == 1, f'Missing or duplicate historical panel: {round_name}/{panel_name}')
                ids = zf.read(matches[0]).decode().splitlines()
                ix = meta_ids.get_indexer(ids)
                require((ix >= 0).all() and len(ids) == len(set(ids)) == 50,
                        f'Invalid archived validation IDs: {round_name}/{panel_name}')
                historical_rank_audit[f'{round_name}/{panel_name}'] = {
                    'N_hits': int(outcome['hits']),
                    'panel_rows_in_global_top_50': {
                        method: int(np.sum(scores[ix] >= 1 - 49 / len(meta)))
                        for method, scores in {
                            'DEL_mean': del_rank,
                            'released_label_adaptation': adapted_rank,
                            'released_label_neighborhood_only': neighbor_rank,
                        }.items()
                    },
                    'median_panel_rank_fraction': {
                        method: float(np.median(scores[ix]))
                        for method, scores in {
                            'DEL_mean': del_rank,
                            'released_label_adaptation': adapted_rank,
                            'released_label_neighborhood_only': neighbor_rank,
                        }.items()
                    },
                }
    report['historical_panel_rank_audit'] = {
        'method': 'Each archived phase-one panel ID was looked up in the validation universe and ranked using the current DEL, released-label adaptation, and released-label-neighborhood scores. N_hits is retained as an aggregate panel result; no member of a partially successful panel is assigned an individual label.',
        'panel_count': len(historical_rank_audit),
        'panels': historical_rank_audit,
    }
    for name, score in panels.items():
        ids = diverse_top(meta, score, fps)
        selected_sets[name] = set(meta.CatalogID.iloc[ids].astype(str))
        outdir = OUT / name
        outdir.mkdir(exist_ok=True)
        submission = outdir / f'{name}.txt'
        submission.write_text('\n'.join(meta.CatalogID.iloc[ids].astype(str))+'\n')
        details = meta.iloc[ids][['CatalogID', 'canonical', 'scaffold', 'SMILES']].copy()
        details['panel_score'] = score[ids]
        details['DEL_ECFP_rank_fraction'] = ecfp_rank[ids]
        details['DEL_FCFP_rank_fraction'] = fcfp_rank[ids]
        details['adapted_rank_fraction'] = adapted_rank[ids]
        details['label_neighbor_rank_fraction'] = neighbor_rank[ids]
        details.to_csv(outdir / 'selected_details.csv', index=False, lineterminator='\n', float_format='%.8g')
        require(len(ids) == 50 and len(set(submission.read_text().splitlines())) == 50,
                f'{name} submission should contain 50 unique IDs.')
        max_pairwise = max(max(DataStructs.BulkTanimotoSimilarity(
            fps[i], [fps[j] for j in ids if j != i])) for i in ids)
        require(max_pairwise < .65, f'{name} panel violated the ECFP diversity cutoff.')
        report['panels'][name] = {
            'rows': len(ids), 'unique_scaffolds': int(meta.scaffold.iloc[ids].nunique()),
            'median_DEL_rank_fraction': float(np.median(del_rank[ids])),
            'median_adapted_rank_fraction': float(np.median(adapted_rank[ids])),
            'maximum_pairwise_ECFP4_Tanimoto': float(max_pairwise),
            'submission_sha256': hashlib.sha256(submission.read_bytes()).hexdigest(),
            'submission_file': str(submission.relative_to(ROOT)),
        }
        with zipfile.ZipFile(outdir / f'{name}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(submission, submission.name)
        report['panels'][name]['overlap_with_other_panels'] = {}
    for a in selected_sets:
        for b in selected_sets:
            if a != b:
                report['panels'][a]['overlap_with_other_panels'][b] = len(selected_sets[a]&selected_sets[b])
    bundle = OUT / 'active_learning_validation_panels.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in panels:
            path = OUT/name/f'{name}.txt'
            z.write(path, f'{name}.txt')
        z.writestr('README.txt',
                   'Three proposed 50-ID validation panels for local review.\n'
                   'DEL_support prioritizes DEL model scores.\n'
                   'Experimental_adaptation uses the released experimental labels plus DEL features.\n'
                   'Model_disagreement prioritizes adapted scores that disagree with DEL.\n'
                   'No files in this archive have been uploaded or submitted.\n')
    report['bundle'] = {'file': str(bundle.relative_to(ROOT)),
                        'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}
    (ROOT / 'reports/validation_panels.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def require(ok, message):
    if not ok:
        raise ValueError(message)


if __name__ == '__main__':
    main()
