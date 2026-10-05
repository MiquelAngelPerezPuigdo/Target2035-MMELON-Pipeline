"""Diagnose phase-one score transfer and grouped adaptation on released labels."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from rdkit.ML.Cluster import Butina
from scipy import sparse
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL_DIR = REPO / 'local_del/models/frozen_round03'
TRAINED_MODEL_DIR = REPO / 'local_del/models'
sys.path.insert(0, str(REPO / 'local_del/vendor'))
import lightgbm as lgb

RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(REPO))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint_matrix(bits, channel):
    return sparse.csr_matrix(np.unpackbits(bits[:, channel, :], axis=1).astype(np.float32))


def predict_frozen(bits, descriptors, channel, seeds):
    x = fingerprint_matrix(bits, channel)
    if channel == 1:
        x = sparse.hstack([x, sparse.csr_matrix(descriptors)], format='csr')
    predictions = []
    for seed in seeds:
        directory = MODEL_DIR if channel == 0 else TRAINED_MODEL_DIR
        path = directory / f"{'ecfp' if channel == 0 else 'fcfp'}_{seed}.txt"
        model = lgb.Booster(model_file=str(path))
        predictions.append(model.predict(x, num_threads=4))
    return np.mean(predictions, axis=0)


def ligand_scores(bits):
    ligands = json.loads((MODEL_DIR / 'verified_ligands.json').read_text())
    fps = [[DataStructs.CreateFromBinaryText(row[ch].tobytes()) for row in bits]
           for ch in (0, 1)]
    signals = []
    for ligand in ligands:
        from local_del.features import featurize
        packed = featurize(ligand['SMILES'])[2]
        qecfp = DataStructs.CreateFromBinaryText(packed[0].tobytes())
        qfcfp = DataStructs.CreateFromBinaryText(packed[1].tobytes())
        ec_tan = np.asarray(DataStructs.BulkTanimotoSimilarity(qecfp, fps[0]), dtype=np.float32)
        fc_tan = np.asarray(DataStructs.BulkTanimotoSimilarity(qfcfp, fps[1]), dtype=np.float32)
        ec_tversky = np.asarray(DataStructs.BulkTverskySimilarity(qecfp, fps[0], .2, 1), dtype=np.float32)
        signals.append(.65 * ec_tan + .20 * fc_tan + .15 * ec_tversky)
    return np.max(np.stack(signals, axis=1), axis=1), ligands


def tanimoto_block(q, r):
    overlap = (q @ r.T).toarray().astype(np.float32, copy=False)
    qn = np.asarray(q.sum(axis=1), dtype=np.float32).reshape(-1, 1)
    rn = np.asarray(r.sum(axis=1), dtype=np.float32).reshape(1, -1)
    return overlap / np.maximum(qn + rn - overlap, 1.0)


def grouped_similarity_features(train, query):
    """Max ECFP similarity to each class; exclude a training row from itself."""
    all_train = np.asarray(train['y']) == 1
    idx_pos = np.flatnonzero(all_train)
    idx_neg = np.flatnonzero(~all_train)
    ref = train['x']
    pos = ref[idx_pos]
    neg = ref[idx_neg]

    qsim_pos = tanimoto_block(query['x'], pos)
    qsim_neg = tanimoto_block(query['x'], neg)
    qpos = qsim_pos.max(axis=1)
    qneg = qsim_neg.max(axis=1)
    qposmean = np.exp(8 * (qsim_pos - 1)).sum(axis=1) / max(pos.shape[0], 1)
    qnegmean = np.exp(8 * (qsim_neg - 1)).sum(axis=1) / max(neg.shape[0], 1)
    out = np.column_stack([qpos, qneg, qpos - qneg, qposmean, qnegmean])

    # Training features are leave-one-out so exact self matches cannot inflate them.
    tsim_pos = tanimoto_block(ref, pos)
    tsim_neg = tanimoto_block(ref, neg)
    for row, label in enumerate(train['y']):
        if label:
            col = int(np.flatnonzero(idx_pos == row)[0])
            tsim_pos[row, col] = 0
        else:
            col = int(np.flatnonzero(idx_neg == row)[0])
            tsim_neg[row, col] = 0
    tpos = tsim_pos.max(axis=1)
    tneg = tsim_neg.max(axis=1)
    tposmean = np.exp(8 * (tsim_pos - 1)).sum(axis=1) / max(pos.shape[0] - 1, 1)
    tnegmean = np.exp(8 * (tsim_neg - 1)).sum(axis=1) / max(neg.shape[0] - 1, 1)
    train_features = np.column_stack([tpos, tneg, tpos - tneg, tposmean, tnegmean])
    return train_features, out


def summary(y, score):
    order = np.argsort(-score, kind='stable')
    precision, recall, _ = precision_recall_curve(y, score)
    return {
        'roc_auc': float(roc_auc_score(y, score)),
        'average_precision': float(average_precision_score(y, score)),
        'trapezoidal_pr_auc': float(auc(recall, precision)),
        'positives_in_top_50': int(y[order[:50]].sum()),
        'positives_in_top_100': int(y[order[:100]].sum()),
        'positives_in_top_250': int(y[order[:250]].sum()),
        'positives_in_top_500': int(y[order[:500]].sum()),
        'recall_at_50': float(y[order[:50]].sum() / y.sum()),
        'precision_at_50': float(y[order[:50]].mean()),
    }


def linear_score(model, x):
    """Evaluate the fitted logistic linear predictor with bounded row-wise sums."""
    return np.sum(x * model.coef_.reshape(1, -1), axis=1) + model.intercept_[0]


def build_groups(bits, threshold=.65):
    fps = [DataStructs.CreateFromBinaryText(row[0].tobytes()) for row in bits]
    distances = []
    for i, fp in enumerate(fps[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1:])
        distances.extend(1.0 - score for score in similarities)
    clusters = Butina.ClusterData(distances, len(fps), 1.0 - threshold,
                                  isDistData=True, reordering=True)
    groups = np.empty(len(fps), dtype=np.int32)
    for group, members in enumerate(clusters):
        groups[list(members)] = group
    return groups, len(clusters)


def main():
    start = time.time()
    table = pd.read_parquet(ROOT / 'data/labeled_compounds.parquet')
    bits = np.load(ROOT / 'data/labeled_bits.npy', mmap_mode='r')
    descriptors = np.load(ROOT / 'data/labeled_descriptors.npy', mmap_mode='r')
    y = table.Label.to_numpy(dtype=np.int8)
    require(len(table) == 6580 and int(y.sum()) == 39, 'Unexpected release contents.')
    require(table.EligibleForInitialFit.all(), 'Expected all released labels to be eligible.')

    ecfp = predict_frozen(bits, descriptors, 0, [2035, 2036])
    fcfp = predict_frozen(bits, descriptors, 1, [2035, 2036])
    ligand, ligands = ligand_scores(bits)
    g_score = .95 * ligand + .05 * ecfp
    matrix = fingerprint_matrix(bits, 0)
    groups, group_n = build_groups(bits)

    data = {'x': matrix, 'y': y}
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2605)
    oof = {key: np.full(len(table), np.nan, dtype=np.float32)
           for key in ['labels_only', 'del_adapted']}
    fold_receipts = []
    del_features = np.column_stack([logit(np.clip(ecfp, 1e-6, 1 - 1e-6)),
                                    logit(np.clip(fcfp, 1e-6, 1 - 1e-6)), ligand])
    for fold, (tr, te) in enumerate(folds.split(np.zeros(len(y)), y, groups), 1):
        require(y[tr].sum() > 0 and y[te].sum() > 0,
                f'Fold {fold} has no positives in train or holdout.')
        train = {'x': matrix[tr], 'y': y[tr]}
        query = {'x': matrix[te]}
        train_nn, test_nn = grouped_similarity_features(train, query)
        scaler_nn = StandardScaler().fit(train_nn)
        labels_model = LogisticRegression(C=.03, class_weight='balanced', max_iter=2000,
                                          solver='liblinear', random_state=fold)
        labels_model.fit(scaler_nn.transform(train_nn), y[tr])
        oof['labels_only'][te] = linear_score(labels_model, scaler_nn.transform(test_nn))

        stack_train = np.column_stack([del_features[tr], train_nn])
        stack_test = np.column_stack([del_features[te], test_nn])
        scaler = StandardScaler().fit(stack_train)
        model = LogisticRegression(C=.03, class_weight='balanced', max_iter=2000,
                                   solver='liblinear', random_state=fold)
        model.fit(scaler.transform(stack_train), y[tr])
        oof['del_adapted'][te] = linear_score(model, scaler.transform(stack_test))
        fold_receipts.append({
            'fold': fold, 'train_rows': int(len(tr)), 'holdout_rows': int(len(te)),
            'train_positives': int(y[tr].sum()), 'holdout_positives': int(y[te].sum()),
            'holdout_group_count': int(len(np.unique(groups[te]))),
        })
        fixed = {
            'frozen_DEL_ECFP': ecfp[te], 'frozen_DEL_FCFP': fcfp[te],
            'historical_G_ligand_plus_DEL': g_score[te],
            'release_positive_neighbor': oof['labels_only'][te],
            'DEL_plus_release_adaptation': oof['del_adapted'][te],
        }
        fold_receipts[-1]['metrics'] = {
            name: summary(y[te], score) for name, score in fixed.items()}

    require(all(np.isfinite(a).all() for a in oof.values()), 'Missing grouped OOF scores.')
    score_methods = {
        'frozen_DEL_ECFP': ecfp,
        'frozen_DEL_FCFP': fcfp,
        'historical_G_ligand_plus_DEL': g_score,
        'release_positive_neighbor': oof['labels_only'],
        'DEL_plus_release_adaptation': oof['del_adapted'],
    }
    metrics = {name: summary(y, score) for name, score in score_methods.items()}
    metrics['random_selected_sample_reference'] = {
        'binder_fraction': float(y.mean()), 'interpretation': 'Observed participant-selected release fraction; not remaining-pool prevalence.'}

    # Ranked individual positives are useful for interpreting where transfer succeeds.
    pos_ix, neg_ix = np.flatnonzero(y), np.flatnonzero(y == 0)
    pos_similarity = tanimoto_block(matrix[pos_ix], matrix[pos_ix])
    np.fill_diagonal(pos_similarity, 0)
    neg_similarity = tanimoto_block(matrix[pos_ix], matrix[neg_ix]).max(axis=1)
    nearest_pos = pos_similarity.max(axis=1)
    positive_rows = []
    for name, score in score_methods.items():
        ranks = pd.Series(score).rank(method='min', ascending=False).astype(int)
        for pos_number, i in enumerate(pos_ix):
            positive_rows.append({
                'Method': name, 'CatalogID': table.CatalogID.iloc[i],
                'rank': int(ranks.iloc[i]), 'score': float(score[i]),
                'nearest_other_positive_similarity': float(nearest_pos[pos_number]),
                'nearest_negative_similarity': float(neg_similarity[pos_number]),
                'structure_status': table.StructureStatus.iloc[i],
            })
    positive_frame = pd.DataFrame(positive_rows)
    positive_frame.to_csv(ROOT / 'reports/released_positive_ranks.csv', index=False,
                          lineterminator='\n', float_format='%.8g')
    pd.DataFrame(fold_receipts).to_csv(ROOT / 'reports/grouped_folds.csv', index=False,
                                       lineterminator='\n')

    oof_frame = table[['CatalogID', 'Label', 'ParentNonChiral', 'MurckoScaffold', 'StructureStatus']].copy()
    oof_frame['ECFP4_cluster'] = groups
    oof_frame['frozen_DEL_ECFP'] = ecfp
    oof_frame['frozen_DEL_FCFP'] = fcfp
    oof_frame['historical_G'] = g_score
    oof_frame['release_neighbor_oof'] = oof['labels_only']
    oof_frame['DEL_adaptation_oof'] = oof['del_adapted']
    oof_frame.to_parquet(ROOT / 'data/release_oof_scores.parquet', index=False)

    top50 = {}
    for name, score in score_methods.items():
        order = np.argsort(-score, kind='stable')[:50]
        top50[name] = table.iloc[order].CatalogID.tolist()
    pairwise = {}
    for a in top50:
        for b in top50:
            if a < b:
                pairwise[f'{a}__{b}'] = len(set(top50[a]) & set(top50[b]))

    receipt = {
        'status': 'passed', 'as_of': '2026-10-05', 'rows': len(table),
        'positives': int(y.sum()), 'negatives': int((y == 0).sum()),
        'positive_fraction_selected_release': float(y.mean()),
        'leakage_grouping': {'method': 'Butina ECFP4, Tanimoto similarity threshold >= 0.65',
                             'groups': group_n, 'folds': fold_receipts,
                             'splitter': 'StratifiedGroupKFold, 5 folds, seed 2605'},
        'full_release_ranking_diagnostic': metrics,
        'grouped_cv_fold_mean': {
            name: {key: float(np.mean([fold['metrics'][name][key] for fold in fold_receipts]))
                  for key in ['roc_auc', 'average_precision', 'trapezoidal_pr_auc',
                              'positives_in_top_50', 'precision_at_50']}
            for name in score_methods},
        'grouped_cv_total_hits_in_each_fold_top_50_of_250': {
            name: int(sum(fold['metrics'][name]['positives_in_top_50'] for fold in fold_receipts))
            for name in score_methods},
        'top_50_method_overlap': pairwise,
        'model_definition': {
            'frozen_DEL_ECFP': 'Mean of phase-one ECFP DEL LightGBM seeds 2035 and 2036.',
            'frozen_DEL_FCFP': 'Mean of phase-one FCFP DEL LightGBM seeds 2035 and 2036.',
            'historical_G_ligand_plus_DEL': '95% maximum verified PGK2 ligand similarity (0.65 ECFP Tanimoto + 0.20 FCFP Tanimoto + 0.15 ECFP Tversky) and 5% frozen ECFP DEL score.',
            'release_positive_neighbor': 'Five-fold grouped OOF logistic model, C=0.03, on leave-one-out max/mean ECFP similarities to released binders and non-binders.',
            'DEL_plus_release_adaptation': 'Five-fold grouped OOF logistic model, C=0.03, using frozen ECFP/FCFP logits, ligand prior and leave-one-out positive/negative ECFP neighborhood summaries.'},
        'structure_sensitivity': {
            'constitutional_discrepancy_count': int(table.StructureStatus.eq('constitution_differs').sum()),
            'all_eight_are_label_zero': bool(table.loc[table.StructureStatus.eq('constitution_differs'), 'Label'].eq(0).all()),
            'released_structures_used_for_features': True,
            'score_evaluated_without_these_rows': {name: summary(y[table.StructureStatus.ne('constitution_differs')],
                                                                  score[table.StructureStatus.ne('constitution_differs')])
                                                    for name, score in score_methods.items()}},
        'interpretation_limits': [
            'The released set was selected from participant top-50 lists; its binder fraction and metrics are not estimates for the remaining test pool.',
            'Organizer chemical-series labels are unavailable. ECFP clusters are an evaluation leakage guard, not official series labels.',
            'Grouped OOF scores are internal comparisons on a small, selected release; do not use alone to spend a blind-test submission.',
            'Validation and test submissions were not made; paid compute was not used.'],
        'source_hashes': {
            'labels': hashlib.sha256((ROOT / 'data/raw/active_learning_data.csv').read_bytes()).hexdigest(),
            'fresh_feature_receipt': hashlib.sha256((ROOT / 'reports/labeled_features.json').read_bytes()).hexdigest(),
            'ECFP_models': {str(seed): hashlib.sha256((MODEL_DIR / f'ecfp_{seed}.txt').read_bytes()).hexdigest()
                            for seed in (2035, 2036)},
            'FCFP_models': {str(seed): hashlib.sha256((TRAINED_MODEL_DIR / f'fcfp_{seed}.txt').read_bytes()).hexdigest()
                            for seed in (2035, 2036)}},
        'seconds': round(time.time() - start, 2),
    }
    (ROOT / 'reports/release_diagnosis.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
