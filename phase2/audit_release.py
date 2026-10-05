"""Ingest the supplied Active Learning release and reconcile it with phase one."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RDLogger.DisableLog('rdApp.*')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def structure(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    require(molecule is not None, f'Invalid SMILES: {smiles}')
    parent = max(Chem.GetMolFrags(molecule, asMols=True), key=lambda item: item.GetNumHeavyAtoms())
    return (Chem.MolToSmiles(molecule, isomericSmiles=True),
            Chem.MolToSmiles(parent, isomericSmiles=False),
            MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False) or '__ACYCLIC__')


def preserve_input(source, name):
    destination = ROOT / 'data/raw' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha256(destination) == sha256(source), f'Existing raw snapshot differs: {destination}')
    elif destination.resolve() != source.resolve():
        shutil.copy2(source, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/raw/active_learning_data.csv')
    parser.add_argument('--template', type=Path, default=ROOT / 'data/raw/active_learning_submission_example.csv')
    parser.add_argument('--as-of', required=True, help='User-facing audit date, YYYY-MM-DD.')
    args = parser.parse_args()
    labels = pd.read_csv(args.labels, dtype={'CatalogID': str})
    template = pd.read_csv(args.template, dtype={'CatalogID': str})
    original = pd.read_csv(REPO / 'Val-Test-set/PGK2_Test_split.csv', dtype={'CatalogID': str})
    validation = pd.read_csv(REPO / 'Val-Test-set/PGK2_Validation_split.csv', dtype={'CatalogID': str})
    historical = pd.read_csv(REPO / 'docs/recovery/test_selected_negatives.csv', dtype={'CatalogID': str})
    require(list(labels.columns) == ['CatalogID', 'Label', 'RDKit_SMILES'], 'Unexpected release schema.')
    require(list(template.columns) == ['CatalogID', 'Score', 'Sel_50'], 'Unexpected template schema.')
    for name, table in [('release', labels), ('template', template), ('original test', original)]:
        require(not table.isna().any().any(), f'Missing values in {name}.')
        require(table.CatalogID.is_unique, f'Duplicate IDs in {name}.')
    require(labels.Label.isin([0, 1]).all(), 'Labels must be binary.')
    require(template.Sel_50.isin([0, 1]).all(), 'Template flags must be binary-valued.')
    require(np.isfinite(template.Score.to_numpy()).all(), 'Non-finite template scores.')
    counts = labels.Label.value_counts().to_dict()
    require(len(labels) == 6580 and counts.get(1) == 39 and counts.get(0) == 6541,
            'Release does not match the supplied organizer announcement.')
    require(int(template.Sel_50.sum()) == 50, 'Template must have 50 example flags.')
    released, candidates, old_ids = set(labels.CatalogID), set(template.CatalogID), set(original.CatalogID)
    require(released <= old_ids, 'Released ID outside original test universe.')
    require(candidates == old_ids - released, 'Template does not match old test minus ALL released IDs.')
    require(not released & set(validation.CatalogID), 'Released labels overlap validation IDs.')

    # Joining by ID establishes the exact evidence universe. Do not reuse structure
    # features for changed constitutions. Released structures are the current
    # modeling reference; retain discrepancies for a sensitivity analysis.
    merged = labels.merge(original, on='CatalogID', validate='one_to_one').rename(columns={'SMILES': 'Original_SMILES'})
    new_structures = merged.RDKit_SMILES.map(structure)
    old_structures = merged.Original_SMILES.map(structure)
    merged['CanonicalIsomeric'] = new_structures.map(lambda item: item[0])
    merged['ParentNonChiral'] = new_structures.map(lambda item: item[1])
    merged['MurckoScaffold'] = new_structures.map(lambda item: item[2])
    merged['OriginalCanonicalIsomeric'] = old_structures.map(lambda item: item[0])
    merged['OriginalParentNonChiral'] = old_structures.map(lambda item: item[1])
    different = merged.CanonicalIsomeric.ne(merged.OriginalCanonicalIsomeric)
    changed_parent = merged.ParentNonChiral.ne(merged.OriginalParentNonChiral)
    merged['StructureStatus'] = np.where(changed_parent, 'constitution_differs',
                                        np.where(different, 'stereochemistry_differs', 'matches'))
    merged['EligibleForInitialFit'] = True
    merged['EvidenceSource'] = 'organizer_released_blind_test_assay_label'
    disagreements = merged.loc[different, ['CatalogID', 'Label', 'StructureStatus', 'Original_SMILES', 'RDKit_SMILES']]
    conflicts = merged.groupby('ParentNonChiral').Label.nunique()
    require(not conflicts.gt(1).any(), 'Conflicting assay labels for identical non-chiral parents.')
    compared = historical.merge(labels[['CatalogID', 'Label']], on='CatalogID', how='left',
                                validate='one_to_one', suffixes=('_reported', '_official'))
    require(compared.Label_official.notna().all(), 'Historical selected test compound absent from release.')
    require(compared.Label_reported.eq(compared.Label_official).all(), 'Historical feedback contradicts official labels.')

    labels_copy = preserve_input(args.labels, 'active_learning_data.csv')
    template_copy = preserve_input(args.template, 'active_learning_submission_example.csv')
    data_dir = ROOT / 'data'
    report_dir = ROOT / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(data_dir / 'labeled_compounds.parquet', index=False)
    # Template values are examples, not predictions. The candidate table contains
    # identities and source structures only and preserves the template's row order.
    candidate_table = template[['CatalogID']].merge(original, on='CatalogID', how='left', validate='one_to_one', sort=False)
    require(candidate_table.CatalogID.tolist() == template.CatalogID.tolist(), 'Candidate order changed.')
    candidate_table.to_parquet(data_dir / 'candidate_universe.parquet', index=False)
    disagreements.to_csv(report_dir / 'structure_discrepancies.csv', index=False, lineterminator='\n')
    inputs = []
    for path, source_url in [(labels_copy, 'https://www.synapse.org/Synapse:syn77795911'),
                             (template_copy, 'https://www.synapse.org/Synapse:syn77796557')]:
        inputs.append({'path': str(path.relative_to(REPO)), 'source_url': source_url,
                       'bytes': path.stat().st_size, 'sha256': sha256(path),
                       'provided_by_user_on': args.as_of})
    output_hashes = {str(p.relative_to(REPO)): sha256(p) for p in
                     [data_dir / 'labeled_compounds.parquet', data_dir / 'candidate_universe.parquet',
                      report_dir / 'structure_discrepancies.csv']}
    report = {
        'as_of': args.as_of, 'status': 'passed', 'inputs': inputs,
        'released_rows': len(labels), 'released_positive': int(counts[1]), 'released_negative': int(counts[0]),
        'positive_fraction_in_selected_release': float(labels.Label.mean()),
        'original_test_rows': len(original), 'eligible_prediction_rows': len(template),
        'template_columns_in_order': list(template.columns),
        'template_flag_dtype': str(template.Sel_50.dtype), 'template_example_selected': int(template.Sel_50.sum()),
        'removed_ids_exactly_equal_all_released_ids': True,
        'released_ids_in_prediction_template': len(released & candidates),
        'released_ids_in_validation': len(released & set(validation.CatalogID)),
        'historical_selected_test_negatives_confirmed': len(compared),
        'historical_official_label_conflicts': 0,
        'structure_comparison': {'invalid_smiles': 0, 'isomeric_canonical_differences': int(different.sum()),
                                 'constitutional_differences': int(changed_parent.sum()),
                                 'stereochemical_only_differences': int((different & ~changed_parent).sum()),
                                 'labels_of_discrepant_rows': sorted(merged.loc[different, 'Label'].unique().tolist()),
                                 'unique_nonchiral_parents': int(merged.ParentNonChiral.nunique()),
                                 'conflicting_parent_labels': 0,
                                 'positive_murcko_scaffolds': int(merged.loc[merged.Label.eq(1), 'MurckoScaffold'].nunique()),
                                 'organizer_chemical_series_membership_available': False},
        'initial_fit_policy': 'Use all released labels with freshly generated features from released structures. Preserve discrepancies and source stereochemistry; group duplicate parents in splits. Compare inclusion/exclusion of eight constitutional discrepancies as a sensitivity analysis.',
        'initial_fit_eligible_rows': int(merged.EligibleForInitialFit.sum()),
        'selection_bias': 'These compounds were selected by participant workflows. Their hit fraction does not estimate prevalence in the remaining library.',
        'template_values_are_predictions': False,
        'phase_two_models_trained': 0, 'phase_two_test_submissions_made': 0,
        'outputs_sha256': output_hashes,
    }
    (report_dir / 'release_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
