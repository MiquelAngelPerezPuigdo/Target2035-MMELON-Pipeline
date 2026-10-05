"""Rebuild phase-two features from released SMILES without phase-one cache reuse."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from local_del.features import featurize, DESC_NAMES
from phase2.audit_release import sha256, require


def main():
    source = ROOT / 'data/labeled_compounds.parquet'
    table = pd.read_parquet(source)
    results = [featurize(s) for s in table.RDKit_SMILES]
    require(all(item is not None for item in results), 'Feature generation failed.')
    require([item[0] for item in results] == table.ParentNonChiral.tolist(),
            'Feature parents disagree with the release audit.')
    bits = np.stack([item[2] for item in results])
    descriptors = np.stack([item[3] for item in results])
    require(np.isfinite(descriptors).all(), 'Non-finite descriptors.')
    require(table.EligibleForInitialFit.all(), 'Some released labels are excluded.')
    checked = []
    for i in np.flatnonzero(table.StructureStatus.eq('constitution_differs')):
        old = featurize(table.iloc[i].Original_SMILES)
        require(old is not None and not np.array_equal(old[2], bits[i]),
                'Constitutional discrepancy did not change fingerprints.')
        checked.append(table.iloc[i].CatalogID)
    outputs = [ROOT / 'data/labeled_bits.npy', ROOT / 'data/labeled_descriptors.npy']
    np.save(outputs[0], bits)
    np.save(outputs[1], descriptors)
    row_hash = hashlib.sha256('\n'.join(table.CatalogID).encode()).hexdigest()
    receipt = {
        'input': str(source.relative_to(ROOT.parent)), 'input_sha256': sha256(source),
        'rows': len(table), 'row_order_catalog_ids_sha256': row_hash,
        'structure_source': 'RDKit_SMILES from organizer active_learning_data.csv',
        'phase_one_feature_cache_reused': False,
        'rdkit_version': __import__('rdkit').__version__,
        'fingerprints': ['ECFP4 2048-bit', 'FCFP4 2048-bit'],
        'packed_bits_shape': list(bits.shape), 'descriptor_names': DESC_NAMES,
        'chirality': False, 'parent_policy': 'largest fragment; no tautomer/charge normalization',
        'constitutional_discrepancies_with_changed_fingerprints': checked,
        'outputs_sha256': {str(p.relative_to(ROOT.parent)): sha256(p) for p in outputs},
        'status': 'passed', 'models_trained': 0,
    }
    (ROOT / 'reports/labeled_features.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
