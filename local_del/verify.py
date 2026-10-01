"""Independent integrity checks for features, scientific splits, models and deliverables."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import lightgbm as lgb
from features import featurize,ECFP,FCFP
from train import matrix,eligible,split_groups


def main():
    checks=[]
    assert featurize('this is not a molecule') is None
    a=featurize('CCO');b=featurize('OCC');c=featurize('CCN')
    assert a[0]==b[0] and np.array_equal(a[2],b[2])
    for k,g in enumerate([ECFP,FCFP]):
        packed=DataStructs.TanimotoSimilarity(DataStructs.CreateFromBinaryText(a[2][k].tobytes()),
                  DataStructs.CreateFromBinaryText(c[2][k].tobytes()))
        native=DataStructs.TanimotoSimilarity(g.GetFingerprint(Chem.MolFromSmiles('CCO')),
                                            g.GetFingerprint(Chem.MolFromSmiles('CCN')))
        assert abs(packed-native)<1e-12
    checks.append('SMILES invariance, invalid-SMILES rejection and packed/native fingerprint Tanimoto parity')
    t=pd.read_parquet(ROOT/'features/training_meta.parquet')
    v=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    good,_=eligible(t);held=split_groups(t)
    assert np.array_equal(good,np.load(ROOT/'data/eligible.npy'))
    assert not set(t.loc[good&held,'scaffold'])&set(t.loc[good&~held,'scaffold'])
    assert t.loc[good,'canonical'].is_unique
    assert not set(t.loc[good&t.source.eq('library_unlabeled'),'library'])-set(t.loc[t.source.str.startswith('selection'),'library'])
    checks.append('No shared scaffold across DEL holdout; no duplicate training parents; no negatives inferred from absent libraries')
    sample=np.random.default_rng(42).choice(len(v),100,replace=False)
    for kind in ['ecfp','fcfp']:
        x=matrix('validation',kind)[sample]
        for seed in [2035,2036]:
            model=lgb.Booster(model_file=str(ROOT/'models'/f'{kind}_{seed}.txt'))
            actual=model.predict(x,num_threads=2)
            expected=np.load(ROOT/'models'/f'validation_{kind}_{seed}.npy')[sample]
            assert np.allclose(actual,expected,rtol=1e-8,atol=1e-10)
    checks.append('Four serialized trained models reproduce saved predictions on 100 randomly chosen candidates')
    manifest=json.loads((ROOT/'submissions/round01/manifest.json').read_text())
    assert len(manifest['sets'])==3
    valid_ids=set(v.CatalogID)
    for name,item in manifest['sets'].items():
        path=ROOT/'submissions/round01'/item['file']
        ids=path.read_text().splitlines()
        assert len(ids)==50 and len(set(ids))==50 and set(ids)<=valid_ids
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        assert v.set_index('CatalogID').loc[ids,'canonical'].nunique()==50
    checks.append('Exactly three submission files; each contains 50 unique valid catalog IDs and 50 distinct parent structures; hashes verified')
    result={'passed':checks,'official_challenge_feedback_received':False}
    (ROOT/'reports/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
