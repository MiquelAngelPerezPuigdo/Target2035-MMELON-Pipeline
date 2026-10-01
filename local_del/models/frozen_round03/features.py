"""Reproducible RDKit features, stored as packed bits to bound memory."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,hashlib
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/Users/perezmi/Target2035-MMELON-Pipeline')
RDLogger.DisableLog('rdApp.*')
ECFP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
FCFP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False,
 atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
DESC_NAMES=['MW','LogP','TPSA','HBA','HBD','RotB','AromaticRings','FractionCSP3','FormalCharge','HeavyAtoms','Rings']


def featurize(smiles):
    try:
        m=Chem.MolFromSmiles(smiles)
        if m is None:return None
        parts=Chem.GetMolFrags(m,asMols=True)
        m=max(parts,key=lambda x:x.GetNumHeavyAtoms())
        if m.GetNumHeavyAtoms()==0:return None
        canonical=Chem.MolToSmiles(m,isomericSmiles=False)
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
        # Acyclic molecules share the empty scaffold and stay together in held-out evaluation.
        scaffold=scaffold or '__ACYCLIC__'
        packed=np.stack([np.packbits(ECFP.GetFingerprintAsNumPy(m)),np.packbits(FCFP.GetFingerprintAsNumPy(m))])
        desc=np.asarray([Descriptors.MolWt(m),Crippen.MolLogP(m),rdMolDescriptors.CalcTPSA(m),
            Lipinski.NumHAcceptors(m),Lipinski.NumHDonors(m),Lipinski.NumRotatableBonds(m),
            rdMolDescriptors.CalcNumAromaticRings(m),rdMolDescriptors.CalcFractionCSP3(m),
            Chem.GetFormalCharge(m),m.GetNumHeavyAtoms(),rdMolDescriptors.CalcNumRings(m)],dtype=np.float32)
        return canonical,scaffold,packed,desc
    except Exception:
        return None


def build(name, df):
    prefix=ROOT/'features'/name
    manifest=prefix.with_suffix('.json')
    input_hash=hashlib.sha256(pd.util.hash_pandas_object(df,index=True).values.tobytes()).hexdigest()
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous.get('input_dataframe_sha256')==input_hash:
            print('Using completed feature cache',name,flush=True);return
        # Derived sparse matrices depend on both feature data and row order.
        for kind in ['ecfp','fcfp']:
            stale=ROOT/'features'/f'{name}_{kind}.npz'
            if stale.exists():stale.unlink()
    start=time.time()
    packed=np.lib.format.open_memmap(str(prefix)+'_bits.npy',mode='w+',dtype=np.uint8,shape=(len(df),2,256))
    desc=np.lib.format.open_memmap(str(prefix)+'_desc.npy',mode='w+',dtype=np.float32,shape=(len(df),len(DESC_NAMES)))
    keys=[];scaffolds=[];valid=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for j,result in enumerate(pool.map(featurize,df.SMILES.astype(str),chunksize=128)):
            if result is None:
                keys.append('');scaffolds.append('');valid.append(False);packed[j]=0;desc[j]=0
            else:
                k,s,b,x=result;keys.append(k);scaffolds.append(s);valid.append(True);packed[j]=b;desc[j]=x
            if (j+1)%25000==0:print(name,j+1,'/',len(df),'seconds',round(time.time()-start),flush=True)
    df=df.copy();df['canonical']=keys;df['scaffold']=scaffolds;df['valid']=valid
    df.to_parquet(str(prefix)+'_meta.parquet',index=False)
    packed.flush();desc.flush()
    result={'input_dataframe_sha256':input_hash,'rows':len(df),'valid':sum(valid),'invalid':len(df)-sum(valid),'seconds':time.time()-start,
        'rdkit':__import__('rdkit').__version__,'fingerprints':['ECFP4 2048-bit','FCFP4 2048-bit'],
        'chirality':False,'descriptors':DESC_NAMES,'structure':'largest fragment; no tautomer/charge normalization'}
    manifest.write_text(json.dumps(result,indent=2)+'\n');print(name,result,flush=True)


if __name__=='__main__':
    import sys
    names=sys.argv[1:] or ['training','validation']
    for name in names:
        df=(pd.read_parquet(ROOT/'data/training_raw.parquet') if name=='training'
            else pd.read_csv(SOURCE/'Val-Test-set'/('PGK2_'+name.capitalize()+'_split.csv')))
        build(name,df)
