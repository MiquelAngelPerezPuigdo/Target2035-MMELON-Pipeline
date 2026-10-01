"""Generate reduced-graph pharmacophore fingerprints for scaffold hopping."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs

ROOT=Path(__file__).resolve().parent

def main():
    report={}
    for split in ["validation","test"]:
        meta=pd.read_parquet(ROOT/f"features/{split}_meta.parquet")
        path=ROOT/f"features/{split}_erg.npy"
        out=open_memmap(path,mode="w+",dtype=np.float32,shape=(len(meta),315))
        start=time.time()
        for begin in range(0,len(meta),10000):
            smiles=meta.SMILES.iloc[begin:begin+10000]
            out[begin:begin+len(smiles)]=np.asarray([rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(s)) for s in smiles],dtype=np.float32)
        out.flush()
        report[split]={"rows":len(meta),"features":315,"seconds":time.time()-start,"file":str(path.relative_to(ROOT))}
    (ROOT/"reports/erg_features.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
