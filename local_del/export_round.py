"""Export exactly three reviewable validation sets; never submits to Synapse."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parent


def percentile(x):return rankdata(x,method='average')/len(x)


def similarity_signal(s):
    return .5*s[:,0]+.5*s[:,1]-.15*s[:,2]-.15*s[:,3]


def select(d,score,n=50):
    # Preserve hit yield: no scaffold diversity cap. Remove duplicate constitutional structures.
    ranking=d.assign(Score=score).loc[d.valid].sort_values(['Score','CatalogID'],ascending=[False,True])
    return ranking.drop_duplicates('canonical').head(n)


def main():
    d=pd.read_parquet(ROOT/'features/validation_meta.parquet')
    a=np.load(ROOT/'models/validation_ecfp.npy')
    b=np.load(ROOT/'models/validation_fcfp.npy')
    s=np.load(ROOT/'models/validation_similarity.npy')
    # This third set blends trained model evidence with local labelled-neighbour evidence.
    sim=similarity_signal(s)
    c=.35*percentile(a)+.15*percentile(b)+.50*percentile(sim)
    scores={'A_ECFP':a,'B_FCFP':b,'C_NEIGHBORS':c}
    d['ECFP_score']=a;d['FCFP_score']=b;d['similarity_signal']=sim
    d['positive_ECFP_similarity']=s[:,0];d['positive_FCFP_similarity']=s[:,1]
    d['control_ECFP_similarity']=s[:,2];d['control_FCFP_similarity']=s[:,3]
    desc=np.load(ROOT/'features/validation_desc.npy',mmap_mode='r')
    d['MW']=desc[:,0];d['LogP']=desc[:,1];d['TPSA']=desc[:,2]
    out=ROOT/'submissions/round01';out.mkdir(exist_ok=True,parents=True)
    info={'round':1,'official_submissions_made':0,'reported_remaining_before_round':'approximately 46 (user estimate)',
        'deadline':'2026-09-30; confirmed on live challenge wiki on 2026-09-11',
        'sets':{},'overlap':{},'seed_stability':{}}
    selections={}
    for name,score in scores.items():
        assert len(score)==len(d) and np.isfinite(score).all() and np.ptp(score)>0
        selected=select(d,score)
        assert len(selected)==50 and selected.CatalogID.nunique()==50 and selected.canonical.nunique()==50
        assert set(selected.CatalogID)<=set(d.CatalogID)
        path=out/f'Team_MMELON_R01_{name}.txt'
        path.write_text('\n'.join(selected.CatalogID.astype(str))+'\n')
        selected.to_csv(out/f'{name}_candidate_details.csv',index=False)
        # Verify actual serialized files, not just their source DataFrames.
        lines=path.read_text().splitlines()
        assert len(lines)==50 and len(set(lines))==50 and all(x.strip()==x and x for x in lines)
        selections[name]=set(lines)
        info['sets'][name]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'count':len(lines),'unique_scaffolds':selected.scaffold.nunique(),
            'score_range':[float(selected.Score.min()),float(selected.Score.max())],
            'median_MW':float(selected.MW.median()),'median_LogP':float(selected.LogP.median()),
            'median_positive_ECFP_similarity':float(selected.positive_ECFP_similarity.median())}
    for i,j in itertools.combinations(scores,2):info['overlap'][i+'__'+j]=len(selections[i]&selections[j])
    info['unique_candidates_across_round']=len(set.union(*selections.values()))
    for kind in ['ecfp','fcfp']:
        p=np.load(ROOT/'models'/f'validation_{kind}_2035.npy');q=np.load(ROOT/'models'/f'validation_{kind}_2036.npy')
        ps=set(select(d,p).CatalogID);qs=set(select(d,q).CatalogID)
        info['seed_stability'][kind]={'spearman':float(spearmanr(p,q).statistic),'top50_overlap':len(ps&qs)}
    d.assign(C_NEIGHBORS_score=c).to_parquet(ROOT/'models/validation_rankings.parquet',index=False)
    # Audit similarity performance using references that exclude held-out scaffolds.
    h=pd.read_parquet(ROOT/'reports/ecfp_heldout.parquet')
    ids=np.load(ROOT/'models/similarity_heldout_ids.npy')
    assert np.array_equal(ids,h.row.to_numpy())
    hs=similarity_signal(np.load(ROOT/'models/heldout_similarity.npy'))
    info['heldout_similarity']={'roc_auc':float(roc_auc_score(h.label,hs)),
        'average_precision':float(average_precision_score(h.label,hs)),
        'warning':'No ASMS labels used; DEL proxy metrics are not expected challenge hit rates.'}
    (out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    results=ROOT/'reports/official_results.json'
    if not results.exists():results.write_text(json.dumps({'round01':{k:{'submission_id':None,'hits':None,'chemical_series':None} for k in scores}},indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
