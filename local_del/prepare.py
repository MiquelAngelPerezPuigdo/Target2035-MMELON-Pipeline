"""Prepare bounded local DEL experiments; source repository is read-only."""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import json
import numpy as np
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('DEL_SOURCE_ROOT', str(ROOT.parent))).expanduser().resolve()
SEED = 2035


def main():
    rng = np.random.default_rng(SEED)
    data = ROOT / 'data'
    counts = ['count_PGK2', 'count_PGK2_with_inhibitor', 'count_NTC']
    zcols = ['zscore_PGK2', 'zscore_PGK2_with_inhibitor', 'zscore_NTC']
    target = data / 'selection_dedup.parquet'
    if not target.exists():
        print('Deduplicating the supplied selection by exact supplied SMILES', flush=True)
        (pl.scan_parquet(SOURCE / 'PGK2_selection.parquet')
         .filter(pl.col('SMILES').is_not_null() & (pl.col('SMILES').str.len_chars() > 0))
         .group_by('SMILES').agg(
             pl.col('compound').first(), *[pl.col(c).sum() for c in counts],
             *[(pl.col(c).sum() / pl.col(c).count().sqrt()).alias(c) for c in zcols],
             pl.col('historic_hits').max(), pl.len().alias('encoding_rows'))
         .sink_parquet(target, compression='zstd'))
    d = pl.read_parquet(target)
    p, i, n, h = [pl.col(c) for c in counts + ['historic_hits']]
    strict = (p >= 3) & (i < .1*p) & (n == 0) & (h < 5)
    broad = (p >= 3) & (i <= .25*p) & (n <= .1*p) & (h < 5)
    neg = ((p <= 1) | (i >= p) | (n >= p) | (h >= 10)) & ~broad
    d = d.with_columns(strict.cast(pl.Int8).alias('strict'), broad.cast(pl.Int8).alias('broad'),
                       pl.col('compound').str.split('-').list.first().alias('library'))
    audit = {'seed': SEED, 'deduplicated_rows': len(d), 'strict_positives': int(d['strict'].sum()),
             'broad_positives': int(d['broad'].sum()), 'selection_negative_pool': d.filter(neg).height}
    lib_summary = d.group_by('library').agg(pl.len(),pl.col('strict').sum(),pl.col('broad').sum()).sort('library')
    lib_summary.write_csv(ROOT/'reports/library_audit.csv')
    positive = d.filter(pl.col('broad') == 1).with_columns(pl.lit('selection_positive').alias('source'))
    # Per-library negative sampling helps avoid learning library identity as activity.
    neg_parts=[]
    for lib in lib_summary['library']:
        pool=d.filter((pl.col('library')==lib)&neg)
        quota=max(2500, min(25000, positive.filter(pl.col('library')==lib).height*3))
        neg_parts.append(pool.sample(n=min(quota,len(pool)),seed=SEED).with_columns(pl.lit('selection_negative').alias('source')))
    selected=pl.concat([positive]+neg_parts)
    # NTC supplement has unresolved count aggregation; use presence only, never sum its counts.
    ntc = (pl.read_parquet(SOURCE/'PGK2_NTC_supplement.parquet')
           .filter(pl.col('SMILES').is_not_null() & (pl.col('SMILES').str.len_chars()>0))
           .unique('SMILES', maintain_order=True))
    overlap=positive.join(ntc.select('SMILES'),on='SMILES',how='semi')
    audit['positive_ntc_presence_overlap']=len(overlap)
    selected=selected.with_columns(pl.col('SMILES').is_in(ntc['SMILES'].implode()).alias('ntc_present'))
    ntc_only=ntc.join(d.select('SMILES'),on='SMILES',how='anti')
    ntc_only=ntc_only.sample(n=min(30000,len(ntc_only)),seed=SEED).select('compound','SMILES')
    ntc_only=ntc_only.with_columns(pl.lit('ntc_presence').alias('source'),pl.lit(True).alias('ntc_present'))
    # Two-stage random library sampling: groups sampled proportional to their row count.
    # Ten groups per library; 500 compounds per group. These are weak, unlabeled negatives.
    enum_parts=[]
    for f in sorted((SOURCE/'OpeDELLibrary').glob('*.parquet')):
        pf=pq.ParquetFile(f)
        weights=np.array([pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)],dtype=float)
        groups=rng.choice(pf.num_row_groups,size=min(10,pf.num_row_groups),replace=False,p=weights/weights.sum())
        for g in groups:
            table=pf.read_row_group(int(g),columns=['compound','SMILES'])
            indices=rng.choice(len(table),size=min(500,len(table)),replace=False)
            enum_parts.append(pl.from_arrow(table.take(indices)))
        print('Sampled background', f.name, flush=True)
    enum=pl.concat(enum_parts).filter(pl.col('SMILES').is_not_null() & (pl.col('SMILES').str.len_chars()>0)).unique('SMILES',maintain_order=True)
    enum=enum.join(d.select('SMILES'),on='SMILES',how='anti').join(ntc.select('SMILES'),on='SMILES',how='anti')
    enum=enum.with_columns(pl.lit('library_unlabeled').alias('source'),pl.lit(False).alias('ntc_present'))
    train=pl.concat([selected,ntc_only,enum],how='diagonal_relaxed').with_columns(
        pl.col('library').fill_null(pl.col('compound').str.split('-').list.first()),
        pl.col('strict').fill_null(0), pl.col('broad').fill_null(0))
    # Sort ensures deterministic downstream ordering regardless of parallel group-by order.
    train=train.sort(['source','SMILES'])
    train.write_parquet(data/'training_raw.parquet',compression='zstd')
    audit['training_rows']=len(train)
    audit['sources']=dict(train.group_by('source').len().iter_rows())
    audit['supplement_handling']='Presence flag only; observed only in NTC used as downweighted negatives; overlapping positives retained at reduced weight.'
    audit['enumeration_handling']='10 random row groups/library, weighted by group size, 500 rows/group. Absent from selection and NTC treated as weak unlabeled negatives.'
    (ROOT/'reports/data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__':
    main()
