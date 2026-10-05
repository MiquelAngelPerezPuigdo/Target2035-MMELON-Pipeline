# PGK2 Active Learning release audit

The supplied October 5 files contain the new experimental labels and the updated prediction universe. The release has 6,580 evaluated compounds, including 39 binders. The template contains exactly the original test library minus all 6,580 evaluated IDs, so the Active Learning prediction pool has 178,052 compounds.

## Verified file contents

| Check | Result |
| --- | ---: |
| Released labeled compounds | 6,580 |
| Label 1 binders | 39 |
| Label 0 non-binders | 6,541 |
| Remaining test candidates | 178,052 |
| Released IDs still in the template | 0 |
| Released IDs outside the original test split | 0 |
| Released IDs in the validation split | 0 |
| Duplicate IDs or missing source values | 0 |
| Our historical selected negatives confirmed by official labels | 100 of 100 |
| Disagreements with our historical selected test feedback | 0 |

The 0.593% binder fraction describes compounds selected by participant workflows. It is not a prevalence estimate for the remaining test library. No chemical-series labels are supplied.

The raw release has `CatalogID`, `Label`, and `RDKit_SMILES`. The example has `CatalogID`, `Score`, and `Sel_50`. The example flags 50 compounds with dummy 0/1 scores and floating-point flags. Those values are not model outputs. Final exports must use the example's eligible IDs, finite model scores, and 50 integer selection flags.

## Structure differences

RDKit successfully parsed every released structure. Comparing canonical structures by CatalogID against the original test file found ten differences, all in label-zero rows. Eight differ in constitution, such as aromatic versus saturated rings, and two differ only in stereochemistry. These are not merely reordered SMILES strings. [Individual discrepancies](reports/structure_discrepancies.csv).

The initial modeling policy excludes the eight constitutional discrepancies pending clarification. It preserves the released and original structures, labels, and comparison flags rather than silently changing either source. The prepared table therefore has 6,572 eligible rows with all 39 positives retained. Two stereo differences are equivalent under the historical non-chiral parent representation; source stereochemistry is retained for other representations.

The release has 6,569 unique non-chiral largest-fragment parent structures and no conflicting labels within those groups. Duplicate parents must stay together in evaluation splits. The positives have 39 distinct exact Murcko scaffolds. That does not imply 39 organizer chemical series, since structurally related compounds can have different exact scaffolds.

## Preserved data and replay

The original Downloads files remain unchanged. Byte-identical local snapshots are at `phase2/data/raw/active_learning_data.csv` and `phase2/data/raw/active_learning_submission_example.csv`. The prepared labeled table is `phase2/data/labeled_compounds.parquet`; the candidate table is `phase2/data/candidate_universe.parquet`. Data files are ignored by Git, while the audit script, discrepancy report, hashes, and source metadata are versioned.

Run the audit on the preserved snapshots with:

```sh
venv/bin/python phase2/audit_release.py --as-of 2026-10-05
```

It checks the original split, template complement, labels, historical feedback, structures, and duplicate parents. It refuses to replace a raw snapshot with different bytes. It generates audit and preparation outputs, and does not fit a model or submit a file. [Audit receipt](reports/release_audit.json), [source entity metadata](reports/source_entities.json).

The [label file](https://www.synapse.org/Synapse:syn77795911) and [template](https://www.synapse.org/Synapse:syn77796557) currently have Synapse version-two metadata. Local file hashes are recorded independently; the source metadata does not by itself prove a byte-for-byte match to the remote file handle.

## Organizer requirements recorded

The announcement provided by the user reports 237 blind-test submissions from 127 teams. Its 6,580 evaluated compounds and 39 true hits match the local data. It states that the blind leaderboard is unofficial during ongoing investigations, Prospective invitations are paused, and final workflows must make significant use of DEL data as a key component. The Active Learning test limit is two submissions per team.

It also reminds teams with at least two chemical series that the stated authorship and top-five Prospective invitations require a Synapse writeup following the official template. This is attributed to the supplied announcement; the audit makes no eligibility determination for our team. [Recorded organizer update](organizer_update.json).

The next task is to compare frozen DEL baselines and experimental-label adaptation using grouped structural holdouts. Keep the released sample's selection bias visible and use top-cutoff precision and chemical transfer diagnostics when choosing the two final submissions.
