# First Active Learning model comparison

This offline experiment checks transfer of the recovered DEL models to the organizer-released assay labels and compares release-label adaptation under chemistry-grouped holdouts. The holdouts are an internal diagnostic on a participant-selected set, not an estimate of performance in the remaining test library. No validation or test submission was made.

## Evidence and grouping

The release has 6,580 participant-selected compounds: 39 binders and 6,541 non-binders (0.593% within this selected release). All 100 prior selected test negatives are among those official labels. This fraction does not estimate hit prevalence among the 178,052 remaining test candidates.

ECFP4 Butina grouping at Tanimoto ≥0.65 produced 5,466 groups. Five stratified group folds held out 6–10 binders each. Related structures assigned to one Butina cluster stayed together. ECFP groups guard against close-analogue leakage; they are not the organizers' chemical-series labels.

## Held-out model comparison

Each value averages five held-out folds. The final column totals binders in each fold’s top 50 ranking, for 250 selections across five folds. It is a comparison within this release, not a forecast for a single 50-molecule test submission.

| Method | ROC AUC | Average precision | Trapezoidal PR AUC | Hits in fold top 50 (of 250) |
|---|---:|---:|---:|---:|
| Frozen DEL ECFP | 0.715 | 0.0198 | 0.0149 | 4/250 |
| Frozen DEL FCFP + descriptors | 0.658 | 0.0204 | 0.0141 | 1/250 |
| Historical G: ligand prior + DEL | 0.534 | 0.0098 | 0.0077 | 2/250 |
| Released-label neighborhood | 0.918 | 0.3579 | 0.3345 | 27/250 |
| DEL + released-label adaptation | 0.923 | 0.3439 | 0.3205 | 27/250 |

The experimental-neighborhood comparator ranked 27 binders in those 250 held-out picks. Adding frozen DEL logits and the ligand prior also ranked 27; average precision and trapezoidal PR AUC were lower, while ROC AUC was slightly higher. DEL adaptation therefore did not improve the key top-50 recovery measure in this diagnostic. It remains important to compare explicit DEL-led methods in validation, and any final workflow still needs significant DEL use.

For context, applying fixed phase-one rankings to the full labeled release recovered 1 of 39 binders in the top 50 with frozen ECFP DEL, 1 with frozen FCFP DEL, and 0 with historical ligand-led G. These are retrospective ranks on the selected feedback set; they do not measure expected hits in the remaining test pool.

## Validation panels prepared

Three local lists contain 50 unique validation CatalogIDs each, with no within-list ECFP4 pair above 0.65 Tanimoto and at most two molecules per exact Murcko scaffold. The panels overlap by 0 (DEL vs adaptation), 0 (DEL vs disagreement), and 2 (adaptation vs disagreement). They separate DEL support, label adaptation, and high adaptation score with weak DEL rank. These are for validation feedback and review, not blind-test files.

The combined archive is [active_learning_validation_panels.zip](submissions/panels/active_learning_validation_panels.zip) (SHA-256 `02df9977d55c263f9e58a161dd469b1f0e120f0db9fbf5d4597080b0c8cef58a`). The three 50-ID files are in `phase2/submissions/panels/`; detail CSVs provide their model ranks and structures. All remain local. The account allowance has not been checked and no submissions were uploaded.

## Decision and limits

Use the validation feedback to see whether DEL-led candidates find binders that the adapted rankings miss, and whether the disagreement list adds distinct official series. Because no new DEL adaptation lift appeared under grouped holdouts, do not treat the current release-neighborhood model alone as an eligible final workflow. Keep the final ranking meaningfully informed by DEL models or DEL-derived evidence and show that contribution through a removal ablation.

The official series assignments were not released, so local ECFP/Murcko diversity is only a selection heuristic. The 39 positives were selected by participants, the 100 historical negatives overlap this release, and the label neighborhoods may identify chemistry favored by prior participant rankings. The validation list feedback is adaptive and should be declared in the writeup.

Machine-readable metrics and fold counts: [release_diagnosis.json](reports/release_diagnosis.json). Individual binder ranks: [released_positive_ranks.csv](reports/released_positive_ranks.csv). Panel definitions and hashes: [validation_panels.json](reports/validation_panels.json). All models and panel scorers used fresh features for the released SMILES; source records and phase-one assets remain intact.
