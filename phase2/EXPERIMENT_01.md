# First Active Learning model comparison

This experiment checks transfer of the recovered DEL models to the organizer-released assay labels, compares release-label adaptation under chemistry-grouped holdouts, and uses the resulting rankings to construct validation panels. The holdouts are an internal diagnostic on a participant-selected set, not an estimate of performance in the remaining test library.

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

The combined archive is [active_learning_validation_panels.zip](submissions/panels/active_learning_validation_panels.zip) (SHA-256 `02df9977d55c263f9e58a161dd469b1f0e120f0db9fbf5d4597080b0c8cef58a`). The three 50-ID files are in `phase2/submissions/panels/`; detail CSVs provide their model ranks and structures. They were uploaded to the challenge project and submitted through the **Blind validation** queue: DEL support (`syn77821634`, submission `9783554`), Experimental adaptation (`syn77821633`, submission `9783555`), and Model disagreement (`syn77821632`, submission `9783556`). All three scored **0 hits**. Two panels overlap on two IDs, so this is 148 distinct candidates with zero recovered hits. The scores and email evidence are tracked in `status.json`.

This outcome continues the earlier validation pattern: score emails for submissions `9778878`, `9779675`, and `9780474` also reported zero hits. The six observed zero-hit validation scores invalidate the current ranking/selection approach as a useful validation strategy. Do not submit another panel from these rankings; first audit historical selection methods, split and identifier handling, score calibration, and whether the selected candidates cover the validation compounds as expected.

Three earlier attempts using the same 50-ID text format were mistakenly sent to the **Active learning** queue. Two received format-rejection emails; the third outcome has not been confirmed. These are not valid test predictions. Do not make further test-queue attempts until the Synapse submission history and account allowance are reconciled. See `status.json` for the exact uncertainty and event counts.

## Decision and limits

The six observed validation submissions all returned zero hits, so the three current ranking panels have failed their external check. Do not keep selecting from these rankings or spend more submissions on them. First reproduce the historical predictions and submission IDs, inspect candidate-universe joins and score direction, and compare the old and current panels against the exact validation/test CatalogID universes. Only after identifying a concrete correction should a new validation experiment be considered. Any eventual final ranking must remain meaningfully informed by DEL models or DEL-derived evidence and show that contribution through a removal ablation.

The official series assignments were not released, so local ECFP/Murcko diversity is only a selection heuristic. The 39 positives were selected by participants, the 100 historical negatives overlap this release, and the label neighborhoods may identify chemistry favored by prior participant rankings. The validation list feedback is adaptive and should be declared in the writeup. The account's total submission history and remaining challenge allowance are not yet reconciled.

Machine-readable metrics and fold counts: [release_diagnosis.json](reports/release_diagnosis.json). Individual binder ranks: [released_positive_ranks.csv](reports/released_positive_ranks.csv). Panel definitions and hashes: [validation_panels.json](reports/validation_panels.json). All models and panel scorers used fresh features for the released SMILES; source records and phase-one assets remain intact.
