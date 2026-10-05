# PGK2 Active Learning preparation

The next campaign starts from experimental feedback, 6,580 released blind-test assay labels, and preserved DEL baselines. The release contains 39 binders and 6,541 non-binders. The main objective is to find new hits across chemical series in the remaining test library. Phase-one top-50 transfer failed, so improved internal DEL AUC alone is not a reason to spend a test submission.

## Official schedule and requirements

Checked October 5, 2026 through the public Synapse wiki API. The [timeline](https://www.synapse.org/Synapse:syn75349604/wiki/641065) lists Active Learning from October 1–31, with 100 validation credits reset after the additional data release and two test submissions. Deadline is October 31 at 23:59 UTC, which is November 1 at 00:59 in Zurich. These are published allowances; the current account balance has not been checked.

The organizer announcement supplied on October 5 and the [actual release](https://www.synapse.org/Synapse:syn77795911) provide labels for all 6,580 evaluated compounds, including negatives. The [updated template](https://www.synapse.org/Synapse:syn77796557) removes all 6,580 IDs from both scoring and selection, leaving exactly 178,052 candidates. Molecules outside the labeled release remain unlabeled. The announcement requires significant DEL use as a key component of the final workflow, rather than a token contribution from a DEL model.

The [submission format](https://www.synapse.org/Synapse:syn75349604/wiki/641045) remains 50 CatalogIDs in a validation text file, or a full test CSV flagging exactly 50 compounds. The released example orders its columns as `CatalogID`, `Score`, `Sel_50`. Use its exact ID universe and row order, with finite model scores and integer 0/1 flags. Its placeholder scores and flags are examples, not predictions. A writeup is accepted through October 31 according to the timeline; preserve methods and provenance as experiments proceed.

## Campaign readiness

The workspace and phase-one evidence are consolidated. Three contrasting 50-ID panels were submitted to Blind validation and all scored zero hits. The full Results page history shows 19 zero-hit and 32 nonzero results across 51 phase-one validation submissions (163 hits total); the strongest panels were ligand-informed. Three additional 50-ID lists were mistakenly sent to the Active learning test queue; two were rejected for format, and the third outcome remains unconfirmed. See the mapped [validation history](reports/validation_history.json) and [status.json](status.json).

- [x] Recover all locally available phase-one code and result records.
- [x] Preserve original submission archives, test files, models, and caches.
- [x] Recover the chat history and correct the quota assumptions.
- [ ] Reconcile the account's actual Active Learning credits and all three mistaken test-queue attempts before any further test submission.
- [x] Obtain the official released labels and current submission templates.
- [x] Confirm that all released compounds are removed from the scoring and selection universe.
- [x] Audit IDs, structures, sources, and label consistency; preserve eight constitutional discrepancy flags and use fresh released-structure features.
- [x] Compare phase-two baselines, submit three validation panels, and record that each scored zero hits.
- [ ] Diagnose why the current three panels failed despite the strong ligand-informed phase-one validation results before fitting or submitting another candidate list.

## Data ingestion

The completed [release audit](DATA_AUDIT.md) and [machine-readable receipt](reports/release_audit.json) record input hashes, schemas, label counts, candidate membership, and agreement with our 100 prior selected test negatives. The prepared labeled table has an `EligibleForInitialFit` field and preserves both original and released structures. All 6,580 rows remain eligible, including eight non-binders with changed constitutions. Features are regenerated from the released structures, and the flags remain available for an inclusion/exclusion sensitivity analysis. Two stereochemical differences remain eligible for the historical non-chiral representation, with the source stereochemistry preserved. The 39 positives are all retained.

All 100 historical selected test negatives already occur in the organizer release. Do not append them again and double their weight. There are no conflicting labels among identical non-chiral parent structures in this release. Group duplicate parents across splits. The 39 positives have 39 distinct exact Murcko scaffolds, which is not an organizer chemical-series count.

Keep downloaded inputs in ignored `phase2/data/` with source URL, download date, license or challenge access conditions, SHA-256, and schema recorded in a tracked manifest. Join by CatalogID first; investigate missing, duplicate, or changed structures before joining by standardized parent. Preserve original SMILES and stereochemistry even when reproducing the historical non-chiral representation.

Assign separate evidence sources to organizer-released positives, logically forced validation labels, our two zero-hit selected test lists, DEL proxy labels, and unlabeled candidates. The compact historical labels are under `docs/recovery/`. Never label every member of a partly successful bag positive. Keep original bag counts so models can use aggregate constraints. Do not add NTC supplement counts without resolving their semantics.

Split confirmed labeled compounds by scaffold or chemical series before model comparison, grouping related structures and preventing duplicate parents from crossing the split. Sparse positive counts can make estimates unstable; report the number of held-out positives. Keep an untouched comparison set where the released sample permits it, and disclose the selection bias of participant-discovered hits.

## Model comparisons

See the [October 5 site review](SITE_REVIEW.md) and [first model comparison](EXPERIMENT_01.md). Fresh released-structure features passed checks, but all three validation panels scored zero. The current candidate-ranking approach has failed its external check; do not spend more credits on its rankings.

1. Use the mapped 51-submission validation history and archived candidate lists as adaptive evidence. Preserve aggregate hit-count constraints; do not turn every member of a partially successful panel into a positive label. The [panel overlap audit](reports/validation_panel_history_overlap.json) found 29 already-known negatives in DEL_support and no historical positives in any current panel.
2. Audit validation and test CatalogID universes, joins, score direction, and selected-ID exports. Queue acceptance and score emails show the current lists were evaluated; investigate why the current rankings missed even historically inferable positives.
3. Compare the successful ligand-informed phase-one selections against current candidate ranks and chemical-family coverage across the validation pool and remaining test pool. Treat grouped cross-validation on the participant-selected release as insufficient evidence of transfer.
4. Only after finding a concrete correction, run a small, documented offline comparison and decide whether another validation experiment is worth a credit. Keep the final workflow substantially DEL-based and require a DEL-removal ablation.

These are proposed comparisons, not completed models or guarantees of improvement. Use the Mac and memory-bounded caches. Preserve phase-one assets and write new results under `phase2/`.

## Validation budget proposal

Do not treat the published allowance as a target. The three current panels all returned zero hits, while the 51 earlier evaluations include both 19 zero-hit and 32 nonzero results. Pause new validation submissions until the current failure is understood and the actual account history is reconciled. The historical request for 50 extra blind-phase lists is superseded by the new phase; it is not an instruction to spend 50 new credits immediately.

## Final test preparation

Before the first test, freeze the model code, input hashes, ranking, selection rules, and offline checks. Confirm all required IDs, finite scores, binary flags, exactly 50 eligible unique selections, and the intended diversity policy. Compare selection stability across fits and inspect unsupported ligand branches, including the ligand-47 failure from phase one.

Prepare the second submission after interpreting the first result under current rules. Avoid submitting overlapping lists together before learning whether the first worked. Preserve the two final files and receipts separately. Scores remain ranking confidence unless calibration has been evaluated.

## Participation questions

The supplied October 5 organizer announcement says the blind-test leaderboard is unofficial pending investigations and Prospective invitations are paused. It reminds teams with at least two identified chemical series that authorship and top-five invitations depend on a correctly formatted Synapse writeup, and excludes workflows without significant DEL use. This does not establish our eligibility. Preserve the announcement's status separately from historical Incentives wording. [Organizer update](organizer_update.json).
