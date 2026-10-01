# PGK2 Active Learning preparation

The next campaign starts from experimental feedback, the newly released blind-test positives, and preserved DEL baselines. The main objective is to find new hits across chemical series in the test library. Phase-one top-50 transfer failed, so improved internal DEL AUC alone is not a reason to spend a test submission.

## Official schedule and requirements

Checked October 1, 2026 through the public Synapse wiki API. The [timeline](https://www.synapse.org/Synapse:syn75349604/wiki/641065) lists Active Learning from October 1–31, with 100 validation credits reset after the additional data release and two test submissions. Deadline is October 31 at 23:59 UTC, which is November 1 at 00:59 in Zurich. These are published allowances; the current account balance has not been checked.

The [overview](https://www.synapse.org/Synapse:syn75349604/wiki/641044) describes releasing true positives identified in participant blind-test predictions, then adjusting models to retrieve new hits from the same library. It still requires a DEL-ML step. Released positives do not turn unreported molecules into known negatives. Verify whether revealed hits must be excluded from the 50 selections and whether they remain in the full scoring universe; the recovered pages do not settle that detail.

The [submission format](https://www.synapse.org/Synapse:syn75349604/wiki/641045) remains 50 CatalogIDs in a validation text file, or a full test CSV with `CatalogID`, `Sel_50`, and `Score`, flagging exactly 50 compounds. Phase-one row counts must not be assumed for the released phase-two template. A writeup is accepted through October 31 according to the timeline; preserve methods and provenance as experiments proceed.

## Campaign readiness

The workspace and phase-one evidence are consolidated. No phase-two model has been trained and no phase-two submission has been made during this recovery. [status.json](status.json) is the separate campaign ledger.

- [x] Recover all locally available phase-one code and result records.
- [x] Preserve original submission archives, test files, models, and caches.
- [x] Recover the chat history and correct the quota assumptions.
- [ ] Check the account's actual Active Learning credits and registration.
- [ ] Obtain the official released labels and current submission templates.
- [ ] Confirm the eligible selection universe and treatment of revealed positives.
- [ ] Audit IDs, structures, sources, and label reliability.
- [ ] Compare phase-two baselines before preparing validation panels.

## Data ingestion

Keep downloaded inputs in ignored `phase2/data/` with source URL, download date, license or challenge access conditions, SHA-256, and schema recorded in a tracked manifest. Join by CatalogID first; investigate missing, duplicate, or changed structures before joining by standardized parent. Preserve original SMILES and stereochemistry even when reproducing the historical non-chiral representation.

Assign separate evidence sources to organizer-released positives, logically forced validation labels, our two zero-hit selected test lists, DEL proxy labels, and unlabeled candidates. The compact historical labels are under `docs/recovery/`. Never label every member of a partly successful bag positive. Keep original bag counts so models can use aggregate constraints. Do not add NTC supplement counts without resolving their semantics.

Split confirmed labeled compounds by scaffold or chemical series before model comparison, grouping related structures and preventing duplicate parents from crossing the split. Sparse positive counts can make estimates unstable; report the number of held-out positives. Keep an untouched comparison set where the released sample permits it, and disclose the selection bias of participant-discovered hits.

## Model comparisons

1. Replay the frozen G ranking, original DEL LightGBM ensemble, and aggregate-count method as baselines on the current candidate universe.
2. Fit a regularized model to reliable experimental labels with DEL predictions as an explicit feature or separate ensemble component. Compare against the same model without those experimental labels to measure their contribution.
3. Evaluate a method that preserves unlabeled status, such as positive-unlabeled learning or heavily downweighted background examples, if label coverage permits it.
4. Compare ECFP, FCFP/descriptors, and ErG before committing to heavier embeddings. ChemBERTa and 3D features are optional experiments whose value must be measured on unseen chemical families.
5. Assess precision and enrichment near the selection cutoff, chemical-series coverage, uncertainty across fits, and sensitivity to domain shift. Report ROC AUC and PR AUC as supporting metrics rather than the sole decision.

These are proposed comparisons, not completed models or guarantees of improvement. Use the Mac and memory-bounded caches. Preserve phase-one assets and write new results under `phase2/`.

## Validation budget proposal

Confirm the balance first. If 100 fresh validations are available, use the following allocation as a starting plan and review it after every batch.

| Purpose | Maximum initial allocation |
| --- | ---: |
| Designed diagnostics for chemistry coverage, label transfer, and model disagreements | 20 |
| Compare promising representations and combinations | 30 |
| Improve series coverage and robust candidate selection | 20 |
| Reserved confirmation before the two test submissions | 30 |

Each panel should answer a recorded question. Track list overlap, known-label exclusions, intended chemistry, and expected decision before submission. Use a few contrasting lists per batch, with a concise feedback form, to reduce manual effort. Stop splitting bags once the expected information no longer helps choose a model or discover a new series. The historical request for 50 extra blind-phase lists is superseded by the new phase; it is not an instruction to spend 50 new credits immediately.

## Final test preparation

Before the first test, freeze the model code, input hashes, ranking, selection rules, and offline checks. Confirm all required IDs, finite scores, binary flags, exactly 50 eligible unique selections, and the intended diversity policy. Compare selection stability across fits and inspect unsupported ligand branches, including the ligand-47 failure from phase one.

Prepare the second submission after interpreting the first result under current rules. Avoid submitting overlapping lists together before learning whether the first worked. Preserve the two final files and receipts separately. Scores remain ranking confidence unless calibration has been evaluated.

## Participation questions

The later Prospective phase invites the top five groups from Blind Test and the top five from Active Learning according to [Incentives](https://www.synapse.org/Synapse:syn75349604/wiki/641066). Our result does not establish an invitation. Authorship wording currently differs between overview and Incentives, and no numerical better-than-random criterion was established in this recovery. Keep that uncertainty separate from the scientific plan.
