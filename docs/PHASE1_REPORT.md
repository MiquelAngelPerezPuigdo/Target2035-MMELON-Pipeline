# PGK2 phase one analysis and results

The September 11, 2026 campaign used this Mac to train DEL-supervised models and adapt predictions using validation feedback. It produced 21 rounds containing 48 evaluated validation lists. The best single list found 25 hits in three chemical series. Both evaluated blind-test lists returned zero hits, covering 100 distinct selected molecules. Those test outcomes are the key evidence for planning Active Learning.

This report combines the recovered Codex conversations, saved code, original submission archives, and the [feedback ledger](../local_del/reports/official_results.json). Feedback was reported by the user; most submission IDs were not saved and the ledger has not been reconciled against the complete Synapse account history. [PHASE1_RESULTS.csv](PHASE1_RESULTS.csv) contains every recorded outcome.

## Data and preparation

The raw DEL selection was deduplicated by exact supplied SMILES into 7,487,569 records. Counts were summed, supplied z-scores combined with unweighted Stouffer aggregation, and historical-hit counts aggregated by maximum. The local classifiers used count-derived labels, not z-scores. There were 26,466 strict and 26,829 broader positive proxy labels. The prepared training table contained 203,419 rows; structural conflicts and duplicate parent structures reduced eligibility to 193,091. [Data audit](../local_del/reports/data_audit.json), [model metrics](../local_del/reports/model_metrics.json).

Strict positives required target counts at least three, inhibitor counts below 10% of target, no NTC counts, and fewer than five historical hits. Broader positives allowed inhibitor counts up to 25% and NTC counts up to 10% of target. These labels represent DEL evidence, not confirmed off-DNA binding. Missing enrichment does not establish inactivity.

Selection negatives were sampled by library. Enumeration background was sampled through row groups and treated as weak evidence; libraries absent from the supplied selection were excluded from background training negatives. The NTC supplement supplied a presence flag and downweighted negatives. Its counts were not summed with selection counts because overlap semantics were unresolved.

Structures used the largest fragment and non-chiral parent representation. Tautomer and charge normalization were not applied. Validation contained 244,328 compounds and test contained 184,632. The full enumerated library was not used as 890 million independent training examples.

## Methods actually evaluated

| Method family | What it used | Recorded outcome |
| --- | --- | --- |
| DEL LightGBM baselines | ECFP4 or FCFP4 with descriptors, count labels, two seeds | Both initial top-50 validation lists had zero hits |
| DEL neighbors and kernels | Positive and control reference fingerprints, Tanimoto and asymmetric Tversky, weighted support | Initial neighbor list had one hit; several expanded lists had zero |
| Public crystal ligand similarity | Verified compounds 21 and 47, plus a DEL-trained contribution | G found 25 hits in three series; the ligand-47 diagnostic J had zero |
| New cores and adaptive bags | Alternative cores, overlap constraints, subdivision of successful lists | K found 21 hits in two series; O found four hits in three series |
| ErG pharmacophores | Alternative representation for scaffold transfer | AH found two hits in one series |
| ChemBERTa embeddings | Local pretrained embedding similarity and adaptive diagnostics | AI found one hit; subsequent constraints isolated an individual hit |
| Crystal shape and consensus | Ligand-21 3D shape and mixed series portfolios | AR had zero, AS had one; AT and AU had zero |
| Aggregate-count model | Regularized fits to bag counts and logically forced labels, then authorized T1 negatives | Revised T2 improved ranking metrics but found zero selected hits |

The G formula was dominated by known-ligand similarity, with a 5% DEL ensemble contribution. It was not evidence that a new MMELON model learned OOD binding. The root MMELON scripts and notebooks belong to the earlier HPC effort; the local campaign replaced that approach. [OOD interpretation](../local_del/reports/ood/INTERPRETATION.md).

The expanded audit compared all validation molecules with 26,544 references and 2,022 controls. Apple MPS was used for weighted fingerprint density calculations; the saved audit records about 12.1 GB of driver-managed unified memory at peak. The DEL-positive median molecular weight was 583 versus 332 for validation, with nine versus four median rotatable bonds. That mismatch motivated size-aware similarity; it did not establish successful transfer. [Expanded findings](../local_del/reports/deep_audit/FINDINGS.md).

## Validation results

| Round | Sets and reported hits | Chemical series in the same order |
| --- | --- | --- |
| 1 | A 0, B 0, C 1 | 0, 0, 1 |
| 2 | D 0, E 0, F 0 | 0, 0, 0 |
| 3 | G 25, H 0, I 1 | 3, 0, 1 |
| 4 | J 0, K 21, L 1 | 0, 2, 1 |
| 5 | M 21, N 8, O 4 | 2, 2, 3 |
| 6 | P 2, Q 3, R 1 | 1, 2, 1 |
| 7 | S 13, T 6, U 2 | 1, 2, 1 |
| 8 | V 13, W 5, X 1 | 1, 1, 1 |
| 9 | Y 1, Z 0, AA 0 | 1, 0, 0 |
| 10 | AB 0, AC 1, AD 2 | 0, 1, 2 |
| 11 | AE 8, AF 7, AG 6 | 1, 1, 2 |
| 12 | AH 2 | 1 |
| 13–15 | AI 1, AJ 1, AK 1 | 1, 1, 1 |
| 16 | AL 1, AM 1 | 1, 1 |
| 17 | AN 1, AO 0 | 1, 0 |
| 18 | AP 0, AQ 0 | 0, 0 |
| 19 | AR 0, AS 1 | 0, 1 |
| 20 | AT 0, AU 0 | 0, 0 |
| 21 | AV 0 | 0 |

The lists overlap and were chosen adaptively. Their hit counts cannot be added to obtain a unique hit total, and the table is not an independent estimate of generalization. Organizer series counts differ from our Murcko scaffold counts. Historical p-values are preserved in the results CSV; they do not remove the effects of adaptive selection.

Constraint propagation over the saved 48 lists involves 1,247 distinct validation IDs. It forces eight positive and 628 negative individual labels, leaving 611 unresolved. These are conditional deductions from user-reported aggregate outcomes and exact saved list membership. They are not organizer-released labels. Seven positives were forced by AF; the eighth was isolated through the ChemBERTa diagnostic sequence. [Inference audit](../local_del/reports/validation_label_inference.json).

## Blind test results

| Evaluated file | Selected hits | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: |
| T1 FROZEN G | 0 of 50 | 0.7012738793 | 0.0020381789 |
| T2 REVISED COUNTS | 0 of 50 | 0.7400418828 | 0.0045254012 |

The original diverse T2 file was prepared but replaced before submission. It must not be recorded as an evaluated third test. Revised T2 had zero selected overlap with T1; all 100 selected compounds therefore have user-reported negative outcomes. The first T1 submission ID, 9780475, appears in the user's pasted leaderboard; the second ID remains unknown.

T1's ranking favored ligand 47 for 31 of 50 selected compounds, despite the earlier zero-hit J diagnostic. After T1 failed, the user explicitly authorized using its feedback to revise T2. The replacement combined six regularized count-model fits, ligand-21 similarity, and the frozen DEL ensemble. It excluded T1 parents, constrained repeated aromatic cores, and selected 48 distinct Murcko scaffolds. Its higher ROC and PR AUC did not produce hits at the selected cutoff. The exact revised file hash is saved in [verification.json](../local_del/submissions/blind_test_revised/verification.json).

## What failed and what remains useful

DEL scaffold-holdout ROC AUC was about 0.969 for ECFP and 0.963 for FCFP; both had perfect top-50 precision against their noisy internal labels. Their first ASMS validation lists had zero hits. Internal DEL metrics therefore failed to predict practical assay hit recovery. Likewise, the strong adaptive known-ligand validation results did not transfer to the blind-test top 50.

The saved chemical-coverage audit already showed fewer close ligand analogs in test: maximum reference-ligand ECFP similarity was 0.3611 in test versus 0.5625 in validation. Test rankings also shifted toward ligand 47. This was a warning about coverage, not proof of zero expected hits. The campaign spent many queries splitting successful bags to identify individual labels, leaving limited independent evidence for broader transfer. The aggregate-count method arrived only after T1 failed.

Useful assets include the frozen baselines, complete candidate features, count constraints, eight forced validation positives, 628 forced validation negatives, 100 selected test negatives, reference-ligand provenance, and exact submission archives. Active Learning should compare these against models trained on the newly released experimental labels, keeping uncertain observations unlabeled.

## Corrections and unresolved evidence

The ledger's `remaining_now: 0` is an obsolete estimate. The final recovered conversation corrected the phase-one official limit to 100 and discussed approximately 50 unused validations. The user requested 50 additional lists in batches of ten. No completed extra lists or evaluation results were found in the recovered workspace; that request was not fulfilled in the available history. We recovered 48 evaluated local lists, not 100.

As of October 1, the official [timeline](https://www.synapse.org/Synapse:syn75349604/wiki/641065) specifies October 1–31 Active Learning and refreshed validation credits after the additional data release. Actual current account credits have not been read. The [overview](https://www.synapse.org/Synapse:syn75349604/wiki/641044) describes releasing true-positive labels from participant blind-test predictions; it does not imply that every other molecule becomes a confirmed negative.

Historical claims about authorship were too definite. On October 1 the overview says better than random in Blind Test **or** Active Learning, while [Incentives](https://www.synapse.org/Synapse:syn75349604/wiki/641066) says **and**. No eligibility guarantee follows from our AUCs or zero-hit outcome. The October 5 forum review resolves this as **or**, with an unadjusted p < 0.01 threshold and the best of two submissions considered. See the [current review](../phase2/SITE_REVIEW.md); this does not establish eligibility for our zero-hit blind submissions.

## October 5 experimental label release

The organizer release now contains 6,580 participant-selected test compounds with positive and negative assay labels. All 100 compounds selected in our two recorded test files are present and labeled zero, confirming the user-reported outcomes. All released IDs have been removed from the Active Learning template. The supplied organizer announcement also makes the blind leaderboard unofficial during investigations and pauses Prospective invitations. The phase-one ledger remains unchanged as historical evidence. Current preparation and requirements are recorded in the [phase-two audit](../phase2/DATA_AUDIT.md).
