# Expanded local analysis after two validation rounds

Five submitted sets returned zero hits. The original Neighbors set returned one hit in one cluster, with p-value 0.05201186181800377. These results do not establish reliable generalization. The user reports 42 validations remaining before round 3.

## Computation actually completed

- Audited all **7,487,569 deduplicated DEL selection records**, including library-level count distributions and building-block marginals for three synthesis positions. This was a full-table audit, not a claim to have fitted a neural network to every row.
- Compared **all 244,328 validation compounds** with **26,544 reference molecules** and **2,022 controls**. No validation shortlist was used.
- Saved the top 32 neighbours per compound for four structural/chemical-feature similarity definitions, with reference identities for further analysis.
- Ran full weighted similarity-density calculations on the **Apple MPS GPU**, holding both complete candidate fingerprint arrays on the GPU. Peak GPU driver-managed unified memory was **12,061,442,048 bytes** (about 12.1 GB); the configured GPU cap was 14.3 GB. The kernel stage took approximately **50 seconds**, excluding preparation and the separate CPU neighbour calculation. Eight CPU workers performed the broader neighbour calculation in approximately 485 seconds. These are different workloads, not a controlled GPU/CPU speed comparison.
- Verified GPU densities against independent RDKit CPU calculations, including a further 16 fingerprint/molecule checks when creating round 3. Full candidate coverage, finite scores, and exact submission-file integrity were checked.

## What the data show

The DEL-positive molecules are much larger and more flexible than validation compounds. Median molecular weight is **583 versus 332**, and median rotatable-bond count is **9 versus 4**. Standard whole-molecule similarity penalizes unmatched portions of larger DEL structures; asymmetric Tversky similarity (query weight 1, reference-only weight 0.2) explicitly tests whether a smaller query matches part of a larger DEL compound. This is a hypothesis about transfer, not evidence of improved hit recovery.

The new GPU density scores compare weighted support across the **entire positive bank** against NTC control support. They use capped read-count reliability, moderated library balancing, and separate public-selective DEL evidence. This avoids relying only on three nearest neighbours. Scores remain uncalibrated ranking signals.

The [public CACHE #7 reference page](https://cache-challenge.org/challenges/finding-selective-inhibitors-of-phosphoglycerate-kinase-2) supplies PGK2/PGK1-selective DEL candidates and two PGK2-bound ligand structures. The downloaded DEL file has 1,393 rows; 1,366 unique reference structures are flagged as public-selective in the combined bank. Those DEL candidates are **not confirmed off-DNA hits**. Compounds 21 and 47 were transcribed from the public figure and checked against the deposited ligand heavy-atom connectivity before scoring. The separate CACHE #7 non-quinazoline restriction is not applied to this DREAM challenge.

Raw similarity distances are preserved in the known-ligand ranking. Percentile averaging can compress differences between especially close and moderately close matches in the extreme tail, so it is not used for that ranking.

## What the validation feedback actually tells us

The five zero-hit sets contain **196 distinct CatalogIDs**. Assuming the submitted files match the saved ZIP archives and evaluations are consistent, their overlap with the one-hit set leaves **22 possible identities** for the earlier hit. No one of those compounds is labelled positive.

Round 3 excludes the 196 previously evaluated zero-hit IDs from **validation selection only**, so submission slots are not spent repeating known assay outcomes. Unfiltered whole-library model rankings are preserved. No positive compound is inferred from the aggregate one-hit count. These are adaptive validation experiments; they cannot be presented as independent blind-test evidence.

## Round 3

- **G — Known ligands:** similarity to the two verified PGK2 crystal ligands, with a DEL-trained model contribution.
- **H — Selective DEL:** full-bank kernel support from the public PGK2-selective DEL references relative to control support.
- **I — Size-aware kernel:** read- and library-weighted full-positive-bank kernel support relative to controls, using size-aware and conventional similarity.

Each file contains 50 unique valid CatalogIDs and 50 distinct non-chiral parent structures. The archive is `submissions/round03_validation_batch.zip`. These three validation submissions will leave **39**, based on the user's corrected quota.

The analysis, new scores, source data and model formulas are saved locally. No HPC or cloud inference was used, and no submissions were made automatically. The next decision depends on the three official results.
