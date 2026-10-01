# Local PGK2 hit discovery

This workspace replaces the MMELON approach with small, DEL-supervised local models. Computation uses this Mac only. Original repository files are reference inputs and remain unchanged. No submissions have been made automatically.

## First batch

Upload the three `.txt` files under `submissions/round01/` to the **validation** queue, one file per submission. Each has exactly 50 CatalogIDs, with no header. The `Team_MMELON` filename prefix is retained from the original repository; rename that prefix if your registered team uses a different name.

- **A_ECFP:** Two LightGBM models using ECFP4 fingerprints and strict count-based DEL labels, averaged over seeds and negative samples.
- **B_FCFP:** Two LightGBM models using feature-based Morgan fingerprints and molecular descriptors, slightly broader labels, and moderated library/evidence weighting.
- **C_NEIGHBORS:** 50% percentile-ranked DEL nearest-neighbour evidence, 35% A, and 15% B. Similarity uses strong positive DEL references with a modest penalty for similarity to control-associated compounds. This is a DEL-supervised ensemble, not docking.

Selections maximize each ranking without a scaffold-diversity quota. Duplicate non-chiral parent structures are removed within each set. Overlap between sets is intentional: replacing high-ranked candidates just to make sets disjoint would confound the model comparison. The manifest records overlap and seed stability.

Return each set's hit count, chemical-series count if available, and preferably its submission ID. Results and the latest user-reported quota are tracked in `reports/official_results.json`. The user corrected the quota to **45 remaining before round 2**. Submitting round 2's three sets will leave 42. Reserve at least 10 for later confirmation and final selection.

## Round 1 feedback and round 2

The user reported A=0 hits, B=0 hits, and C (Neighbors)=1 hit in 1 chemical cluster, with p-value 0.05201186181800377. This makes the neighbor approach a provisional lead, with weak evidence; the strong DEL holdout metrics did not translate to hits for the two fingerprint-model sets.

`round02.py` creates three further validation sets without changing round 1's ZIP. It expands the positive reference bank to all eligible strict DEL-positive compounds without NTC supplement overlap. It compares structural neighbors, chemical-feature neighbors, and a control-adjusted blend. The expensive expanded-reference comparison uses a union of the top 20,000 candidates from each of five initial signals (60,022 unique candidates in this run). This prescreen can miss candidates outside that union. Round 1's aggregate feedback selects the model family only: no individual candidate labels are inferred, and prior set membership does not cause forced retention or exclusion. New candidates, overlaps, exact scoring formulas, reference counts, and integrity checks are recorded in round 2's manifest.

## Data and labels

Inputs are the local `PGK2_selection.parquet`, `PGK2_NTC_supplement.parquet`, OpenDEL enumeration files, and validation SMILES. No external activity labels or test labels were used.

- Exact supplied SMILES are deduplicated by summing counts, Stouffer-combining supplied z-scores, and taking maximum historical-hit counts. **The trained classifiers do not use z-scores.**
- Strict positives: target count >= 3, inhibitor count < 10% of target, NTC count = 0, historical hits < 5.
- Broader positives: target count >= 3, inhibitor count <= 25% of target, NTC count <= 10% of target, historical hits < 5.
- Selection negatives are sampled by library from weak target observations or control-associated/promiscuous observations. These remain noisy proxy labels.
- Background enumeration compounds are sampled through random row groups, not by loading the entire library. Absence from selection is treated as uncertain and downweighted. Enumerated libraries with no supplied target-selection rows are excluded from training negatives.
- NTC supplement counts are not added to selection counts because duplicate/overlap semantics are unresolved. NTC-only presence supplies downweighted negative examples; overlap with positives reduces their weight.
- Structures use the largest fragment and non-chiral fingerprints. Conflicting positive/negative parent structures are removed, and duplicate parent structures are collapsed. Tautomer and charge standardization are not performed.

## Validation and limitations

The internal holdout assigns entire non-chiral Bemis–Murcko scaffolds using a fixed hash. Holdout scaffolds never occur in model training or similarity reference sets. Library weights are computed from each training subset alone. Final models are fit to all eligible DEL training data after internal evaluation. Two seeds are averaged per model family.

**Internal metrics use noisy DEL-derived labels; they are not ASMS hit rates.** A good DEL holdout score cannot establish that a model generalizes to the challenge's different chemical space. All 244,328 validation compounds were scored without activity labels. Scores are ranking/confidence outputs and are not calibrated binding probabilities. The first batch is an experiment, not a claim of improvement over earlier submissions.

Other limitations: negatives are sampled rather than exhaustive; scaffold splitting does not remove every form of chemical similarity; normalized DEL counts and missing observations can bias labels; NTC supplement aggregation remains unresolved; chemical analogs and stereoisomers may differ biologically even when represented identically here.

## Reproduce and inspect

The existing repository's Python environment supplies RDKit, Polars, NumPy, SciPy, and scikit-learn. LightGBM 4.6.0 is installed locally in `vendor/`; its OpenMP runtime comes from the existing PyTorch installation. No HPC or cloud inference is used. Scripts use at most four workers each and packed fingerprints to keep memory bounded.

`run_local.sh` runs preparation, featurization, training, similarity, and export. It overwrites this experiment's generated models and round files; preserve a completed round before rerunning. The fixed prepared Parquet files and recorded hashes are the exact replay inputs; regenerating an initial parallel deduplication file may change row ordering and hence a seeded sample. Feature and similarity caches validate their inputs. To reproduce fits on the saved prepared dataset, run `train.py`, `similarity.py`, and `export_round.py` using the environment variables in `run_local.sh`.

`verify.py` checks fingerprint correctness, train/holdout separation, saved-model replay, and actual submission files. `reports/` contains data audits, internal metrics, verification, and provenance. Candidate-detail CSV files are for inspection; only the three 50-line `.txt` files are validation submissions.

## Challenge references

Live pages checked September 11, 2026:

- [Overview and extended September 30 deadline](https://www.synapse.org/Synapse:syn75349604/wiki/641044)
- [Submission formats](https://www.synapse.org/Synapse:syn75349604/wiki/641045)
- [Baseline discussion](https://www.synapse.org/Synapse:syn75349604/discussion/threadId=14646)
- [Unresolved NTC supplement questions](https://www.synapse.org/Synapse:syn75349604/discussion/threadId=14860)

Both blind validation and blind test now end September 30. Validation permits 50 candidates per file; the blind test permits only two submissions. This first batch is for validation only.
