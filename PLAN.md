# 🧬 PLAN: Fine-Tuning MAMMAL for DREAM x CACHE Target 2035

This document outlines our strategy for handling the massive amount of DEL data (890M molecules) and building a state-of-the-art model for out-of-distribution (OOD) hit prediction on PGK2 using the multi-modal model **MAMMAL**.

---

## 🎯 The Core Idea & OOD Generalization
Traditional machine learning models (e.g. LightGBM, FNNs trained on Morgan Fingerprints) are highly prone to overfitting on the chemical space of the training libraries. They perform poorly when evaluating unseen scaffolds (OOD), which is the exact scenario in this challenge (testing on the 400K ASMS library).

To solve this, we will train **MAMMAL** using a multi-modal combinatorial formulation:
1. **Protein Target Context**: Feed the target protein sequence (PGK2) to MAMMAL's encoder.
2. **Combinatorial Building Blocks (BBs)**: Instead of just the final compound, we feed the SMILES of the individual building blocks ($BB_1$, $BB_2$, $BB_3$) that make up the molecule. This allows the model to learn localized combinatorial binding patterns.
3. **Whole Molecule SMILES**: Feed the final deprotected enumerated SMILES string.
4. **MAMMAL Multi-Modal Tokenizer**: Merge all of these in a single sequence-to-sequence prompt!

---

## 🛠️ Step-by-Step Implementation Roadmap

### Phase 1: Data Housekeeping & Deduplication
Raw DNA-Encoded Library (DEL) selection files contain duplicates due to **protecting group redundancy** and **codon redundancy**. To avoid data leakage and signal inflation, we must deduplicate the raw data.
- **Deduplication strategy**: Group the selection file by `SMILES`.
- **Aggregation rules**:
  - `*_count` columns (reads) $\rightarrow$ Sum the values across duplicate entries.
  - `*_zscore` columns $\rightarrow$ Combine using **unweighted Stouffer's method**:
    $$Z_{\text{combined}} = \frac{\sum_{i=1}^n Z_i}{\sqrt{n}}$$
- **Tooling**: We use highly optimized **Polars** (adapted from `DREAM_DEL_deduplication`) to process millions of rows in seconds.

### Phase 2: Target Label Engineering (Experimental Score Mapping)
We have three experimental conditions:
- $Z_{\text{PGK2}}$: Target alone.
- $Z_{\text{with\_inhibitor}}$: Target pre-incubated with a known competitive ATP-site inhibitor.
- $Z_{\text{NTC}}$: No Target Control (baseline matrix binders).

To train MAMMAL, we must map these scores to a single classification label or continuous target:
1. **Target Affinity Score (TAS)**:
   $$TAS = Z_{\text{PGK2}} - Z_{\text{NTC}}$$
2. **Orthosteric specificity (ATP binding site)**:
   A compound is a specific ATP-site binder if its enrichment drops when pre-incubated with the inhibitor:
   $$Score = Z_{\text{PGK2}} - \max(Z_{\text{NTC}}, Z_{\text{with\_inhibitor}})$$
3. **Binary Activity Label**:
   A robust classification label can be computed via thresholding:
   $$\text{Active} = (Z_{\text{PGK2}} \ge \theta_{\text{high}}) \land (Z_{\text{NTC}} \le \theta_{\text{low}}) \land (Z_{\text{with\_inhibitor}} \le \theta_{\text{low}})$$

### Phase 3: Combinatorial Building Block Mapping
Every compound ID (e.g. `qDOS11-0012-0045-0210`) represents:
$$\text{Library ID} - \text{BB}_1\text{\_ID} - \text{BB}_2\text{\_ID} - \text{BB}_3\text{\_ID}$$
Using the building block SMILES from `OpenDEL_libraries.zip`:
1. Parse each compound ID.
2. Lookup the chemical structures of $BB_1, BB_2, BB_3$.
3. Concatenate them in the MAMMAL prompt alongside the full compound.

### Phase 4: Prompt Engineering for MAMMAL
The prompt will utilize MAMMAL's multi-modal capabilities:
```text
<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_CLASS><SENTINEL_ID_0>
<MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>[PGK2 Sequence]<SEQUENCE_NATURAL_END>
<@TOKENIZER-TYPE=SMILES>
<MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SEQUENCE_NATURAL_START>[BB1_SMILES].[BB2_SMILES].[BB3_SMILES]<SEQUENCE_NATURAL_END>
<MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SEQUENCE_NATURAL_START>[Compound_SMILES]<SEQUENCE_NATURAL_END>
<EOS>
```
The model's target for fine-tuning will predict the binary classification `<1>` or `<0>` token, or regress on the continuous score.

### Phase 5: High-Performance Training Pipeline
- **Imbalance Mitigation**: DEL data is extremely sparse (mostly inactives). We will build a balanced training dataset by keeping all active hits (e.g., $100\text{k} - 500\text{k}$) and downsampling the negatives/unobserved compounds.
- **Streaming Data Loader**: We'll implement a chunked data loader to feed the PyTorch / Lightning pipeline efficiently.
