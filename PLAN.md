# 🧬 PLAN: Fine-Tuning MAMMAL for DREAM x CACHE Target 2035

This document outlines our strategy for handling the massive amount of DEL data (890M molecules) and building a state-of-the-art model for out-of-distribution (OOD) hit prediction on PGK2 using the multi-modal model **MAMMAL**.

---

## 🤯 Why This Looks Impossible (But Isn't)

The library has **~898 million enumerated compounds**. Fine-tuning a 458M-parameter transformer on a billion molecules seems computationally insane — *if* each molecule were an independent data point.

It is **not**. A DEL is built combinatorially from a tiny shared set of building blocks:

$$N_{\text{compounds}} = |BB_1| \times |BB_2| \times |BB_3| \approx 10^9, \qquad\text{but}\qquad \text{information} \sim |BB_1| + |BB_2| + |BB_3| \approx \text{a few thousand.}$$

The apparent billion-scale problem is really only a **few-thousand-fragment** problem in disguise. By explicitly encoding the building blocks, MAMMAL learns the *combinatorial grammar* of binding rather than memorizing 898M strings.

**Consequence:** we do **not** need to discard data to make training feasible. Every compound reinforces signal about the fragments it shares with others, so the design goal of this repo is to **embrace all the data as important**. Downsampling (Phase 5) is a purely optional speed/convenience knob — set `sample_size` to `0`/`None` to train on the full deduplicated library.

---

## 🎯 The Core Idea & OOD Generalization
Traditional machine learning models (e.g. LightGBM, FNNs trained on Morgan Fingerprints) are highly prone to overfitting on the chemical space of the training libraries. They perform poorly when evaluating unseen scaffolds (OOD), which is the exact scenario in this challenge (testing on the 400K ASMS library).

To solve this, we will train **MAMMAL** using a multi-modal combinatorial formulation:
1. **Protein Target Context**: Feed the target protein sequence (PGK2) to MAMMAL's encoder.
2. **Combinatorial Building Blocks (BBs)**: Instead of just the final compound, we feed the SMILES of the individual building blocks ($BB_1$, $BB_2$, $BB_3$) that make up the molecule. This allows the model to learn localized combinatorial binding patterns — the key to why a "billion-molecule" fine-tune is actually tractable.
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
- **Embrace all data (default philosophy)**: Thanks to the combinatorial encoding, the full deduplicated library is tractable — every compound reinforces shared building-block signal. Set `sample_size=0`/`None` to train on everything.
- **Imbalance Mitigation (optional knob)**: DEL data is extremely sparse (mostly inactives). For faster iteration you *may* keep all active hits and downsample negatives/unobserved compounds via `sample_size`, but this is a convenience, **not** a requirement of the method.
- **Streaming Data Loader**: We'll implement a chunked data loader to feed the PyTorch / Lightning pipeline efficiently, so even the full library never has to sit in memory at once.

### Phase 6: Compact Combinatorial & Validation Reverse-Engineering
To optimize data density and peak model performance, we utilize the **Compact Combinatorial Mode**:
1. **Redundancy Reduction**: We only feed the protein sequence and the building block SMILES ($BB_1, BB_2, BB_3$) once. Instead of the massive, redundant full compound SMILES string, we feed the short, unique combinatorial building block identifier codes (e.g. `0012-0045-0210`).
2. **Reverse Engineering Validation Compounds**: Because validation compounds are only supplied as full synthesized SMILES strings, we run an on-the-fly chemical reverse-engineering module at inference time. Using **RDKit**, we check for exact substructure overlap with our library's building blocks, then apply a custom asymmetric **Tversky similarity** ($\alpha=0.0$, $\beta=1.0$) comparison on Morgan fingerprints to resolve each validation SMILES back to the closest Cycle 1, Cycle 2, and Cycle 3 building block IDs and structures. This reconstructs the combinatorial prompt perfectly so MAMMAL evaluates them using its specialized combinatorial rules.

