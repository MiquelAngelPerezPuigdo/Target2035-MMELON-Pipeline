# 🧬 PLAN: Fine-Tuning MMELON for DREAM x CACHE Target 2035

This document outlines our strategy for handling the massive amount of DEL data (890M molecules) and building a state-of-the-art model for out-of-distribution (OOD) hit prediction on PGK2 using the multi-modal late-fusion model **MMELON**.

---

## 🤯 Why This Looks Impossible (But Isn't)

The library has **~898 million enumerated compounds**. Fine-tuning a molecular transformer on a billion molecules seems computationally insane — *if* each molecule were an independent data point.

It is **not**. A DEL is built combinatorially from a tiny shared set of building blocks:

$$N_{\text{compounds}} = |BB_1| \times |BB_2| \times |BB_3| \approx 10^9, \qquad\text{but}\qquad \text{information} \sim |BB_1| + |BB_2| + |BB_3| \approx \text{a few thousand.}$$

The apparent billion-scale problem is really only a **few-thousand-fragment** problem in disguise. By explicitly encoding the building blocks, MMELON learns the *combinatorial grammar* of binding rather than memorizing 898M strings.

**Consequence:** we do **not** need to discard data to make training feasible. Every compound reinforces signal about the fragments it shares with others, so the design goal of this repo is to **embrace all the data as important** by utilizing our **Late-Fusion Double-Caching Strategy**.

---

## 🎯 The Core Idea & OOD Generalization
Traditional machine learning models (e.g. LightGBM, FNNs trained on Morgan Fingerprints) are highly prone to overfitting on the chemical space of the training libraries. They perform poorly when evaluating unseen scaffolds (OOD), which is the exact scenario in this challenge (testing on the 400K ASMS library).

To solve this, we will train **MMELON** using a decoupled late-fusion combinatorial formulation:
1. **Multi-View Caching**: Pass all physical building blocks through MMELON's pre-trained molecular encoders once to cache their dense $768$-dimensional multi-view representation.
2. **Combinatorial Integration**: For each enumerated DEL compound, look up its component building block embeddings and average them instantly in PyTorch:
   $$E_{\text{compound}} = \text{Mean}(E_{\text{BB}_1}, E_{\text{BB}_2}, E_{\text{BB}_3})$$
3. **Pocket Specificity Classifier**: Train a lightweight PyTorch MLP head on top of these pre-computed embeddings to target competitive, pocket-specific binding.

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

### Phase 2: Target Label Engineering (Pocket Specificity Squash)
We have three experimental conditions:
- $Z_{\text{PGK2}}$: Target alone.
- $Z_{\text{with\_inhibitor}}$: Target pre-incubated with a known competitive ATP-site inhibitor.
- $Z_{\text{NTC}}$: No Target Control (baseline matrix binders).

To train MMELON, we map these competitive metrics into a continuous 0-1 sigmoid target:
$$S = Z_{\text{PGK2}} - \max(Z_{\text{NTC}}, Z_{\text{with\_inhibitor}})$$
Which isolates orthosteric ATP-pocket binders. We then squash this difference via a Sigmoid:
$$\text{Score} = \sigma\left(\frac{S - \beta}{\tau}\right)$$
This scoring module (`scoring.py`) acts as our primary play arena where different parameters should be tried during training.

### Phase 3: Building Block Multi-View Embedding Caching
Instead of running deep neural networks (Graphs, SMILES Transformers, Molecular Images) over 890 million records in real-time, we:
1. Parse physical building block parquet composition tables from OpenDEL.
2. Extract multi-view embeddings for the ~3,000 building blocks exactly once.
3. Cache them to a compressed file (`mmelon_bb_embeddings.npz`) on disk.

### Phase 4: Fast MLP Head Training
During the main SLURM jobs on the Kuma EPFL cluster, we:
1. Bypass all heavy molecular encoders completely.
2. For each DEL sequence, retrieve the building block identifiers, look up their pre-computed embeddings, and average them.
3. Train a Multi-Layer Perceptron (MLP) head in PyTorch. Training throughput reaches **50,000+ samples/second**, completing a full training run in under 4 hours.

### Phase 5: High-Precision Validation & Inference
At inference time, validation molecules are supplied as full SMILES strings with no predefined building blocks:
- **Direct Multi-view Inference**: We pass validation compounds directly into the pre-trained MMELON encoder to extract their exact multi-view embedding and feed them into our trained MLP head.
- **Robust Generalization**: This guarantees zero tokenizer mismatches, zero mapping hacks, and zero manual tuning at validation time—ensuring peak performance!

