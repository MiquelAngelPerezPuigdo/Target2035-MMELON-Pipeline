# 🧬 DREAM x CACHE Target 2035 — MMELON Preprocessing & Target Engineering

Welcome to the unified repository for the **DREAM x CACHE Target 2035 Drug Discovery Challenge**.

This repository is designed for **collaborative development**:
- **Workflow & Data Engineering (Local/Dev)**: Validating schemas, cleaning raw DEL datasets, performing Polars-based structural deduplication, and mapping/modulating customizable target scores.
- **Model Fine-Tuning & Inference (HPC)**: Powering ultra-large scale training loops with a memory-efficient, on-the-fly tokenized PyTorch `Dataset` for **MMELON** (Multi-view Molecular Embedding with Late Fusion).

---

## 🤯 The Big (Seemingly Crazy) Idea

At first glance, this project looks **impossible**. The BCM OpenDEL library contains **~898 million enumerated compounds**. Fine-tuning a molecular transformer like MMELON on nearly a billion molecules sounds absurd — the raw compute, memory, and I/O would be prohibitive if every molecule were a truly independent data point.

**But it isn't absurd — because of the combinatorial nature of a DNA-Encoded Library.**

A DEL is not 898M independent molecules. It is a small set of physical **building blocks (BBs)** combined in 2–3 reaction cycles:

$$N_{\text{compounds}} = |BB_1| \times |BB_2| \times |BB_3| \quad\text{(e.g. } 1000 \times 1000 \times 1000 = 10^9\text{)}$$

The *true* information content of the library scales with the **sum** of the building blocks ($|BB_1| + |BB_2| + |BB_3| \approx$ a few thousand), **not** their product (a billion). Every one of the 898M "molecules" is just a re-combination of a tiny shared vocabulary of fragments.

### Why this changes everything
- **Feed the model what actually varies.** By explicitly encoding the individual building blocks ($BB_1, BB_2, BB_3$) alongside the assembled compound, MMELON learns the *combinatorial grammar* of binding — how fragments contribute to activity — rather than memorizing a billion unique strings.
- **The effective learning problem is small.** The model only needs to internalize how a few thousand fragments interact, so the giant dataset collapses into a tractable fine-tuning task. The scale is an illusion created by combinatorics.
- **This makes it realistic to embrace *all* the data as important.** Because every compound reinforces signal about its shared building blocks, we don't *have* to throw data away. Downsampling (see below) is offered as a convenience/speed knob — **not a requirement**. The philosophy of this repo is that all 898M enumerated compounds carry usable enrichment signal, and the combinatorial encoding lets the model absorb it efficiently.
- **It naturally targets out-of-distribution (OOD) performance.** A model that understands fragment-level combinatorial rules generalizes to *unseen* recombinations and scaffolds — precisely what the challenge's OOD ASMS test set demands, and precisely where fingerprint baselines fail.
- **Validation Reverse-Engineering:** Since validation molecules are given as full SMILES strings, our pipeline reverse-engineers them using RDKit substructure filters and asymmetric **Tversky Similarity matching** ($\alpha=0.0$, $\beta=1.0$) back to the closest matching physical library building blocks. This translates validation compounds back into the combinatorial language of the fine-tuned MMELON model.

> **In one sentence:** it looks impossible because of the ~898M compound count, but the library's combinatorial structure means the real problem is only a few thousand building blocks wide — so we can (and do) treat the entire dataset as valuable rather than discarding it.

---

## 📁 Repository Map

```
CACHE Challenge/
├── README.md                      # This master guide
├── PLAN.md                        # Master strategy, math formulations & roadmap
├── scoring.py                     # Custom scoring engine mapping 3 conditions to 0-1
├── preprocess_del.py              # Polars deduplication & PyTorch Dataset class
├── run_pipeline.py                # Command-Line executable pipeline entrypoint
├── run_validation.py              # Unified executable validation & submission entrypoint
├── pgk2_sequence.fasta            # Human PGK2 FASTA protein sequence (P07205)
├── test_pipeline.py               # Automated local validation test suite
└── Target2035_Aircheck_Utils/     # Official challenge utility repo (fingerprints/evaluation)
```

---

## 🛠️ Step 1: Environment Setup

Create and configure a unified environment (e.g. named `mammal_env`) with all PyTorch, Polars, and MAMMAL dependencies.

```bash
# Create and activate a conda environment
conda create -n mammal_env python=3.10 -y
conda activate mammal_env

# Install PyTorch with your required CUDA runtime (e.g. CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install Polars, Pandas, and HuggingFace dependencies
pip install polars pandas pyarrow fastparquet scikit-learn tqdm

# Install the IBM Biomed Multi-Alignment package (MAMMAL)
pip install biomed-multi-alignment
```

---

## 💾 Step 2: Download the Databases (HPC / Production)

To run the pipeline on the full dataset, download the target database files from [AIRCHECK Platform](https://aircheck.ai/challenges) and place them in the repository root directory:

1. **`PGK2_selection.parquet`**: The raw selection file containing sequence read counts and Z-scores across PGK2, PGK2 + Inhibitor, and NTC (approx. 7.7M rows).
2. **`OpenDEL_libraries.zip`**: Contains sub-library composition directories. Unzip it so that the building block tables are available at:
   `OpenDEL-libraries/building_blocks/*.parquet`

---

## 🚀 Step 3: Run the Pipeline

The unified script `run_pipeline.py` provides a Command-Line Interface (CLI) to clean, deduplicate, engineering labels, and inspect PyTorch Dataset dimensions.

```bash
# Display all configurable CLI options
python run_pipeline.py --help
```

### Example Commands:

#### 1. Train under Tier 1 (Conservative Binary Hard Labeling)
Assigns a strict `1` or `0` label using conservative thresholds:
```bash
python run_pipeline.py \
  --selection-file PGK2_selection.parquet \
  --scoring-scheme tier1 \
  --score-threshold 0.5 \
  --sample-size 250000 \
  --output-dir processed_data
```

#### 2. Train under Tier 2 (Continuous Soft Sigmoid Specificity Targets) — *Recommended*
Computes pocket-specificity scores:
$$S = Z_{\text{PGK2}} - \max(Z_{\text{NTC}}, Z_{\text{inhibitor}})$$
and squashes them to a $[0, 1]$ target range using a sigmoid with adjustable temperature ($\tau$) and bias ($\beta$). You can optionally add `--compact` to utilize the compact combinatorial prompt format:
```bash
python run_pipeline.py \
  --selection-file PGK2_selection.parquet \
  --scoring-scheme tier2 \
  --sample-size 300000 \
  --compact \
  --output-dir processed_data
```

#### 3. Train under Tier 3 (Bayesian Read Count Log-Ratios)
Models enrichment directly from sequencing reads, using pseudo-counts to filter out low-read count noise:
```bash
python run_pipeline.py \
  --selection-file PGK2_selection.parquet \
  --scoring-scheme tier3 \
  --sample-size 250000 \
  --output-dir processed_data
```

---

## 🔬 Local Validation Test Suite

If you make modifications to the scoring formulas or prompt sequences and want to verify they won't break the HPC training loop, you can run our automated pipeline test suite **locally without downloading any databases**:

```bash
# Run local mock tests (creates synthetic parquets & verifies full MAMMAL tokenization)
python test_pipeline.py
```

---

## � Step 4: Run Inference & Prepare Submission

Once your model has been fine-tuned, you can use the unified validation script `run_validation.py` to run feedforward inference on CACHE validation or test split CSV/Parquets. It automatically ranks candidates, tags the top 50, and generates the exact required challenge submission outputs inside a `/submissions/` directory:

1. **Validation Split submission (`Team_MAMMAL_submission_validation.txt`)**: A list of exactly 50 CatalogIDs (one per line).
2. **Test Split submission (`Team_MAMMAL_submission_test.csv`)**: A formatted 3-column CSV (`CatalogID`, `Sel_50`, `Score`).

### Running validation & inference:
If you fine-tuned your model in `--compact` mode, you should also pass `--compact` during validation so the full smiles are reverse-engineered into the correct building-block representation on-the-fly:
```bash
python run_validation.py \
  --model-dir /path/to/fine-tuned-checkpoint/ \
  --validation-file PGK2_CACHE_Val_Test_Set.csv \
  --fasta-file pgk2_sequence.fasta \
  --compact \
  --bb-glob "OpenDEL-libraries/building_blocks/*.parquet" \
  --output-dir submissions
```

### Running validation with automated local evaluation (if local labels are available):
If you have a local gold standard CSV containing `RandomID`, `Label`, and `Cluster`, you can supply it using `--gold-file` to automatically calculate `ROC-AUC`, `PR-AUC`, `Hit count @50`, `Unique clusters hit`, and statistical `Poisson-Binomial p-value`:
```bash
python run_validation.py \
  --model-dir /path/to/fine-tuned-checkpoint/ \
  --validation-file PGK2_CACHE_Val_Test_Set.csv \
  --gold-file PGK2_Gold_Standard.csv \
  --output-dir submissions
```

---

## �💡 How to Feed the Preprocessed Data into MAMMAL Training

Once `run_pipeline.py` has run and generated the preprocessed parquets, your collaborator can load the PyTorch Dataset directly in their training script:

```python
from preprocess_del import BuildingBlockMapper, LargeMAMMALDataset, load_pgk2_sequence
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from torch.utils.data import DataLoader

# 1. Initialize Lookups and Tokenizer
bb_mapper = BuildingBlockMapper(bb_files_glob="OpenDEL-libraries/building_blocks/*.parquet")
tokenizer_op = ModularTokenizerOp.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m")
pgk2_sequence = load_pgk2_sequence()

# 2. Instantiate large-scale train dataset
train_dataset = LargeMAMMALDataset(
    selection_parquet_path="processed_data/PGK2_selection_deduplicated.parquet",
    bb_mapper=bb_mapper,
    protein_sequence=pgk2_sequence,
    tokenizer_op=tokenizer_op,
    scoring_scheme="tier2",            # Try "tier1", "tier2", or "tier3"
    score_threshold_labeling=0.5,      # Decision threshold
    sample_size=300000,                # Downsample inactives to keep loop balanced and fast
)

# 3. Feed to standard DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
```
