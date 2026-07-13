# 🧬 DREAM x CACHE Target 2035 — MAMMAL Preprocessing & Target Engineering

Welcome to the unified repository for the **DREAM x CACHE Target 2035 Drug Discovery Challenge**.

This repository is designed for **collaborative development**:
- **Workflow & Data Engineering (Local/Dev)**: Validating schemas, cleaning raw DEL datasets, performing Polars-based structural deduplication, and mapping/modulating customizable target scores.
- **Model Fine-Tuning & Inference (HPC)**: Powering ultra-large scale training loops with a memory-efficient, on-the-fly tokenized PyTorch `Dataset` for **MAMMAL** (Molecular Aligned Multi-Modal Architecture and Language).

---

## 📁 Repository Map

```
CACHE Challenge/
├── README.md                      # This master guide
├── PLAN.md                        # Master strategy, math formulations & roadmap
├── scoring.py                     # Custom scoring engine mapping 3 conditions to 0-1
├── preprocess_del.py              # Polars deduplication & PyTorch Dataset class
├── run_pipeline.py                # Command-Line executable pipeline entrypoint
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
and squashes them to a $[0, 1]$ target range using a sigmoid with adjustable temperature ($\tau$) and bias ($\beta$):
```bash
python run_pipeline.py \
  --selection-file PGK2_selection.parquet \
  --scoring-scheme tier2 \
  --sample-size 300000 \
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

## 💡 How to Feed the Preprocessed Data into MAMMAL Training

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
