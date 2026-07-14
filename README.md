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

Create and configure a unified environment (e.g. named `mmelon311`) with all PyTorch, PyG, and MMELON dependencies.

```bash
# Create and activate a conda environment
conda create -n mmelon311 python=3.11 -y
conda activate mmelon311

# Install PyTorch with your required CUDA runtime (e.g. CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0

# Install PyTorch Geometric dependencies
pip install -f https://data.pyg.org/whl/torch-2.1.0+cu121.html "pyg_lib==0.4.0+pt21cu121" "torch_scatter==2.1.2+pt21cu121" "torch_cluster==1.6.3+pt21cu121" "torch_spline_conv==1.2.2+pt21cu121"

# Install Polars, Pandas, and other dependencies
pip install polars pandas pyarrow fastparquet scikit-learn tqdm rdkit notebook ipykernel

# Install the pre-trained MMELON multi-view framework
pip install git+https://github.com/jmorrone/biomed-multi-view.git
```

---

## 💾 Step 2: Download the Databases (HPC / Production)

To run the pipeline on the full dataset, download the target database files from [AIRCHECK Platform](https://aircheck.ai/challenges) and place them in the repository root directory:

1. **`PGK2_selection.parquet`**: The raw selection file containing sequence read counts and Z-scores across PGK2, PGK2 + Inhibitor, and NTC (approx. 7.7M rows).
2. **`OpenDEL_libraries.zip`**: Contains sub-library composition directories. Unzip it so that the building block tables are available at:
   `OpenDEL-libraries/building_blocks/*.parquet`

---

## 🚀 Step 3: Run the Pipeline & Caching

The unified script `run_pipeline.py` cleans selection data, extracts pre-computed MMELON embeddings for all physical building blocks, and trains a highly scalable MLP classification head.

```bash
# Train MLP head over Sigmoid Specificity Targets on all deduplicated sequences
python run_pipeline.py \
  --selection-file PGK2_selection.parquet \
  --scoring-scheme tier2 \
  --score-threshold 0.5 \
  --sample-size 0 \
  --output-dir processed_data
```

---

## 🔬 Local Validation Test Suite

If you make modifications to the scoring formulas or pipeline structures, you can run our automated pipeline test suite **locally without downloading any databases**:

```bash
# Run local mock tests (verifies Polar deduplication, RDKit reverse-engineering, caching & training)
python test_pipeline.py
```

---

## � Step 4: Run Inference & Prepare Submission

Once your prediction head is trained, you can use the unified validation script `run_validation.py` to run feedforward inference on CACHE validation or test split CSV/Parquets. It automatically ranks candidates, tags the top 50, and generates the exact required challenge submission outputs inside a `/submissions/` directory:

1. **Validation Split submission (`Team_MMELON_submission_validation.txt`)**: A list of exactly 50 CatalogIDs (one per line).
2. **Test Split submission (`Team_MMELON_submission_test.csv`)**: A formatted 3-column CSV (`CatalogID`, `Sel_50`, `Score`).

### Running validation & inference:
```bash
python run_validation.py \
  --model-file processed_data/mmelon_mlp_head.pt \
  --validation-file PGK2_CACHE_Val_Test_Set.csv \
  --output-dir submissions
```

### Running validation with automated local evaluation (if local labels are available):
If you have a local gold standard CSV containing `RandomID`, `Label`, and `Cluster`, you can supply it using `--gold-file` to automatically calculate `ROC-AUC`, `PR-AUC`, `Hit count @50`, `Unique clusters hit`, and statistical `Poisson-Binomial p-value`:
```bash
python run_validation.py \
  --model-file processed_data/mmelon_mlp_head.pt \
  --validation-file PGK2_CACHE_Val_Test_Set.csv \
  --gold-file PGK2_Gold_Standard.csv \
  --output-dir submissions
```

---

## 💡 How to Load Preprocessed Data into Custom Training Loops

Once `run_pipeline.py` has run and generated the pre-computed building block embeddings, loading the dataset is trivial:

```python
from preprocess_del import CombinatorialMMELONDataset
from torch.utils.data import DataLoader

# Instantiate fast combinatorial dataset
train_dataset = CombinatorialMMELONDataset(
    selection_parquet_path="processed_data/PGK2_selection_deduplicated.parquet",
    bb_embeddings_path="processed_data/mmelon_bb_embeddings.npz",
    scoring_scheme="tier2",
    score_threshold_labeling=0.5,
)

# Feed to high-throughput DataLoader (batch size 4096 is fully feasible!)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=4)
```
