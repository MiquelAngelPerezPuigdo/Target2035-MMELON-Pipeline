#!/bin/bash
#SBATCH --job-name=mmelon_pipeline
#SBATCH --output=logs/mmelon_pipeline_%j.out
#SBATCH --error=logs/mmelon_pipeline_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048
# By default, Euler GPU nodes are only accessible if you belong to a shareholder group.
# If you are using the ETH-wide "public" share, GPUs are not available.
# Since bcm_binary trains in just seconds on CPU, we disable --gpus to guarantee compatibility.
# (If you belong to a shareholder group with GPUs, you can add #SBATCH --gpus=1 and #SBATCH -A <your_share_group> here)

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules on Euler to ensure python3 is available
module load stack/2024-06 python/3.11.6

# Euler compute nodes block external internet access by default.
# We must load 'eth_proxy' to allow pip to download and install packages over the internet!
module load eth_proxy

# Activate virtual environment or create it on Euler if it doesn't exist or is corrupted.
if [ ! -d "venv_euler" ] || [ ! -f "venv_euler/bin/pip" ]; then
  echo "Creating fresh Python virtual environment on Euler..."
  rm -rf venv_euler
  python3 -m venv venv_euler
  source venv_euler/bin/activate
  python3 -m ensurepip --upgrade || true
  pip install --upgrade pip wheel setuptools
  pip install numpy pandas polars torch rdkit pyarrow fastparquet tqdm scikit-learn
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
  pip install git+https://github.com/jmorrone/biomed-multi-view.git
else
  source venv_euler/bin/activate
fi

# Step 1: Execute the unified pipeline training
# Running with bcm_hybrid (Strictly-filtered high-confidence actives + Soft-Sigmoid continuous relative affinity)
# Set --sample-size to 0 to train over the FULL deduplicated DEL dataset (no downsampling!).
python3 run_pipeline.py \
  --selection-file PGK2_selection.parquet \
  --bb-glob "OpeDELLibrary/BBids_SMILES/*.csv" \
  --scoring-scheme bcm_hybrid \
  --score-threshold 0.5 \
  --sample-size 0 \
  --output-dir processed_data

# Step 2: Run inference & validation to prepare the final challenge submission files
# This uses the trained MLP model on the CACHE validation/test datasets.
# We run it on the validation split file first to generate the validation output.
python3 run_validation.py \
  --model-file processed_data/mmelon_mlp_head.pt \
  --validation-file Val-Test-set/PGK2_Validation_split.csv \
  --bb-glob "OpeDELLibrary/BBids_SMILES/*.csv" \
  --output-dir submissions

