#!/bin/bash
# DREAM x CACHE Target 2035: Euler HPC One-Click Deployment Script

echo "=============================================================="
echo "🚀 Starting automated deployment of MMELON Pipeline to Euler..."
echo "=============================================================="

# Define Euler connection details
USER_HOST="perezmi@euler.ethz.ch"
REMOTE_DIR="~/Target2035-MMELON-Pipeline"

echo "📂 Step 1/3: Copying updated pipeline files to Euler..."
scp submit_euler.sh preprocess_del.py run_pipeline.py run_validation.py scoring.py test_pipeline.py ${USER_HOST}:${REMOTE_DIR}/

if [ $? -ne 0 ]; then
  echo "❌ Error copying files. Make sure you entered the correct password."
  exit 1
fi

echo "✔ Files successfully synced!"

echo "🧹 Step 2/3: Cleaning up old virtual environment on Euler compute nodes..."
ssh ${USER_HOST} "rm -rf ${REMOTE_DIR}/venv_euler && mkdir -p ${REMOTE_DIR}/logs"

if [ $? -ne 0 ]; then
  echo "❌ Error cleaning remote directory."
  exit 1
fi

echo "✔ Remote environment cleared!"

echo "🚀 Step 3/3: Submitting Slurm training job to Euler batch queue..."
ssh ${USER_HOST} "cd ${REMOTE_DIR} && sbatch submit_euler.sh"

if [ $? -ne 0 ]; then
  echo "❌ Error submitting batch job."
  exit 1
fi

echo "=============================================================="
echo "🎉 DEDEPLOYMENT SUCCESSFUL! Your job is now queued on Euler."
echo "=============================================================="
echo "To check your job status in real-time, run:"
echo "  ssh ${USER_HOST} \"squeue --me\""
echo "=============================================================="
