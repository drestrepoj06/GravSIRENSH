#!/bin/bash
#SBATCH --job-name=Train_SH_siren.py       # Name of your job
#SBATCH --output=logs/%x_%j.out        # Output file (%x=job name, %j=job ID)
#SBATCH --error=logs/%x_%j.err         # Error file
#SBATCH --time=48:00:00                # Max wall time (hh:mm:ss)
#SBATCH --partition=gpu                # GPU partition name (depends on cluster)
#SBATCH --gres=gpu:1                   # Number of GPUs (e.g., 1 GPU)
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --mem=32G                      # Memory for this job
#SBATCH --mail-type=END,FAIL           # Email notification (optional)
#SBATCH --mail-user=you@wur.nl         # Your email (optional)

# --- Environment setup ---
module purge
module load cuda/12.1  # adjust to your cluster's version
module load anaconda   # or python/3.10 etc.

# Activate your conda or venv
source activate gravity_env

# Go to your project directory
cd ~/projects/GravSIREN/

# --- Run your script ---
echo "Starting training on $HOSTNAME at $(date)"
python train_gravity.py --config configs/train_siren.json
echo "Finished at $(date)"