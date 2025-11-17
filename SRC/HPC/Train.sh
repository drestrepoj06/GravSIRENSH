#!/bin/bash
#SBATCH --job-name=train_sh_siren
#SBATCH --output=/lustre/scratch/WUR/ESG/restr001/GravSIRENSH/Logs/%x_%A_%a.out
#SBATCH --error=/lustre/scratch/WUR/ESG/restr001/GravSIRENSH/Logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4

# --- Activate virtual environment ---
source /lustre/scratch/WUR/ESG/restr001/virtual_envs/gravity_env/bin/activate

# --- Move to training directory ---
cd /lustre/scratch/WUR/ESG/restr001/GravSIRENSH/SRC/Training_test

# --- Diagnostics ---
echo "Running on host: $HOSTNAME"
python - <<'EOF'
import torch, platform
print(f"Python {platform.python_version()}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
EOF

# --- Modes for job array ---
modes=("U" "g_direct" "g_indirect" "U_g_direct" "U_g_indirect")
mode=${modes[$SLURM_ARRAY_TASK_ID]}

echo "Starting Train_SH_siren.py for mode=${mode} at $(date)"

export WANDB_API_KEY="5d0df57b839b8b36807c8c81e3fb6a49225a3dd6"

python Train_SH_siren.py --mode "${mode}"

echo "Finished mode=${mode} at $(date)"
