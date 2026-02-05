#!/bin/bash
#SBATCH --job-name=nextmol
#SBATCH --output=/home2/nishanth.r/nextmol_%j.log
#SBATCH --error=/home2/nishanth.r/nextmol_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --nodelist=gnode118

# ============================================================================
# SLURM Job Script for NExT-Mol Experiment
# ============================================================================
#
# Submit with: sbatch slurm_job.sh
#

echo "============================================================================"
echo "NExT-Mol Experiment - SLURM Job"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

# Activate conda environment
INSTALL_DIR="/scratch/nishanth.r/miniconda3"
ENV_NAME="nextmol"

source "$INSTALL_DIR/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Python: $(which python)"
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")'
nvidia-smi
echo ""

# Change to experiment directory
cd /scratch/nishanth.r/nextmol_experiment

# Run the experiment
bash run_experiment.sh

echo ""
echo "Job completed at: $(date)"

