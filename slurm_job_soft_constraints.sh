#!/bin/bash
#SBATCH --job-name=soft_constraints
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/soft_constraints_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/soft_constraints_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode118

# ============================================================================
# SLURM Job Script for Soft Constraints Experiment
# Experiment: 23-02-2026-Exp-1(soft_restrictions)
# ============================================================================
#
# Submit with: sbatch slurm_job_soft_constraints.sh
#

echo "============================================================================"
echo "Soft Constraints Experiment 23-02-2026-Exp-1 - SLURM Job"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""


PROJ=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
cd "$PROJ"
mkdir -p logs
source venv/bin/activate

echo "Python: $(which python)"
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")'
nvidia-smi
echo ""

EXP_DIR="experiments/23-02-2026-Exp-1(soft_restrictions)"

python training/train_soft_constraints.py \
    --data data/qm9_selfies.jsonl \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --hidden_dim 256 \
    --num_layers 6 \
    --timesteps 1000 \
    --geometry_weight 1.0 \
    --planarity_weight 5.0 \
    --chirality_weight 3.0 \
    --ring_strain_weight 2.0 \
    --exp_dir "${EXP_DIR}" \
    2>&1 | tee "${EXP_DIR}/logs/training_$(date +%Y%m%d_%H%M).log"
echo ""
echo "Job completed at: $(date)"
