#!/bin/bash
#SBATCH --job-name=nextmol
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/nextmol_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/nextmol_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
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


PROJ=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
cd "$PROJ"
mkdir -p logs
source venv/bin/activate

echo "Python: $(which python)"
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")'
nvidia-smi
echo ""

# Change to experiment directory

# Run the experiment
# Kill current run, then:
python training/train_conformer.py \
    --geometry_weight 0.1 \
    --epochs 200 \
    --data data/qm9_selfies.jsonl \
    2>&1 | tee training_$(date +%Y%m%d_%H%M).log
echo ""
echo "Job completed at: $(date)"

