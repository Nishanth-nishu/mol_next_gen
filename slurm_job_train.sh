#!/bin/bash
#SBATCH --job-name=nextmol_train
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/nextmol_train_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/nextmol_train_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --nodelist=gnode118

# ============================================================================
# Job: ConformerDiffusion Training  (curriculum fix — geometry_weight=0.1)
#
# Submit: sbatch slurm_job_train.sh
# Monitor: tail -f /home2/nishanth.r/nextmol_train_<jobid>.log
# ============================================================================

echo "=========================================="
echo "NExT-Mol  |  Conformer Training"
echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "=========================================="

PROJ=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
cd "$PROJ"
mkdir -p logs
source venv/bin/activate

echo "Python : $(which python)"
python -c 'import torch; print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Configuration ────────────────────────────────────────────────────────────
EPOCHS=200
GEOMETRY_WEIGHT=0.1
DATA="data/qm9_selfies.jsonl"
CHECKPOINT_DIR="checkpoints"
LOG_FILE="$PROJ/logs/nextmol_train_${SLURM_JOB_ID}.log"

mkdir -p "$CHECKPOINT_DIR" plots

echo "Config:"
echo "  Epochs          : $EPOCHS"
echo "  Geometry weight : $GEOMETRY_WEIGHT  (curriculum ramp over first 40%)"
echo "  Data            : $DATA"
echo "  Checkpoints     : $CHECKPOINT_DIR/"
echo "  Log             : $LOG_FILE"
echo ""

# ── Stage 1: Train ───────────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 1: Training  (epochs=$EPOCHS, geometry_weight=$GEOMETRY_WEIGHT)"
echo "============================================================================"

python training/train_conformer.py \
    --geometry_weight $GEOMETRY_WEIGHT \
    --epochs          $EPOCHS \
    --data            $DATA \
    --checkpoint_dir  $CHECKPOINT_DIR

echo ""
echo "Training complete."
echo ""

# ── Stage 2: Generate plots ───────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 2: Generating training dashboard plots"
echo "============================================================================"

if [ -f "$CHECKPOINT_DIR/conformer_best.pt" ]; then
    python visualization/visualize_training.py \
        --checkpoint $CHECKPOINT_DIR/conformer_best.pt \
        --output     plots/
    echo "  Dashboard saved to plots/"
else
    echo "  No checkpoint found — skipping plots."
fi

echo ""
echo "=========================================="
echo "JOB COMPLETED  |  $(date)"
echo "=========================================="
