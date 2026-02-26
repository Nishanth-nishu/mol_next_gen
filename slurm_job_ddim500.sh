#!/bin/bash
#SBATCH --job-name=nextmol_ddim500
#SBATCH --output=/home2/nishanth.r/nextmol_ddim500_%j.log
#SBATCH --error=/home2/nishanth.r/nextmol_ddim500_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --nodelist=gnode118

# ============================================================================
# Job: DDIM-500  —  Balanced quality/speed with 500 diffusion steps
#
# NOTE: This job assumes a trained checkpoint already exists at:
#       checkpoints/conformer_best.pt
#       If not, run slurm_job.sh first to train the base model.
#
# Submit: sbatch slurm_job_ddim500.sh
# ============================================================================

echo "=========================================="
echo "NExT-Mol  |  DDIM-500 Generation"
echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "=========================================="

cd /scratch/nishanth.r/mol_next_gen
source venv/bin/activate

echo "Python : $(which python)"
python -c 'import torch; print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR="outputs/ddim500"
CHECKPOINT_DIR="checkpoints"
DATA_DIR="data"
NUM_MOLECULES=1000
DDIM_STEPS=500           # 500-step — quality/speed tradeoff

mkdir -p "$OUTPUT_DIR"

echo "Config:"
echo "  DDIM steps     : $DDIM_STEPS  (balanced quality/speed)"
echo "  Num molecules  : $NUM_MOLECULES"
echo "  Checkpoint     : $CHECKPOINT_DIR/conformer_best.pt"
echo "  Output dir     : $OUTPUT_DIR"
echo ""

# ── Stage 1: Generate ─────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 1: Generating molecules  (DDIM steps = $DDIM_STEPS)"
echo "============================================================================"

python generation/generate_nextmol.py \
    --num_molecules   $NUM_MOLECULES \
    --selfies_data    $DATA_DIR/qm9_selfies.jsonl \
    --conformer_model $CHECKPOINT_DIR/conformer_best.pt \
    --output          $OUTPUT_DIR/generated_ddim500.sdf \
    --ddim_steps      $DDIM_STEPS \
    --guidance_scale  1.0

echo ""
echo "Generation done. SDF: $OUTPUT_DIR/generated_ddim500.sdf"
echo ""

# ── Stage 2: Evaluate ─────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 2: Evaluating results"
echo "============================================================================"

python evaluation/evaluate_validity.py \
    --generated  $OUTPUT_DIR/generated_ddim500.sdf \
    --reference  $DATA_DIR/qm9_selfies.jsonl \
    --output     $OUTPUT_DIR/evaluation_ddim500.json

echo ""
echo "Evaluation done. Results: $OUTPUT_DIR/evaluation_ddim500.json"
echo ""

# ── Stage 3: Visualize ────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 3: Rendering training dashboard"
echo "============================================================================"

LOG_FILE=$(ls /home2/nishanth.r/nextmol_*.log 2>/dev/null | sort | tail -1)
if [ -n "$LOG_FILE" ]; then
    python visualization/train_dashboard.py \
        --log    "$LOG_FILE" \
        --output plots/ddim500/
    echo "  Dashboard saved to plots/ddim500/"
else
    echo "  No training log found — skipping dashboard."
fi

echo ""
echo "=========================================="
echo "JOB COMPLETED  |  $(date)"
echo "=========================================="
