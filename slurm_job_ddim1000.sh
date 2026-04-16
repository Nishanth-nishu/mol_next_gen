#!/bin/bash
#SBATCH --job-name=nextmol_ddim1000
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/nextmol_ddim1000_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/nextmol_ddim1000_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist=gnode118

# ============================================================================
# Job: DDIM-1000  —  High-quality generation with full 1000 diffusion steps
#
# NOTE: This job assumes a trained checkpoint already exists at:
#       checkpoints/conformer_best.pt
#       If not, run slurm_job.sh first to train the base model.
#
# Submit: sbatch slurm_job_ddim1000.sh
# ============================================================================

echo "=========================================="
echo "NExT-Mol  |  DDIM-1000 Generation"
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

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR="outputs/ddim1000"
CHECKPOINT_DIR="checkpoints"
DATA_DIR="data"
NUM_MOLECULES=1000
DDIM_STEPS=1000          # Full 1000-step diffusion — highest quality

mkdir -p "$OUTPUT_DIR"

echo "Config:"
echo "  DDIM steps     : $DDIM_STEPS  (full diffusion — slowest, best quality)"
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
    --output          $OUTPUT_DIR/generated_ddim1000.sdf \
    --ddim_steps      $DDIM_STEPS \
    --guidance_scale  1.0

echo ""
echo "Generation done. SDF: $OUTPUT_DIR/generated_ddim1000.sdf"
echo ""

# ── Stage 2: Evaluate ─────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 2: Evaluating results"
echo "============================================================================"

python evaluation/evaluate_validity.py \
    --generated  $OUTPUT_DIR/generated_ddim1000.sdf \
    --reference  $DATA_DIR/qm9_selfies.jsonl \
    --output     $OUTPUT_DIR/evaluation_ddim1000.json

echo ""
echo "Evaluation done. Results: $OUTPUT_DIR/evaluation_ddim1000.json"
echo ""

# ── Stage 3: Visualize ────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 3: Rendering training dashboard"
echo "============================================================================"

LOG_FILE=$(ls "$PROJ"/logs/nextmol_*.log 2>/dev/null | sort | tail -1)
if [ -n "$LOG_FILE" ]; then
    python visualization/train_dashboard.py \
        --log    "$LOG_FILE" \
        --output plots/ddim1000/
    echo "  Dashboard saved to plots/ddim1000/"
else
    echo "  No training log found — skipping dashboard."
fi

echo ""
echo "=========================================="
echo "JOB COMPLETED  |  $(date)"
echo "=========================================="
