#!/bin/bash
#SBATCH --job-name=nextmol_large
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/nextmol_large_%j.log
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/nextmol_large_%j.log
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode118

# ============================================================================
# Job: Large Model  —  Deep chemistry learning with scaled-up architecture
#
# Architecture:
#   HIDDEN_DIM  = 1024   (4× base)
#   NUM_LAYERS  = 12     (2× base)
#   EDGE_DIM    = 64     (2× base, proportional to hidden_dim)
#   TIME_DIM    = 256    (2× base)
#   BATCH_SIZE  = 32     (smaller to fit larger model in GPU memory)
#
# Training:
#   EPOCHS      = 1000   (10× base — deep convergence)
#   DDIM_STEPS  = 50     (fast sampling at evaluation time)
#
# Submit: sbatch slurm_job_large.sh
# ============================================================================

echo "=========================================="
echo "NExT-Mol  |  LARGE MODEL Training"
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
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR="outputs/large"
CHECKPOINT_DIR="checkpoints/large"
DATA_DIR="data"

# --- Model architecture (scaled up for deep chemistry learning) ---
HIDDEN_DIM=1024      # 4× base (256) — much richer representations
NUM_LAYERS=12        # 2× base (6)   — more message-passing depth
EDGE_DIM=64          # 2× base (32)  — richer bond representations
TIME_DIM=256         # 2× base (128) — richer timestep embeddings

# --- Training ---
EPOCHS=1000          # 10× base — full convergence
BATCH_SIZE=32        # Smaller to fit large model in VRAM
LR=5e-5              # Lower LR for large model stability
MAX_ATOMS=15

# --- Generation ---
NUM_MOLECULES=1000
DDIM_STEPS=50

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

echo "Architecture:"
echo "  hidden_dim  = $HIDDEN_DIM"
echo "  num_layers  = $NUM_LAYERS"
echo "  edge_dim    = $EDGE_DIM"
echo "  time_dim    = $TIME_DIM"
echo ""
echo "Training:"
echo "  epochs      = $EPOCHS"
echo "  batch_size  = $BATCH_SIZE"
echo "  lr          = $LR"
echo "  max_atoms   = $MAX_ATOMS"
echo ""

# ── Stage 1: Prepare data (skip if already done) ───────────────────────────
if [ ! -f "$DATA_DIR/qm9_selfies.jsonl" ]; then
    echo "============================================================================"
    echo "STAGE 1: Preparing SELFIES data"
    echo "============================================================================"
    python training/prepare_selfies_data.py \
        --input  /scratch/nishanth.r/egnn/data/qm9_100k.jsonl \
        --output $DATA_DIR/qm9_selfies.jsonl \
        --build_vocab \
        --vocab_path $DATA_DIR/selfies_vocab.json
    echo "Data preparation complete."
else
    echo "STAGE 1: Data already prepared — skipping."
fi
echo ""

# ── Stage 2: Train large model ─────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 2: Training large model  ($EPOCHS epochs)"
echo "============================================================================"

python training/train_conformer.py \
    --data       $DATA_DIR/qm9_selfies.jsonl \
    --epochs     $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr         $LR \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --edge_dim   $EDGE_DIM \
    --time_dim   $TIME_DIM \
    --max_atoms  $MAX_ATOMS \
    --save_dir   $CHECKPOINT_DIR

echo "Training complete!"
echo ""

# ── Stage 3: Generate ─────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 3: Generating molecules"
echo "============================================================================"

python generation/generate_nextmol.py \
    --num_molecules   $NUM_MOLECULES \
    --selfies_data    $DATA_DIR/qm9_selfies.jsonl \
    --conformer_model $CHECKPOINT_DIR/conformer_best.pt \
    --output          $OUTPUT_DIR/generated_large.sdf \
    --ddim_steps      $DDIM_STEPS \
    --guidance_scale  1.0

echo "Generation done: $OUTPUT_DIR/generated_large.sdf"
echo ""

# ── Stage 4: Evaluate ─────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 4: Evaluating generated molecules"
echo "============================================================================"

python evaluation/evaluate_validity.py \
    --generated  $OUTPUT_DIR/generated_large.sdf \
    --reference  $DATA_DIR/qm9_selfies.jsonl \
    --output     $OUTPUT_DIR/evaluation_large.json

echo "Evaluation done: $OUTPUT_DIR/evaluation_large.json"
echo ""

# ── Stage 5: Dashboard ────────────────────────────────────────────────────
echo "============================================================================"
echo "STAGE 5: Rendering training dashboard"
echo "============================================================================"

python visualization/train_dashboard.py \
    --log    "$PROJ/logs/nextmol_large_${SLURM_JOB_ID}.log" \
    --output plots/large/

echo "  Dashboard saved to plots/large/"
echo ""

echo "=========================================="
echo "JOB COMPLETED  |  $(date)"
echo "=========================================="
