#!/bin/bash
# ============================================================================
# run_experiment.sh — Full NExT-Mol Experiment Pipeline
# ============================================================================
#
# This script runs the complete pipeline:
# 1. Prepare SELFIES data from QM9
# 2. Train conformer diffusion model
# 3. Generate molecules
# 4. Evaluate validity
#
# Usage:
#   ./run_experiment.sh [--prepare] [--train] [--generate] [--evaluate]
#

set -e

# ============================================================================
# Environment Setup
# ============================================================================

# Load CUDA module (HPC systems)
module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true

# Set CUDA library paths (adjust if needed)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

INSTALL_DIR="/scratch/nishanth.r/miniconda3"
ENV_NAME="nextmol"

# Activate conda
source "$INSTALL_DIR/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Python: $(which python)"
echo "CUDA visible: $CUDA_VISIBLE_DEVICES"
echo "Testing PyTorch..."
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")' 2>/dev/null || echo "Warning: PyTorch CUDA check failed"
echo ""

# ============================================================================
# Configuration
# ============================================================================
QM9_DATA="/scratch/nishanth.r/egnn/data/qm9_100k.jsonl"  # Your QM9 path
OUTPUT_DIR="outputs"
CHECKPOINT_DIR="checkpoints"
DATA_DIR="data"

# Training hyperparameters
EPOCHS=100
BATCH_SIZE=64
HIDDEN_DIM=256
NUM_LAYERS=6
MAX_ATOMS=15

# Generation parameters
NUM_MOLECULES=1000
DDIM_STEPS=50

# Create directories
mkdir -p $OUTPUT_DIR $CHECKPOINT_DIR $DATA_DIR

# Parse arguments
DO_PREPARE=false
DO_TRAIN=false
DO_GENERATE=false
DO_EVALUATE=false

if [ $# -eq 0 ]; then
    # Run all stages if no arguments
    DO_PREPARE=true
    DO_TRAIN=true
    DO_GENERATE=true
    DO_EVALUATE=true
else
    for arg in "$@"; do
        case $arg in
            --prepare)
                DO_PREPARE=true
                ;;
            --train)
                DO_TRAIN=true
                ;;
            --generate)
                DO_GENERATE=true
                ;;
            --evaluate)
                DO_EVALUATE=true
                ;;
            *)
                echo "Unknown argument: $arg"
                exit 1
                ;;
        esac
    done
fi

echo "============================================================================"
echo "                     NExT-Mol Experiment Pipeline"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  QM9 Data: $QM9_DATA"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Num Layers: $NUM_LAYERS"
echo ""

# ============================================================================
# STAGE 1: Prepare SELFIES Data
# ============================================================================
if $DO_PREPARE; then
    echo "============================================================================"
    echo "STAGE 1: Preparing SELFIES Data"
    echo "============================================================================"
    
    python training/prepare_selfies_data.py \
        --input $QM9_DATA \
        --output $DATA_DIR/qm9_selfies.jsonl \
        --build_vocab \
        --vocab_path $DATA_DIR/selfies_vocab.json
    
    echo ""
    echo "Data preparation complete!"
    echo ""
fi

# ============================================================================
# STAGE 2: Train Conformer Diffusion
# ============================================================================
if $DO_TRAIN; then
    echo "============================================================================"
    echo "STAGE 2: Training Conformer Diffusion Model"
    echo "============================================================================"
    
    python training/train_conformer.py \
        --data $DATA_DIR/qm9_selfies.jsonl \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --num_layers $NUM_LAYERS \
        --max_atoms $MAX_ATOMS \
        --save_dir $CHECKPOINT_DIR
    
    echo ""
    echo "Training complete!"
    echo ""
fi

# ============================================================================
# STAGE 3: Generate Molecules
# ============================================================================
if $DO_GENERATE; then
    echo "============================================================================"
    echo "STAGE 3: Generating Molecules (NExT-Mol Approach)"
    echo "============================================================================"
    
    python generation/generate_nextmol.py \
        --num_molecules $NUM_MOLECULES \
        --selfies_data $DATA_DIR/qm9_selfies.jsonl \
        --conformer_model $CHECKPOINT_DIR/conformer_best.pt \
        --output $OUTPUT_DIR/generated_nextmol.sdf \
        --ddim_steps $DDIM_STEPS
    
    echo ""
    echo "Generation complete!"
    echo ""
fi

# ============================================================================
# STAGE 4: Evaluate Results
# ============================================================================
if $DO_EVALUATE; then
    echo "============================================================================"
    echo "STAGE 4: Evaluating Results"
    echo "============================================================================"
    
    python evaluation/evaluate_validity.py \
        --generated $OUTPUT_DIR/generated_nextmol.sdf \
        --reference $DATA_DIR/qm9_selfies.jsonl \
        --output $OUTPUT_DIR/evaluation_results.json
    
    echo ""
    echo "Evaluation complete!"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "                         Experiment Complete!"
echo "============================================================================"
echo ""
echo "Output files:"
echo "  Data:        $DATA_DIR/qm9_selfies.jsonl"
echo "  Model:       $CHECKPOINT_DIR/conformer_best.pt"
echo "  Molecules:   $OUTPUT_DIR/generated_nextmol.sdf"
echo "  Evaluation:  $OUTPUT_DIR/evaluation_results.json"
echo ""
echo "Key insight: NExT-Mol achieves near-100% validity by pre-validating"
echo "topology (SELFIES) before generating 3D coordinates!"
echo ""
