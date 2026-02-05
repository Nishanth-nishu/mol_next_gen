#!/bin/bash
# ============================================================================
# setup_environment.sh — Install Miniconda and Required Packages
# ============================================================================
#
# This script:
# 1. Downloads and installs Miniconda (if not present)
# 2. Creates a conda environment for NExT-Mol
# 3. Installs all required packages
#
# Usage:
#   bash setup_environment.sh
#
# ============================================================================

set -e

# Configuration
INSTALL_DIR="/scratch/nishanth.r/miniconda3"
ENV_NAME="nextmol"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================================"
echo "           NExT-Mol Environment Setup"
echo "============================================================================"
echo ""
echo "Install directory: $INSTALL_DIR"
echo "Environment name: $ENV_NAME"
echo ""

# ============================================================================
# Step 1: Install Miniconda (if not present or incomplete)
# ============================================================================
# Check if conda.sh exists (proper installation marker)
if [ ! -f "$INSTALL_DIR/etc/profile.d/conda.sh" ]; then
    echo "============================================================================"
    echo "Step 1: Installing/Reinstalling Miniconda..."
    echo "============================================================================"
    
    # Remove incomplete installation if exists
    if [ -d "$INSTALL_DIR" ]; then
        echo "Removing incomplete Miniconda installation..."
        rm -rf "$INSTALL_DIR"
    fi
    
    # Download Miniconda
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    
    echo "Downloading Miniconda..."
    wget -q --show-progress "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"
    
    echo "Installing Miniconda to $INSTALL_DIR..."
    bash "$MINICONDA_INSTALLER" -b -p "$INSTALL_DIR"
    
    # Clean up installer
    rm "$MINICONDA_INSTALLER"
    
    echo "Miniconda installed successfully!"
else
    echo "Step 1: Miniconda already installed at $INSTALL_DIR"
fi

# ============================================================================
# Step 2: Initialize conda
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 2: Initializing conda..."
echo "============================================================================"

# Source conda
source "$INSTALL_DIR/etc/profile.d/conda.sh"
conda init bash 2>/dev/null || true

echo "Conda version: $(conda --version)"

# ============================================================================
# Step 3: Create environment (if not exists)
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 3: Creating conda environment '$ENV_NAME'..."
echo "============================================================================"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Updating..."
    conda activate $ENV_NAME
else
    echo "Creating new environment with Python 3.10..."
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME
fi

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# ============================================================================
# Step 4: Install PyTorch with CUDA
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 4: Installing PyTorch with CUDA..."
echo "============================================================================"

# Install PyTorch (CUDA 11.8 version - adjust if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ============================================================================
# Step 5: Install RDKit
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 5: Installing RDKit..."
echo "============================================================================"

conda install -c conda-forge rdkit -y

echo "RDKit version: $(python -c 'from rdkit import Chem; print(Chem.rdBase.rdkitVersion)')"

# ============================================================================
# Step 6: Install other dependencies
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 6: Installing other dependencies..."
echo "============================================================================"

pip install selfies>=2.1.0
pip install transformers>=4.30.0
pip install tqdm
pip install pandas
pip install numpy
pip install matplotlib
pip install scipy

echo ""
echo "Installed packages:"
pip list | grep -E "(torch|rdkit|selfies|transformers|tqdm|pandas|numpy)"

# ============================================================================
# Step 7: Update run_experiment.sh with correct paths
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 7: Updating run_experiment.sh..."
echo "============================================================================"

# Create updated run script
cat > "$SCRIPT_DIR/run_experiment.sh" << 'SCRIPT_EOF'
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
INSTALL_DIR="/scratch/nishanth.r/miniconda3"
ENV_NAME="nextmol"

# Activate conda
source "$INSTALL_DIR/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ============================================================================
# Configuration
# ============================================================================
QM9_DATA="/scratch/nishanth.r/new_egnn/egnn/data/qm9_optimized.jsonl"  # Your QM9 path
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
SCRIPT_EOF

chmod +x "$SCRIPT_DIR/run_experiment.sh"
echo "Updated run_experiment.sh with correct conda paths"

# ============================================================================
# Done!
# ============================================================================
echo ""
echo "============================================================================"
echo "                    Setup Complete!"
echo "============================================================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  source /scratch/nishanth.r/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate nextmol"
echo ""
echo "To run the experiment:"
echo "  cd /scratch/nishanth.r/nextmol_experiment"
echo "  bash run_experiment.sh"
echo ""
echo "Or for SLURM jobs, use the batch script (see slurm_job.sh)"
echo ""
