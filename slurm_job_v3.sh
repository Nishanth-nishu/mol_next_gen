#!/bin/bash
#SBATCH --job-name=nextmol_v3
#SBATCH --output=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/slurm_v3_%j.out
#SBATCH --error=/scratch/nishanth.r/nextmol_experiment/mol_next_gen/logs/slurm_v3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --nodelist=gnode118

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Started    : $(date)"
echo "=========================================="

PROJ=/scratch/nishanth.r/nextmol_experiment/mol_next_gen
EXP_DIR="$PROJ/experiments/09-03-2026-Exp-3(stable_full_constraints)"

# Create experiment directory tree
mkdir -p "$EXP_DIR/logs"
mkdir -p "$EXP_DIR/checkpoints"
mkdir -p "$EXP_DIR/plots"
mkdir -p "$EXP_DIR/evaluation"
mkdir -p "$EXP_DIR/molecules"

# Copy SLURM stdout/err into experiment logs dir (symlink)
ln -sf "$PROJ/logs/slurm_v3_${SLURM_JOB_ID}.out" "$EXP_DIR/logs/slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
ln -sf "$PROJ/logs/slurm_v3_${SLURM_JOB_ID}.err" "$EXP_DIR/logs/slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true

# ── Python virtual environment (pip, no conda) ──────────────────────────────
VENV="$PROJ/venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Please create it first from the project root:"
    echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$VENV/bin/activate"

echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

cd $PROJ

python3 training/train_v3.py \
    --data         data/qm9_100k.jsonl  \
    --max_atoms    50 \
    --val_split    0.1 \
    --epochs       300 \
    --batch_size   64 \
    --lr           3e-4 \
    --warmup       5 \
    --hidden_dim   512 \
    --num_layers   10 \
    --timesteps    1000 \
    --num_rbf      20 \
    --time_dim     256 \
    --geometry_weight  1.0 \
    --num_generate 500 \
    --exp_dir      "$EXP_DIR"

echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="
