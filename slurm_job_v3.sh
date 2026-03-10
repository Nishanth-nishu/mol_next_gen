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
VENV=/scratch/nishanth.r/nextmol_venv

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
    echo "Installing requirements (--no-cache-dir into scratch) ..."
    "$VENV/bin/pip" install --no-cache-dir --upgrade pip
    "$VENV/bin/pip" install --no-cache-dir \
        'torch>=2.0' \
        rdkit \
        'selfies>=2.1.0' \
        'transformers>=4.30.0' \
        tqdm pandas numpy matplotlib
else
    echo "Reusing existing venv at $VENV"
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
    --edge_dim     64 \
    --time_dim     256 \
    --geometry_weight  0.1 \
    --num_generate 500 \
    --exp_dir      "$EXP_DIR"

echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="
