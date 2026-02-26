"""
plot_soft_constraints.py — Soft Constraints Experiment Visualization
Experiment: 23-02-2026-Exp-1 (soft_restrictions)

Usage:
    python visualization/plot_soft_constraints.py
    python visualization/plot_soft_constraints.py --eval_json ../evaluation/training_history.json

Reads training_history.json from the experiment evaluation/ dir and produces:
    - plots/loss_curves.png     — total, MSE, geometry losses
    - plots/rmsd_progression.png — RMSD over training
    - plots/constraint_overview.png — per-constraint loss panel (if data present)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


EXP_ROOT = Path(__file__).parent.parent   # experiments/23-02-2026-Exp-1/


def load_history(eval_json: str) -> list:
    with open(eval_json) as f:
        return json.load(f)


def plot_loss_curves(history: list, plots_dir: Path):
    """Main 2x2 loss curve panel."""
    epochs    = [h['epoch']       for h in history]
    train_tot = [h['train_total'] for h in history]
    val_tot   = [h['val_total']   for h in history]
    train_mse = [h['train_mse']   for h in history]
    val_mse   = [h['val_mse']     for h in history]
    train_geo = [h['train_geo']   for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Experiment: 23-02-2026-Exp-1 — Soft Chemical Constraints',
                 fontsize=14, fontweight='bold', y=1.01)

    COLORS = {'train': '#4C72B0', 'val': '#DD8452', 'geo': '#55A868', 'constraint': '#C44E52'}
    ALPHA = 0.85

    # --- Total loss ---
    ax = axes[0, 0]
    ax.plot(epochs, train_tot, color=COLORS['train'], alpha=ALPHA, label='Train total')
    ax.plot(epochs, val_tot,   color=COLORS['val'],   alpha=ALPHA, label='Val total',
            linestyle='--')
    ax.set_title('Total Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(framealpha=0.6)
    ax.grid(True, alpha=0.25)
    ax.set_yscale('log')

    # --- MSE loss ---
    ax = axes[0, 1]
    ax.plot(epochs, train_mse, color=COLORS['train'], alpha=ALPHA, label='Train MSE')
    ax.plot(epochs, val_mse,   color=COLORS['val'],   alpha=ALPHA, label='Val MSE',
            linestyle='--')
    ax.set_title('Diffusion MSE Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend(framealpha=0.6)
    ax.grid(True, alpha=0.25)

    # --- Geometry loss ---
    ax = axes[1, 0]
    ax.plot(epochs, train_geo, color=COLORS['geo'], alpha=ALPHA, label='Train Geo')
    ax.set_title('Geometry Constraint Loss (Total)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(framealpha=0.6)
    ax.grid(True, alpha=0.25)

    # --- Train / Val ratio ---
    ax = axes[1, 1]
    ratio = [v / max(t, 1e-8) for t, v in zip(train_tot, val_tot)]
    ax.plot(epochs, ratio, color='purple', alpha=ALPHA)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='ratio = 1')
    ax.set_title('Val / Train Loss Ratio', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Ratio')
    ax.legend(framealpha=0.6)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = plots_dir / 'loss_curves.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_rmsd(history: list, plots_dir: Path):
    """RMSD progression plot."""
    rmsd_data = [(h['epoch'], h['rmsd_mean'], h['rmsd_std'])
                 for h in history if h.get('rmsd_mean', 0) > 0]
    if not rmsd_data:
        print("  No RMSD data found, skipping rmsd_progression.png")
        return

    epochs, means, stds = zip(*rmsd_data)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, means, 'o-', color='#8172B2', linewidth=2, markersize=5, label='RMSD mean')
    ax.fill_between(epochs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color='#8172B2', label='±1 std')
    ax.set_title('RMSD Progression (Kabsch-aligned) — Exp-1 Soft Constraints',
                 fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSD (Å)')
    ax.legend(framealpha=0.6)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = plots_dir / 'rmsd_progression.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_constraint_overview(history: list, plots_dir: Path):
    """
    Per-constraint loss panel.
    Keys looked for: 'planarity', 'chirality', 'ring_strain' in history dicts.
    Falls back gracefully if keys are absent (older log format).
    """
    keys = ['planarity', 'chirality', 'ring_strain']
    labels = {
        'planarity':   ('Planarity Loss',  '#e76f51'),
        'chirality':   ('Chirality Loss',  '#2a9d8f'),
        'ring_strain': ('Ring Strain Loss','#e9c46a'),
    }
    present = [k for k in keys if any(k in h for h in history)]
    if not present:
        print("  No per-constraint loss data in history; skipping constraint_overview.png")
        return

    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5))
    if len(present) == 1:
        axes = [axes]

    for ax, key in zip(axes, present):
        vals = [h.get(key, 0.0) for h in history]
        label, color = labels[key]
        ax.plot(epochs, vals, color=color, linewidth=2)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.25)

    fig.suptitle('Soft Constraint Loss Breakdown — Exp-1', fontweight='bold')
    plt.tight_layout()
    out = plots_dir / 'constraint_overview.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def main():
    default_eval = str(EXP_ROOT / 'evaluation' / 'training_history.json')
    parser = argparse.ArgumentParser(description='Plot Soft Constraints Experiment Results')
    parser.add_argument('--eval_json', type=str, default=default_eval,
                        help='Path to training_history.json')
    parser.add_argument('--plots_dir', type=str, default=str(EXP_ROOT / 'plots'),
                        help='Directory to save plots')
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.eval_json):
        print(f"History file not found: {args.eval_json}")
        print("Run training first: python training/train_soft_constraints.py")
        return

    print(f"Loading history from {args.eval_json}")
    history = load_history(args.eval_json)
    print(f"Loaded {len(history)} epochs")

    print("Generating plots...")
    plot_loss_curves(history, plots_dir)
    plot_rmsd(history, plots_dir)
    plot_constraint_overview(history, plots_dir)
    print(f"\nAll plots saved to {plots_dir}")


if __name__ == '__main__':
    main()
