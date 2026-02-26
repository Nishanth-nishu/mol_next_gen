"""
visualize_training.py — Training Metrics Visualizer for mol_next_gen

Parses training logs OR checkpoint history files and generates plots:
  - Train / Val Loss curves
  - RMSD progression (every 5 epochs)
  - 3D Validity rates (fully valid / bonds / clash-free, every 10 epochs)
  - Bond length error
  - Learning rate schedule

Usage:
    # From SLURM log:
    python visualization/visualize_training.py --log jepa_train_<JOBID>.log --output plots/

    # From checkpoint:
    python visualization/visualize_training.py --checkpoint checkpoints/conformer_best.pt --output plots/
"""

import re
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
PANEL_BG  = '#161b22'
BORDER    = '#30363d'
TEXT_MAIN = '#e6edf3'
TEXT_DIM  = '#8b949e'

COLORS = {
    'train_loss':  '#58a6ff',
    'val_loss':    '#f78166',
    'train_mse':   '#79c0ff',
    'val_mse':     '#ffa198',
    'train_geo':   '#d2a8ff',
    'rmsd':        '#7ee787',
    'fully_valid': '#3fb950',
    'bond_valid':  '#e3b341',
    'clash_free':  '#58a6ff',
    'bond_error':  '#f78166',
    'lr':          '#79c0ff',
    'accent':      '#ffa657',
}

def _style(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT_MAIN)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, alpha=0.5, linewidth=0.7)


# ─────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────

def parse_training_log(log_path: str) -> dict:
    """
    Parse train_conformer.py output log.

    Handles both old format:
        Epoch N: train_loss=X.XXXX, val_loss=Y.YYYY
    And new format (with component breakdown):
        Epoch N: train=X (mse=A geo=B), val=Y (mse=C)
    Plus:
        → 3D Validity: fully_valid=X%, bonds=Y%, no_clash=Z%, bond_err=W Å
    """
    m = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_mse': [], 'train_geo': [], 'val_mse': [],
        'rmsd_epochs': [], 'rmsd_mean': [], 'rmsd_std': [],
        'validity_epochs': [], 'fully_valid_rate': [],
        'bond_valid_rate': [], 'clash_free_rate': [], 'bond_error': [],
        'lr_epochs': [], 'lr': [],
    }

    # Old format: Epoch N: train_loss=X, val_loss=Y
    pat_old   = re.compile(r'Epoch\s+(\d+):\s+train_loss=([0-9.]+),\s+val_loss=([0-9.]+)')
    # New format: Epoch N: train=X (mse=A geo=B), val=Y (mse=C)
    pat_new   = re.compile(
        r'Epoch\s+(\d+):\s+train=([0-9.]+)\s*\(mse=([0-9.]+)\s+geo=([0-9.]+)\),\s*val=([0-9.]+)\s*\(mse=([0-9.]+)\)'
    )
    pat_rmsd     = re.compile(r'rmsd=([0-9.]+).([0-9.]+)')
    pat_validity = re.compile(
        r'fully_valid=([0-9.]+)%.*?bonds=([0-9.]+)%.*?no_clash=([0-9.]+)%.*?bond_err=([0-9.]+)'
    )
    pat_lr = re.compile(r'\blr[=\s]+([0-9.e+\-]+)', re.IGNORECASE)

    current_epoch = None
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            # Try new format first
            e = pat_new.search(line)
            if e:
                current_epoch = int(e.group(1))
                m['epoch'].append(current_epoch)
                m['train_loss'].append(float(e.group(2)))
                m['train_mse'].append(float(e.group(3)))
                m['train_geo'].append(float(e.group(4)))
                m['val_loss'].append(float(e.group(5)))
                m['val_mse'].append(float(e.group(6)))
            else:
                e = pat_old.search(line)
                if e:
                    current_epoch = int(e.group(1))
                    m['epoch'].append(current_epoch)
                    m['train_loss'].append(float(e.group(2)))
                    m['val_loss'].append(float(e.group(3)))
                    m['train_mse'].append(float('nan'))
                    m['train_geo'].append(float('nan'))
                    m['val_mse'].append(float('nan'))

            if current_epoch is not None:
                r = pat_rmsd.search(line)
                if r:
                    m['rmsd_epochs'].append(current_epoch)
                    m['rmsd_mean'].append(float(r.group(1)))
                    m['rmsd_std'].append(float(r.group(2)))

                lr_m = pat_lr.search(line)
                if lr_m:
                    try:
                        m['lr_epochs'].append(current_epoch)
                        m['lr'].append(float(lr_m.group(1)))
                    except ValueError:
                        pass

            v = pat_validity.search(line)
            if v and current_epoch is not None:
                m['validity_epochs'].append(current_epoch)
                m['fully_valid_rate'].append(float(v.group(1)))
                m['bond_valid_rate'].append(float(v.group(2)))
                m['clash_free_rate'].append(float(v.group(3)))
                m['bond_error'].append(float(v.group(4)))

    return m


def parse_checkpoint_history(ckpt_path: str) -> dict:
    """Load history dict saved inside a .pt checkpoint."""
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    history = ckpt.get('history', [])
    m = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_mse': [], 'train_geo': [], 'val_mse': [],
        'rmsd_epochs': [], 'rmsd_mean': [], 'rmsd_std': [],
        'validity_epochs': [], 'fully_valid_rate': [],
        'bond_valid_rate': [], 'clash_free_rate': [], 'bond_error': [],
        'lr_epochs': [], 'lr': [],
    }
    for h in history:
        ep = h.get('epoch', 0)
        m['epoch'].append(ep)
        m['train_loss'].append(h.get('train_loss', 0.0))
        m['val_loss'].append(h.get('val_loss', 0.0))
        m['train_mse'].append(h.get('train_mse', float('nan')))
        m['train_geo'].append(h.get('train_geo', float('nan')))
        m['val_mse'].append(h.get('val_mse', float('nan')))
        if h.get('rmsd_mean', 0) > 0:
            m['rmsd_epochs'].append(ep)
            m['rmsd_mean'].append(h['rmsd_mean'])
            m['rmsd_std'].append(h.get('rmsd_std', 0))
        if 'lr' in h:
            m['lr_epochs'].append(ep)
            m['lr'].append(h['lr'])
    return m


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def _no_data(ax, msg='No data yet'):
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, color=TEXT_DIM, fontsize=11)


def plot_loss_curves(metrics: dict, output_dir: str):
    if not metrics['epoch']:
        print("  No epoch data found."); return

    has_components = any(not np.isnan(v) for v in metrics.get('train_mse', [float('nan')]))

    nrows = 2 if has_components else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 5 * nrows))
    if nrows == 1:
        axes = [axes]
    _style(fig, axes)

    ax = axes[0]
    ax.semilogy(metrics['epoch'], metrics['train_loss'],
                color=COLORS['train_loss'], lw=2.2, label='Train (total)')
    ax.semilogy(metrics['epoch'], metrics['val_loss'],
                color=COLORS['val_loss'], lw=2, ls='--', label='Val (total)')
    best_i = int(np.argmin(metrics['val_loss']))
    ax.scatter([metrics['epoch'][best_i]], [metrics['val_loss'][best_i]],
               color=COLORS['accent'], s=100, zorder=5,
               label=f"Best val {metrics['val_loss'][best_i]:.4f} @ ep{metrics['epoch'][best_i]}")
    ax.set_title('Training Loss — Total (MSE + Geometry curriculum)', fontsize=14,
                 fontweight='bold', pad=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)

    if has_components:
        ax2 = axes[1]
        mse_t = np.array(metrics['train_mse'], dtype=float)
        geo_t = np.array(metrics['train_geo'], dtype=float)
        mse_v = np.array(metrics['val_mse'], dtype=float)
        eps   = metrics['epoch']

        # Only plot where we have real data (not nan)
        valid = ~np.isnan(mse_t)
        if valid.any():
            ax2.plot(np.array(eps)[valid], mse_t[valid],
                     color=COLORS['train_mse'], lw=2, label='Train MSE')
            ax2.plot(np.array(eps)[valid], mse_v[valid],
                     color=COLORS['val_mse'], lw=2, ls='--', label='Val MSE')
            ax2.fill_between(np.array(eps)[valid], mse_t[valid], mse_v[valid],
                             alpha=0.08, color=COLORS['train_mse'])
        if ~np.isnan(geo_t).all():
            ax2.plot(np.array(eps)[valid], geo_t[valid],
                     color=COLORS['train_geo'], lw=1.8, ls=':', label='Train Geo')

        ax2.set_title('Loss Component Breakdown — MSE (diffusion) vs Geometry',
                      fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
        ax2.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)

    plt.tight_layout()
    p = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_rmsd(metrics: dict, output_dir: str):
    if not metrics['rmsd_mean']:
        print("  No RMSD data."); return

    rmsd    = np.array(metrics['rmsd_mean'])
    std     = np.array(metrics['rmsd_std'])
    epochs  = metrics['rmsd_epochs']

    fig, ax = plt.subplots(figsize=(10, 5))
    _style(fig, [ax])

    ax.plot(epochs, rmsd, color=COLORS['rmsd'], lw=2.2, marker='o', ms=5)
    ax.fill_between(epochs, rmsd - std, rmsd + std,
                    color=COLORS['rmsd'], alpha=0.18)
    best_i = int(np.argmin(rmsd))
    ax.scatter([epochs[best_i]], [rmsd[best_i]], color=COLORS['accent'],
               s=110, zorder=5, marker='*',
               label=f"Best {rmsd[best_i]:.3f} Å  @ ep{epochs[best_i]}")

    ax.set_title('Kabsch-RMSD vs Ground Truth', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSD (Å)')
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)
    plt.tight_layout()
    p = os.path.join(output_dir, 'rmsd_progression.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_validity_metrics(metrics: dict, output_dir: str):
    fv = metrics.get('fully_valid_rate', [])
    if not fv:
        print("  No validity data."); return

    val_eps = metrics.get('validity_epochs', list(range(1, len(fv) + 1)))
    bv = metrics.get('bond_valid_rate', [])
    cf = metrics.get('clash_free_rate', [])
    be = metrics.get('bond_error', [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    _style(fig, [ax1, ax2])

    # Rates
    ax1.plot(val_eps, fv, color=COLORS['fully_valid'], lw=2.2, marker='s', ms=6, label='Fully Valid')
    if bv:
        ax1.plot(val_eps[:len(bv)], bv, color=COLORS['bond_valid'], lw=2, marker='o', ms=5, label='Bonds Valid')
    if cf:
        ax1.plot(val_eps[:len(cf)], cf, color=COLORS['clash_free'], lw=2, marker='^', ms=5,
                 ls=':', label='Clash-Free')
    ax1.set_ylim(-5, 105)
    ax1.set_title('3D Validity Rates', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Rate (%)')
    ax1.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)

    # Bond error
    if be:
        be_eps = val_eps[:len(be)]
        ax2.plot(be_eps, be, color=COLORS['bond_error'], lw=2.2, marker='D', ms=5)
        ax2.fill_between(be_eps, be, alpha=0.15, color=COLORS['bond_error'])
        ax2.axhline(0.2, color=COLORS['accent'], lw=1.2, ls='--', label='Tolerance 0.2 Å')
        ax2.set_title('Mean Bond Length Error', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Error (Å)')
        ax2.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)
    else:
        _no_data(ax2, 'Bond error not logged')

    plt.tight_layout()
    p = os.path.join(output_dir, 'validity_metrics.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_lr_schedule(metrics: dict, output_dir: str):
    if not metrics.get('lr'):
        print("  No LR data."); return

    fig, ax = plt.subplots(figsize=(10, 4))
    _style(fig, [ax])
    ax.plot(metrics['lr_epochs'], metrics['lr'], color=COLORS['lr'], lw=2)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
    plt.tight_layout()
    p = os.path.join(output_dir, 'lr_schedule.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_dashboard(metrics: dict, output_dir: str, title: str = ''):
    """3x2 dashboard: loss, MSE/geo split, RMSD, validity, bond error, LR."""
    has_components = any(not np.isnan(v) for v in metrics.get('train_mse', [float('nan')]))

    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(title or 'NExT-Mol Training Dashboard',
                 color=TEXT_MAIN, fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.28,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    def _ax(r, c):
        a = fig.add_subplot(gs[r, c])
        _style(fig, [a]); return a

    # (0,0) Total Loss
    ax = _ax(0, 0)
    if metrics['epoch']:
        ax.semilogy(metrics['epoch'], metrics['train_loss'],
                    color=COLORS['train_loss'], lw=2, label='Train (total)')
        ax.semilogy(metrics['epoch'], metrics['val_loss'],
                    color=COLORS['val_loss'], lw=2, ls='--', label='Val (total)')
        best_i = int(np.argmin(metrics['val_loss']))
        ax.scatter([metrics['epoch'][best_i]], [metrics['val_loss'][best_i]],
                   color=COLORS['accent'], s=80, zorder=5)
        ax.set_title('Total Loss (log)', fontweight='bold')
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)
    else:
        _no_data(ax, 'No loss data')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')

    # (0,1) MSE vs Geo component
    ax = _ax(0, 1)
    if has_components:
        mse_t = np.array(metrics['train_mse'], dtype=float)
        geo_t = np.array(metrics['train_geo'], dtype=float)
        mse_v = np.array(metrics['val_mse'],   dtype=float)
        eps   = np.array(metrics['epoch'])
        valid = ~np.isnan(mse_t)
        if valid.any():
            ax.plot(eps[valid], mse_t[valid], color=COLORS['train_mse'], lw=2, label='Train MSE')
            ax.plot(eps[valid], mse_v[valid], color=COLORS['val_mse'], lw=2, ls='--', label='Val MSE')
        if ~np.isnan(geo_t).all():
            ax.plot(eps[valid], geo_t[valid], color=COLORS['train_geo'], lw=1.8, ls=':', label='Geo loss')
        ax.set_title('MSE vs Geometry Loss (curriculum ramp)', fontweight='bold')
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)
    else:
        _no_data(ax, 'No component data\n(upgrade train_conformer.py)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')

    # (1,0) RMSD
    ax = _ax(1, 0)
    if metrics['rmsd_mean']:
        rmsd = np.array(metrics['rmsd_mean'])
        ax.plot(metrics['rmsd_epochs'], rmsd, color=COLORS['rmsd'], lw=2, marker='o', ms=4)
        ax.fill_between(metrics['rmsd_epochs'],
                        rmsd - np.array(metrics['rmsd_std']),
                        rmsd + np.array(metrics['rmsd_std']),
                        color=COLORS['rmsd'], alpha=0.18)
        best_i = int(np.argmin(rmsd))
        ax.scatter([metrics['rmsd_epochs'][best_i]], [rmsd[best_i]],
                   color=COLORS['accent'], s=90, zorder=5, marker='*',
                   label=f"Best {rmsd[best_i]:.3f} Å")
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)
    else:
        _no_data(ax, 'RMSD every 5 epochs')
    ax.set_title('Kabsch-RMSD vs Ground Truth (↓)', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSD (Å)')

    # (1,1) Validity rates
    ax = _ax(1, 1)
    fv = metrics.get('fully_valid_rate', [])
    if fv:
        eps = metrics['validity_epochs']
        ax.plot(eps, fv, color=COLORS['fully_valid'], lw=2, marker='s', ms=5,
                label='Fully Valid')
        bv = metrics.get('bond_valid_rate', [])
        cf = metrics.get('clash_free_rate', [])
        if bv:
            ax.plot(eps[:len(bv)], bv, color=COLORS['bond_valid'], lw=1.8,
                    marker='o', ms=4, label='Bonds Valid')
        if cf:
            ax.plot(eps[:len(cf)], cf, color=COLORS['clash_free'], lw=1.8,
                    marker='^', ms=4, ls=':', label='Clash-Free')
        ax.set_ylim(-5, 105)
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)
    else:
        _no_data(ax, 'Validity every 10 epochs')
    ax.set_title('3D Validity Rates (↑)', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Rate (%)')

    # (2,0) Bond length error
    ax = _ax(2, 0)
    be = metrics.get('bond_error', [])
    if be:
        eps = metrics['validity_epochs'][:len(be)]
        ax.plot(eps, be, color=COLORS['bond_error'], lw=2, marker='D', ms=4)
        ax.fill_between(eps, be, alpha=0.15, color=COLORS['bond_error'])
        ax.axhline(0.2, color=COLORS['accent'], lw=1.2, ls='--',
                   label='0.2 Å tolerance')
        ax.set_title('Mean Bond Length Error (↓)', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE (Å)')
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)
    else:
        _no_data(ax, 'Bond error not logged yet')

    # (2,1) Learning rate
    ax = _ax(2, 1)
    if metrics.get('lr'):
        ax.plot(metrics['lr_epochs'], metrics['lr'], color=COLORS['lr'], lw=2)
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
    else:
        _no_data(ax, 'No LR data')

    p = os.path.join(output_dir, 'training_overview.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--log',        type=str, help='Path to SLURM / console training log')
    parser.add_argument('--checkpoint', type=str, help='Path to .pt checkpoint (loads history)')
    parser.add_argument('--output',     type=str, default='plots/', help='Output directory')
    args = parser.parse_args()

    if not args.log and not args.checkpoint:
        print("Provide --log or --checkpoint"); sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    if args.log:
        print(f"Parsing log: {args.log}")
        metrics = parse_training_log(args.log)
        title = f"NExT-Mol — {os.path.basename(args.log)} ({len(metrics['epoch'])} epochs)"
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        metrics = parse_checkpoint_history(args.checkpoint)
        title = f"NExT-Mol — {os.path.basename(args.checkpoint)}"

    epochs_done = len(metrics['epoch'])
    print(f"  Epochs: {epochs_done} | RMSD points: {len(metrics['rmsd_mean'])} | "
          f"Validity points: {len(metrics['fully_valid_rate'])}")
    print()

    plot_loss_curves(metrics, args.output)
    plot_rmsd(metrics, args.output)
    plot_validity_metrics(metrics, args.output)
    plot_lr_schedule(metrics, args.output)
    plot_dashboard(metrics, args.output, title=title)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == '__main__':
    main()
