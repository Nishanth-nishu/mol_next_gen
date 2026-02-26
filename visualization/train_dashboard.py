"""
train_dashboard.py — Live Training Visualization Dashboard

Parses a SLURM/console training log from train_conformer.py and renders:
  1. Train / Val Loss (log scale)
  2. RMSD progression (Kabsch-aligned, every 5 epochs)
  3. 3D Validity rates (fully_valid, bonds, clash_free — every 10 epochs)
  4. Bond Error (Å)
  5. Learning Rate schedule
  6. Summary statistics box

Usage:
    # One-shot (generate plots from finished or partial log):
    python visualization/train_dashboard.py --log jepa_train_2579106.log --output plots/

    # Live watch mode (re-renders every 30s):
    python visualization/train_dashboard.py --log jepa_train_2579106.log --output plots/ --watch

    # Specify custom refresh interval:
    python visualization/train_dashboard.py --log jepa_train_2579106.log --output plots/ --watch --interval 60
"""

import re
import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — works in SLURM/SSH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = '#0d1117'
PANEL_BG   = '#161b22'
BORDER     = '#30363d'
TEXT_MAIN  = '#e6edf3'
TEXT_DIM   = '#8b949e'

COLORS = {
    'train_loss':   '#58a6ff',   # blue
    'val_loss':     '#f78166',   # salmon
    'rmsd':         '#d2a8ff',   # purple
    'fully_valid':  '#3fb950',   # green
    'bond_valid':   '#e3b341',   # amber
    'clash_free':   '#58a6ff',   # blue
    'bond_error':   '#f78166',   # salmon
    'lr':           '#79c0ff',   # light blue
    'accent':       '#ffa657',   # orange
}

def _apply_theme(fig, axes_list):
    """Apply dark theme to figure and all axes."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_list:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT_MAIN)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, alpha=0.5, linewidth=0.7)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOG PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> dict:
    """
    Parse train_conformer.py output log.

    Expected patterns:
        Epoch N: train_loss=X.XXXX, val_loss=Y.YYYY
        Epoch N: train_loss=X.XXXX, val_loss=Y.YYYY, rmsd=A.AAA±B.BBB Å
        → 3D Validity: fully_valid=X%, bonds=Y%, no_clash=Z%, bond_err=W Å
        Scheduler lr=...   (from scheduler.get_last_lr)
    """
    m = {
        'epoch':           [],
        'train_loss':      [],
        'val_loss':        [],
        'rmsd_epochs':     [],
        'rmsd_mean':       [],
        'rmsd_std':        [],
        'validity_epochs': [],
        'fully_valid':     [],
        'bond_valid':      [],
        'clash_free':      [],
        'bond_error':      [],
        'lr_epochs':       [],
        'lr':              [],
    }

    pat_epoch    = re.compile(r'Epoch\s+(\d+):\s+train_loss=([0-9.]+),\s+val_loss=([0-9.]+)')
    pat_rmsd     = re.compile(r'rmsd=([0-9.]+)±([0-9.]+)')
    pat_validity = re.compile(
        r'fully_valid=([0-9.]+)%.*?bonds=([0-9.]+)%.*?no_clash=([0-9.]+)%.*?bond_err=([0-9.]+)'
    )
    pat_lr       = re.compile(r'lr[=\s]+([0-9.e+-]+)', re.IGNORECASE)

    current_epoch = None

    try:
        with open(log_path, 'r', errors='replace') as f:
            for line in f:
                # ── Epoch summary line ──────────────────────────────────────
                e = pat_epoch.search(line)
                if e:
                    current_epoch = int(e.group(1))
                    m['epoch'].append(current_epoch)
                    m['train_loss'].append(float(e.group(2)))
                    m['val_loss'].append(float(e.group(3)))

                    # RMSD embedded in same line
                    r = pat_rmsd.search(line)
                    if r:
                        m['rmsd_epochs'].append(current_epoch)
                        m['rmsd_mean'].append(float(r.group(1)))
                        m['rmsd_std'].append(float(r.group(2)))

                    # LR embedded in same line
                    lr_m = pat_lr.search(line)
                    if lr_m and current_epoch is not None:
                        try:
                            m['lr_epochs'].append(current_epoch)
                            m['lr'].append(float(lr_m.group(1)))
                        except ValueError:
                            pass

                # ── Validity line (immediately follows epoch line) ──────────
                v = pat_validity.search(line)
                if v and current_epoch is not None:
                    m['validity_epochs'].append(current_epoch)
                    m['fully_valid'].append(float(v.group(1)))
                    m['bond_valid'].append(float(v.group(2)))
                    m['clash_free'].append(float(v.group(3)))
                    m['bond_error'].append(float(v.group(4)))

    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {log_path}")
        sys.exit(1)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _no_data(ax, label='No data yet'):
    ax.text(0.5, 0.5, label, ha='center', va='center',
            transform=ax.transAxes, color=TEXT_DIM, fontsize=12)


def plot_loss(ax, m):
    """Train and validation loss curves."""
    if not m['epoch']:
        _no_data(ax, 'No epoch data yet'); return

    ax.semilogy(m['epoch'], m['train_loss'], color=COLORS['train_loss'],
                lw=2, label='Train Loss')
    ax.semilogy(m['epoch'], m['val_loss'], color=COLORS['val_loss'],
                lw=2, linestyle='--', label='Val Loss')

    # Best val marker
    best_idx = int(np.argmin(m['val_loss']))
    ax.scatter([m['epoch'][best_idx]], [m['val_loss'][best_idx]],
               color=COLORS['accent'], s=80, zorder=5,
               label=f"Best val {m['val_loss'][best_idx]:.4f} @ ep{m['epoch'][best_idx]}")

    ax.set_title('Loss (log scale)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8, framealpha=0.3,
              facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN)


def plot_rmsd(ax, m):
    """Kabsch-RMSD every 5 epochs."""
    if not m['rmsd_mean']:
        _no_data(ax, 'RMSD logged every 5 epochs…'); return

    rmsd = np.array(m['rmsd_mean'])
    std  = np.array(m['rmsd_std'])
    eps  = m['rmsd_epochs']

    ax.plot(eps, rmsd, color=COLORS['rmsd'], lw=2, marker='o', ms=5)
    ax.fill_between(eps, rmsd - std, rmsd + std,
                    color=COLORS['rmsd'], alpha=0.18)

    best_i = int(np.argmin(rmsd))
    ax.scatter([eps[best_i]], [rmsd[best_i]], color=COLORS['accent'],
               s=100, zorder=5, marker='*',
               label=f"Best {rmsd[best_i]:.3f}Å @ ep{eps[best_i]}")

    ax.set_title('Kabsch-RMSD vs Ground Truth', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSD (Å)')
    ax.legend(fontsize=8, framealpha=0.3,
              facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN)


def plot_validity_rates(ax, m):
    """3D validity rates every 10 epochs."""
    if not m['fully_valid']:
        _no_data(ax, '3D validity logged every 10 epochs…'); return

    eps = m['validity_epochs']
    ax.plot(eps, m['fully_valid'], color=COLORS['fully_valid'],
            lw=2.2, marker='s', ms=6, label='Fully Valid')
    ax.plot(eps[:len(m['bond_valid'])], m['bond_valid'],
            color=COLORS['bond_valid'], lw=1.8, marker='o', ms=5,
            label='Bonds Valid')
    ax.plot(eps[:len(m['clash_free'])], m['clash_free'],
            color=COLORS['clash_free'], lw=1.8, marker='^', ms=5,
            linestyle=':', label='Clash-Free')

    ax.set_ylim(-5, 105)
    ax.set_title('3D Validity Rates', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rate (%)')
    ax.legend(fontsize=8, framealpha=0.3,
              facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN)


def plot_bond_error(ax, m):
    """Mean bond length error over epochs."""
    if not m['bond_error']:
        _no_data(ax, 'Bond error logged every 10 epochs…'); return

    eps = m['validity_epochs'][:len(m['bond_error'])]
    be  = m['bond_error']

    ax.plot(eps, be, color=COLORS['bond_error'], lw=2, marker='D', ms=5)
    ax.fill_between(eps, be, alpha=0.15, color=COLORS['bond_error'])
    ax.axhline(0.2, color=COLORS['accent'], lw=1.2, ls='--',
               label='Tolerance 0.2 Å')

    ax.set_title('Mean Bond Length Error', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (Å)')
    ax.legend(fontsize=8, framealpha=0.3,
              facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN)


def plot_lr(ax, m):
    """LR schedule."""
    if not m['lr']:
        # Synthesise expected cosine annealing if we have epoch info
        if m['epoch']:
            n = len(m['epoch'])
            _no_data(ax, 'LR not logged — see scheduler'); return
        _no_data(ax, 'No LR data'); return

    ax.plot(m['lr_epochs'], m['lr'], color=COLORS['lr'], lw=2)
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))


def plot_summary(ax, m, log_path):
    """Summary statistics text box."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    epochs_done = len(m['epoch'])
    best_val    = min(m['val_loss']) if m['val_loss'] else float('nan')
    best_ep_val = m['epoch'][int(np.argmin(m['val_loss']))] if m['val_loss'] else '—'
    best_rmsd   = min(m['rmsd_mean']) if m['rmsd_mean'] else float('nan')
    best_ep_r   = m['rmsd_epochs'][int(np.argmin(m['rmsd_mean']))] if m['rmsd_mean'] else '—'
    fv_last     = m['fully_valid'][-1] if m['fully_valid'] else float('nan')
    be_last     = m['bond_error'][-1] if m['bond_error'] else float('nan')

    lines = [
        f"{'─'*30}",
        f"  Log:  {os.path.basename(log_path)}",
        f"  Updated: {datetime.now().strftime('%H:%M:%S')}",
        f"{'─'*30}",
        f"  Epochs completed : {epochs_done}",
        f"  Best val loss    : {best_val:.5f}  (ep {best_ep_val})",
        f"  Best RMSD        : {best_rmsd:.3f} Å  (ep {best_ep_r})",
        f"  Fully valid (last): {fv_last:.1f}%",
        f"  Bond error (last) : {be_last:.3f} Å",
        f"{'─'*30}",
    ]

    ax.text(0.05, 0.95, '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left',
            fontfamily='monospace',
            color=TEXT_MAIN,
            bbox=dict(facecolor=PANEL_BG, edgecolor=BORDER,
                      boxstyle='round,pad=0.5'))

    ax.set_title('Training Summary', fontweight='bold', color=TEXT_MAIN)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render(log_path: str, output_dir: str):
    """Parse log and save all plots."""
    os.makedirs(output_dir, exist_ok=True)
    m = parse_log(log_path)

    epochs_done = len(m['epoch'])
    print(f"  Parsed {epochs_done} epochs | "
          f"RMSD points: {len(m['rmsd_mean'])} | "
          f"Validity points: {len(m['fully_valid'])}")

    # ── Individual plots ────────────────────────────────────────────────────
    _plots = [
        ('loss_curves.png',      plot_loss,           'Loss Curves'),
        ('rmsd_progression.png', plot_rmsd,           'RMSD Progression'),
        ('validity_rates.png',   plot_validity_rates, '3D Validity Rates'),
        ('bond_error.png',       plot_bond_error,     'Bond Length Error'),
        ('lr_schedule.png',      plot_lr,             'LR Schedule'),
    ]

    for fname, fn, title in _plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        _apply_theme(fig, [ax])
        fn(ax, m)
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ {path}")

    # ── Combined dashboard (2×3 grid) ───────────────────────────────────────
    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor(DARK_BG)

    fig.suptitle(
        f'NExT-Mol Training Dashboard  —  {os.path.basename(log_path)}  '
        f'({epochs_done} epochs)',
        color=TEXT_MAIN, fontsize=15, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    axes_map = [
        (0, 0, plot_loss),
        (0, 1, plot_rmsd),
        (0, 2, plot_validity_rates),
        (1, 0, plot_bond_error),
        (1, 1, plot_lr),
        (1, 2, lambda ax, m: plot_summary(ax, m, log_path)),
    ]

    all_axes = []
    for row, col, fn in axes_map:
        ax = fig.add_subplot(gs[row, col])
        all_axes.append(ax)
        fn(ax, m)

    _apply_theme(fig, [ax for ax in all_axes if ax.get_visible()])

    dashboard_path = os.path.join(output_dir, 'training_dashboard.png')
    fig.savefig(dashboard_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Dashboard: {dashboard_path}")

    return epochs_done


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NExT-Mol live training dashboard'
    )
    parser.add_argument(
        '--log', required=True,
        help='Path to SLURM/console training log'
    )
    parser.add_argument(
        '--output', default='plots/',
        help='Directory to save PNG plots (default: plots/)'
    )
    parser.add_argument(
        '--watch', action='store_true',
        help='Keep running and re-render on each interval'
    )
    parser.add_argument(
        '--interval', type=int, default=30,
        help='Seconds between re-renders in --watch mode (default: 30)'
    )
    args = parser.parse_args()

    if args.watch:
        print(f"[watch] Monitoring {args.log} — refreshing every {args.interval}s")
        print(f"        Ctrl-C to stop.\n")
        while True:
            try:
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"[{ts}] Rendering …")
                n = render(args.log, args.output)
                print(f"[{ts}] Done ({n} epochs). Sleeping {args.interval}s …\n")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n[watch] Stopped.")
                break
            except Exception as exc:
                print(f"[warn] Render failed: {exc}. Retrying in {args.interval}s …")
                time.sleep(args.interval)
    else:
        print(f"Rendering dashboard from: {args.log}")
        render(args.log, args.output)
        print("Done.")


if __name__ == '__main__':
    main()
