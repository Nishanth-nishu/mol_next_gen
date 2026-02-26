"""
plot_results.py — Evaluation Results Dashboard for mol_next_gen

Reads evaluation_results.json (output of evaluate_validity.py) and plots:
  - Key metrics bar chart (validity, uniqueness, diversity, novelty)
  - Property distributions (MW, LogP, HBD, HBA, rotatable, TPSA)
  - Radar chart summary

Usage:
    python visualization/plot_results.py \
        --results outputs/evaluation_results.json \
        --output  plots/results/
"""

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

METRIC_COLORS = {
    'validity':          '#3fb950',
    'uniqueness':        '#58a6ff',
    'diversity':         '#d2a8ff',
    'scaffold_diversity':'#e3b341',
    'novelty':           '#ffa657',
}

PROP_COLORS = ['#58a6ff', '#3fb950', '#d2a8ff', '#e3b341', '#f78166', '#79c0ff']

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
# LOAD
# ─────────────────────────────────────────────────────────────

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_key_metrics(results: dict, output_dir: str):
    metrics = {
        'Validity':           results.get('validity', 0) * 100,
        'Uniqueness':         results.get('uniqueness', 0) * 100,
        'Diversity':          results.get('diversity', 0) * 100,
        'Scaffold\nDiversity': results.get('scaffold_diversity', 0) * 100,
        'Novelty':            results.get('novelty', 0) * 100,
    }

    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = list(METRIC_COLORS.values())

    fig, ax = plt.subplots(figsize=(11, 5))
    _style(fig, [ax])

    bars = ax.bar(labels, values, color=colors, edgecolor='none', width=0.55)
    ax.set_ylim(0, 115)
    ax.axhline(100, color=BORDER, lw=1, ls='--')
    ax.set_ylabel('Score (%)'); ax.set_xlabel('')
    ax.set_title('Evaluation Metrics — Generated Molecules', fontsize=14,
                 fontweight='bold', pad=12)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom',
                color=TEXT_MAIN, fontsize=11, fontweight='bold')

    # Annotations
    n = results.get('num_generated', '?')
    ax.text(0.99, 0.97, f"n = {n:,}" if isinstance(n, int) else f"n = {n}",
            transform=ax.transAxes, ha='right', va='top',
            color=TEXT_DIM, fontsize=10)

    plt.tight_layout()
    p = os.path.join(output_dir, 'key_metrics.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_property_distributions(results: dict, output_dir: str):
    props = results.get('properties', {})
    if not props:
        print("  No property data."); return

    prop_labels = {
        'mw':        'Mol. Weight (Da)',
        'logp':      'LogP',
        'hbd':       'H-Bond Donors',
        'hba':       'H-Bond Acceptors',
        'rotatable': 'Rotatable Bonds',
        'tpsa':      'TPSA (Å²)',
    }
    prop_list = [k for k in prop_labels if k in props]
    n = len(prop_list)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    _style(fig, axes.flatten())
    fig.suptitle('Molecular Property Distributions', color=TEXT_MAIN,
                 fontsize=14, fontweight='bold', y=1.01)

    for idx, key in enumerate(prop_list):
        ax = axes.flatten()[idx]
        data = props[key]
        mean = data.get('mean', 0)
        std  = data.get('std', 0)
        lo   = data.get('min', mean - 3 * std)
        hi   = data.get('max', mean + 3 * std)

        # Simulate a distribution from mean/std/min/max
        x = np.linspace(lo, hi, 300)
        # Truncated Gaussian approximation
        from scipy.stats import truncnorm
        try:
            a_val = (lo - mean) / (std + 1e-9)
            b_val = (hi - mean) / (std + 1e-9)
            dist  = truncnorm(a_val, b_val, loc=mean, scale=std)
            y     = dist.pdf(x)
        except Exception:
            # Fallback: simple Gaussian
            y = np.exp(-0.5 * ((x - mean) / (std + 1e-9)) ** 2)
            y /= (y.max() + 1e-9)

        color = PROP_COLORS[idx % len(PROP_COLORS)]
        ax.fill_between(x, y, color=color, alpha=0.4)
        ax.plot(x, y, color=color, lw=2)
        ax.axvline(mean, color='#ffa657', lw=1.5, ls='--',
                   label=f'μ={mean:.2f}  σ={std:.2f}')
        ax.set_title(prop_labels[key], fontweight='bold')
        ax.set_xlabel(prop_labels[key])
        ax.set_ylabel('Density')
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER,
                  labelcolor=TEXT_MAIN, fontsize=8)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes.flatten()[idx].set_visible(False)

    plt.tight_layout()
    p = os.path.join(output_dir, 'property_distributions.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_radar(results: dict, output_dir: str):
    """Radar/spider chart of key metrics (0–1 scale)."""
    categories = ['Validity', 'Uniqueness', 'Diversity', 'Scaffold Div.', 'Novelty']
    values_raw  = [
        results.get('validity', 0),
        results.get('uniqueness', 0),
        results.get('diversity', 0),
        results.get('scaffold_diversity', 0),
        results.get('novelty', 0),
    ]
    values = values_raw + [values_raw[0]]  # close the polygon

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    ax.plot(angles, values, color='#58a6ff', lw=2.5)
    ax.fill(angles, values, color='#58a6ff', alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories,
                      color=TEXT_MAIN, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color=TEXT_DIM, fontsize=8)
    ax.grid(color=BORDER, alpha=0.6)
    ax.spines['polar'].set_edgecolor(BORDER)
    ax.set_title('Performance Radar', color=TEXT_MAIN,
                 fontsize=14, fontweight='bold', pad=20)

    # Score labels on points
    for angle, val, cat in zip(angles[:-1], values_raw, categories):
        ax.text(angle, val + 0.07, f'{val*100:.0f}%',
                ha='center', va='center', color=TEXT_MAIN, fontsize=9, fontweight='bold')

    p = os.path.join(output_dir, 'performance_radar.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_dashboard(results: dict, output_dir: str):
    """Combined dashboard."""
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle('NExT-Mol Evaluation Dashboard', color=TEXT_MAIN,
                 fontsize=16, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    # Top-left: bar metrics (span 2 cols)
    ax_bar = fig.add_subplot(gs[0, :2])
    _style(fig, [ax_bar])
    metrics = {
        'Validity':    results.get('validity', 0) * 100,
        'Uniqueness':  results.get('uniqueness', 0) * 100,
        'Diversity':   results.get('diversity', 0) * 100,
        'Scaf. Div.':  results.get('scaffold_diversity', 0) * 100,
        'Novelty':     results.get('novelty', 0) * 100,
    }
    bars = ax_bar.bar(list(metrics), list(metrics.values()),
                      color=list(METRIC_COLORS.values()), edgecolor='none', width=0.55)
    ax_bar.set_ylim(0, 115)
    ax_bar.axhline(100, color=BORDER, lw=1, ls='--')
    ax_bar.set_title('Key Metrics', fontweight='bold')
    ax_bar.set_ylabel('Score (%)')
    for bar, val in zip(bars, metrics.values()):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, f'{val:.1f}%',
                    ha='center', va='bottom', color=TEXT_MAIN, fontsize=10, fontweight='bold')

    # Top-right: radar
    ax_r = fig.add_subplot(gs[0, 2], polar=True)
    ax_r.set_facecolor(PANEL_BG)
    categories = ['Validity', 'Uniqueness', 'Diversity', 'Scaf.Div.', 'Novelty']
    vals = [results.get(k, 0) for k in
            ['validity', 'uniqueness', 'diversity', 'scaffold_diversity', 'novelty']]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    vals_c = vals + [vals[0]]; angles_c = angles + angles[:1]
    ax_r.plot(angles_c, vals_c, color='#58a6ff', lw=2)
    ax_r.fill(angles_c, vals_c, color='#58a6ff', alpha=0.22)
    ax_r.set_thetagrids(np.degrees(angles), categories, color=TEXT_MAIN, fontsize=9)
    ax_r.set_ylim(0, 1); ax_r.grid(color=BORDER, alpha=0.6)
    ax_r.spines['polar'].set_edgecolor(BORDER)
    ax_r.set_facecolor(PANEL_BG)
    ax_r.set_title('Radar', color=TEXT_MAIN, fontweight='bold', pad=15)

    # Bottom: property summaries
    props = results.get('properties', {})
    prop_keys = ['mw', 'logp', 'hbd', 'hba', 'rotatable', 'tpsa']
    prop_names = ['MW (Da)', 'LogP', 'HBD', 'HBA', 'Rotatable', 'TPSA']
    sub_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax in sub_axes:
        _style(fig, [ax])

    # Show 3 properties in bottom row as bullet points
    lines = []
    for k, name in zip(prop_keys, prop_names):
        if k in props:
            d = props[k]
            lines.append(f"{name:>12s}:  μ = {d.get('mean', 0):7.2f}   σ = {d.get('std', 0):.2f}")

    for i, ax in enumerate(sub_axes):
        chunk = lines[i*2:(i+1)*2]
        ax.axis('off')
        ax.text(0.05, 0.7, '\n'.join(chunk),
                transform=ax.transAxes, fontsize=10,
                va='top', fontfamily='monospace', color=TEXT_MAIN,
                bbox=dict(facecolor=PANEL_BG, edgecolor=BORDER,
                          boxstyle='round,pad=0.5', alpha=0.8))

    p = os.path.join(output_dir, 'evaluation_dashboard.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--results', required=True, help='Path to evaluation_results.json')
    parser.add_argument('--output',  default='plots/results/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading: {args.results}")
    results = load_results(args.results)
    print(f"  Validity={results.get('validity',0)*100:.1f}%  "
          f"Uniqueness={results.get('uniqueness',0)*100:.1f}%  "
          f"Novelty={results.get('novelty',0)*100:.1f}%")
    print()

    plot_key_metrics(results, args.output)
    plot_property_distributions(results, args.output)
    plot_radar(results, args.output)
    plot_dashboard(results, args.output)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == '__main__':
    main()
