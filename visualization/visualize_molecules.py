"""
visualize_molecules.py — Conformer Quality Visualizer for mol_next_gen

Reads generated SDF or JSON files and plots:
  - Bond length distributions (overall + per bond type)
  - 3D coordinate scatter (PCA projection)
  - Per-molecule RMSD bar chart
  - Bond error deviation heatmap

Usage:
    python visualization/visualize_molecules.py \
        --sdf   outputs/generated_nextmol.sdf \
        --output plots/molecules/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
PANEL_BG  = '#161b22'
BORDER    = '#30363d'
TEXT_MAIN = '#e6edf3'
TEXT_DIM  = '#8b949e'

BOND_COLORS = {
    1: '#58a6ff',   # single — blue
    2: '#3fb950',   # double — green
    3: '#d2a8ff',   # triple — purple
    4: '#e3b341',   # aromatic — amber
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
# IDEAL BOND LENGTHS (chemistry reference)
# ─────────────────────────────────────────────────────────────
IDEAL_BONDS = {
    # (atom1, atom2, bond_order): ideal_length_Angstrom
    (6, 6, 1): 1.54, (6, 6, 2): 1.34, (6, 6, 4): 1.40,
    (6, 7, 1): 1.47, (6, 7, 2): 1.29, (6, 7, 4): 1.34,
    (6, 8, 1): 1.43, (6, 8, 2): 1.22,
    (6, 1, 1): 1.09, (7, 1, 1): 1.01,
    (8, 1, 1): 0.96, (6, 9, 1): 1.35,
    (6, 17, 1): 1.77, (6, 16, 1): 1.82,
}

def _ideal_bond(a1, a2, order):
    key = (min(a1, a2), max(a1, a2), order)
    return IDEAL_BONDS.get(key, 1.50)


# ─────────────────────────────────────────────────────────────
# SDF PARSER
# ─────────────────────────────────────────────────────────────

def _parse_sdf(sdf_path: str):
    """Parse SDF file. Returns list of molecule dicts."""
    mols = []
    current = None

    with open(sdf_path, 'r', errors='replace') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        name = lines[i].strip() if i < len(lines) else ''
        i += 3  # skip name, comment, program lines
        if i >= len(lines):
            break

        # Counts line: aaabbb...
        counts = lines[i].split()
        i += 1
        if len(counts) < 2:
            continue
        try:
            num_atoms = int(counts[0])
            num_bonds = int(counts[1])
        except ValueError:
            continue

        atoms = []
        for _ in range(num_atoms):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                sym = parts[3]
                atoms.append({'x': x, 'y': y, 'z': z, 'sym': sym})
            except ValueError:
                pass

        bonds = []
        for _ in range(num_bonds):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 3:
                continue
            try:
                a1 = int(parts[0]) - 1
                a2 = int(parts[1]) - 1
                order = int(parts[2])
                bonds.append({'a1': a1, 'a2': a2, 'order': order})
            except ValueError:
                pass

        # Skip to $$$$
        while i < len(lines) and '$$$$' not in lines[i]:
            i += 1
        i += 1

        mols.append({'name': name, 'atoms': atoms, 'bonds': bonds})

    return mols


ATOM_NUMS = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}


def _bond_lengths(mols):
    """Extract (bond_order, actual_len, ideal_len, error) tuples."""
    records = []
    for mol in mols:
        atoms = mol['atoms']
        for b in mol['bonds']:
            a1, a2 = b['a1'], b['a2']
            if a1 >= len(atoms) or a2 >= len(atoms):
                continue
            at1, at2 = atoms[a1], atoms[a2]
            dx = at1['x'] - at2['x']
            dy = at1['y'] - at2['y']
            dz = at1['z'] - at2['z']
            dist = (dx**2 + dy**2 + dz**2) ** 0.5
            z1 = ATOM_NUMS.get(at1['sym'], 6)
            z2 = ATOM_NUMS.get(at2['sym'], 6)
            ideal = _ideal_bond(z1, z2, b['order'])
            records.append((b['order'], dist, ideal, abs(dist - ideal)))
    return records


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_bond_distribution(mols, output_dir):
    records = _bond_lengths(mols)
    if not records:
        print("  No bond data."); return

    by_order = defaultdict(list)
    for order, dist, ideal, err in records:
        by_order[order].append(dist)

    orders = sorted(by_order)
    nplots = len(orders)
    fig, axes = plt.subplots(1, nplots, figsize=(5 * nplots, 5), squeeze=False)
    _style(fig, axes[0])

    bond_names = {1: 'Single', 2: 'Double', 3: 'Triple', 4: 'Aromatic'}
    ideal_map  = {1: 1.54, 2: 1.34, 3: 1.20, 4: 1.40}

    for idx, order in enumerate(orders):
        ax = axes[0][idx]
        _style(fig, [ax])
        dists = by_order[order]
        color = BOND_COLORS.get(order, '#58a6ff')
        ax.hist(dists, bins=40, color=color, alpha=0.85, edgecolor='none')
        ideal = ideal_map.get(order, 1.5)
        ax.axvline(ideal, color='#ffa657', lw=1.8, ls='--',
                   label=f'Ideal {ideal:.2f} Å')
        ax.set_title(f'{bond_names.get(order, str(order))} Bonds  (n={len(dists)})',
                     fontweight='bold')
        ax.set_xlabel('Bond Length (Å)'); ax.set_ylabel('Count')
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)

    fig.suptitle('Bond Length Distributions', color=TEXT_MAIN,
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, 'bond_distributions.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_bond_error_summary(mols, output_dir):
    records = _bond_lengths(mols)
    if not records:
        return

    errors = [e for _, _, _, e in records]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _style(fig, [ax1, ax2])

    ax1.hist(errors, bins=60, color='#58a6ff', alpha=0.85, edgecolor='none')
    ax1.axvline(0.2, color='#ffa657', lw=1.8, ls='--', label='0.2 Å tolerance')
    pct_valid = 100 * sum(e < 0.2 for e in errors) / len(errors)
    ax1.set_title(f'Bond Error Distribution  ({pct_valid:.1f}% within 0.2 Å)',
                  fontweight='bold')
    ax1.set_xlabel('|actual − ideal| (Å)'); ax1.set_ylabel('Count')
    ax1.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)

    # Per-molecule mean error
    mol_errors = []
    for mol in mols:
        bl = _bond_lengths([mol])
        if bl:
            mol_errors.append(np.mean([e for _, _, _, e in bl]))
    if mol_errors:
        ax2.hist(mol_errors, bins=40, color='#d2a8ff', alpha=0.85, edgecolor='none')
        ax2.axvline(0.2, color='#ffa657', lw=1.8, ls='--', label='0.2 Å threshold')
        ax2.set_title('Per-Molecule Mean Bond Error', fontweight='bold')
        ax2.set_xlabel('Mean Error (Å)'); ax2.set_ylabel('# Molecules')
        ax2.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)

    plt.tight_layout()
    p = os.path.join(output_dir, 'bond_error_summary.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_3d_scatter(mols, output_dir, n_mols=200):
    """PCA projection of 3D coordinates."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        # Manual PCA
        pass

    all_coords = []
    for mol in mols[:n_mols]:
        for at in mol['atoms']:
            all_coords.append([at['x'], at['y'], at['z']])

    if not all_coords:
        return

    coords = np.array(all_coords)
    # Centre
    coords -= coords.mean(axis=0)
    # Manual PCA (SVD)
    _, _, Vt = np.linalg.svd(coords, full_matrices=False)
    proj = coords @ Vt[:2].T   # (N, 2)

    fig, ax = plt.subplots(figsize=(8, 8))
    _style(fig, [ax])
    ax.scatter(proj[:, 0], proj[:, 1], s=2, c='#58a6ff', alpha=0.3)
    ax.set_title(f'3D Atom Positions (PCA projection, first {n_mols} mols)',
                 fontweight='bold')
    ax.set_xlabel('PC1 (Å)'); ax.set_ylabel('PC2 (Å)')
    plt.tight_layout()
    p = os.path.join(output_dir, 'coords_pca.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


def plot_atom_composition(mols, output_dir):
    """Atom type frequency."""
    from collections import Counter
    counts = Counter()
    for mol in mols:
        for at in mol['atoms']:
            counts[at['sym']] += 1

    if not counts:
        return

    syms  = sorted(counts, key=lambda s: -counts[s])[:12]
    vals  = [counts[s] for s in syms]
    colors = ['#58a6ff', '#3fb950', '#d2a8ff', '#e3b341', '#f78166',
              '#79c0ff', '#ffa657', '#ff7b72', '#a5d6ff', '#7ee787',
              '#d2a8ff', '#ffa198']

    fig, ax = plt.subplots(figsize=(10, 5))
    _style(fig, [ax])
    bars = ax.bar(syms, vals, color=colors[:len(syms)], edgecolor='none')
    ax.set_title('Atom Type Composition', fontweight='bold')
    ax.set_xlabel('Element'); ax.set_ylabel('Count')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f'{v:,}', ha='center', va='bottom', color=TEXT_DIM, fontsize=8)
    plt.tight_layout()
    p = os.path.join(output_dir, 'atom_composition.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {p}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize generated conformer quality')
    parser.add_argument('--sdf',    type=str, help='SDF file of generated molecules')
    parser.add_argument('--output', type=str, default='plots/molecules/')
    args = parser.parse_args()

    if not args.sdf:
        print("Provide --sdf"); sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading SDF: {args.sdf}")
    mols = _parse_sdf(args.sdf)
    print(f"  Loaded {len(mols)} molecules")

    plot_bond_distribution(mols, args.output)
    plot_bond_error_summary(mols, args.output)
    plot_3d_scatter(mols, args.output)
    plot_atom_composition(mols, args.output)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == '__main__':
    main()
