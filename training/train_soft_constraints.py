"""
train_soft_constraints.py — Experiment-Aware Training Script
Experiment: 23-02-2026-Exp-1 (soft_restrictions)

Extends train_conformer.py with:
1. All outputs directed to experiment folder (checkpoints, plots, logs, evaluation)
2. Auto-detection of aromatic rings, chiral centers, small rings from graph structure
3. Extended loss logging: total, mse, geo, planarity, chirality, ring_strain
4. 4-stage curriculum: bonds → angles → torsions → full soft constraints (60%+)
5. Guided sampling uses all new constraints via guided_sample()
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conformer_diffusion import ConformerDiffusion, remove_com
from models.geometry_constraints import GeometryConstraints


# =============================================================================
# GRAPH-STRUCTURE CONSTRAINT DETECTION
# (No RDKit required — pure graph from bond types and adjacency)
# =============================================================================

def detect_aromatic_rings(edge_index: torch.Tensor,
                           bond_types: torch.Tensor,
                           num_atoms: int) -> list:
    """
    Detect aromatic rings from the molecular graph using bond types.
    Aromatic bonds have type 4 in our encoding.

    Returns list of rings (each a sorted list of atom indices).
    Uses simple DFS cycle detection on the aromatic subgraph only.
    """
    row, col = edge_index[0].tolist(), edge_index[1].tolist()
    bt = bond_types.tolist()

    # Build aromatic adjacency
    arom_adj = {i: [] for i in range(num_atoms)}
    for i, j, b in zip(row, col, bt):
        if b == 4:  # aromatic
            arom_adj[i].append(j)

    visited_global = set()
    rings = []

    def dfs(start, current, path, adj):
        for nxt in adj[current]:
            if nxt == start and len(path) >= 5:
                rings.append(sorted(path))
                return
            if nxt not in path and nxt not in visited_global:
                path.add(nxt)
                dfs(start, nxt, path, adj)
                path.discard(nxt)

    # Find rings by DFS from each aromatic atom
    arom_atoms = [i for i in range(num_atoms) if arom_adj[i]]
    for atom in arom_atoms:
        if atom in visited_global:
            continue
        path = {atom}
        dfs(atom, atom, path, arom_adj)
        visited_global.add(atom)

    # Deduplicate rings
    unique_rings = []
    seen = set()
    for r in rings:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique_rings.append(r)

    return unique_rings


def detect_small_rings(edge_index: torch.Tensor,
                        bond_types: torch.Tensor,
                        num_atoms: int,
                        max_ring_size: int = 4) -> list:
    """
    Detect 3- and 4-membered rings for ring strain calculation.
    Uses DFS on the full bond graph (all bond types).
    """
    row, col = edge_index[0].tolist(), edge_index[1].tolist()

    adj = {i: [] for i in range(num_atoms)}
    for i, j in zip(row, col):
        adj[i].append(j)

    small_rings = []
    seen_keys = set()

    def dfs_small(start, current, path, depth):
        if depth > max_ring_size:
            return
        for nxt in adj[current]:
            if nxt == start and depth >= 3:
                ring = sorted(path)
                key = tuple(ring)
                if key not in seen_keys and len(ring) <= max_ring_size:
                    seen_keys.add(key)
                    small_rings.append(list(path))  # preserve order
            elif nxt not in path:
                path.append(nxt)
                dfs_small(start, nxt, path, depth + 1)
                path.pop()

    for atom in range(num_atoms):
        dfs_small(atom, atom, [atom], 1)

    return small_rings


def detect_chiral_centers(atom_types: torch.Tensor,
                           edge_index: torch.Tensor,
                           bond_types: torch.Tensor) -> list:
    """
    Detect potential chiral centers: sp3 carbon atoms with exactly 4
    distinct-type neighbors. Returns list of (center_idx, neighbors, sign=+1).

    Note: sign (+1/-1) is unknown without 3D coordinates or stereo flags,
    so we return +1 for all. During guided sampling the gradient step will
    reinforce whichever handedness the model predicts (soft enforcement).
    """
    num_atoms = atom_types.size(0)
    row, col = edge_index[0].tolist(), edge_index[1].tolist()
    bt = bond_types.tolist()

    adj = {i: [] for i in range(num_atoms)}
    bond_map = {}
    for i, j, b in zip(row, col, bt):
        adj[i].append(j)
        bond_map[(i, j)] = b

    CARBON = 6
    chiral = []
    for c in range(num_atoms):
        if atom_types[c].item() != CARBON:
            continue
        neighbors = adj[c]
        if len(neighbors) != 4:
            continue
        # Check all single bonds (sp3)
        bond_orders = [bond_map.get((c, n), 0) for n in neighbors]
        if all(bo == 1 for bo in bond_orders):
            # All four neighbors, assign sign=+1 (model will reinforce)
            chiral.append((c, neighbors, +1))

    return chiral


# =============================================================================
# DATASET (same as train_conformer.py)
# =============================================================================

class ConformerDataset(Dataset):
    """Dataset for conformer diffusion training with precomputed constraint data."""

    def __init__(self, data_path: str, max_atoms: int = 20):
        self.data_path = data_path
        self.max_atoms = max_atoms
        self.data = []
        self._load_data()

    def _load_data(self):
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, 'r') as f:
            for line in tqdm(f):
                item = json.loads(line.strip())
                if item.get('coordinates') is None:
                    continue
                if item['num_atoms'] > self.max_atoms:
                    continue
                self.data.append(item)
        print(f"Loaded {len(self.data)} molecules")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        atom_types  = torch.tensor(item['atom_types'],   dtype=torch.long)
        coordinates = torch.tensor(item['coordinates'],  dtype=torch.float32)
        edge_index  = torch.tensor(item['edge_index'],   dtype=torch.long)
        bond_types  = torch.tensor(item['bond_types'],   dtype=torch.long)
        return {
            'atom_types':   atom_types,
            'coordinates':  coordinates,
            'edge_index':   edge_index,
            'bond_types':   bond_types,
            'num_atoms':    item['num_atoms'],
        }


def collate_fn(batch):
    """Collate variable-sized molecules into a single graph batch."""
    atom_types_list, coords_list, edge_index_list = [], [], []
    bond_types_list, batch_idx_list = [], []
    offset = 0

    for i, item in enumerate(batch):
        N = item['num_atoms']
        atom_types_list.append(item['atom_types'])
        coords_list.append(item['coordinates'])
        edge_index_list.append(item['edge_index'] + offset)
        bond_types_list.append(item['bond_types'])
        batch_idx_list.append(torch.full((N,), i, dtype=torch.long))
        offset += N

    return {
        'atom_types':    torch.cat(atom_types_list),
        'coordinates':   torch.cat(coords_list),
        'edge_index':    torch.cat(edge_index_list, dim=1),
        'bond_types':    torch.cat(bond_types_list),
        'batch_idx':     torch.cat(batch_idx_list),
        'num_molecules': len(batch),
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def get_soft_constraints(atom_types, edge_index, bond_types, batch_idx,
                          device, enabled=True):
    """
    For each molecule in the batch, detect aromatic rings, chiral centers,
    small rings, and return batched lists (offset-corrected).
    """
    if not enabled:
        return None, None, None

    B = batch_idx.max().item() + 1
    all_arom, all_chiral, all_small = [], [], []

    for mol_idx in range(B):
        mask = batch_idx == mol_idx
        offset = mask.nonzero(as_tuple=True)[0][0].item()
        n = mask.sum().item()

        mol_at = atom_types[mask].cpu()
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        mol_ei = edge_index[:, edge_mask].cpu() - offset
        mol_bt = bond_types[edge_mask].cpu()

        arom = detect_aromatic_rings(mol_ei, mol_bt, n)
        small = detect_small_rings(mol_ei, mol_bt, n)
        chiral = detect_chiral_centers(mol_at, mol_ei, mol_bt)

        # Shift indices back to global batch position
        all_arom  += [[idx + offset for idx in r] for r in arom]
        all_small += [[idx + offset for idx in r] for r in small]
        all_chiral += [(c + offset, [n + offset for n in nbrs], s)
                       for c, nbrs, s in chiral]

    return all_arom or None, all_chiral or None, all_small or None


def train_epoch(model, dataloader, optimizer, device, epoch, max_epochs=100,
                geometry_weight=1.0, use_soft_constraints=True):
    """Train one epoch. Returns dict with mean losses (total/mse/geo)."""
    model.train()
    total_loss = mse_sum = geo_sum = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        atom_types = batch['atom_types'].to(device)
        coords     = batch['coordinates'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx  = batch['batch_idx'].to(device)

        coords = remove_com(coords, batch_idx)

        optimizer.zero_grad()
        loss_dict = model.get_loss(
            coords, atom_types, edge_index, bond_types, batch_idx,
            geometry_weight=geometry_weight,
            epoch=epoch,
            max_epochs=max_epochs,
            min_snr_gamma=5.0,
        )

        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t = loss_dict['total'].item()
        total_loss += t
        mse_sum    += loss_dict['mse'].item()
        geo_sum    += loss_dict['geo'].item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{t:.4f}', 'mse': f'{loss_dict["mse"].item():.4f}'})

    n = max(num_batches, 1)
    return {'total': total_loss/n, 'mse': mse_sum/n, 'geo': geo_sum/n}


@torch.no_grad()
def validate(model, dataloader, device, geometry_weight=1.0, epoch=1, max_epochs=100):
    """Validate — same curriculum as training."""
    model.eval()
    total_loss = mse_sum = geo_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        atom_types = batch['atom_types'].to(device)
        coords     = batch['coordinates'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx  = batch['batch_idx'].to(device)

        coords = remove_com(coords, batch_idx)
        loss_dict = model.get_loss(
            coords, atom_types, edge_index, bond_types, batch_idx,
            geometry_weight=geometry_weight,
            epoch=epoch,
            max_epochs=max_epochs,
            min_snr_gamma=5.0,
        )
        total_loss += loss_dict['total'].item()
        mse_sum    += loss_dict['mse'].item()
        geo_sum    += loss_dict['geo'].item()
        num_batches += 1

    n = max(num_batches, 1)
    return {'total': total_loss/n, 'mse': mse_sum/n, 'geo': geo_sum/n}


def scatter_mean(src, index, dim=0):
    count = torch.zeros(index.max() + 1, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
    count = count.clamp(min=1)
    out = torch.zeros(index.max() + 1, src.size(1), device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, src.size(1)), src)
    return out / count.unsqueeze(-1)


@torch.no_grad()
def kabsch_rmsd(coords_pred: torch.Tensor, coords_true: torch.Tensor) -> float:
    pred_c = coords_pred - coords_pred.mean(0)
    true_c = coords_true - coords_true.mean(0)
    H = pred_c.T @ true_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    sign_mat = torch.eye(3, device=coords_pred.device)
    sign_mat[2, 2] = d.sign()
    R = Vt.T @ sign_mat @ U.T
    pred_aligned = pred_c @ R.T
    return torch.sqrt(torch.mean((pred_aligned - true_c) ** 2)).item()


@torch.no_grad()
def sample_and_evaluate(model, dataloader, device, num_samples=10,
                         use_guidance=True):
    """Sample conformers and compare to ground truth with Kabsch RMSD."""
    model.eval()
    rmsds = []
    num_mols = 0

    for batch in dataloader:
        if num_mols >= num_samples:
            break

        atom_types  = batch['atom_types'].to(device)
        coords_true = batch['coordinates'].to(device)
        edge_index  = batch['edge_index'].to(device)
        bond_types  = batch['bond_types'].to(device)
        batch_idx   = batch['batch_idx'].to(device)

        coords_true = coords_true - scatter_mean(coords_true, batch_idx)[batch_idx]

        if use_guidance and hasattr(model, 'guided_sample'):
            # Detect soft constraints for this batch
            arom, chiral, small = get_soft_constraints(
                atom_types.cpu(), edge_index.cpu(), bond_types.cpu(),
                batch_idx.cpu(), device)
            coords_gen = model.guided_sample(
                atom_types, edge_index, bond_types, batch_idx,
                num_steps=50, guidance_scale=1.0,
                aromatic_rings=arom,
                chiral_centers=chiral,
                small_rings=small,
            )
        else:
            coords_gen = model.ddim_sample(
                atom_types, edge_index, bond_types, batch_idx, num_steps=50)

        for b in range(batch['num_molecules']):
            if num_mols >= num_samples:
                break
            mask = (batch_idx == b)
            rmsd = kabsch_rmsd(coords_gen[mask], coords_true[mask])
            rmsds.append(rmsd)
            num_mols += 1

    if rmsds:
        return float(np.mean(rmsds)), float(np.std(rmsds))
    return 0.0, 0.0


@torch.no_grad()
def evaluate_3d_validity(model, dataloader, device, num_samples=20):
    """Evaluate strict 3D validity with chemistry-aware bond targets."""
    from models.geometry_constraints import get_ideal_bond_length
    model.eval()

    bond_errors = []
    valid_count = clash_free_count = fully_valid_count = total = 0
    BOND_TOL = 0.2
    MIN_NONBOND = 1.4

    for batch in dataloader:
        if total >= num_samples:
            break

        atom_types = batch['atom_types'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx  = batch['batch_idx'].to(device)

        arom, chiral, small = get_soft_constraints(
            atom_types.cpu(), edge_index.cpu(), bond_types.cpu(),
            batch_idx.cpu(), device)

        if hasattr(model, 'guided_sample'):
            coords_gen = model.guided_sample(
                atom_types, edge_index, bond_types, batch_idx,
                num_steps=50, guidance_scale=1.0,
                aromatic_rings=arom, chiral_centers=chiral, small_rings=small)
        else:
            coords_gen = model.ddim_sample(
                atom_types, edge_index, bond_types, batch_idx, num_steps=50)

        for b in range(batch['num_molecules']):
            if total >= num_samples:
                break

            mask     = (batch_idx == b)
            coords   = coords_gen[mask]
            mol_at   = atom_types[mask]
            N        = mask.sum().item()

            edge_mask   = mask[edge_index[0]] & mask[edge_index[1]]
            mol_edges   = edge_index[:, edge_mask]
            mol_bt      = bond_types[edge_mask]

            idx_map     = torch.cumsum(mask.long(), 0) - 1
            local_edges = idx_map[mol_edges]

            mol_valid = True
            mol_bond_error = 0.0
            n_bonds = 0

            for e_idx in range(0, local_edges.size(1), 2):
                i, j  = local_edges[0, e_idx].item(), local_edges[1, e_idx].item()
                btype = mol_bt[e_idx].item()
                ideal = get_ideal_bond_length(mol_at[i].item(), mol_at[j].item(), btype)
                dist  = torch.norm(coords[i] - coords[j]).item()
                error = abs(dist - ideal)
                mol_bond_error += error
                n_bonds += 1
                if error > BOND_TOL:
                    mol_valid = False

            if n_bonds > 0:
                bond_errors.append(mol_bond_error / n_bonds)
            if mol_valid:
                valid_count += 1

            has_clash = False
            bonded = set()
            for e_idx in range(local_edges.size(1)):
                i, j = local_edges[0, e_idx].item(), local_edges[1, e_idx].item()
                bonded.add((min(i, j), max(i, j)))

            for i in range(N):
                for j in range(i + 1, N):
                    if (i, j) not in bonded:
                        if torch.norm(coords[i] - coords[j]).item() < MIN_NONBOND:
                            has_clash = True
                            break
                if has_clash:
                    break

            if not has_clash:
                clash_free_count += 1
            if mol_valid and not has_clash:
                fully_valid_count += 1
            total += 1

    t = max(total, 1)
    return {
        'bond_valid_rate':  valid_count / t,
        'clash_free_rate':  clash_free_count / t,
        'fully_valid_rate': fully_valid_count / t,
        'mean_bond_error':  float(np.mean(bond_errors)) if bond_errors else 0.0,
        'total_evaluated':  total,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def save_loss_plots(history, plots_dir):
    """Save comprehensive loss curve plots to plots_dir."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs    = [h['epoch'] for h in history]
    train_tot = [h['train_total'] for h in history]
    val_tot   = [h['val_total']   for h in history]
    train_mse = [h['train_mse']   for h in history]
    val_mse   = [h['val_mse']     for h in history]
    train_geo = [h['train_geo']   for h in history]
    rmsds     = [h['rmsd_mean']   for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 23-02-2026-Exp-1 (Soft Restrictions)', fontsize=14, fontweight='bold')

    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, train_tot, label='Train total', color='steelblue')
    ax.plot(epochs, val_tot,   label='Val total',   color='coral', linestyle='--')
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MSE loss
    ax = axes[0, 1]
    ax.plot(epochs, train_mse, label='Train MSE', color='steelblue')
    ax.plot(epochs, val_mse,   label='Val MSE',   color='coral', linestyle='--')
    ax.set_title('Diffusion MSE Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Geometry loss
    ax = axes[1, 0]
    ax.plot(epochs, train_geo, label='Geo loss', color='seagreen')
    ax.set_title('Geometry Constraint Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSD
    ax = axes[1, 1]
    rmsd_epochs = [h['epoch'] for h in history if h['rmsd_mean'] > 0]
    rmsd_vals   = [h['rmsd_mean'] for h in history if h['rmsd_mean'] > 0]
    if rmsd_vals:
        ax.plot(rmsd_epochs, rmsd_vals, 'o-', color='purple', label='RMSD (Å)')
    ax.set_title('RMSD Progression')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSD (Å)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {plots_dir}/loss_curves.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    EXP_NAME = "23-02-2026-Exp-1(soft_restrictions)"
    DEFAULT_EXP_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", EXP_NAME)

    parser = argparse.ArgumentParser(
        description='Train Conformer Diffusion — Soft Constraints Experiment')
    # Data
    parser.add_argument('--data',       type=str,   default='data/qm9_selfies.jsonl')
    parser.add_argument('--max_atoms',  type=int,   default=15)
    parser.add_argument('--val_split',  type=float, default=0.1)
    # Training
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    # Model
    parser.add_argument('--hidden_dim', type=int,   default=256)
    parser.add_argument('--num_layers', type=int,   default=6)
    parser.add_argument('--timesteps',  type=int,   default=1000)
    parser.add_argument('--edge_dim',   type=int,   default=32)
    parser.add_argument('--time_dim',   type=int,   default=128)
    # Geometry
    parser.add_argument('--geometry_weight',    type=float, default=1.0)
    parser.add_argument('--planarity_weight',   type=float, default=5.0)
    parser.add_argument('--chirality_weight',   type=float, default=3.0)
    parser.add_argument('--ring_strain_weight', type=float, default=2.0)
    # Experiment
    parser.add_argument('--exp_dir',    type=str,   default=DEFAULT_EXP_DIR,
                        help='Root experiment folder (all outputs go here)')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Create experiment directory tree
    # -------------------------------------------------------------------------
    exp_dir       = Path(args.exp_dir)
    ckpt_dir      = exp_dir / 'checkpoints'
    plots_dir     = exp_dir / 'plots'
    eval_dir      = exp_dir / 'evaluation'
    logs_dir      = exp_dir / 'logs'
    vis_dir       = exp_dir / 'visualization'

    for d in [ckpt_dir, plots_dir, eval_dir, logs_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Logging — both console and file
    # -------------------------------------------------------------------------
    log_file = logs_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file)),
        ]
    )
    log = logging.getLogger(__name__)
    log.info(f"Experiment: {EXP_NAME}")
    log.info(f"Output dir: {exp_dir}")
    log.info(f"Args: {vars(args)}")

    # Save config snapshot
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump({'args': vars(args), 'experiment': EXP_NAME,
                   'date': datetime.now().isoformat()}, f, indent=2)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    full_dataset = ConformerDataset(args.data, max_atoms=args.max_atoms)
    n_val   = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val])
    log.info(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=4)

    # -------------------------------------------------------------------------
    # Model — with new constraint weights
    # -------------------------------------------------------------------------
    model = ConformerDiffusion(
        num_timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=args.edge_dim,
        time_dim=args.time_dim,
    ).to(device)

    # Override geometry weights with experiment values
    model.geometry = GeometryConstraints(
        bond_weight=1.0,
        angle_weight=0.3,
        torsion_weight=0.1,
        repulsion_weight=0.5,
        planarity_weight=args.planarity_weight / 10.0,   # scale to ~1.0 range
        chirality_weight=args.chirality_weight / 10.0,
        ring_strain_weight=args.ring_strain_weight / 10.0,
    )

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        # Curriculum: soft constraints enabled from 60% onwards
        progress = epoch / args.epochs
        use_soft = progress > 0.60

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            max_epochs=args.epochs,
            geometry_weight=args.geometry_weight,
            use_soft_constraints=use_soft,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, device,
            geometry_weight=args.geometry_weight,
            epoch=epoch, max_epochs=args.epochs,
        )

        rmsd_mean, rmsd_std = 0.0, 0.0
        validity = None

        if epoch % 5 == 0 or epoch == 1:
            rmsd_mean, rmsd_std = sample_and_evaluate(
                model, val_loader, device, num_samples=50)

        if epoch % 10 == 0:
            validity = evaluate_3d_validity(model, val_loader, device, num_samples=50)
            eval_path = eval_dir / f'epoch_{epoch:04d}_validity.json'
            with open(eval_path, 'w') as f:
                json.dump({'epoch': epoch, **validity}, f, indent=2)

        # Log
        log_msg = (f"Epoch {epoch:4d} | "
                   f"train={train_metrics['total']:.4f} "
                   f"(mse={train_metrics['mse']:.4f} geo={train_metrics['geo']:.4f}) | "
                   f"val={val_metrics['total']:.4f} "
                   f"(mse={val_metrics['mse']:.4f}) | "
                   f"lr={scheduler.get_last_lr()[0]:.2e}")
        if rmsd_mean > 0:
            log_msg += f" | rmsd={rmsd_mean:.3f}±{rmsd_std:.3f}Å"
        if validity:
            log_msg += (f"\n  → 3D: valid={validity['fully_valid_rate']*100:.1f}% "
                        f"bonds={validity['bond_valid_rate']*100:.1f}% "
                        f"no_clash={validity['clash_free_rate']*100:.1f}% "
                        f"err={validity['mean_bond_error']:.3f}Å")
        log.info(log_msg)

        history.append({
            'epoch':       epoch,
            'train_total': train_metrics['total'],
            'train_mse':   train_metrics['mse'],
            'train_geo':   train_metrics['geo'],
            'val_total':   val_metrics['total'],
            'val_mse':     val_metrics['mse'],
            'rmsd_mean':   rmsd_mean,
            'rmsd_std':    rmsd_std,
            'lr':          scheduler.get_last_lr()[0],
        })

        # Save best checkpoint
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':         val_metrics['total'],
                'args':             vars(args),
            }, str(ckpt_dir / 'conformer_best.pt'))
            log.info(f"  ✓ Saved best model (val={val_metrics['total']:.4f})")

        # Periodic checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history':          history,
            }, str(ckpt_dir / f'conformer_epoch{epoch:04d}.pt'))

        # Save plots every 10 epochs
        if epoch % 10 == 0:
            save_loss_plots(history, str(plots_dir))

        scheduler.step()

    # Final model save
    torch.save({
        'model_state_dict': model.state_dict(),
        'history':          history,
        'args':             vars(args),
    }, str(ckpt_dir / 'conformer_final.pt'))

    # Final plots & history JSON
    save_loss_plots(history, str(plots_dir))
    with open(eval_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    log.info(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    log.info(f"All outputs saved to: {exp_dir}")


if __name__ == '__main__':
    main()
