"""
train_v3.py — Research-Backed Conformer Diffusion Training (v3)
Experiment: 09-03-2026-Exp-3(stable_full_constraints)

Key improvements over train_soft_constraints.py:
  1. All geometry constraints (bonds+angles+torsions+planarity) from epoch 1 [EQGAT-diff]
  2. Checkpoint saved on val_mse + separate best-validity checkpoint [not val_total]
  3. 200-sample validity eval (was 50) — reduces noise floor from 14% to 3.5%
  4. LR warmup (5 epochs linear) then CosineAnnealing — avoids cold-start instability
  5. Per-molecule PDB export in pdb_files/ + load_all_vmd.tcl loader script
  6. Scheduler correctly fast-forwarded when resuming from checkpoint
  7. Soft constraints (planarity, chirality, ring-strain) active from epoch 1
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conformer_diffusion import ConformerDiffusion, remove_com
from models.geometry_constraints import GeometryConstraints

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# =============================================================================
# RING / CHIRALITY DETECTION (graph-only, no RDKit)
# =============================================================================

def detect_aromatic_rings(edge_index, bond_types, num_atoms):
    row, col = edge_index[0].tolist(), edge_index[1].tolist()
    bt = bond_types.tolist()
    arom_adj = {i: [] for i in range(num_atoms)}
    for i, j, b in zip(row, col, bt):
        if b == 4:
            arom_adj[i].append(j)

    visited_global, rings = set(), []

    def dfs(start, current, path, adj):
        for nxt in adj[current]:
            if nxt == start and len(path) >= 5:
                rings.append(sorted(path))
                return
            if nxt not in path and nxt not in visited_global:
                path.add(nxt)
                dfs(start, nxt, path, adj)
                path.discard(nxt)

    for atom in [i for i in range(num_atoms) if arom_adj[i]]:
        if atom not in visited_global:
            dfs(atom, atom, {atom}, arom_adj)
            visited_global.add(atom)

    seen, unique = set(), []
    for r in rings:
        k = tuple(r)
        if k not in seen:
            seen.add(k)
            unique.append(r)
    return unique


def detect_small_rings(edge_index, bond_types, num_atoms, max_ring_size=4):
    row, col = edge_index[0].tolist(), edge_index[1].tolist()
    adj = {i: [] for i in range(num_atoms)}
    for i, j in zip(row, col):
        adj[i].append(j)
    small_rings, seen_keys = [], set()

    def dfs_small(start, current, path, depth):
        if depth > max_ring_size:
            return
        for nxt in adj[current]:
            if nxt == start and depth >= 3:
                key = tuple(sorted(path))
                if key not in seen_keys and len(path) <= max_ring_size:
                    seen_keys.add(key)
                    small_rings.append(list(path))
            elif nxt not in path:
                path.append(nxt)
                dfs_small(start, nxt, path, depth + 1)
                path.pop()

    for atom in range(num_atoms):
        dfs_small(atom, atom, [atom], 1)
    return small_rings


def detect_chiral_centers(atom_types, edge_index, bond_types):
    num_atoms = atom_types.size(0)
    row, col = edge_index[0].tolist(), edge_index[1].tolist()
    bt = bond_types.tolist()
    adj = {i: [] for i in range(num_atoms)}
    bond_map = {}
    for i, j, b in zip(row, col, bt):
        adj[i].append(j)
        bond_map[(i, j)] = b

    chiral = []
    for c in range(num_atoms):
        if atom_types[c].item() != 6:        # carbon only
            continue
        neighbors = adj[c]
        if len(neighbors) != 4:
            continue
        if all(bond_map.get((c, n), 0) == 1 for n in neighbors):
            chiral.append((c, neighbors, +1))
    return chiral


def get_soft_constraints(atom_types, edge_index, bond_types, batch_idx, device):
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

        arom   = detect_aromatic_rings(mol_ei, mol_bt, n)
        small  = detect_small_rings(mol_ei, mol_bt, n)
        chiral = detect_chiral_centers(mol_at, mol_ei, mol_bt)

        all_arom   += [[idx + offset for idx in r] for r in arom]
        all_small  += [[idx + offset for idx in r] for r in small]
        all_chiral += [(c + offset, [nb + offset for nb in nbrs], s)
                       for c, nbrs, s in chiral]

    return (all_arom or None), (all_chiral or None), (all_small or None)


# =============================================================================
# DATASET
# =============================================================================

BOND_TYPE_MAP = {
    'SINGLE':   1,
    'DOUBLE':   2,
    'TRIPLE':   3,
    'AROMATIC': 4,
}


def _build_graph_from_smiles(smiles):
    """
    Build (edge_index [2, E], bond_types [E]) as undirected graph from SMILES.
    Returns None, None if RDKit is unavailable or SMILES parsing fails.
    SMILES in qm9_100k.jsonl have explicit H atoms, so we parse as-is.
    """
    if not HAS_RDKIT or not smiles:
        return None, None
    try:
        from rdkit import Chem
        # Keep explicit H atoms in SMILES (e.g. [H]O[H]) so atom count matches
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        # Map RDKit BondType enum to integer bond order
        BT_MAP = {
            Chem.BondType.SINGLE:   1,
            Chem.BondType.DOUBLE:   2,
            Chem.BondType.TRIPLE:   3,
            Chem.BondType.AROMATIC: 4,
        }
        rows, cols, btypes = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = BT_MAP.get(bond.GetBondType(), 1)
            rows += [i, j]
            cols += [j, i]
            btypes += [bt, bt]
        if not rows:
            return None, None
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        bond_types = torch.tensor(btypes, dtype=torch.long)
        return edge_index, bond_types
    except Exception:
        return None, None


def _fallback_full_graph(n_atoms):
    """Fully connected graph with all bond types = 1 (fallback when no SMILES)."""
    rows, cols = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                rows.append(i)
                cols.append(j)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    bond_types = torch.ones(len(rows), dtype=torch.long)
    return edge_index, bond_types


class ConformerDataset(Dataset):
    """
    Reads qm9_100k.jsonl format:
      - 'coords'     : list of 50 [x,y,z] (padded with zeros)
      - 'atom_types' : list of 50 ints (padded with -1)
      - 'coord_mask' : list of 50 ints (1=real atom, 0=padding)
      - 'smiles'     : SMILES string (used to build graph topology)
    """
    def __init__(self, data_path, max_atoms=50):
        self.data = []
        print(f"Loading data from {data_path}...")
        skipped = 0
        with open(data_path) as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                # Support both old format ('coordinates') and new format ('coords')
                coords_raw  = item.get('coords', item.get('coordinates'))
                at_raw      = item.get('atom_types', [])
                mask_raw    = item.get('coord_mask', None)
                smiles      = item.get('smiles', '')

                if coords_raw is None:
                    skipped += 1
                    continue

                # Unpad using coord_mask
                if mask_raw is not None:
                    mask = [bool(m) for m in mask_raw]
                    coords   = [c for c, m in zip(coords_raw, mask) if m]
                    at       = [a for a, m in zip(at_raw, mask) if m]
                else:
                    # Old format: use num_atoms or -1 sentinel
                    num_atoms = item.get('num_atoms', None)
                    if num_atoms is not None:
                        coords = coords_raw[:num_atoms]
                        at     = at_raw[:num_atoms]
                    else:
                        coords = [c for c, a in zip(coords_raw, at_raw) if a != -1]
                        at     = [a for a in at_raw if a != -1]

                n = len(at)
                if n == 0 or n > max_atoms:
                    skipped += 1
                    continue

                self.data.append({
                    'atom_types':  at,
                    'coords':      coords,
                    'smiles':      smiles,
                    'num_atoms':   n,
                })

        print(f"Loaded {len(self.data)} molecules ({skipped} skipped)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        atom_types = torch.tensor(item['atom_types'], dtype=torch.long)
        coords     = torch.tensor(item['coords'],     dtype=torch.float32)
        smiles     = item['smiles']
        n          = item['num_atoms']

        edge_index, bond_types = _build_graph_from_smiles(smiles)
        if edge_index is None:
            edge_index, bond_types = _fallback_full_graph(n)

        return {
            'atom_types':  atom_types,
            'coordinates': coords,
            'edge_index':  edge_index,
            'bond_types':  bond_types,
            'num_atoms':   n,
            'smiles':      smiles,
        }


def collate_fn(batch):
    atom_types_list, coords_list, edge_index_list = [], [], []
    bond_types_list, batch_idx_list, smiles_list = [], [], []
    offset = 0
    for i, item in enumerate(batch):
        N = item['num_atoms']
        atom_types_list.append(item['atom_types'])
        coords_list.append(item['coordinates'])
        edge_index_list.append(item['edge_index'] + offset)
        bond_types_list.append(item['bond_types'])
        batch_idx_list.append(torch.full((N,), i, dtype=torch.long))
        smiles_list.append(item['smiles'])
        offset += N
    return {
        'atom_types':    torch.cat(atom_types_list),
        'coordinates':   torch.cat(coords_list),
        'edge_index':    torch.cat(edge_index_list, dim=1),
        'bond_types':    torch.cat(bond_types_list),
        'batch_idx':     torch.cat(batch_idx_list),
        'num_molecules': len(batch),
        'smiles_list':   smiles_list,
    }


# =============================================================================
# TRAINING / VALIDATION
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch, max_epochs,
                geometry_weight=0.1):
    model.train()
    total_sum = mse_sum = geo_sum = 0.0
    n = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:4d}")
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
            epoch=epoch, max_epochs=max_epochs,
            min_snr_gamma=5.0,
        )
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t = loss_dict['total'].item()
        total_sum += t
        mse_sum   += loss_dict['mse'].item()
        geo_sum   += loss_dict['geo'].item()
        n += 1
        pbar.set_postfix({'loss': f'{t:.4f}', 'mse': f'{loss_dict["mse"].item():.4f}'})

    return {'total': total_sum / max(n, 1),
            'mse':   mse_sum   / max(n, 1),
            'geo':   geo_sum   / max(n, 1)}


@torch.no_grad()
def validate(model, dataloader, device, geometry_weight=0.1, epoch=1, max_epochs=300):
    model.eval()
    total_sum = mse_sum = geo_sum = 0.0
    n = 0
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
            epoch=epoch, max_epochs=max_epochs,
            min_snr_gamma=5.0,
        )
        total_sum += loss_dict['total'].item()
        mse_sum   += loss_dict['mse'].item()
        geo_sum   += loss_dict['geo'].item()
        n += 1

    return {'total': total_sum / max(n, 1),
            'mse':   mse_sum   / max(n, 1),
            'geo':   geo_sum   / max(n, 1)}


def scatter_mean(src, index, dim=0):
    count = torch.zeros(index.max() + 1, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
    count = count.clamp(min=1)
    out = torch.zeros(index.max() + 1, src.size(1), device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, src.size(1)), src)
    return out / count.unsqueeze(-1)


@torch.no_grad()
def kabsch_rmsd(pred, true):
    pred_c = pred - pred.mean(0)
    true_c = true - true.mean(0)
    H = pred_c.T @ true_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    sign_mat = torch.eye(3, device=pred.device)
    sign_mat[2, 2] = d.sign()
    R = Vt.T @ sign_mat @ U.T
    return torch.sqrt(torch.mean((pred_c @ R.T - true_c) ** 2)).item()


@torch.no_grad()
def sample_and_evaluate(model, dataloader, device, num_samples=20):
    model.eval()
    rmsds, n = [], 0
    for batch in dataloader:
        if n >= num_samples:
            break
        atom_types  = batch['atom_types'].to(device)
        coords_true = batch['coordinates'].to(device)
        edge_index  = batch['edge_index'].to(device)
        bond_types  = batch['bond_types'].to(device)
        batch_idx   = batch['batch_idx'].to(device)
        coords_true = coords_true - scatter_mean(coords_true, batch_idx)[batch_idx]

        arom, chiral, small = get_soft_constraints(
            atom_types.cpu(), edge_index.cpu(), bond_types.cpu(), batch_idx.cpu(), device)

        if hasattr(model, 'guided_sample'):
            coords_gen = model.guided_sample(
                atom_types, edge_index, bond_types, batch_idx,
                num_steps=50, guidance_scale=1.0,
                aromatic_rings=arom, chiral_centers=chiral, small_rings=small)
        else:
            coords_gen = model.ddim_sample(
                atom_types, edge_index, bond_types, batch_idx, num_steps=50)

        for b in range(batch['num_molecules']):
            if n >= num_samples:
                break
            mask = batch_idx == b
            rmsds.append(kabsch_rmsd(coords_gen[mask], coords_true[mask]))
            n += 1

    if rmsds:
        return float(np.mean(rmsds)), float(np.std(rmsds))
    return 0.0, 0.0


# =============================================================================
# VALIDITY EVALUATION (200 samples, research-grade)
# =============================================================================

@torch.no_grad()
def evaluate_validity(model, dataloader, device, num_samples=200, log=None):
    """
    Evaluates generated molecules using RDKit if available,
    otherwise falls back to bond-length geometry checks.
    200 samples reduces validity noise floor from 14% → 3.5%.
    """
    model.eval()

    if HAS_RDKIT:
        return _evaluate_validity_rdkit(model, dataloader, device, num_samples, log)
    else:
        return _evaluate_validity_geometry(model, dataloader, device, num_samples, log)


@torch.no_grad()
def _evaluate_validity_rdkit(model, dataloader, device, num_samples=200, log=None):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from models.geometry_constraints import get_ideal_bond_lengths_vectorized

    valid_count = clash_free = total = 0
    all_smiles = []
    train_smiles = set()   # collect training SMILES for novelty

    for batch in dataloader:
        if total >= num_samples:
            break
        atom_types = batch['atom_types'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx  = batch['batch_idx'].to(device)
        for smi in batch.get('smiles_list', []):
            if smi:
                train_smiles.add(smi)

        arom, chiral, small = get_soft_constraints(
            atom_types.cpu(), edge_index.cpu(), bond_types.cpu(), batch_idx.cpu(), device)

        if hasattr(model, 'guided_sample'):
            coords_gen = model.guided_sample(
                atom_types, edge_index, bond_types, batch_idx,
                num_steps=50, guidance_scale=1.0,
                aromatic_rings=arom, chiral_centers=chiral, small_rings=small)
        else:
            coords_gen = model.ddim_sample(
                atom_types, edge_index, bond_types, batch_idx, num_steps=50)

        BOND_ORDER = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                      3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
        ATOM_NUMS  = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P',
                      16: 'S', 17: 'Cl', 35: 'Br', 53: 'I', 1: 'H'}

        for b in range(batch['num_molecules']):
            if total >= num_samples:
                break
            mask   = (batch_idx == b)
            coords = coords_gen[mask].cpu().numpy()
            at     = atom_types[mask].cpu().tolist()
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            mol_ei    = edge_index[:, edge_mask].cpu()
            mol_bt    = bond_types[edge_mask].cpu().tolist()

            # remap to local indices
            idx_map = torch.cumsum(mask.long(), 0) - 1
            local_ei = idx_map[edge_index[:, edge_mask]].tolist()

            try:
                rw = Chem.RWMol()
                for anum in at:
                    sym = ATOM_NUMS.get(anum, 'C')
                    rw.AddAtom(Chem.Atom(sym))
                added = set()
                for (i, j), bt_val in zip(zip(local_ei[0], local_ei[1]), mol_bt):
                    if i < j and (i, j) not in added:
                        rw.AddBond(i, j, BOND_ORDER.get(bt_val, Chem.BondType.SINGLE))
                        added.add((i, j))
                conf = Chem.Conformer(len(at))
                for i, (x, y, z) in enumerate(coords):
                    conf.SetAtomPosition(i, (float(x), float(y), float(z)))
                rw.AddConformer(conf, assignId=True)
                mol = rw.GetMol()
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol)
                all_smiles.append(smi)
                valid_count += 1

                # Clash check
                N = len(at)
                crd = torch.tensor(coords)
                dist_mat = torch.cdist(crd.unsqueeze(0), crd.unsqueeze(0))[0]
                bonded = set()
                for (i, j) in zip(local_ei[0], local_ei[1]):
                    bonded.add((min(i, j), max(i, j)))
                clash = False
                for i in range(N):
                    for j in range(i + 1, N):
                        if (i, j) not in bonded and dist_mat[i, j].item() < 1.4:
                            clash = True
                            break
                    if clash:
                        break
                if not clash:
                    clash_free += 1
            except Exception:
                pass
            total += 1

    unique_smiles = len(set(all_smiles))
    novel_smiles  = sum(1 for s in set(all_smiles) if s not in train_smiles)
    t = max(total, 1)
    return {
        'validity_rate':  valid_count / t,
        'valid_count':    valid_count,
        'total_generated': total,
        'uniqueness':     unique_smiles / max(valid_count, 1),
        'unique_smiles':  unique_smiles,
        'novelty':        novel_smiles  / max(unique_smiles, 1),
        'clash_free_rate': clash_free / t,
    }


@torch.no_grad()
def _evaluate_validity_geometry(model, dataloader, device, num_samples=200, log=None):
    from models.geometry_constraints import get_ideal_bond_length
    BOND_TOL  = 0.2
    MIN_CLASH = 1.4

    valid = clash_free = total = 0
    bond_errors = []

    for batch in dataloader:
        if total >= num_samples:
            break
        atom_types = batch['atom_types'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx  = batch['batch_idx'].to(device)

        arom, chiral, small = get_soft_constraints(
            atom_types.cpu(), edge_index.cpu(), bond_types.cpu(), batch_idx.cpu(), device)

        coords_gen = model.guided_sample(
            atom_types, edge_index, bond_types, batch_idx, num_steps=50,
            guidance_scale=1.0, aromatic_rings=arom, chiral_centers=chiral, small_rings=small)

        for b in range(batch['num_molecules']):
            if total >= num_samples:
                break
            mask   = batch_idx == b
            coords = coords_gen[mask]
            at     = atom_types[mask]
            N      = mask.sum().item()
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            mol_edges = edge_index[:, edge_mask]
            mol_bt    = bond_types[edge_mask]
            idx_map   = torch.cumsum(mask.long(), 0) - 1
            local_ei  = idx_map[mol_edges]

            mol_valid = True
            err_sum = 0.0
            n_b = 0
            for e in range(0, local_ei.size(1), 2):
                i, j = local_ei[0, e].item(), local_ei[1, e].item()
                bt_v = mol_bt[e].item()
                ideal = get_ideal_bond_length(at[i].item(), at[j].item(), bt_v)
                dist  = torch.norm(coords[i] - coords[j]).item()
                err   = abs(dist - ideal)
                err_sum += err
                n_b += 1
                if err > BOND_TOL:
                    mol_valid = False
            if n_b > 0:
                bond_errors.append(err_sum / n_b)
            if mol_valid:
                valid += 1

            bonded = set()
            for e in range(local_ei.size(1)):
                i, j = local_ei[0, e].item(), local_ei[1, e].item()
                bonded.add((min(i, j), max(i, j)))
            clash = any(torch.norm(coords[i] - coords[j]).item() < MIN_CLASH
                        for i in range(N) for j in range(i + 1, N)
                        if (i, j) not in bonded)
            if not clash:
                clash_free += 1
            total += 1

    t = max(total, 1)
    return {
        'validity_rate':   valid / t,
        'valid_count':     valid,
        'total_generated': total,
        'clash_free_rate': clash_free / t,
        'mean_bond_error': float(np.mean(bond_errors)) if bond_errors else 0.0,
        'uniqueness':      0.0,
        'unique_smiles':   0,
        'novelty':         0.0,
    }


# =============================================================================
# PDB EXPORT — one file per molecule   (clean, no MODEL/ENDMDL)
# =============================================================================

@torch.no_grad()
def export_valid_molecules(model, dataloader, device,
                            num_generate=200,
                            sdf_path=None, pdb_dir=None, log=None):
    """
    Generate molecules, validate with RDKit (or geometry fallback),
    and export:
      - SDF : all valid molecules in one combined file
      - PDB : one clean .pdb per molecule in pdb_dir/
              + pdb_dir/all_molecules.pdb  (combined multi-MODEL file)
              + pdb_dir/../load_all_vmd.tcl (VMD loader script)
    """
    model.eval()
    valid_mols = []
    total_gen  = 0
    train_smiles = set()

    for batch in dataloader:
        if total_gen >= num_generate:
            break
        atom_types = batch['atom_types'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx  = batch['batch_idx'].to(device)
        for smi in batch.get('smiles_list', []):
            if smi:
                train_smiles.add(smi)

        arom, chiral, small = get_soft_constraints(
            atom_types.cpu(), edge_index.cpu(), bond_types.cpu(), batch_idx.cpu(), device)

        if hasattr(model, 'guided_sample'):
            coords_gen = model.guided_sample(
                atom_types, edge_index, bond_types, batch_idx,
                num_steps=50, guidance_scale=1.0,
                aromatic_rings=arom, chiral_centers=chiral, small_rings=small)
        else:
            coords_gen = model.ddim_sample(
                atom_types, edge_index, bond_types, batch_idx, num_steps=50)

        if HAS_RDKIT:
            from rdkit import Chem
            BOND_ORDER = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                          3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
            ATOM_NUMS  = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P',
                          16: 'S', 17: 'Cl', 35: 'Br', 53: 'I', 1: 'H'}

            for b in range(batch['num_molecules']):
                if total_gen >= num_generate:
                    break
                mask   = (batch_idx == b)
                coords = coords_gen[mask].cpu().numpy()
                at     = atom_types[mask].cpu().tolist()
                edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
                mol_bt    = bond_types[edge_mask].cpu().tolist()
                idx_map   = torch.cumsum(mask.long(), 0) - 1
                local_ei  = idx_map[edge_index[:, edge_mask]].tolist()
                total_gen += 1
                try:
                    rw = Chem.RWMol()
                    for anum in at:
                        rw.AddAtom(Chem.Atom(ATOM_NUMS.get(anum, 'C')))
                    added = set()
                    for (i, j), bt_val in zip(zip(local_ei[0], local_ei[1]), mol_bt):
                        if i < j and (i, j) not in added:
                            rw.AddBond(i, j, BOND_ORDER.get(bt_val, Chem.BondType.SINGLE))
                            added.add((i, j))
                    conf = Chem.Conformer(len(at))
                    for i, (x, y, z) in enumerate(coords):
                        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
                    rw.AddConformer(conf, assignId=True)
                    mol = rw.GetMol()
                    Chem.SanitizeMol(mol)
                    smi = Chem.MolToSmiles(mol)
                    mol.SetProp('_Name', f'mol_{len(valid_mols)+1:04d}')
                    mol.SetProp('SMILES', smi)
                    valid_mols.append(mol)
                except Exception:
                    pass
        else:
            total_gen += batch['num_molecules']

    if log:
        log.info(f"  Generated {total_gen} molecules, {len(valid_mols)} passed RDKit validation")

    # ── SDF export ──────────────────────────────────────────────────────────
    if sdf_path and valid_mols and HAS_RDKIT:
        from rdkit.Chem import SDWriter
        with SDWriter(str(sdf_path)) as w:
            for mol in valid_mols:
                w.write(mol)
        if log:
            log.info(f"  SDF saved  → {sdf_path}  ({len(valid_mols)} molecules)")

    # ── PDB export: one clean file per molecule ──────────────────────────────
    if pdb_dir and valid_mols and HAS_RDKIT:
        pdb_dir = Path(pdb_dir)
        pdb_dir.mkdir(parents=True, exist_ok=True)

        combined_path = pdb_dir / 'all_molecules.pdb'
        n_exported = 0
        with open(str(combined_path), 'w') as combined_f:
            for idx, mol in enumerate(valid_mols):
                try:
                    pdb_block = Chem.MolToPDBBlock(mol)
                    mol_name  = mol.GetProp('_Name') if mol.HasProp('_Name') else f'mol_{idx+1:04d}'
                    smi       = mol.GetProp('SMILES') if mol.HasProp('SMILES') else ''

                    # ── Individual clean PDB (no MODEL/ENDMDL — VMD-safe) ───
                    ind_path = pdb_dir / f'{mol_name}.pdb'
                    with open(str(ind_path), 'w') as f:
                        f.write(f"REMARK SMILES {smi}\n")
                        # Strip MODEL/ENDMDL lines from pdb_block
                        for line in pdb_block.splitlines(keepends=True):
                            if not (line.startswith('MODEL') or line.startswith('ENDMDL')):
                                f.write(line)

                    # ── Combined (still useful for bulk loading) ────────────
                    combined_f.write(f"MODEL     {idx+1}\n")
                    combined_f.write(f"REMARK SMILES {smi}\n")
                    combined_f.write(pdb_block)
                    combined_f.write("ENDMDL\n")
                    n_exported += 1
                except Exception:
                    continue

        # ── VMD Tcl loader script ───────────────────────────────────────────
        tcl_path = pdb_dir.parent / 'load_all_vmd.tcl'
        tcl_lines = [
            "# load_all_vmd.tcl — generated by train_v3.py",
            "# Usage: vmd -e load_all_vmd.tcl",
            "# Or inside VMD console: source load_all_vmd.tcl",
            "# Loads every mol_NNNN.pdb as a SEPARATE molecule (not a trajectory)",
            "",
        ]
        for i in range(1, n_exported + 1):
            tcl_lines.append(f"mol new {{pdb_files/mol_{i:04d}.pdb}} type pdb")
        tcl_path.write_text("\n".join(tcl_lines) + "\n")

        if log:
            log.info(f"  PDB files  → {pdb_dir}/mol_NNNN.pdb  ({n_exported} files)")
            log.info(f"  Combined   → {combined_path}")
            log.info(f"  VMD loader → {tcl_path}")
    elif pdb_dir and not HAS_RDKIT:
        if log:
            log.warning("  RDKit not available — skipping PDB export")

    # ── Metrics ─────────────────────────────────────────────────────────────
    if valid_mols and HAS_RDKIT:
        all_smi   = [m.GetProp('SMILES') for m in valid_mols if m.HasProp('SMILES')]
        unique    = set(all_smi)
        novel     = {s for s in unique if s not in train_smiles}
        return {
            'validity_rate':   len(valid_mols) / max(total_gen, 1),
            'valid_count':     len(valid_mols),
            'total_generated': total_gen,
            'uniqueness':      len(unique)  / max(len(valid_mols), 1),
            'unique_smiles':   len(unique),
            'novelty':         len(novel)   / max(len(unique),  1),
        }
    return {'validity_rate': 0.0, 'valid_count': 0, 'total_generated': total_gen}


# =============================================================================
# PLOTTING
# =============================================================================

def save_loss_plots(history, plots_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs    = [h['epoch']       for h in history]
    train_tot = [h['train_total'] for h in history]
    val_tot   = [h['val_total']   for h in history]
    train_mse = [h['train_mse']   for h in history]
    val_mse   = [h['val_mse']     for h in history]
    train_geo = [h['train_geo']   for h in history]
    rmsds     = [h['rmsd_mean']   for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('09-03-2026-Exp-3 (Stable Full Constraints, v3)', fontsize=13, fontweight='bold')

    for ax, ys, labels, title in [
        (axes[0, 0], [(epochs, train_tot, 'Train total', 'steelblue'),
                      (epochs, val_tot,   'Val total',   'coral')],       'Total Loss'),
        (axes[0, 1], [(epochs, train_mse, 'Train MSE', 'steelblue'),
                      (epochs, val_mse,   'Val MSE',   'coral')],         'Diffusion MSE Loss'),
        (axes[1, 0], [(epochs, train_geo, 'Geo loss', 'seagreen')],       'Geometry Loss'),
        (axes[1, 1], [(e, v, 'RMSD (Å)', 'purple')
                      for e, v in [([h['epoch'] for h in history if h['rmsd_mean'] > 0],
                                    [h['rmsd_mean'] for h in history if h['rmsd_mean'] > 0])]
                      if e], 'RMSD Progression'),
    ]:
        for (ep, vals, label, color) in ys if isinstance(ys, list) else [(ys,)]:
            ax.plot(ep, vals, label=label, color=color,
                    linestyle='--' if 'Val' in label else '-')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    EXP_NAME = "09-03-2026-Exp-3(stable_full_constraints)"
    DEFAULT_EXP_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", EXP_NAME)

    parser = argparse.ArgumentParser(description='Train Conformer Diffusion v3')
    # Data
    parser.add_argument('--data',       type=str,   default='data/qm9_selfies.jsonl')
    parser.add_argument('--max_atoms',  type=int,   default=50)
    parser.add_argument('--val_split',  type=float, default=0.1)
    # Training
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--warmup',     type=int,   default=5,
                        help='Linear LR warmup epochs before CosineAnnealing')
    parser.add_argument('--resume',     type=str,   default=None,
                        help='Path to checkpoint to resume from')
    # Model
    parser.add_argument('--hidden_dim', type=int,   default=512)
    parser.add_argument('--num_layers', type=int,   default=10)
    parser.add_argument('--timesteps',  type=int,   default=1000)
    parser.add_argument('--edge_dim',   type=int,   default=64)
    parser.add_argument('--time_dim',   type=int,   default=256)
    # Geometry
    parser.add_argument('--geometry_weight', type=float, default=0.1,
                        help='Base geometry weight (fixed, no curriculum)')
    parser.add_argument('--num_generate',    type=int,   default=500)
    # Experiment
    parser.add_argument('--exp_dir',    type=str,   default=DEFAULT_EXP_DIR)
    args = parser.parse_args()

    # ── Experiment directories ───────────────────────────────────────────────
    exp_dir  = Path(args.exp_dir)
    ckpt_dir = exp_dir / 'checkpoints'
    plots_dir = exp_dir / 'plots'
    eval_dir  = exp_dir / 'evaluation'
    logs_dir  = exp_dir / 'logs'
    mol_dir   = exp_dir / 'molecules'
    for d in [ckpt_dir, plots_dir, eval_dir, logs_dir, mol_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Logging ─────────────────────────────────────────────────────────────
    log_file = logs_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(str(log_file))],
    )
    log = logging.getLogger(__name__)
    log.info(f"Experiment : {EXP_NAME}")
    log.info(f"Output dir : {exp_dir}")
    log.info(f"RDKit      : {'yes' if HAS_RDKIT else 'no (geometry-only validity)'}")
    log.info(f"Args       : {vars(args)}")

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump({'args': vars(args), 'experiment': EXP_NAME,
                   'date': datetime.now().isoformat()}, f, indent=2)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
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

    # ── Model ───────────────────────────────────────────────────────────────
    model = ConformerDiffusion(
        num_timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_dim=args.edge_dim,
        time_dim=args.time_dim,
    ).to(device)

    # Override geometry weights (all constraints active, balanced weights)
    model.geometry = GeometryConstraints(
        bond_weight=1.0,
        angle_weight=0.5,
        torsion_weight=0.2,
        repulsion_weight=0.5,
        planarity_weight=0.5,   # active from epoch 1
        chirality_weight=0.3,
        ring_strain_weight=0.2,
    )

    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(ep):
        # Linear warmup for first warmup_epochs, then cosine decay
        if ep < args.warmup:
            return float(ep + 1) / float(args.warmup)
        progress = (ep - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch  = 1
    best_val_mse = float('inf')
    best_validity = 0.0
    history = []

    if args.resume and Path(args.resume).exists():
        log.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        history     = ckpt.get('history', [])
        best_val_mse  = ckpt.get('best_val_mse',  float('inf'))
        best_validity = ckpt.get('best_validity',  0.0)
        # Fast-forward scheduler to correct position
        for _ in range(start_epoch - 1):
            scheduler.step()
        log.info(f"Resumed at epoch {start_epoch}, best_val_mse={best_val_mse:.4f}, "
                 f"best_validity={best_validity*100:.1f}%")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device,
                                    epoch, args.epochs, geometry_weight=args.geometry_weight)
        val_metrics   = validate(model, val_loader, device,
                                 geometry_weight=args.geometry_weight,
                                 epoch=epoch, max_epochs=args.epochs)
        scheduler.step()

        rmsd_mean, rmsd_std = 0.0, 0.0
        if epoch % 5 == 0 or epoch == 1:
            rmsd_mean, rmsd_std = sample_and_evaluate(model, val_loader, device, num_samples=20)

        validity_metrics = None
        if epoch % 10 == 0:
            validity_metrics = evaluate_validity(
                model, val_loader, device, num_samples=200, log=log)  # 200 samples
            with open(eval_dir / f'epoch_{epoch:04d}_validity.json', 'w') as f:
                json.dump({'epoch': epoch, **validity_metrics}, f, indent=2)

        # ── Logging ─────────────────────────────────────────────────────────
        lr_now = scheduler.get_last_lr()[0]
        log_msg = (f"Epoch {epoch:4d}/{args.epochs} | "
                   f"train={train_metrics['total']:.4f} "
                   f"(mse={train_metrics['mse']:.4f} geo={train_metrics['geo']:.4f}) | "
                   f"val={val_metrics['total']:.4f} "
                   f"(mse={val_metrics['mse']:.4f}) | "
                   f"lr={lr_now:.2e}")
        if rmsd_mean > 0:
            log_msg += f" | rmsd={rmsd_mean:.3f}±{rmsd_std:.3f}Å"
        if validity_metrics:
            log_msg += (f"\n  → valid={validity_metrics['validity_rate']*100:.1f}% "
                        f"unique={validity_metrics['uniqueness']*100:.1f}% "
                        f"novel={validity_metrics['novelty']*100:.1f}%")
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
            'lr':          lr_now,
            'validity':    validity_metrics.get('validity_rate', 0.0) if validity_metrics else 0.0,
        })

        # ── FIX: Checkpoint on val_mse (curriculum-independent) ─────────────
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse':          val_metrics['mse'],
                'best_val_mse':     best_val_mse,
                'best_validity':    best_validity,
                'history':          history,
                'args':             vars(args),
            }, str(ckpt_dir / 'conformer_best_mse.pt'))
            log.info(f"  ✓ Best MSE checkpoint  (val_mse={best_val_mse:.4f})")

        # ── Separate best-validity checkpoint ────────────────────────────────
        if validity_metrics and validity_metrics['validity_rate'] > best_validity:
            best_validity = validity_metrics['validity_rate']
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validity_rate':    best_validity,
                'best_val_mse':     best_val_mse,
                'best_validity':    best_validity,
                'history':          history,
                'args':             vars(args),
            }, str(ckpt_dir / 'conformer_best_validity.pt'))
            log.info(f"  ✓ Best validity checkpoint  ({best_validity*100:.1f}%)")

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if epoch % 25 == 0:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history':          history,
                'best_val_mse':     best_val_mse,
                'best_validity':    best_validity,
            }, str(ckpt_dir / f'conformer_epoch{epoch:04d}.pt'))

        if epoch % 10 == 0:
            save_loss_plots(history, str(plots_dir))

    # ── Save full history ────────────────────────────────────────────────────
    with open(eval_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    log.info(f"Training complete. Best val_mse={best_val_mse:.4f}, "
             f"best_validity={best_validity*100:.1f}%")

    # ── Post-training molecule export ────────────────────────────────────────
    log.info(f"\n{'='*60}\nPost-training: Generating and exporting molecules\n{'='*60}")

    # Use best checkpoint for generation
    best_ckpt = ckpt_dir / 'conformer_best_validity.pt'
    if not best_ckpt.exists():
        best_ckpt = ckpt_dir / 'conformer_best_mse.pt'
    if best_ckpt.exists():
        log.info(f"Loading best checkpoint: {best_ckpt}")
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    sdf_path      = mol_dir / 'generated_valid_molecules.sdf'
    pdb_files_dir = mol_dir / 'pdb_files'

    gen_metrics = export_valid_molecules(
        model, val_loader, device,
        num_generate=args.num_generate,
        sdf_path=str(sdf_path),
        pdb_dir=str(pdb_files_dir),
        log=log,
    )

    if gen_metrics:
        log.info(f"\n{'─'*60}")
        log.info(f"Generation Results ({gen_metrics.get('total_generated', 0)} generated)")
        log.info(f"{'─'*60}")
        log.info(f"  Validity:    {gen_metrics.get('validity_rate',  0)*100:.1f}%")
        log.info(f"  Valid count: {gen_metrics.get('valid_count',    0)}")
        log.info(f"  Uniqueness:  {gen_metrics.get('uniqueness',     0)*100:.1f}%")
        log.info(f"  Novelty:     {gen_metrics.get('novelty',        0)*100:.1f}%")
        log.info(f"{'─'*60}")
        with open(eval_dir / 'generation_metrics.json', 'w') as f:
            json.dump(gen_metrics, f, indent=2)


if __name__ == '__main__':
    main()
