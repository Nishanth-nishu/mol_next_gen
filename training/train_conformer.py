"""
train_conformer.py — Train Conformer Diffusion Model

This script trains the E(3)-equivariant conformer diffusion model.

Key difference from old approach:
- Old: Train to generate geometry → optimize to coords → infer bonds
- New: Train to generate coords directly for FIXED valid graphs

The molecular graph (atoms, bonds) is provided as input and never changes.
The model only learns to denoise the 3D coordinates.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conformer_diffusion import ConformerDiffusion


# =============================================================================
# DATASET
# =============================================================================

class ConformerDataset(Dataset):
    """
    Dataset for conformer diffusion training.
    
    Each item contains:
    - atom_types: (N,) atomic numbers
    - edge_index: (2, E) bond connections
    - bond_types: (E,) bond orders
    - coordinates: (N, 3) ground truth 3D positions
    """
    
    def __init__(self, data_path: str, max_atoms: int = 20):
        self.data_path = data_path
        self.max_atoms = max_atoms
        self.data = []
        
        self._load_data()
    
    def _load_data(self):
        """Load JSONL data."""
        print(f"Loading data from {self.data_path}...")
        
        with open(self.data_path, 'r') as f:
            for line in tqdm(f):
                item = json.loads(line.strip())
                
                # Skip if no coordinates
                if item.get('coordinates') is None:
                    continue
                
                # Skip if too many atoms
                if item['num_atoms'] > self.max_atoms:
                    continue
                
                self.data.append(item)
        
        print(f"Loaded {len(self.data)} molecules")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert to tensors
        atom_types = torch.tensor(item['atom_types'], dtype=torch.long)
        coordinates = torch.tensor(item['coordinates'], dtype=torch.float32)
        
        edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
        bond_types = torch.tensor(item['bond_types'], dtype=torch.long)
        
        return {
            'atom_types': atom_types,
            'coordinates': coordinates,
            'edge_index': edge_index,
            'bond_types': bond_types,
            'num_atoms': item['num_atoms']
        }


def collate_fn(batch):
    """
    Collate batch with variable-sized molecules.
    Creates batch indices for each atom.
    """
    atom_types_list = []
    coords_list = []
    edge_index_list = []
    bond_types_list = []
    batch_idx_list = []
    
    offset = 0
    
    for i, item in enumerate(batch):
        N = item['num_atoms']
        
        atom_types_list.append(item['atom_types'])
        coords_list.append(item['coordinates'])
        
        # Shift edge indices
        edge_index_list.append(item['edge_index'] + offset)
        bond_types_list.append(item['bond_types'])
        
        # Batch assignment
        batch_idx_list.append(torch.full((N,), i, dtype=torch.long))
        
        offset += N
    
    return {
        'atom_types': torch.cat(atom_types_list),
        'coordinates': torch.cat(coords_list),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'bond_types': torch.cat(bond_types_list),
        'batch_idx': torch.cat(batch_idx_list),
        'num_molecules': len(batch)
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch, max_epochs=100):
    """Train for one epoch with curriculum-based geometry learning."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        atom_types = batch['atom_types'].to(device)
        coords = batch['coordinates'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx = batch['batch_idx'].to(device)
        
        # Center coordinates (important for equivariance)
        coords = coords - scatter_mean(coords, batch_idx, dim=0)[batch_idx]
        
        # Forward with curriculum geometry learning
        optimizer.zero_grad()
        loss = model.get_loss(
            coords, atom_types, edge_index, bond_types, batch_idx,
            geometry_weight=0.3,
            epoch=epoch,
            max_epochs=max_epochs
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def scatter_mean(src, index, dim=0):
    """Compute mean by scattering."""
    count = torch.zeros(index.max() + 1, device=src.device)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
    count = count.clamp(min=1)
    
    out = torch.zeros(index.max() + 1, src.size(1), device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand(-1, src.size(1)), src)
    
    return out / count.unsqueeze(-1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        atom_types = batch['atom_types'].to(device)
        coords = batch['coordinates'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx = batch['batch_idx'].to(device)
        
        coords = coords - scatter_mean(coords, batch_idx, dim=0)[batch_idx]
        
        loss = model.get_loss(coords, atom_types, edge_index, bond_types, batch_idx)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def kabsch_rmsd(coords_pred: torch.Tensor, coords_true: torch.Tensor) -> float:
    """
    Compute RMSD after optimal alignment using Kabsch algorithm.
    
    Args:
        coords_pred: (N, 3) predicted coordinates
        coords_true: (N, 3) ground truth coordinates
    
    Returns:
        RMSD in Angstroms
    """
    # Center both
    pred_centered = coords_pred - coords_pred.mean(dim=0)
    true_centered = coords_true - coords_true.mean(dim=0)
    
    # Compute optimal rotation using SVD (Kabsch algorithm)
    H = pred_centered.T @ true_centered  # 3x3
    U, S, Vt = torch.linalg.svd(H)
    
    # Handle reflection case
    d = torch.det(Vt.T @ U.T)
    sign_matrix = torch.eye(3, device=coords_pred.device)
    sign_matrix[2, 2] = d.sign()
    
    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T
    
    # Rotate predicted to align with true
    pred_aligned = pred_centered @ R.T
    
    # Compute RMSD
    rmsd = torch.sqrt(torch.mean((pred_aligned - true_centered) ** 2)).item()
    
    return rmsd


@torch.no_grad()
def sample_and_evaluate(model, dataloader, device, num_samples=10, use_guidance=True):
    """Sample conformers and compare to ground truth with proper alignment."""
    model.eval()
    
    rmsds = []
    num_mols = 0
    
    for batch in dataloader:
        if num_mols >= num_samples:
            break
        
        atom_types = batch['atom_types'].to(device)
        coords_true = batch['coordinates'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx = batch['batch_idx'].to(device)
        
        # Center ground truth
        coords_true = coords_true - scatter_mean(coords_true, batch_idx, dim=0)[batch_idx]
        
        # Sample using DDIM (with or without guidance)
        if use_guidance and hasattr(model, 'guided_sample'):
            coords_gen = model.guided_sample(
                atom_types, edge_index, bond_types, batch_idx, 
                num_steps=50, guidance_scale=1.0  # Higher guidance for better geometry
            )
        else:
            coords_gen = model.ddim_sample(
                atom_types, edge_index, bond_types, batch_idx, num_steps=50
            )
        
        # Compute RMSD per molecule
        for b in range(batch['num_molecules']):
            if num_mols >= num_samples:
                break
            
            mask = (batch_idx == b)
            c_true = coords_true[mask]
            c_gen = coords_gen[mask]
            
            # Use Kabsch-aligned RMSD
            rmsd = kabsch_rmsd(c_gen, c_true)
            rmsds.append(rmsd)
            num_mols += 1
    
    if rmsds:
        mean_rmsd = np.mean(rmsds)
        std_rmsd = np.std(rmsds)
        return mean_rmsd, std_rmsd
    
    return 0.0, 0.0


@torch.no_grad()
def evaluate_3d_validity(model, dataloader, device, num_samples=20):
    """
    Evaluate STRICT 3D validity of generated conformers.
    
    Uses chemistry-aware bond length targets (no fallbacks).
    
    Returns:
        bond_valid_rate: Fraction of molecules with valid bond lengths
        clash_free_rate: Fraction of molecules without steric clashes
        fully_valid_rate: Fraction of molecules passing ALL validity checks
        mean_bond_error: Mean deviation from ideal bond lengths
    """
    from models.geometry_constraints import get_ideal_bond_length
    
    model.eval()
    
    bond_errors = []
    valid_count = 0
    clash_free_count = 0
    fully_valid_count = 0
    total = 0
    
    BOND_TOL = 0.2  # Strict: 0.2Å tolerance
    MIN_NONBOND = 1.4  # Minimum non-bonded distance
    
    for batch in dataloader:
        if total >= num_samples:
            break
        
        atom_types = batch['atom_types'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx = batch['batch_idx'].to(device)
        
        # Generate using guided sampling with higher guidance
        if hasattr(model, 'guided_sample'):
            coords_gen = model.guided_sample(
                atom_types, edge_index, bond_types, batch_idx,
                num_steps=50, guidance_scale=1.0  # Full guidance
            )
        else:
            coords_gen = model.ddim_sample(
                atom_types, edge_index, bond_types, batch_idx, num_steps=50
            )
        
        # Evaluate per molecule
        for b in range(batch['num_molecules']):
            if total >= num_samples:
                break
            
            mask = (batch_idx == b)
            coords = coords_gen[mask]
            mol_atom_types = atom_types[mask]
            N = mask.sum().item()
            
            # Get edges for this molecule
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            mol_edges = edge_index[:, edge_mask]
            mol_bond_types = bond_types[edge_mask]
            
            # Remap indices to local
            idx_map = torch.cumsum(mask.long(), 0) - 1
            local_edges = idx_map[mol_edges]
            
            # Check bond lengths with chemistry-aware targets
            mol_valid = True
            mol_bond_error = 0.0
            n_bonds = 0
            
            for e_idx in range(0, local_edges.size(1), 2):  # Skip duplicates
                i, j = local_edges[0, e_idx].item(), local_edges[1, e_idx].item()
                btype = mol_bond_types[e_idx].item()
                
                # Use chemistry-aware ideal distance
                a1 = mol_atom_types[i].item()
                a2 = mol_atom_types[j].item()
                ideal = get_ideal_bond_length(a1, a2, btype)
                
                dist = torch.norm(coords[i] - coords[j]).item()
                error = abs(dist - ideal)
                
                mol_bond_error += error
                n_bonds += 1
                
                if error > BOND_TOL:
                    mol_valid = False
            
            if n_bonds > 0:
                bond_errors.append(mol_bond_error / n_bonds)
            if mol_valid:
                valid_count += 1
            
            # Check steric clashes
            has_clash = False
            bonded_pairs = set()
            for e_idx in range(local_edges.size(1)):
                i, j = local_edges[0, e_idx].item(), local_edges[1, e_idx].item()
                bonded_pairs.add((min(i, j), max(i, j)))
            
            for i in range(N):
                for j in range(i + 1, N):
                    if (i, j) not in bonded_pairs:
                        dist = torch.norm(coords[i] - coords[j]).item()
                        if dist < MIN_NONBOND:
                            has_clash = True
                            break
                if has_clash:
                    break
            
            if not has_clash:
                clash_free_count += 1
            
            # Fully valid: bonds OK AND no clashes
            if mol_valid and not has_clash:
                fully_valid_count += 1
            
            total += 1
    
    bond_valid_rate = valid_count / total if total > 0 else 0.0
    clash_free_rate = clash_free_count / total if total > 0 else 0.0
    fully_valid_rate = fully_valid_count / total if total > 0 else 0.0
    mean_bond_error = np.mean(bond_errors) if bond_errors else 0.0
    
    return {
        'bond_valid_rate': bond_valid_rate,
        'clash_free_rate': clash_free_rate,
        'fully_valid_rate': fully_valid_rate,
        'mean_bond_error': mean_bond_error,
        'total_evaluated': total
    }



# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Conformer Diffusion')
    parser.add_argument('--data', type=str, default='data/qm9_selfies.jsonl',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of equivariant layers')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--max_atoms', type=int, default=15,
                        help='Maximum atoms per molecule')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    full_dataset = ConformerDataset(args.data, max_atoms=args.max_atoms)
    
    # Split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Model
    model = ConformerDiffusion(
        num_timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train with curriculum geometry learning
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, max_epochs=args.epochs)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Sample and compute RMSD (every 5 epochs for meaningful updates)
        rmsd_mean, rmsd_std = 0.0, 0.0
        validity = None
        
        if epoch % 5 == 0 or epoch == 1:
            rmsd_mean, rmsd_std = sample_and_evaluate(model, val_loader, device, num_samples=20)
        
        # Evaluate 3D validity (every 10 epochs)
        if epoch % 10 == 0:
            validity = evaluate_3d_validity(model, val_loader, device, num_samples=30)
        
        # Log
        log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        if rmsd_mean > 0:
            log_msg += f", rmsd={rmsd_mean:.3f}±{rmsd_std:.3f}Å"
        if validity:
            log_msg += f"\n  → 3D Validity: fully_valid={validity['fully_valid_rate']*100:.1f}%, bonds={validity['bond_valid_rate']*100:.1f}%, no_clash={validity['clash_free_rate']*100:.1f}%, bond_err={validity['mean_bond_error']:.3f}Å"
        print(log_msg)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'rmsd_mean': rmsd_mean,
            'rmsd_std': rmsd_std,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'conformer_best.pt'))
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        
        # Save checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, os.path.join(args.save_dir, f'conformer_epoch{epoch}.pt'))
        
        scheduler.step()
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'args': vars(args)
    }, os.path.join(args.save_dir, 'conformer_final.pt'))
    
    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
