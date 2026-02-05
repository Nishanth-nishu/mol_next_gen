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

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
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
        
        # Forward
        optimizer.zero_grad()
        loss = model.get_loss(coords, atom_types, edge_index, bond_types, batch_idx)
        
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
def sample_and_evaluate(model, dataloader, device, num_samples=5):
    """Sample conformers and compare to ground truth."""
    model.eval()
    
    total_rmsd = 0
    num_mols = 0
    
    for batch in dataloader:
        if num_mols >= num_samples:
            break
        
        atom_types = batch['atom_types'].to(device)
        coords_true = batch['coordinates'].to(device)
        edge_index = batch['edge_index'].to(device)
        bond_types = batch['bond_types'].to(device)
        batch_idx = batch['batch_idx'].to(device)
        
        # Center
        coords_true = coords_true - scatter_mean(coords_true, batch_idx, dim=0)[batch_idx]
        
        # Sample
        coords_gen = model.ddim_sample(
            atom_types, edge_index, bond_types, batch_idx, num_steps=50
        )
        coords_gen = coords_gen - scatter_mean(coords_gen, batch_idx, dim=0)[batch_idx]
        
        # RMSD per molecule
        for b in range(batch['num_molecules']):
            mask = (batch_idx == b)
            c_true = coords_true[mask]
            c_gen = coords_gen[mask]
            
            rmsd = torch.sqrt(torch.mean((c_true - c_gen) ** 2)).item()
            total_rmsd += rmsd
            num_mols += 1
    
    return total_rmsd / num_mols if num_mols > 0 else 0


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
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Sample RMSD (every 10 epochs)
        rmsd = 0
        if epoch % 10 == 0:
            rmsd = sample_and_evaluate(model, val_loader, device, num_samples=10)
        
        # Log
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, rmsd={rmsd:.3f}Å")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'rmsd': rmsd,
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
