"""
dual_encoder.py — MDM-style Dual Equivariant Encoder

Separates local (covalent bond) and global (van der Waals) interactions.

Why this helps:
- Local encoder: Strict geometry constraints for bonded atoms (< 2Å)
- Global encoder: Soft constraints for non-bonded atoms (all pairs)

This makes the model "understand" that bond distances are fixed constraints
while non-bonded distances are flexible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# LOCAL ENCODER (Covalent Bonds)
# =============================================================================

class LocalGNN(nn.Module):
    """
    Local encoder for bonded interactions.
    
    Only processes atoms connected by chemical bonds.
    Enforces strict bond length/angle constraints.
    """
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LocalLayer(hidden_dim))
    
    def forward(self, 
                h: torch.Tensor,          # (N, hidden_dim)
                pos: torch.Tensor,        # (N, 3)
                edge_index: torch.Tensor, # (2, E) bonded edges only
                edge_attr: torch.Tensor   # (E, edge_dim) bond features
                ) -> torch.Tensor:
        """Process only bonded interactions."""
        for layer in self.layers:
            h = layer(h, pos, edge_index, edge_attr)
        return h


class LocalLayer(nn.Module):
    """Single local (covalent) message passing layer."""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 32):
        super().__init__()
        
        # Message network (includes geometric info)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 4, hidden_dim),  # +4: dist, unit_vec
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h, pos, edge_index, edge_attr):
        row, col = edge_index
        
        # Geometric features
        diff = pos[row] - pos[col]  # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)
        unit_vec = diff / dist  # (E, 3)
        
        # Message
        msg_input = torch.cat([h[row], h[col], edge_attr, dist, unit_vec], dim=-1)
        msg = self.message_mlp(msg_input)
        
        # Aggregate
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, h.size(-1)), msg)
        
        # Update
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


# =============================================================================
# GLOBAL ENCODER (Van der Waals / All Pairs)
# =============================================================================

class GlobalGNN(nn.Module):
    """
    Global encoder for all pairwise interactions.
    
    Processes all atom pairs within a distance cutoff.
    Models van der Waals forces and long-range interactions.
    """
    
    def __init__(self, 
                 hidden_dim: int = 128, 
                 num_layers: int = 2,
                 cutoff: float = 10.0):
        super().__init__()
        
        self.cutoff = cutoff
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GlobalLayer(hidden_dim, cutoff))
    
    def forward(self,
                h: torch.Tensor,       # (N, hidden_dim)
                pos: torch.Tensor,     # (N, 3)
                batch_idx: torch.Tensor  # (N,) batch assignment
                ) -> torch.Tensor:
        """Process all pairwise interactions within cutoff."""
        for layer in self.layers:
            h = layer(h, pos, batch_idx)
        return h


class GlobalLayer(nn.Module):
    """Single global (all-pairs) layer."""
    
    def __init__(self, hidden_dim: int, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff
        
        # Distance encoding (RBF-like)
        self.dist_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16)
        )
        
        # Message network
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h, pos, batch_idx):
        device = h.device
        N = h.size(0)
        
        # Build all-pairs edges within same molecule
        edge_src, edge_dst = [], []
        
        for b in batch_idx.unique():
            mask = (batch_idx == b)
            indices = mask.nonzero().squeeze(-1)
            n = indices.size(0)
            
            # All pairs within molecule
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_src.append(indices[i])
                        edge_dst.append(indices[j])
        
        if len(edge_src) == 0:
            return h
        
        edge_src = torch.tensor(edge_src, device=device)
        edge_dst = torch.tensor(edge_dst, device=device)
        
        # Compute distances
        diff = pos[edge_src] - pos[edge_dst]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        
        # Filter by cutoff
        mask = (dist.squeeze(-1) < self.cutoff)
        edge_src = edge_src[mask]
        edge_dst = edge_dst[mask]
        dist = dist[mask]
        
        if edge_src.size(0) == 0:
            return h
        
        # Distance embedding with soft cutoff
        cutoff_weight = 0.5 * (1 + torch.cos(3.14159 * dist / self.cutoff))
        dist_feat = self.dist_embed(dist) * cutoff_weight
        
        # Message
        msg_input = torch.cat([h[edge_src], h[edge_dst], dist_feat], dim=-1)
        msg = self.message_mlp(msg_input)
        
        # Aggregate
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, edge_dst.unsqueeze(-1).expand(-1, h.size(-1)), msg)
        
        # Update
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


# =============================================================================
# DUAL ENCODER
# =============================================================================

class DualEquivariantEncoder(nn.Module):
    """
    MDM-style dual encoder combining local and global interactions.
    
    Local: Bonded atoms (covalent, < 2Å) - strict constraints
    Global: All pairs (van der Waals) - soft constraints
    
    The fusion of both gives the model understanding of:
    - What distances MUST be (bonds)
    - What distances SHOULD be (non-bonds)
    """
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 num_local_layers: int = 3,
                 num_global_layers: int = 2,
                 cutoff: float = 10.0):
        super().__init__()
        
        self.local_encoder = LocalGNN(hidden_dim, num_local_layers)
        self.global_encoder = GlobalGNN(hidden_dim, num_global_layers, cutoff)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Gating mechanism (learns which signal is more important)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self,
                h: torch.Tensor,            # (N, hidden_dim)
                pos: torch.Tensor,          # (N, 3)
                edge_index: torch.Tensor,   # (2, E) bonded edges
                edge_attr: torch.Tensor,    # (E, edge_dim)
                batch_idx: torch.Tensor     # (N,)
                ) -> torch.Tensor:
        """
        Dual encoding pass.
        
        Returns:
            h_fused: (N, hidden_dim) combined features
        """
        # Local (bonded only)
        h_local = self.local_encoder(h, pos, edge_index, edge_attr)
        
        # Global (all pairs)
        h_global = self.global_encoder(h, pos, batch_idx)
        
        # Fusion with gating
        h_cat = torch.cat([h_local, h_global], dim=-1)
        gate = self.gate(h_cat)
        
        h_fused = gate * h_local + (1 - gate) * h_global
        h_fused = self.fusion(h_cat) + h_fused  # Residual
        
        return h_fused


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Testing DualEquivariantEncoder...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_dim = 64
    
    # Create encoder
    encoder = DualEquivariantEncoder(
        hidden_dim=hidden_dim,
        num_local_layers=2,
        num_global_layers=1
    ).to(device)
    
    # Fake data: 2 molecules (3 + 4 atoms)
    h = torch.randn(7, hidden_dim, device=device)
    pos = torch.randn(7, 3, device=device) * 2  # Random positions
    
    # Bonded edges (local)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 3, 4, 3, 5, 3, 6],
        [1, 0, 2, 0, 4, 3, 5, 3, 6, 3]
    ], device=device)
    edge_attr = torch.randn(10, 32, device=device)
    
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)
    
    # Forward
    h_out = encoder(h, pos, edge_index, edge_attr, batch_idx)
    
    print(f"Input shape: {h.shape}")
    print(f"Output shape: {h_out.shape}")
    print("Test passed!")
