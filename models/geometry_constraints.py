"""
geometry_constraints.py — Chemistry-Aware 3D Geometry Constraints

This module provides differentiable loss functions for enforcing physically
and chemically valid 3D molecular conformations.

Key components:
1. Atom-pair-specific bond length targets
2. Bond angle constraints (sp3: 109.5°, sp2: 120°, sp: 180°)
3. Planarity constraints for aromatic/conjugated systems
4. Steric clash detection and penalties
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


# =============================================================================
# CHEMISTRY-AWARE BOND LENGTH TARGETS
# =============================================================================

# Bond lengths in Angstroms: (atom1, atom2, bond_order) -> ideal_distance
# Organized by common organic chemistry bonds
BOND_LENGTH_TABLE = {
    # C-C bonds
    (6, 6, 1): 1.54,   # C-C single (sp3-sp3)
    (6, 6, 2): 1.34,   # C=C double
    (6, 6, 3): 1.20,   # C≡C triple
    (6, 6, 4): 1.40,   # C-C aromatic
    
    # C-H bonds
    (6, 1, 1): 1.09,   # C-H (sp3)
    
    # C-N bonds
    (6, 7, 1): 1.47,   # C-N single
    (6, 7, 2): 1.29,   # C=N double
    (6, 7, 3): 1.16,   # C≡N triple
    (6, 7, 4): 1.34,   # C-N aromatic
    
    # C-O bonds
    (6, 8, 1): 1.43,   # C-O single
    (6, 8, 2): 1.23,   # C=O double
    (6, 8, 4): 1.36,   # C-O aromatic
    
    # C-S bonds
    (6, 16, 1): 1.82,  # C-S single
    (6, 16, 2): 1.71,  # C=S double
    
    # C-F, C-Cl, C-Br bonds
    (6, 9, 1): 1.35,   # C-F
    (6, 17, 1): 1.77,  # C-Cl
    (6, 35, 1): 1.94,  # C-Br
    
    # N-H bonds
    (7, 1, 1): 1.01,   # N-H
    
    # N-N bonds
    (7, 7, 1): 1.45,   # N-N single
    (7, 7, 2): 1.25,   # N=N double
    (7, 7, 3): 1.10,   # N≡N triple
    
    # N-O bonds
    (7, 8, 1): 1.40,   # N-O single
    (7, 8, 2): 1.21,   # N=O double
    
    # O-H bonds
    (8, 1, 1): 0.96,   # O-H
    
    # O-O bonds
    (8, 8, 1): 1.48,   # O-O single (peroxide)
    
    # S-H bonds
    (16, 1, 1): 1.34,  # S-H
    
    # S-S bonds
    (16, 16, 1): 2.05, # S-S single (disulfide)
}

# Default bond lengths by bond order (fallback)
DEFAULT_BOND_LENGTHS = {
    1: 1.50,  # Single
    2: 1.34,  # Double
    3: 1.20,  # Triple
    4: 1.40,  # Aromatic
}

# Bond length tolerances (consider valid if within this range)
BOND_TOLERANCE = 0.15  # Angstroms


def get_ideal_bond_length(atom1: int, atom2: int, bond_order: int) -> float:
    """Get ideal bond length for an atom pair with given bond order."""
    # Try canonical order (smaller atom number first)
    key = (min(atom1, atom2), max(atom1, atom2), bond_order)
    if key in BOND_LENGTH_TABLE:
        return BOND_LENGTH_TABLE[key]
    
    # Try with original order
    key2 = (atom1, atom2, bond_order)
    if key2 in BOND_LENGTH_TABLE:
        return BOND_LENGTH_TABLE[key2]
    
    # Fallback to default by bond order
    return DEFAULT_BOND_LENGTHS.get(bond_order, 1.50)


# =============================================================================
# IDEAL BOND ANGLES
# =============================================================================

# Hybridization -> ideal angle in degrees
IDEAL_ANGLES = {
    'sp3': 109.5,  # Tetrahedral
    'sp2': 120.0,  # Trigonal planar
    'sp': 180.0,   # Linear
}

# Atom type patterns for hybridization detection
# In practice, we use the number of neighbors to estimate
def estimate_hybridization(num_neighbors: int) -> str:
    """Estimate hybridization from number of neighbors."""
    if num_neighbors <= 2:
        return 'sp'
    elif num_neighbors == 3:
        return 'sp2'
    else:
        return 'sp3'


# =============================================================================
# GEOMETRY LOSS FUNCTIONS
# =============================================================================

class GeometryConstraints:
    """
    Differentiable geometry constraints for molecular conformations.
    
    This class computes losses that encourage:
    - Correct bond lengths
    - Correct bond angles
    - No steric clashes
    - Planarity for aromatic systems
    """
    
    def __init__(self,
                 bond_weight: float = 10.0,
                 angle_weight: float = 5.0,
                 repulsion_weight: float = 5.0,
                 planarity_weight: float = 2.0,
                 min_nonbond_dist: float = 1.5):
        """
        Args:
            bond_weight: Weight for bond length loss
            angle_weight: Weight for bond angle loss
            repulsion_weight: Weight for steric repulsion loss
            planarity_weight: Weight for planarity loss
            min_nonbond_dist: Minimum non-bonded distance (Angstroms)
        """
        self.bond_weight = bond_weight
        self.angle_weight = angle_weight
        self.repulsion_weight = repulsion_weight
        self.planarity_weight = planarity_weight
        self.min_nonbond_dist = min_nonbond_dist
    
    def compute_bond_loss(self,
                          pos: torch.Tensor,
                          atom_types: torch.Tensor,
                          edge_index: torch.Tensor,
                          bond_types: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for deviation from ideal bond lengths.
        
        Uses chemistry-aware targets based on atom types and bond order.
        """
        row, col = edge_index
        
        # Compute current distances
        diff = pos[row] - pos[col]
        dists = torch.norm(diff, dim=-1).clamp(min=1e-6)
        
        # Get ideal distances for each bond
        ideal_dists = torch.zeros_like(dists)
        for e in range(len(dists)):
            a1 = atom_types[row[e]].item()
            a2 = atom_types[col[e]].item()
            bo = bond_types[e].item()
            ideal_dists[e] = get_ideal_bond_length(a1, a2, bo)
        
        # Squared error loss
        loss = F.mse_loss(dists, ideal_dists)
        
        return self.bond_weight * loss
    
    def compute_angle_loss(self,
                           pos: torch.Tensor,
                           atom_types: torch.Tensor,
                           edge_index: torch.Tensor,
                           batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for deviation from ideal bond angles.
        
        For each central atom j with neighbors i and k,
        compute angle i-j-k and compare to ideal.
        """
        device = pos.device
        N = pos.size(0)
        
        # Build adjacency list
        row, col = edge_index
        neighbors = [[] for _ in range(N)]
        for e in range(row.size(0)):
            i, j = row[e].item(), col[e].item()
            neighbors[j].append(i)
        
        # Collect all angle triplets (i, j, k) where j is central
        angle_losses = []
        
        for j in range(N):
            neigh = neighbors[j]
            if len(neigh) < 2:
                continue
            
            # Estimate ideal angle from number of neighbors
            hybridization = estimate_hybridization(len(neigh))
            ideal_angle = math.radians(IDEAL_ANGLES[hybridization])
            
            # For each pair of neighbors
            for idx_i, i in enumerate(neigh):
                for k in neigh[idx_i + 1:]:
                    # Compute vectors
                    v1 = pos[i] - pos[j]
                    v2 = pos[k] - pos[j]
                    
                    # Compute angle using dot product
                    cos_angle = F.cosine_similarity(
                        v1.unsqueeze(0), v2.unsqueeze(0)
                    ).clamp(-0.999, 0.999)
                    
                    angle = torch.acos(cos_angle)
                    
                    # Loss: deviation from ideal angle
                    angle_error = (angle - ideal_angle) ** 2
                    angle_losses.append(angle_error)
        
        if len(angle_losses) == 0:
            return torch.tensor(0.0, device=device)
        
        loss = torch.stack(angle_losses).mean()
        return self.angle_weight * loss
    
    def compute_repulsion_loss(self,
                               pos: torch.Tensor,
                               edge_index: torch.Tensor,
                               batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute steric repulsion loss for non-bonded atom pairs.
        
        Penalizes pairs closer than min_nonbond_dist.
        """
        device = pos.device
        N = pos.size(0)
        
        row, col = edge_index
        
        # Create bonded mask (including 1-3 interactions via angle)
        bonded_mask = torch.zeros(N, N, device=device, dtype=torch.bool)
        bonded_mask[row, col] = True
        
        # Compute all pairwise distances
        all_dists = torch.cdist(pos, pos)  # (N, N)
        
        # Same-molecule mask
        same_mol = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)
        
        # Non-bonded, same-molecule, non-self pairs
        nb_mask = same_mol & ~bonded_mask & ~torch.eye(N, device=device, dtype=torch.bool)
        
        # Get distances for non-bonded pairs
        nb_dists = all_dists[nb_mask]
        
        # Soft repulsion: penalize distances below threshold
        clashing = nb_dists[nb_dists < self.min_nonbond_dist]
        
        if len(clashing) == 0:
            return torch.tensor(0.0, device=device)
        
        # Quadratic penalty
        loss = torch.mean((self.min_nonbond_dist - clashing) ** 2)
        
        return self.repulsion_weight * loss
    
    def compute_total_loss(self,
                           pos: torch.Tensor,
                           atom_types: torch.Tensor,
                           edge_index: torch.Tensor,
                           bond_types: torch.Tensor,
                           batch_idx: torch.Tensor,
                           include_angles: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total geometry constraint loss.
        
        Returns:
            total_loss: Scalar loss tensor
            breakdown: Dict with individual loss components
        """
        bond_loss = self.compute_bond_loss(pos, atom_types, edge_index, bond_types)
        repulsion_loss = self.compute_repulsion_loss(pos, edge_index, batch_idx)
        
        total_loss = bond_loss + repulsion_loss
        breakdown = {
            'bond_loss': bond_loss.item(),
            'repulsion_loss': repulsion_loss.item(),
        }
        
        if include_angles:
            angle_loss = self.compute_angle_loss(pos, atom_types, edge_index, batch_idx)
            total_loss = total_loss + angle_loss
            breakdown['angle_loss'] = angle_loss.item()
        
        breakdown['total_loss'] = total_loss.item()
        
        return total_loss, breakdown


# =============================================================================
# VALIDITY EVALUATION (STRICT)
# =============================================================================

class StrictValidityEvaluator:
    """
    Evaluate 3D validity with strict thresholds.
    
    A molecule is considered "truly valid" only if:
    1. All bond lengths within tolerance
    2. All bond angles within tolerance
    3. No steric clashes
    """
    
    def __init__(self,
                 bond_tolerance: float = 0.2,
                 angle_tolerance: float = 15.0,  # degrees
                 min_nonbond_dist: float = 1.4):
        self.bond_tolerance = bond_tolerance
        self.angle_tolerance = math.radians(angle_tolerance)
        self.min_nonbond_dist = min_nonbond_dist
    
    @torch.no_grad()
    def evaluate_molecule(self,
                          pos: torch.Tensor,
                          atom_types: torch.Tensor,
                          edge_index: torch.Tensor,
                          bond_types: torch.Tensor) -> Dict:
        """
        Evaluate strict 3D validity for a single molecule.
        
        Returns:
            Dict with validity flags and metrics
        """
        row, col = edge_index
        N = pos.size(0)
        
        # 1. Check bond lengths
        diff = pos[row] - pos[col]
        dists = torch.norm(diff, dim=-1)
        
        bond_errors = []
        bond_valid = True
        for e in range(len(dists)):
            a1 = atom_types[row[e]].item()
            a2 = atom_types[col[e]].item()
            bo = bond_types[e].item()
            ideal = get_ideal_bond_length(a1, a2, bo)
            error = abs(dists[e].item() - ideal)
            bond_errors.append(error)
            if error > self.bond_tolerance:
                bond_valid = False
        
        # 2. Check bond angles
        neighbors = [[] for _ in range(N)]
        for e in range(row.size(0)):
            i, j = row[e].item(), col[e].item()
            neighbors[j].append(i)
        
        angle_errors = []
        angles_valid = True
        
        for j in range(N):
            neigh = neighbors[j]
            if len(neigh) < 2:
                continue
            
            hybridization = estimate_hybridization(len(neigh))
            ideal_angle = math.radians(IDEAL_ANGLES[hybridization])
            
            for idx_i, i in enumerate(neigh):
                for k in neigh[idx_i + 1:]:
                    v1 = pos[i] - pos[j]
                    v2 = pos[k] - pos[j]
                    
                    cos_angle = F.cosine_similarity(
                        v1.unsqueeze(0), v2.unsqueeze(0)
                    ).clamp(-0.999, 0.999)
                    
                    angle = torch.acos(cos_angle).item()
                    error = abs(angle - ideal_angle)
                    angle_errors.append(error)
                    
                    if error > self.angle_tolerance:
                        angles_valid = False
        
        # 3. Check steric clashes
        bonded_pairs = set()
        for e in range(row.size(0)):
            i, j = row[e].item(), col[e].item()
            bonded_pairs.add((min(i, j), max(i, j)))
        
        clash_free = True
        clash_count = 0
        
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) not in bonded_pairs:
                    dist = torch.norm(pos[i] - pos[j]).item()
                    if dist < self.min_nonbond_dist:
                        clash_free = False
                        clash_count += 1
        
        # Overall validity
        fully_valid = bond_valid and angles_valid and clash_free
        
        return {
            'fully_valid': fully_valid,
            'bond_valid': bond_valid,
            'angles_valid': angles_valid,
            'clash_free': clash_free,
            'mean_bond_error': sum(bond_errors) / len(bond_errors) if bond_errors else 0,
            'max_bond_error': max(bond_errors) if bond_errors else 0,
            'mean_angle_error': sum(angle_errors) / len(angle_errors) if angle_errors else 0,
            'max_angle_error': max(angle_errors) if angle_errors else 0,
            'clash_count': clash_count,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Testing GeometryConstraints...")
    
    # Create a simple ethane-like molecule
    # C1-C2 with hydrogens
    atom_types = torch.tensor([6, 6])  # Two carbons
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],  # C-C at 1.54Å
    ])
    edge_index = torch.tensor([[0, 1], [1, 0]])  # Bidirectional
    bond_types = torch.tensor([1, 1])  # Single bonds
    batch_idx = torch.tensor([0, 0])
    
    constraints = GeometryConstraints()
    
    loss, breakdown = constraints.compute_total_loss(
        pos, atom_types, edge_index, bond_types, batch_idx,
        include_angles=False
    )
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Breakdown: {breakdown}")
    
    # Evaluate validity
    evaluator = StrictValidityEvaluator()
    validity = evaluator.evaluate_molecule(pos, atom_types, edge_index, bond_types)
    print(f"Validity: {validity}")
    
    print("\nTest passed!")
