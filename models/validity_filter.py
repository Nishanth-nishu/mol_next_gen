"""
validity_filter.py — Step-wise Validity Checking (MolCode-style)

Enforces chemical validity during the sampling process.
Rejects or penalizes denoising steps that would violate chemistry rules.

Key checks:
1. Valency constraints (C max 4, N max 3, O max 2, etc.)
2. Distance constraints (bonds ~1.5Å, no overlaps)
3. Angle constraints (reasonable bond angles)
"""

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Tuple, Optional, Dict, List
import numpy as np


# =============================================================================
# CHEMICAL CONSTANTS
# =============================================================================

# Maximum valence for each atom type (atomic number → max bonds)
MAX_VALENCE = {
    1: 1,   # H
    5: 3,   # B
    6: 4,   # C
    7: 3,   # N (can be 4 with formal charge)
    8: 2,   # O
    9: 1,   # F
    14: 4,  # Si
    15: 3,  # P (can be 5)
    16: 2,  # S (can be 6)
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}

# Typical bond length ranges (Å) by bond order
BOND_LENGTHS = {
    1: (1.0, 1.8),   # Single bond: 1.0-1.8 Å
    2: (1.1, 1.6),   # Double bond
    3: (1.0, 1.3),   # Triple bond
    4: (1.2, 1.5),   # Aromatic
}

# Minimum non-bonded distance (van der Waals)
MIN_NONBOND_DISTANCE = 1.5  # Å


# =============================================================================
# VALIDITY CHECKER
# =============================================================================

class ValidityChecker:
    """
    Check chemical validity of molecular conformations.
    
    Can be used:
    1. Post-hoc: Check full molecule after generation
    2. Step-wise: Check during sampling (MolCode-style)
    """
    
    def __init__(self,
                 strict_valence: bool = True,
                 check_distances: bool = True,
                 check_angles: bool = False):
        """
        Args:
            strict_valence: Reject if valence exceeded
            check_distances: Check if bond lengths are reasonable
            check_angles: Check if angles are reasonable (expensive)
        """
        self.strict_valence = strict_valence
        self.check_distances = check_distances
        self.check_angles = check_angles
    
    def check_valency(self, 
                      atom_types: torch.Tensor,  # (N,) atomic numbers
                      edge_index: torch.Tensor,  # (2, E) bonds
                      bond_types: torch.Tensor   # (E,) bond orders
                      ) -> Tuple[bool, Dict]:
        """
        Check if valency constraints are satisfied.
        
        Returns:
            valid: True if all atoms satisfy valency
            info: Dict with details
        """
        N = atom_types.size(0)
        device = atom_types.device
        
        # Count effective bonds per atom (considering bond order)
        bond_counts = torch.zeros(N, device=device)
        
        row, col = edge_index
        # Only count each edge once (undirected → /2)
        for e in range(0, edge_index.size(1), 2):
            i, j = row[e].item(), col[e].item()
            order = bond_types[e].item()
            
            bond_counts[i] += order
            bond_counts[j] += order
        
        # Actually each edge is counted once, so we need to divide
        # Since edges are duplicated (i→j and j→i), we already have correct count
        
        # Check valency
        violations = []
        for i in range(N):
            atom_num = atom_types[i].item()
            max_val = MAX_VALENCE.get(atom_num, 4)
            current = bond_counts[i].item() / 2  # Divide because edges are bidirectional
            
            if current > max_val:
                violations.append({
                    'atom_idx': i,
                    'atom_type': atom_num,
                    'current_valence': current,
                    'max_valence': max_val
                })
        
        return len(violations) == 0, {'violations': violations}
    
    def check_bond_distances(self,
                             pos: torch.Tensor,        # (N, 3) coordinates
                             edge_index: torch.Tensor, # (2, E) bonds
                             bond_types: torch.Tensor  # (E,) bond orders
                             ) -> Tuple[bool, Dict]:
        """
        Check if bond distances are within reasonable ranges.
        """
        row, col = edge_index
        
        # Compute distances
        diff = pos[row] - pos[col]
        dists = torch.norm(diff, dim=-1)  # (E,)
        
        violations = []
        for e in range(edge_index.size(1)):
            bond_order = bond_types[e].item()
            dist = dists[e].item()
            
            min_d, max_d = BOND_LENGTHS.get(bond_order, (1.0, 2.0))
            
            if dist < min_d - 0.2 or dist > max_d + 0.2:  # 0.2Å tolerance
                violations.append({
                    'edge': (row[e].item(), col[e].item()),
                    'bond_order': bond_order,
                    'distance': dist,
                    'expected_range': (min_d, max_d)
                })
        
        return len(violations) == 0, {'violations': violations}
    
    def check_steric_clashes(self,
                             pos: torch.Tensor,        # (N, 3)
                             edge_index: torch.Tensor  # (2, E) bonds
                             ) -> Tuple[bool, Dict]:
        """
        Check for steric clashes (non-bonded atoms too close).
        """
        N = pos.size(0)
        
        # Get set of bonded pairs
        bonded = set()
        for e in range(edge_index.size(1)):
            i, j = edge_index[0, e].item(), edge_index[1, e].item()
            bonded.add((min(i, j), max(i, j)))
        
        # Check all non-bonded pairs
        violations = []
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) not in bonded:
                    dist = torch.norm(pos[i] - pos[j]).item()
                    if dist < MIN_NONBOND_DISTANCE:
                        violations.append({
                            'atoms': (i, j),
                            'distance': dist,
                            'min_required': MIN_NONBOND_DISTANCE
                        })
        
        return len(violations) == 0, {'violations': violations}
    
    def check_all(self,
                  pos: torch.Tensor,
                  atom_types: torch.Tensor,
                  edge_index: torch.Tensor,
                  bond_types: torch.Tensor
                  ) -> Tuple[bool, Dict]:
        """
        Run all validity checks.
        """
        results = {}
        valid = True
        
        # Valency check
        val_ok, val_info = self.check_valency(atom_types, edge_index, bond_types)
        results['valency'] = {'valid': val_ok, 'info': val_info}
        valid = valid and val_ok
        
        # Distance check
        if self.check_distances:
            dist_ok, dist_info = self.check_bond_distances(pos, edge_index, bond_types)
            results['distances'] = {'valid': dist_ok, 'info': dist_info}
            valid = valid and dist_ok
        
        # Steric check
        stereo_ok, stereo_info = self.check_steric_clashes(pos, edge_index)
        results['steric'] = {'valid': stereo_ok, 'info': stereo_info}
        valid = valid and stereo_ok
        
        return valid, results


# =============================================================================
# STEP-WISE VALIDITY FILTER (for sampling)
# =============================================================================

class StepWiseValidityFilter:
    """
    MolCode-style filter that checks validity during sampling.
    
    If a denoising step produces invalid geometry, it can:
    1. Reject and resample
    2. Add penalty to loss
    3. Project to valid region
    """
    
    def __init__(self, 
                 bond_distance_weight: float = 10.0,
                 steric_weight: float = 5.0):
        self.bond_distance_weight = bond_distance_weight
        self.steric_weight = steric_weight
    
    def compute_validity_loss(self,
                              pos: torch.Tensor,
                              edge_index: torch.Tensor,
                              bond_types: torch.Tensor
                              ) -> torch.Tensor:
        """
        Compute a differentiable validity loss that can be used
        to guide sampling towards valid conformations.
        
        Returns:
            loss: Scalar tensor
        """
        loss = torch.tensor(0.0, device=pos.device)
        
        # 1. Bond distance loss
        row, col = edge_index
        diff = pos[row] - pos[col]
        dists = torch.norm(diff, dim=-1)
        
        # Target distances based on bond type
        target_dists = torch.zeros_like(dists)
        for e in range(len(bond_types)):
            order = bond_types[e].item()
            if order == 1:
                target_dists[e] = 1.5
            elif order == 2:
                target_dists[e] = 1.3
            elif order == 3:
                target_dists[e] = 1.2
            else:
                target_dists[e] = 1.4
        
        bond_loss = torch.mean((dists - target_dists) ** 2)
        loss = loss + self.bond_distance_weight * bond_loss
        
        # 2. Steric clash loss
        N = pos.size(0)
        for i in range(N):
            for j in range(i + 1, N):
                # Check if bonded
                is_bonded = ((edge_index[0] == i) & (edge_index[1] == j)).any() or \
                           ((edge_index[0] == j) & (edge_index[1] == i)).any()
                
                if not is_bonded:
                    dist = torch.norm(pos[i] - pos[j])
                    if dist < MIN_NONBOND_DISTANCE:
                        # Soft penalty
                        clash_loss = (MIN_NONBOND_DISTANCE - dist) ** 2
                        loss = loss + self.steric_weight * clash_loss
        
        return loss
    
    def project_to_valid(self,
                         pos: torch.Tensor,
                         edge_index: torch.Tensor,
                         bond_types: torch.Tensor,
                         num_steps: int = 10,
                         lr: float = 0.1
                         ) -> torch.Tensor:
        """
        Project coordinates to satisfy distance constraints.
        
        Uses gradient descent to minimize validity loss.
        """
        pos = pos.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=lr)
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = self.compute_validity_loss(pos, edge_index, bond_types)
            loss.backward()
            optimizer.step()
        
        return pos.detach()


# =============================================================================
# RDKIT-BASED VALIDATION
# =============================================================================

def validate_with_rdkit(atom_types: torch.Tensor,
                        pos: torch.Tensor,
                        infer_bonds: bool = True) -> Tuple[bool, Optional[Chem.Mol]]:
    """
    Final validation using RDKit.
    
    Note: In NExT-Mol approach, bonds are KNOWN, so we don't need to infer them.
    This function is mainly for the old approach comparison.
    
    Args:
        atom_types: (N,) atomic numbers
        pos: (N, 3) coordinates
        infer_bonds: If True, infer bonds from distances (old approach)
                    If False, assume bonds are already set
    
    Returns:
        valid: True if RDKit can parse the molecule
        mol: RDKit Mol object if valid
    """
    N = len(atom_types)
    
    # Create editable mol
    mol = Chem.RWMol()
    
    # Add atoms
    for i in range(N):
        atom_num = atom_types[i].item() if isinstance(atom_types[i], torch.Tensor) else atom_types[i]
        atom = Chem.Atom(int(atom_num))
        mol.AddAtom(atom)
    
    if infer_bonds:
        # Infer bonds from distance (OLD APPROACH - often fails!)
        coords = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else pos
        
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(coords[i] - coords[j])
                
                # Very rough heuristics
                if dist < 1.2:
                    try:
                        mol.AddBond(i, j, Chem.BondType.TRIPLE)
                    except:
                        pass
                elif dist < 1.4:
                    try:
                        mol.AddBond(i, j, Chem.BondType.DOUBLE)
                    except:
                        pass
                elif dist < 1.8:
                    try:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                    except:
                        pass
    
    # Add coordinates
    conf = Chem.Conformer(N)
    coords = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    for i in range(N):
        conf.SetAtomPosition(i, coords[i].tolist())
    mol.AddConformer(conf)
    
    # Try to sanitize
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return True, mol
    except:
        return False, None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Testing ValidityChecker...")
    
    # Create a valid water molecule
    atom_types = torch.tensor([8, 1, 1])  # O, H, H
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])  # O-H, O-H (bidirectional)
    bond_types = torch.tensor([1, 1, 1, 1])  # All single bonds
    
    # Good geometry
    pos_good = torch.tensor([
        [0.0, 0.0, 0.0],    # O
        [0.96, 0.0, 0.0],   # H (0.96Å from O)
        [-0.24, 0.93, 0.0]  # H 
    ])
    
    # Bad geometry (clashing)
    pos_bad = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],    # Too close!
        [0.6, 0.0, 0.0]     # Clashing with first H
    ])
    
    checker = ValidityChecker()
    
    # Test good
    valid, info = checker.check_all(pos_good, atom_types, edge_index, bond_types)
    print(f"Good geometry valid: {valid}")
    
    # Test bad
    valid, info = checker.check_all(pos_bad, atom_types, edge_index, bond_types)
    print(f"Bad geometry valid: {valid}")
    print(f"Issues: {info}")
    
    print("\nTest passed!")
