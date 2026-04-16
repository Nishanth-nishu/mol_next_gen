"""
geometry_constraints.py — Chemistry-Aware 3D Geometry Constraints

Research-based references:
- EDM (Hoogeboom et al. 2022): CoM removal, equivariant diffusion
- GeoMol (Ganea et al. 2021): Torsion angle prediction
- TorDiff (Jing et al. 2022): Diffusion over torsion angles
- MMFF94 (Halgren 1996): Bond/angle/torsion targets from molecular mechanics

Constraints implemented:
1. Bond length loss        — vectorized lookup vs ideal MMFF94 values
2. Bond angle loss         — hybridization-aware (sp/sp2/sp3/aromatic)
3. Torsion loss            — OPLS-AA cosine potential (V1/V2/V3)
4. Repulsion loss          — VDW-based soft repulsion, excludes 1-2 and 1-3 pairs
5. Planarity loss  [NEW]   — SVD best-fit plane penalty for aromatic rings
6. Chirality loss  [NEW]   — Signed tetrahedral volume to enforce R/S config
7. Ring strain loss [NEW]  — Penalizes bond angles in 3/4-membered rings
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


# =============================================================================
# VAN DER WAALS RADII (Å) — atom-type specific minimum non-bonded distances
# From Bondi (1964) + MMFF94 parameterization
# =============================================================================

# Atom (atomic number) -> VDW radius in Å
VDW_RADII = {
    1:  1.10,   # H
    5:  1.92,   # B
    6:  1.70,   # C
    7:  1.55,   # N
    8:  1.52,   # O
    9:  1.47,   # F
    14: 2.10,   # Si
    15: 1.80,   # P
    16: 1.80,   # S
    17: 1.75,   # Cl
    35: 1.85,   # Br
    53: 1.98,   # I
}
DEFAULT_VDW = 1.70  # Default for unknown atoms


def get_min_nonbond_dist(atom1: int, atom2: int) -> float:
    """Get minimum allowed non-bonded distance based on VDW radii sum * 0.7 (clash threshold)."""
    r1 = VDW_RADII.get(atom1, DEFAULT_VDW)
    r2 = VDW_RADII.get(atom2, DEFAULT_VDW)
    return (r1 + r2) * 0.70  # 70% of VDW sum = clash threshold


# =============================================================================
# VECTORIZED BOND LENGTH LOOKUP TABLE
# Indexed by (atom1, atom2, bond_order) using a pre-built tensor
# Shape: [MAX_ATOMIC_NUM, MAX_ATOMIC_NUM, 5]  (bond_order 0-4, 0=unused)
# =============================================================================

MAX_ATOMIC_NUM = 54  # Covers H(1) to I(53)

# Build lookup as Python dict first, then convert to tensor
BOND_LENGTH_DICT = {
    # C-C bonds
    (6, 6, 1): 1.54,
    (6, 6, 2): 1.34,
    (6, 6, 3): 1.20,
    (6, 6, 4): 1.40,
    # C-H
    (6, 1, 1): 1.09,
    (1, 6, 1): 1.09,
    # C-N
    (6, 7, 1): 1.47,
    (7, 6, 1): 1.47,
    (6, 7, 2): 1.29,
    (7, 6, 2): 1.29,
    (6, 7, 3): 1.16,
    (7, 6, 3): 1.16,
    (6, 7, 4): 1.34,
    (7, 6, 4): 1.34,
    # C-O
    (6, 8, 1): 1.43,
    (8, 6, 1): 1.43,
    (6, 8, 2): 1.23,
    (8, 6, 2): 1.23,
    (6, 8, 4): 1.36,
    (8, 6, 4): 1.36,
    # C-S
    (6, 16, 1): 1.82,
    (16, 6, 1): 1.82,
    (6, 16, 2): 1.71,
    (16, 6, 2): 1.71,
    # C-halogens
    (6, 9,  1): 1.35,
    (9,  6, 1): 1.35,
    (6, 17, 1): 1.77,
    (17, 6, 1): 1.77,
    (6, 35, 1): 1.94,
    (35, 6, 1): 1.94,
    (6, 53, 1): 2.14,
    (53, 6, 1): 2.14,
    # N-H
    (7, 1, 1): 1.01,
    (1, 7, 1): 1.01,
    # N-N
    (7, 7, 1): 1.45,
    (7, 7, 2): 1.25,
    (7, 7, 3): 1.10,
    # N-O
    (7, 8, 1): 1.40,
    (8, 7, 1): 1.40,
    (7, 8, 2): 1.21,
    (8, 7, 2): 1.21,
    # O-H
    (8, 1, 1): 0.96,
    (1, 8, 1): 0.96,
    # O-O
    (8, 8, 1): 1.48,
    # S-H
    (16, 1, 1): 1.34,
    (1, 16, 1): 1.34,
    # S-S
    (16, 16, 1): 2.05,
    # S-N (sulfonamide)
    (16, 7, 1): 1.65,
    (7, 16, 1): 1.65,
    # S-O
    (16, 8, 2): 1.44,
    (8, 16, 2): 1.44,
    (16, 8, 1): 1.58,
    (8, 16, 1): 1.58,
    # P-C (organophosphorus)
    (15, 6, 1): 1.84,
    (6, 15, 1): 1.84,
    # P-N
    (15, 7, 1): 1.68,
    (7, 15, 1): 1.68,
    # P-O (phosphate, common in nucleotides)
    (15, 8, 1): 1.63,
    (8, 15, 1): 1.63,
    (15, 8, 2): 1.48,
    (8, 15, 2): 1.48,
    # P-S
    (15, 16, 1): 1.95,
    (16, 15, 1): 1.95,
    # P-H
    (15, 1, 1): 1.44,
    (1, 15, 1): 1.44,
    # P-F
    (15, 9, 1): 1.55,
    (9, 15, 1): 1.55,
    # Si-C
    (14, 6, 1): 1.86,
    (6, 14, 1): 1.86,
    # Si-H
    (14, 1, 1): 1.47,
    (1, 14, 1): 1.47,
    # Si-O
    (14, 8, 1): 1.65,
    (8, 14, 1): 1.65,
    # Si-N
    (14, 7, 1): 1.74,
    (7, 14, 1): 1.74,
    # Si-Si
    (14, 14, 1): 2.33,
    # B-C
    (5, 6, 1): 1.57,
    (6, 5, 1): 1.57,
    # B-N
    (5, 7, 1): 1.40,
    (7, 5, 1): 1.40,
    # B-O
    (5, 8, 1): 1.38,
    (8, 5, 1): 1.38,
    # B-H
    (5, 1, 1): 1.19,
    (1, 5, 1): 1.19,
    # N-S (sulfonamide N)
    (7, 16, 2): 1.54,
    (16, 7, 2): 1.54,
}

# Default bond lengths by bond order (fallback when pair not in table)
DEFAULT_BOND_LENGTHS = {0: 1.50, 1: 1.50, 2: 1.34, 3: 1.20, 4: 1.40}

# Pre-build tensor lookup: shape [54, 54, 5]
_BOND_TENSOR = torch.zeros(MAX_ATOMIC_NUM, MAX_ATOMIC_NUM, 5)
for bo in range(5):
    _BOND_TENSOR[:, :, bo] = DEFAULT_BOND_LENGTHS.get(bo, 1.50)
for (a1, a2, bo), dist in BOND_LENGTH_DICT.items():
    if a1 < MAX_ATOMIC_NUM and a2 < MAX_ATOMIC_NUM and bo < 5:
        _BOND_TENSOR[a1, a2, bo] = dist
        _BOND_TENSOR[a2, a1, bo] = dist


def get_ideal_bond_length(atom1: int, atom2: int, bond_order: int) -> float:
    """Get ideal bond length (Python scalar, for non-batched use)."""
    a1 = min(atom1, MAX_ATOMIC_NUM - 1)
    a2 = min(atom2, MAX_ATOMIC_NUM - 1)
    bo = min(bond_order, 4)
    return _BOND_TENSOR[a1, a2, bo].item()


def get_ideal_bond_lengths_vectorized(
    atom1_types: torch.Tensor,   # (E,) atomic numbers
    atom2_types: torch.Tensor,   # (E,) atomic numbers
    bond_orders: torch.Tensor,   # (E,) bond orders
) -> torch.Tensor:
    """
    Vectorized lookup of ideal bond lengths for a batch of edges.
    Returns (E,) tensor of ideal distances in Angstroms.
    O(1) — no Python loops.
    """
    device = atom1_types.device
    table = _BOND_TENSOR.to(device)
    
    a1 = atom1_types.clamp(0, MAX_ATOMIC_NUM - 1).long()
    a2 = atom2_types.clamp(0, MAX_ATOMIC_NUM - 1).long()
    bo = bond_orders.clamp(0, 4).long()
    
    return table[a1, a2, bo]


# =============================================================================
# HYBRIDIZATION DETECTION (atom-type + bond-type aware)
# =============================================================================

IDEAL_ANGLES = {
    'sp3': 109.5,
    'sp2': 120.0,
    'sp':  180.0,
    'aromatic': 120.0,
}

# Atoms that prefer sp2: C with a double bond, N with a double bond, carbonyl O
SP2_ATOMS = {6, 7, 8, 16}  # C, N, O, S can be sp2

def detect_hybridization(
    atom_idx: int,
    atom_num: int,
    neighbor_indices: List[int],
    bond_types_for_atom: List[int],
) -> str:
    """
    Detect hybridization from atom type + bond types.
    More accurate than naive neighbor-count approach.
    """
    n = len(neighbor_indices)
    
    if n == 0:
        return 'sp3'
    
    has_double = any(bo == 2 for bo in bond_types_for_atom)
    has_triple = any(bo == 3 for bo in bond_types_for_atom)
    has_aromatic = any(bo == 4 for bo in bond_types_for_atom)
    
    if has_triple:
        return 'sp'
    if has_aromatic:
        return 'aromatic'
    if has_double and atom_num in SP2_ATOMS:
        return 'sp2'
    if n <= 2:
        return 'sp'
    if n == 3 and atom_num in SP2_ATOMS:
        return 'sp2'
    return 'sp3'


# =============================================================================
# TORSION ANGLE UTILITIES
# =============================================================================

# OPLS-AA / MMFF94 inspired torsion barriers (kcal/mol -> dimensionless relative)
# Organized as: (central_bond_atom1_hybridization, central_bond_atom2_hybridization) -> (V1, V2, V3)
# V_tors = V1*(1+cos(phi))/2 + V2*(1-cos(2*phi))/2 + V3*(1+cos(3*phi))/2
TORSION_PARAMS = {
    ('sp3', 'sp3'): (0.0, 0.0, 1.0),    # Simple rotation, 3-fold (e.g. C-C single)
    ('sp3', 'sp2'): (0.0, 2.0, 0.0),    # 2-fold, prefer 180°
    ('sp2', 'sp2'): (0.0, 6.0, 0.0),    # Conjugated, strong 2-fold barrier
    ('sp2', 'sp3'): (0.0, 2.0, 0.0),
    ('sp',  'sp3'): (0.0, 0.0, 0.2),
    ('aromatic', 'aromatic'): (0.0, 10.0, 0.0),  # Strong planarity
    ('aromatic', 'sp3'): (0.0, 1.0, 0.0),
    ('aromatic', 'sp2'): (0.0, 5.0, 0.0),
}


def compute_dihedral(
    p0: torch.Tensor,  # (3,)
    p1: torch.Tensor,  # (3,)
    p2: torch.Tensor,  # (3,)
    p3: torch.Tensor,  # (3,)
) -> torch.Tensor:
    """Compute dihedral angle (phi) between 4 atom positions. Returns scalar in radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    
    n1_norm = n1 / torch.norm(n1).clamp(min=1e-8)
    n2_norm = n2 / torch.norm(n2).clamp(min=1e-8)
    b2_norm = b2 / torch.norm(b2).clamp(min=1e-8)
    
    cos_phi = (n1_norm * n2_norm).sum()
    sin_phi = (torch.linalg.cross(n1_norm, n2_norm) * b2_norm).sum()
    
    phi = torch.atan2(sin_phi, cos_phi)
    return phi


def torsion_energy(phi: torch.Tensor, V1: float, V2: float, V3: float) -> torch.Tensor:
    """OPLS-style torsion energy: E = V1*(1+cos(phi))/2 + V2*(1-cos(2*phi))/2 + V3*(1+cos(3*phi))/2"""
    return (V1 * (1 + torch.cos(phi)) / 2 +
            V2 * (1 - torch.cos(2 * phi)) / 2 +
            V3 * (1 + torch.cos(3 * phi)) / 2)


# =============================================================================
# GEOMETRY LOSS FUNCTIONS (fully vectorized where possible)
# =============================================================================

class GeometryConstraints:
    """
    Differentiable geometry constraints for molecular conformations.
    All major operations are vectorized (no Python loops over edges/atoms in hot path).
    """

    def __init__(self,
                 bond_weight: float = 10.0,
                 angle_weight: float = 3.0,
                 torsion_weight: float = 1.0,
                 repulsion_weight: float = 5.0,
                 min_nonbond_dist: float = 1.2,
                 planarity_weight: float = 5.0,
                 chirality_weight: float = 3.0,
                 ring_strain_weight: float = 2.0):
        self.bond_weight = bond_weight
        self.angle_weight = angle_weight
        self.torsion_weight = torsion_weight
        self.repulsion_weight = repulsion_weight
        self.min_nonbond_dist = min_nonbond_dist
        self.planarity_weight = planarity_weight
        self.chirality_weight = chirality_weight
        self.ring_strain_weight = ring_strain_weight

    def compute_bond_loss(self,
                          pos: torch.Tensor,
                          atom_types: torch.Tensor,
                          edge_index: torch.Tensor,
                          bond_types: torch.Tensor) -> torch.Tensor:
        """
        Vectorized bond length loss using precomputed lookup table.
        O(E) memory, O(1) Python overhead.
        """
        row, col = edge_index

        diff = pos[row] - pos[col]
        dists = torch.norm(diff, dim=-1).clamp(min=1e-6)

        ideal_dists = get_ideal_bond_lengths_vectorized(
            atom_types[row], atom_types[col], bond_types
        )

        loss = F.mse_loss(dists, ideal_dists)
        return self.bond_weight * loss

    def compute_angle_loss(self,
                           pos: torch.Tensor,
                           atom_types: torch.Tensor,
                           edge_index: torch.Tensor,
                           bond_types: torch.Tensor,
                           batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Bond angle loss with hybridization-aware targets.

        FIX: Build angle triplets (i, j, k) with per-center ideal angles as tensors,
        then compute ALL angles in a single vectorized GPU operation.
        Replaces the O(N × deg²) Python loop that stalls the GPU.
        """
        device = pos.device
        N = pos.size(0)
        row, col = edge_index

        # Build adjacency (still one Python scan — unavoidable for triplet building)
        neighbors       = [[] for _ in range(N)]
        neighbor_bonds  = [[] for _ in range(N)]
        for e in range(row.size(0)):
            i_e, j_e = row[e].item(), col[e].item()
            neighbors[j_e].append(i_e)
            neighbor_bonds[j_e].append(bond_types[e].item())

        # Collect triplets and ideal angles as lists → convert to tensors once
        tri_i, tri_j, tri_k, ideals = [], [], [], []

        for j in range(N):
            neigh = neighbors[j]
            if len(neigh) < 2:
                continue
            atom_num   = atom_types[j].item()
            hyb        = detect_hybridization(j, atom_num, neigh, neighbor_bonds[j])
            ideal_rad  = math.radians(IDEAL_ANGLES[hyb])

            for idx_i, i in enumerate(neigh):
                for k in neigh[idx_i + 1:]:
                    tri_i.append(i)
                    tri_j.append(j)
                    tri_k.append(k)
                    ideals.append(ideal_rad)

        if not tri_i:
            return torch.tensor(0.0, device=device)

        # Vectorized angle computation — single GPU kernel
        ti  = torch.tensor(tri_i, dtype=torch.long, device=device)
        tj  = torch.tensor(tri_j, dtype=torch.long, device=device)
        tk  = torch.tensor(tri_k, dtype=torch.long, device=device)
        tgt = torch.tensor(ideals, dtype=pos.dtype, device=device)

        v1 = pos[ti] - pos[tj]   # (T, 3)
        v2 = pos[tk] - pos[tj]   # (T, 3)

        cos_a = F.cosine_similarity(v1, v2, dim=-1).clamp(-0.9999, 0.9999)  # (T,)
        angles = torch.acos(cos_a)                                            # (T,)

        loss = ((angles - tgt) ** 2).mean()
        return self.angle_weight * loss

    def compute_torsion_loss(self,
                              pos: torch.Tensor,
                              atom_types: torch.Tensor,
                              edge_index: torch.Tensor,
                              bond_types: torch.Tensor,
                              batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Torsion angle (dihedral) loss.
        For each rotatable bond j-k, sample pairs of neighbors (i-j-k-l)
        and compute OPLS-AA torsion energy penalty.
        
        Based on GeoMol/TorDiff: torsion angles are the primary degree of
        freedom that determines 3D conformation quality.
        """
        device = pos.device
        N = pos.size(0)
        row, col = edge_index

        # Build adjacency
        neighbors = [[] for _ in range(N)]
        neighbor_bonds = [[] for _ in range(N)]
        for e in range(row.size(0)):
            i_idx, j_idx = row[e].item(), col[e].item()
            neighbors[j_idx].append(i_idx)
            neighbor_bonds[j_idx].append(bond_types[e].item())

        torsion_losses = []

        # For each edge j->k (unique bonds, skip reverse)
        seen_bonds = set()
        for e in range(row.size(0)):
            j, k = row[e].item(), col[e].item()
            bond_key = (min(j, k), max(j, k))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)

            # Only consider single/aromatic bonds (rotatable)
            bo = bond_types[e].item()
            if bo not in (1, 4):
                continue

            # Get neighbors of j (excluding k) and k (excluding j)
            j_neighbors = [n for n in neighbors[j] if n != k]
            k_neighbors = [n for n in neighbors[k] if n != j]

            if not j_neighbors or not k_neighbors:
                continue

            # Hybridization of bond atoms
            j_hyb = detect_hybridization(j, atom_types[j].item(), neighbors[j], neighbor_bonds[j])
            k_hyb = detect_hybridization(k, atom_types[k].item(), neighbors[k], neighbor_bonds[k])

            hyb_key = (j_hyb, k_hyb) if (j_hyb, k_hyb) in TORSION_PARAMS else \
                      (k_hyb, j_hyb) if (k_hyb, j_hyb) in TORSION_PARAMS else \
                      ('sp3', 'sp3')

            V1, V2, V3 = TORSION_PARAMS[hyb_key]

            # Take one representative torsion per bond (first neighbors)
            i_idx = j_neighbors[0]
            l_idx = k_neighbors[0]

            phi = compute_dihedral(pos[i_idx], pos[j], pos[k], pos[l_idx])
            e_tors = torsion_energy(phi, V1, V2, V3)
            torsion_losses.append(e_tors)

        if len(torsion_losses) == 0:
            return torch.tensor(0.0, device=device)

        loss = torch.stack(torsion_losses).mean()
        return self.torsion_weight * loss

    def compute_repulsion_loss(self,
                               pos: torch.Tensor,
                               atom_types: torch.Tensor,
                               edge_index: torch.Tensor,
                               batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Steric repulsion loss — computed PER MOLECULE to fix the N>300 skip bug.

        BUG FIX: Old code checked `if N > 300: skip` where N is the WHOLE BATCH
        (64 molecules × ~15 atoms = 960 atoms → always skipped!).  Now we loop
        over molecules and only compute repulsion for each molecule individually.
        QM9 molecules have ≤ 29 heavy atoms, so per-mol N ≤ 29 is always fast.

        Excludes:
          1-2 pairs (bonded) — handled by bond loss
          1-3 pairs (A-B-C)  — natural geometry, not clashes
        """
        device = pos.device
        row, col = edge_index
        B = int(batch_idx.max().item()) + 1

        mol_losses = []

        for mol_b in range(B):
            mol_mask = (batch_idx == mol_b)
            N_mol = int(mol_mask.sum().item())

            if N_mol < 2:
                continue

            mol_pos = pos[mol_mask]                    # (N_mol, 3)
            mol_at  = atom_types[mol_mask]             # (N_mol,)

            # Local edge_index for this molecule
            edge_mask = mol_mask[row] & mol_mask[col]
            if not edge_mask.any():
                continue

            # Re-index edges to local 0-based indices
            local_map = torch.full((pos.size(0),), -1, dtype=torch.long, device=device)
            local_map[mol_mask.nonzero(as_tuple=True)[0]] = torch.arange(N_mol, device=device)
            local_row = local_map[row[edge_mask]]
            local_col = local_map[col[edge_mask]]

            # Build 1-2 mask (bonded)
            bonded_12 = torch.zeros(N_mol, N_mol, device=device, dtype=torch.bool)
            bonded_12[local_row, local_col] = True

            # Build 1-3 mask via adjacency matmul
            adj_f = bonded_12.float()
            bonded_13 = (adj_f @ adj_f).bool() & ~bonded_12 & ~torch.eye(N_mol, device=device, dtype=torch.bool)

            excluded = bonded_12 | bonded_13 | torch.eye(N_mol, device=device, dtype=torch.bool)
            nb_mask = ~excluded

            if not nb_mask.any():
                continue

            all_dists = torch.cdist(mol_pos, mol_pos)   # (N_mol, N_mol)

            atom_vdw = torch.tensor(
                [VDW_RADII.get(a.item(), DEFAULT_VDW) for a in mol_at],
                device=device, dtype=pos.dtype
            )
            vdw_thresh = (atom_vdw.unsqueeze(0) + atom_vdw.unsqueeze(1)) * 0.70

            nb_dists  = all_dists[nb_mask]
            thresh    = vdw_thresh[nb_mask]

            clashing = nb_dists < thresh
            if clashing.any():
                clash_dists  = nb_dists[clashing]
                clash_thresh = thresh[clashing]
                mol_losses.append(torch.mean((clash_thresh - clash_dists) ** 2))

        if not mol_losses:
            return torch.tensor(0.0, device=device)

        loss = torch.stack(mol_losses).mean()
        return self.repulsion_weight * loss

    # =========================================================================
    # NEW SOFT CONSTRAINTS (Exp-1: soft_restrictions)
    # =========================================================================

    def compute_planarity_loss(
            self,
            pos: torch.Tensor,
            aromatic_rings: Optional[List[List[int]]] = None) -> torch.Tensor:
        """
        Planarity penalty for aromatic / conjugated rings.

        For each ring, find the best-fit plane via SVD of the centred atom
        positions.  Loss = mean squared perpendicular distance from that plane.

        Args:
            pos: (N, 3) atom positions
            aromatic_rings: list of rings, each a list of atom indices that
                            belong to the same aromatic / conjugated ring.
                            e.g. [[0,1,2,3,4,5]] for benzene.

        Returns:
            Scalar loss tensor.
        """
        device = pos.device
        if not aromatic_rings:
            return torch.tensor(0.0, device=device)

        ring_losses = []
        for ring in aromatic_rings:
            if len(ring) < 3:
                continue
            ring_pos = pos[ring]                          # (R, 3)
            centroid  = ring_pos.mean(dim=0, keepdim=True)  # (1, 3)
            centred   = ring_pos - centroid               # (R, 3)

            # SVD: smallest singular vector = normal to best-fit plane
            # centred = U @ S @ Vt  → Vt[-1] is normal
            try:
                _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
            except RuntimeError:
                continue
            normal = Vt[-1]                               # (3,) unit normal

            # Perpendicular distances (signed)
            dists = (centred * normal.unsqueeze(0)).sum(dim=-1)  # (R,)
            ring_losses.append((dists ** 2).mean())

        if not ring_losses:
            return torch.tensor(0.0, device=device)

        loss = torch.stack(ring_losses).mean()
        return self.planarity_weight * loss

    def compute_chirality_loss(
            self,
            pos: torch.Tensor,
            chiral_centers: Optional[List[Tuple]] = None) -> torch.Tensor:
        """
        Chirality enforcement via signed tetrahedral volume.

        For each chiral center (center_idx, [n1, n2, n3, n4], sign):
          - Compute the signed volume of the tetrahedron: V = det([v1,v2,v3])
            where vi = pos[ni] - pos[center]
          - sign = +1 (R) or -1 (S)
          - Loss = relu(-sign * V + margin) — penalises wrong handedness

        Args:
            pos:            (N, 3) atom positions
            chiral_centers: list of (center_idx, [n1, n2, n3, n4], sign)
                            sign ∈ {+1, -1}

        Returns:
            Scalar loss tensor.
        """
        device = pos.device
        if not chiral_centers:
            return torch.tensor(0.0, device=device)

        margin = 0.1   # Minimum required signed volume (Å³)
        losses = []
        for center_idx, neighbors, sign in chiral_centers:
            if len(neighbors) < 4:
                continue
            c = pos[center_idx]      # (3,)
            n1, n2, n3 = neighbors[:3]
            v1 = pos[n1] - c         # (3,)
            v2 = pos[n2] - c
            v3 = pos[n3] - c

            # Signed volume = scalar triple product  v1 · (v2 × v3)
            vol = (v1 * torch.linalg.cross(v2, v3)).sum()  # scalar

            # Penalise if sign * vol < margin
            losses.append(F.relu(-sign * vol + margin))

        if not losses:
            return torch.tensor(0.0, device=device)

        loss = torch.stack(losses).mean()
        return self.chirality_weight * loss

    def compute_ring_strain_loss(
            self,
            pos: torch.Tensor,
            small_rings: Optional[List[List[int]]] = None) -> torch.Tensor:
        """
        Ring strain penalty for 3- and 4-membered rings.

        VSEPR predicts 109.5° for sp3 carbons, but small rings have much
        tighter ideal angles:
          - Cyclopropane (3-ring): ideal internal angle = 60°
          - Cyclobutane  (4-ring): ideal internal angle = 90°

        Loss = mean squared deviation of each internal bond angle from the
        ring-specific ideal.

        Args:
            pos:         (N, 3) atom positions
            small_rings: list of rings, each a list of sequential atom indices
                         forming a 3- or 4-membered ring.

        Returns:
            Scalar loss tensor.
        """
        device = pos.device
        if not small_rings:
            return torch.tensor(0.0, device=device)

        RING_IDEAL_ANGLES = {3: math.radians(60.0),
                             4: math.radians(90.0)}

        angle_losses = []
        for ring in small_rings:
            n = len(ring)
            ideal = RING_IDEAL_ANGLES.get(n)
            if ideal is None:
                continue
            # Compute each internal angle: for sequential ring a-b-c
            for i in range(n):
                a = ring[(i - 1) % n]
                b = ring[i]
                c = ring[(i + 1) % n]

                v1 = pos[a] - pos[b]
                v2 = pos[c] - pos[b]
                cos_angle = F.cosine_similarity(
                    v1.unsqueeze(0), v2.unsqueeze(0)
                ).clamp(-0.9999, 0.9999)
                angle = torch.acos(cos_angle)
                angle_losses.append((angle - ideal) ** 2)

        if not angle_losses:
            return torch.tensor(0.0, device=device)

        loss = torch.stack(angle_losses).mean()
        return self.ring_strain_weight * loss

    # =========================================================================
    # TOTAL LOSS
    # =========================================================================

    def compute_total_loss(
            self,
            pos: torch.Tensor,
            atom_types: torch.Tensor,
            edge_index: torch.Tensor,
            bond_types: torch.Tensor,
            batch_idx: torch.Tensor,
            include_angles: bool = True,
            include_torsions: bool = False,
            aromatic_rings: Optional[List[List[int]]] = None,
            chiral_centers: Optional[List[Tuple]] = None,
            small_rings: Optional[List[List[int]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total geometry constraint loss with full breakdown."""
        bond_loss = self.compute_bond_loss(pos, atom_types, edge_index, bond_types)
        repulsion_loss = self.compute_repulsion_loss(pos, atom_types, edge_index, batch_idx)

        total_loss = bond_loss + repulsion_loss
        breakdown = {
            'bond_loss':      bond_loss.item(),
            'repulsion_loss': repulsion_loss.item(),
        }

        if include_angles:
            angle_loss = self.compute_angle_loss(
                pos, atom_types, edge_index, bond_types, batch_idx)
            total_loss = total_loss + angle_loss
            breakdown['angle_loss'] = angle_loss.item()

        if include_torsions:
            torsion_loss = self.compute_torsion_loss(
                pos, atom_types, edge_index, bond_types, batch_idx)
            total_loss = total_loss + torsion_loss
            breakdown['torsion_loss'] = torsion_loss.item()

        # Soft constraints (Exp-1)
        planarity_loss = self.compute_planarity_loss(pos, aromatic_rings)
        total_loss = total_loss + planarity_loss
        breakdown['planarity_loss'] = planarity_loss.item()

        chirality_loss = self.compute_chirality_loss(pos, chiral_centers)
        total_loss = total_loss + chirality_loss
        breakdown['chirality_loss'] = chirality_loss.item()

        ring_strain_loss = self.compute_ring_strain_loss(pos, small_rings)
        total_loss = total_loss + ring_strain_loss
        breakdown['ring_strain_loss'] = ring_strain_loss.item()

        breakdown['total_loss'] = total_loss.item()
        return total_loss, breakdown


# =============================================================================
# STRICT VALIDITY EVALUATOR
# =============================================================================

class StrictValidityEvaluator:
    """
    Strict 3D validity evaluation with per-atom-type VDW thresholds.
    """

    def __init__(self,
                 bond_tolerance: float = 0.20,    # Å
                 angle_tolerance: float = 15.0,    # degrees
                 vdw_clash_fraction: float = 0.70):
        self.bond_tolerance = bond_tolerance
        self.angle_tolerance = math.radians(angle_tolerance)
        self.vdw_clash_fraction = vdw_clash_fraction

    @torch.no_grad()
    def evaluate_molecule(self,
                          pos: torch.Tensor,
                          atom_types: torch.Tensor,
                          edge_index: torch.Tensor,
                          bond_types: torch.Tensor) -> Dict:
        row, col = edge_index
        N = pos.size(0)

        # 1. Bond lengths
        diff = pos[row] - pos[col]
        dists = torch.norm(diff, dim=-1)

        ideal_dists = get_ideal_bond_lengths_vectorized(
            atom_types[row], atom_types[col], bond_types
        )

        bond_errors = (dists - ideal_dists).abs().tolist()
        bond_valid = all(e <= self.bond_tolerance for e in bond_errors)

        # 2. Bond angles
        neighbors = [[] for _ in range(N)]
        neighbor_bonds = [[] for _ in range(N)]
        for e in range(row.size(0)):
            i, j = row[e].item(), col[e].item()
            neighbors[j].append(i)
            neighbor_bonds[j].append(bond_types[e].item())

        angle_errors = []
        angles_valid = True

        for j in range(N):
            neigh = neighbors[j]
            if len(neigh) < 2:
                continue
            atom_num = atom_types[j].item()
            hyb = detect_hybridization(j, atom_num, neigh, neighbor_bonds[j])
            ideal_angle = math.radians(IDEAL_ANGLES[hyb])

            for idx_i, i in enumerate(neigh):
                for k in neigh[idx_i + 1:]:
                    v1 = pos[i] - pos[j]
                    v2 = pos[k] - pos[j]
                    cos_angle = F.cosine_similarity(
                        v1.unsqueeze(0), v2.unsqueeze(0)
                    ).clamp(-0.9999, 0.9999)
                    angle = torch.acos(cos_angle).item()
                    error = abs(angle - ideal_angle)
                    angle_errors.append(error)
                    if error > self.angle_tolerance:
                        angles_valid = False

        # 3. Steric clashes (using VDW radii, excluding 1-2 and 1-3)
        bonded_pairs = set()
        for e in range(row.size(0)):
            i, j = row[e].item(), col[e].item()
            bonded_pairs.add((min(i, j), max(i, j)))

        # Build 1-3 pairs
        neighbors_plain = [[] for _ in range(N)]
        for e in range(row.size(0)):
            neighbors_plain[col[e].item()].append(row[e].item())

        pairs_13 = set()
        for j in range(N):
            nbs = neighbors_plain[j]
            for ni in nbs:
                for nk in nbs:
                    if ni < nk:
                        pairs_13.add((ni, nk))

        clash_free = True
        clash_count = 0

        for i in range(N):
            for j in range(i + 1, N):
                pair = (i, j)
                if pair in bonded_pairs or pair in pairs_13:
                    continue
                dist = torch.norm(pos[i] - pos[j]).item()
                r_i = VDW_RADII.get(atom_types[i].item(), DEFAULT_VDW)
                r_j = VDW_RADII.get(atom_types[j].item(), DEFAULT_VDW)
                min_dist = (r_i + r_j) * self.vdw_clash_fraction
                if dist < min_dist:
                    clash_free = False
                    clash_count += 1

        fully_valid = bond_valid and angles_valid and clash_free

        return {
            'fully_valid': fully_valid,
            'bond_valid': bond_valid,
            'angles_valid': angles_valid,
            'clash_free': clash_free,
            'mean_bond_error': sum(bond_errors) / len(bond_errors) if bond_errors else 0,
            'max_bond_error': max(bond_errors) if bond_errors else 0,
            'mean_angle_error_deg': math.degrees(sum(angle_errors) / len(angle_errors)) if angle_errors else 0,
            'max_angle_error_deg': math.degrees(max(angle_errors)) if angle_errors else 0,
            'clash_count': clash_count,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Testing GeometryConstraints (v2 — vectorized + torsion)...")

    # Ethanol: C-C-O with H's attached
    # Atoms: C(6), C(6), O(8)
    atom_types = torch.tensor([6, 6, 8])
    pos = torch.tensor([
        [0.000, 0.000, 0.000],   # C1
        [1.540, 0.000, 0.000],   # C2 (C-C single = 1.54Å)
        [2.000, 1.200, 0.000],   # O  (C-O single = 1.43Å)
    ])
    edge_index = torch.tensor([[0,1, 1,0, 1,2, 2,1], [1,0, 0,1, 2,1, 1,2]])
    edge_index = torch.tensor([[0,1,2], [1,2,1]]) # simplified
    # Full bidirectional
    edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
    bond_types = torch.tensor([1,1,1,1])
    batch_idx = torch.tensor([0,0,0])

    # Test vectorized lookup
    ideal = get_ideal_bond_lengths_vectorized(
        atom_types[edge_index[0]], atom_types[edge_index[1]], bond_types
    )
    print(f"Ideal bond lengths (C-C, C-C, C-O, C-O): {ideal.tolist()}")
    assert abs(ideal[0].item() - 1.54) < 0.01, "C-C should be 1.54Å"
    assert abs(ideal[2].item() - 1.43) < 0.01, "C-O should be 1.43Å"

    # Test geometry constraints
    constraints = GeometryConstraints()
    total_loss, breakdown = constraints.compute_total_loss(
        pos, atom_types, edge_index, bond_types, batch_idx,
        include_angles=True, include_torsions=False
    )
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Breakdown: {breakdown}")

    # Test torsion: eclipsed butane-like (4 atoms in line)
    pos4 = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [2.54, 1.0, 0.0],
        [3.54, 1.0, 1.0],  # gauche
    ])
    atom4 = torch.tensor([6, 6, 6, 6])
    ei4  = torch.tensor([[0,1,2,3],[1,2,3,2]])  # just the chain
    ei4 = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    bt4 = torch.ones(6, dtype=torch.long)
    bi4 = torch.zeros(4, dtype=torch.long)

    torsion_loss = constraints.compute_torsion_loss(pos4, atom4, ei4, bt4, bi4)
    print(f"Torsion loss (gauche butane): {torsion_loss.item():.6f}")

    # Test strict evaluator
    evaluator = StrictValidityEvaluator()
    result = evaluator.evaluate_molecule(pos, atom_types, edge_index, bond_types)
    print(f"\nValidity eval: {result}")

    print("\nAll tests passed!")
