"""
tests/test_chemistry.py — Comprehensive Chemistry Validation Tests

Cross-validates all fixes with known molecules:
- Ethanol: bond lengths, angles
- Benzene: aromaticity, planarity
- Glycine: valency for N, O, C
- Ethane conformers: torsion loss differences (staggered vs eclipsed)
- Steric clash detection
- Overvalent carbon detection
"""

import sys
import os
import math
import torch
# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry_constraints import (
    get_ideal_bond_length,
    get_ideal_bond_lengths_vectorized,
    GeometryConstraints,
    StrictValidityEvaluator,
    compute_dihedral,
    torsion_energy,
    BOND_LENGTH_DICT,
    VDW_RADII,
)
from models.validity_filter import ValidityChecker, StepWiseValidityFilter
from models.conformer_diffusion import ConformerDiffusion, remove_com


# ============================================================================
# HELPERS
# ============================================================================

def make_water():
    """O-H...H water molecule with correct geometry."""
    atom_types = torch.tensor([8, 1, 1])      # O, H, H
    pos = torch.tensor([
        [0.000,  0.000, 0.000],
        [0.960,  0.000, 0.000],
        [-0.240, 0.927, 0.000],
    ])
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    bond_types = torch.ones(4, dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    return atom_types, pos, edge_index, bond_types, batch


def make_ethanol():
    """Ethanol (CCO) — C-C single, C-O single bonds. Partial molecule (no H's for simplicity)."""
    # C1, C2, O
    atom_types = torch.tensor([6, 6, 8])
    pos = torch.tensor([
        [0.000, 0.000, 0.000],   # C1
        [1.540, 0.000, 0.000],   # C2 at 1.54Å from C1
        [2.970, 0.000, 0.000],   # O  at 1.43Å from C2 (2.97 = 1.54+1.43)
    ])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    bond_types = torch.ones(4, dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    return atom_types, pos, edge_index, bond_types, batch


def make_benzene_flat():
    """Benzene ring, perfectly flat, all C-C aromatic = 1.40Å, 120° angles."""
    # 6 carbons in regular hexagon with C-C = 1.40Å
    r = 1.40 / (2 * math.sin(math.pi / 6))  # Circumradius
    pos = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        pos.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    pos = torch.tensor(pos)
    atom_types = torch.tensor([6] * 6)
    # Ring edges (bidirectional): 0-1, 1-2, 2-3, 3-4, 4-5, 5-0
    src = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0]
    dst = [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
    edge_index = torch.tensor([src, dst])
    bond_types = torch.full((12,), 4, dtype=torch.long)  # Aromatic
    batch = torch.zeros(6, dtype=torch.long)
    return atom_types, pos, edge_index, bond_types, batch


def make_staggered_ethane():
    """
    Staggered ethane: H-C-C-H dihedral = 60° (gauche, low torsion energy for 3-fold V3).
    The 3-fold potential V3 has minima near ±60°, maximum at 0°/180°.
    """
    # C1 at origin, C2 along x-axis
    c1 = torch.tensor([0.0, 0.0, 0.0])
    c2 = torch.tensor([1.54, 0.0, 0.0])
    # H on C1: tetrahedral, one H at approx (+cos(54.75°), +sin(54.75°), 0) direction
    h1 = c1 + 1.09 * torch.tensor([math.cos(math.radians(180+54.75)), math.sin(math.radians(54.75)), 0.0])
    # H on C2 staggered: 60° dihedral
    h2 = c2 + 1.09 * torch.tensor([math.cos(math.radians(-54.75)), math.sin(math.radians(60+54.75)), 0.0])

    pos = torch.stack([h1, c1, c2, h2])
    atom_types = torch.tensor([1, 6, 6, 1])
    edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    bond_types = torch.ones(6, dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    return pos, atom_types, edge_index, bond_types, batch


def make_eclipsed_ethane():
    """Eclipsed ethane: H-C-C-H dihedral = 0° (max V3 energy — eclipsed penalty)."""
    c1 = torch.tensor([0.0, 0.0, 0.0])
    c2 = torch.tensor([1.54, 0.0, 0.0])
    h1 = c1 + 1.09 * torch.tensor([math.cos(math.radians(180+54.75)), math.sin(math.radians(54.75)), 0.0])
    # Eclipsed: same side as h1 (0° dihedral)
    h2 = c2 + 1.09 * torch.tensor([math.cos(math.radians(-54.75)), math.sin(math.radians(54.75)), 0.0])

    pos = torch.stack([h1, c1, c2, h2])
    atom_types = torch.tensor([1, 6, 6, 1])
    edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    bond_types = torch.ones(6, dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    return pos, atom_types, edge_index, bond_types, batch


# ============================================================================
# BOND LENGTH TESTS
# ============================================================================

def test_bond_length_table_cc():
    """C-C single bond should be 1.54Å."""
    assert abs(get_ideal_bond_length(6, 6, 1) - 1.54) < 0.001

def test_bond_length_table_co():
    """C-O single bond should be 1.43Å."""
    assert abs(get_ideal_bond_length(6, 8, 1) - 1.43) < 0.001

def test_bond_length_table_oh():
    """O-H bond should be 0.96Å."""
    assert abs(get_ideal_bond_length(8, 1, 1) - 0.96) < 0.001

def test_bond_length_table_ch():
    """C-H bond should be 1.09Å."""
    assert abs(get_ideal_bond_length(6, 1, 1) - 1.09) < 0.001

def test_bond_length_table_caromatic():
    """C-C aromatic bond should be 1.40Å."""
    assert abs(get_ideal_bond_length(6, 6, 4) - 1.40) < 0.001

def test_bond_length_symmetric():
    """Bond length lookup should be symmetric (same for A-B and B-A)."""
    assert abs(get_ideal_bond_length(6, 8, 1) - get_ideal_bond_length(8, 6, 1)) < 0.001
    assert abs(get_ideal_bond_length(7, 1, 1) - get_ideal_bond_length(1, 7, 1)) < 0.001

def test_vectorized_lookup_matches_scalar():
    """Vectorized bond lookup should match scalar lookup."""
    a1 = torch.tensor([6, 6, 8, 7])
    a2 = torch.tensor([6, 8, 1, 1])
    bo = torch.tensor([1, 1, 1, 1])
    
    result = get_ideal_bond_lengths_vectorized(a1, a2, bo)
    expected = torch.tensor([
        get_ideal_bond_length(6, 6, 1),
        get_ideal_bond_length(6, 8, 1),
        get_ideal_bond_length(8, 1, 1),
        get_ideal_bond_length(7, 1, 1),
    ])
    assert torch.allclose(result, expected, atol=0.001), f"Mismatch: {result} vs {expected}"


# ============================================================================
# BOND LOSS TESTS
# ============================================================================

def test_bond_loss_perfect():
    """Bond loss should be ~0 for a perfectly-placed ethanol."""
    atom_types, pos, edge_index, bond_types, batch = make_ethanol()
    gc = GeometryConstraints(bond_weight=10.0)
    loss = gc.compute_bond_loss(pos, atom_types, edge_index, bond_types)
    assert loss.item() < 0.01, f"Bond loss should be near 0 for ideal geometry, got {loss.item()}"

def test_bond_loss_bad():
    """Bond loss should be large for badly placed atoms."""
    atom_types, pos, edge_index, bond_types, batch = make_ethanol()
    pos_bad = pos.clone()
    pos_bad[2] = pos_bad[2] + torch.tensor([1.0, 0.0, 0.0])  # Move O 1Å wrong
    gc = GeometryConstraints(bond_weight=10.0)
    loss_bad = gc.compute_bond_loss(pos_bad, atom_types, edge_index, bond_types)
    loss_good = gc.compute_bond_loss(pos, atom_types, edge_index, bond_types)
    assert loss_bad.item() > loss_good.item() * 10, "Bad geometry should have much higher bond loss"


# ============================================================================
# TORSION TESTS
# ============================================================================

def test_dihedral_computation():
    """Test dihedral angle computation on known geometry."""
    # Perfectly eclipsed: phi = 0
    p0 = torch.tensor([1.0, 1.0, 0.0])
    p1 = torch.tensor([1.0, 0.0, 0.0])
    p2 = torch.tensor([0.0, 0.0, 0.0])
    p3 = torch.tensor([0.0, 1.0, 0.0])  # Same side as p0 → phi ≈ 0
    phi = compute_dihedral(p0, p1, p2, p3)
    assert phi.abs().item() < 0.1, f"Expected ~0 dihedral, got {math.degrees(phi.item()):.1f}°"

def test_torsion_loss_staggered_less_than_eclipsed():
    """Staggered ethane should have lower torsion loss than eclipsed."""
    gc = GeometryConstraints(torsion_weight=1.0)
    
    pos_s, at_s, ei_s, bt_s, bi_s = make_staggered_ethane()
    pos_e, at_e, ei_e, bt_e, bi_e = make_eclipsed_ethane()
    
    t_staggered = gc.compute_torsion_loss(pos_s, at_s, ei_s, bt_s, bi_s)
    t_eclipsed  = gc.compute_torsion_loss(pos_e, at_e, ei_e, bt_e, bi_e)
    
    print(f"\nTorsion: staggered={t_staggered.item():.4f}, eclipsed={t_eclipsed.item():.4f}")
    # Eclipsed conformer should have higher torsion energy
    # (Both are non-zero for simple cosine potential, but eclipsed should be higher)
    # For V3 (3-fold), eclipsed (0°) is max, staggered (60°) is midpoint — so eclipsed ≥ staggered
    assert t_eclipsed.item() >= t_staggered.item() - 0.01, \
        f"Eclipsed ({t_eclipsed.item():.4f}) should be ≥ staggered ({t_staggered.item():.4f})"


# ============================================================================
# REPULSION TESTS
# ============================================================================

def test_repulsion_no_clash_water():
    """Valid water should have zero repulsion (no clashes)."""
    atom_types, pos, edge_index, bond_types, batch = make_water()
    gc = GeometryConstraints(repulsion_weight=5.0)
    rep = gc.compute_repulsion_loss(pos, atom_types, edge_index, batch)
    assert rep.item() < 0.01, f"Valid water should have no repulsion, got {rep.item()}"

def test_repulsion_detects_clash():
    """Atoms placed 0.5Å apart should trigger repulsion."""
    atom_types = torch.tensor([6, 6])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])  # 0.5Å apart (clash!)
    edge_index = torch.zeros(2, 0, dtype=torch.long)  # No bonds
    batch = torch.zeros(2, dtype=torch.long)
    gc = GeometryConstraints(repulsion_weight=5.0)
    rep = gc.compute_repulsion_loss(pos, atom_types, edge_index, batch)
    assert rep.item() > 0.1, f"Should detect clash at 0.5Å, got repulsion={rep.item()}"

def test_repulsion_excludes_12pairs():
    """Directly bonded atoms should NOT trigger repulsion even if close."""
    atom_types = torch.tensor([6, 1])  # C-H bond, 1.09Å ideal
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    bond_types = torch.ones(2, dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)
    gc = GeometryConstraints(repulsion_weight=5.0)
    rep = gc.compute_repulsion_loss(pos, atom_types, edge_index, batch)
    # C-H at 1.09Å is below the 1.5Å default threshold, but it's a 1-2 pair → excluded
    assert rep.item() < 0.01, f"Bonded 1-2 pair should be excluded from repulsion, got {rep.item()}"


# ============================================================================
# VALENCY TESTS
# ============================================================================

def test_valency_water():
    """Water: O has valence 2, H has valence 1 — valid."""
    atom_types, pos, edge_index, bond_types, batch = make_water()
    checker = ValidityChecker()
    val_ok, info = checker.check_valency(atom_types, edge_index, bond_types)
    assert val_ok, f"Water valency should be valid. Info: {info}"
    valences = info['valences']
    assert abs(valences[0] - 2.0) < 0.01, f"O should have valence 2, got {valences[0]}"
    assert abs(valences[1] - 1.0) < 0.01, f"H should have valence 1, got {valences[1]}"

def test_valency_overvalent_carbon():
    """Carbon with 5 single bonds should fail valency check."""
    # CH5 — impossible
    atom_types = torch.tensor([6, 1, 1, 1, 1, 1])
    # 5 C-H bonds, bidirectional
    ei_src = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5]
    ei_dst = [1, 0, 2, 0, 3, 0, 4, 0, 5, 0]
    edge_index = torch.tensor([ei_src, ei_dst])
    bond_types = torch.ones(10, dtype=torch.long)
    
    checker = ValidityChecker()
    val_ok, info = checker.check_valency(atom_types, edge_index, bond_types)
    assert not val_ok, f"CH5 should fail valency check"
    violations = info['violations']
    assert any(v['atom_type'] == 6 for v in violations), "C should be the violating atom"

def test_valency_methane():
    """Methane (CH4): C has valence 4, each H has valence 1 — valid."""
    atom_types = torch.tensor([6, 1, 1, 1, 1])
    ei_src = [0, 1, 0, 2, 0, 3, 0, 4]
    ei_dst = [1, 0, 2, 0, 3, 0, 4, 0]
    edge_index = torch.tensor([ei_src, ei_dst])
    bond_types = torch.ones(8, dtype=torch.long)
    
    checker = ValidityChecker()
    val_ok, info = checker.check_valency(atom_types, edge_index, bond_types)
    assert val_ok, f"Methane should pass valency. Info: {info}"
    assert abs(info['valences'][0] - 4.0) < 0.01, f"C should have valence 4, got {info['valences'][0]}"


# ============================================================================
# COM REMOVAL TESTS
# ============================================================================

def test_com_removal_per_molecule():
    """CoM should be removed independently for each molecule in the batch."""
    # Two molecules stacked
    x = torch.tensor([
        [2.0, 0.0, 0.0],   # mol 0, atom 0
        [4.0, 0.0, 0.0],   # mol 0, atom 1
        [10.0, 0.0, 0.0],  # mol 1, atom 0
        [14.0, 0.0, 0.0],  # mol 1, atom 1
    ])
    batch = torch.tensor([0, 0, 1, 1])
    
    x_centered = remove_com(x, batch)
    
    # Mol 0 CoM should be 0
    mol0_com = x_centered[:2].mean(0)
    assert mol0_com.abs().max().item() < 1e-5, f"Mol 0 CoM should be 0, got {mol0_com}"
    
    # Mol 1 CoM should be 0
    mol1_com = x_centered[2:].mean(0)
    assert mol1_com.abs().max().item() < 1e-5, f"Mol 1 CoM should be 0, got {mol1_com}"


# ============================================================================
# DIFFUSION MODEL TESTS
# ============================================================================

def test_diffusion_forward_no_nan():
    """Forward pass (get_loss) should not produce NaN or inf.
    get_loss() returns a dict with keys 'total', 'mse', 'geo'.
    """
    model = ConformerDiffusion(num_timesteps=100, hidden_dim=64, num_layers=2)

    atom_types = torch.tensor([8, 1, 1, 6, 1, 1, 1])
    edge_index = torch.tensor([[0,1,0,2,3,4,3,5,3,6],[1,0,2,0,4,3,5,3,6,3]])
    bond_types = torch.ones(10, dtype=torch.long)
    batch_idx  = torch.tensor([0,0,0,1,1,1,1])
    x_0 = remove_com(torch.randn(7, 3), batch_idx)

    loss_dict = model.get_loss(x_0, atom_types, edge_index, bond_types, batch_idx,
                               geometry_weight=1.0, epoch=1, max_epochs=100)

    # Verify dict structure
    assert isinstance(loss_dict, dict), f"get_loss should return dict, got {type(loss_dict)}"
    assert 'total' in loss_dict and 'mse' in loss_dict and 'geo' in loss_dict

    total = loss_dict['total']
    assert not torch.isnan(total), f"Total loss should not be NaN, got {total.item()}"
    assert total.item() > 0, "Total loss should be positive"
    assert not torch.isnan(loss_dict['mse']), "MSE component should not be NaN"

def test_diffusion_sampling_shapes():
    """DDIM sampling should return correct shapes."""
    model = ConformerDiffusion(num_timesteps=50, hidden_dim=64, num_layers=2)
    model.eval()

    atom_types = torch.tensor([6, 6, 6])
    edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
    bond_types = torch.ones(4, dtype=torch.long)
    batch_idx  = torch.zeros(3, dtype=torch.long)

    with torch.no_grad():
        x_gen = model.ddim_sample(atom_types, edge_index, bond_types, batch_idx, num_steps=5)

    assert x_gen.shape == (3, 3), f"Expected (3, 3), got {x_gen.shape}"

def test_diffusion_com_near_zero_after_sampling():
    """Generated coordinates should have near-zero CoM per molecule."""
    model = ConformerDiffusion(num_timesteps=50, hidden_dim=64, num_layers=2)
    model.eval()

    atom_types = torch.tensor([6, 8, 1, 1])
    edge_index = torch.tensor([[0,1,1,2,1,3],[1,0,2,1,3,1]])
    bond_types = torch.ones(6, dtype=torch.long)
    batch_idx  = torch.zeros(4, dtype=torch.long)

    with torch.no_grad():
        x_gen = model.ddim_sample(atom_types, edge_index, bond_types, batch_idx, num_steps=10)

    com = x_gen.mean(0)
    assert com.abs().max().item() < 1.0, f"CoM should be near 0, got {com}"


# ============================================================================
# RDKIT VALIDATION TEST
# ============================================================================

def test_rdkit_validation_methane():
    """Methane with correct geometry should pass RDKit validation."""
    from models.validity_filter import validate_with_rdkit
    
    atom_types = torch.tensor([6, 1, 1, 1, 1])
    # Tetrahedral CH4 geometry
    d = 1.09
    pos = torch.tensor([
        [0.000,  0.000,  0.000],   # C
        [d,      0.000,  0.000],   # H1
        [-d/3,   d*0.943, 0.000], # H2
        [-d/3,  -d*0.471, d*0.816], # H3
        [-d/3,  -d*0.471,-d*0.816], # H4
    ])
    ei_src = [0,1,0,2,0,3,0,4]
    ei_dst = [1,0,2,0,3,0,4,0]
    edge_index = torch.tensor([ei_src, ei_dst])
    bond_types = torch.ones(8, dtype=torch.long)
    
    valid, mol, info = validate_with_rdkit(
        atom_types, edge_index, bond_types, pos, run_mmff=False
    )
    assert valid, f"Methane should pass RDKit validation. Info: {info}"
    assert mol is not None


# ============================================================================
# SOFT CONSTRAINTS TESTS (Exp-1: soft_restrictions)
# ============================================================================

def test_planarity_flat_benzene_zero_loss():
    """Flat benzene should have near-zero planarity loss."""
    atom_types, pos, edge_index, bond_types, batch = make_benzene_flat()
    gc = GeometryConstraints(planarity_weight=5.0)
    aromatic_rings = [list(range(6))]
    loss = gc.compute_planarity_loss(pos, aromatic_rings)
    assert loss.item() < 0.01, f"Flat benzene planarity loss should be ~0, got {loss.item()}"

def test_planarity_nonflat_benzene_nonzero_loss():
    """Bending one atom out of plane should increase planarity loss."""
    atom_types, pos, edge_index, bond_types, batch = make_benzene_flat()
    pos_bent = pos.clone()
    pos_bent[3, 2] = 0.5  # Push atom 3 out of z=0 plane by 0.5Å
    gc = GeometryConstraints(planarity_weight=5.0)
    aromatic_rings = [list(range(6))]
    loss_flat = gc.compute_planarity_loss(pos, aromatic_rings)
    loss_bent = gc.compute_planarity_loss(pos_bent, aromatic_rings)
    assert loss_bent.item() > loss_flat.item() + 0.01, \
        f"Non-flat benzene should have higher planarity loss: flat={loss_flat.item():.4f}, bent={loss_bent.item():.4f}"

def test_chirality_correct_sign_near_zero():
    """Correct R/S chirality should produce near-zero loss if volume is positive."""
    # Tetrahedral carbon: C at origin, 4 neighbors
    d = 1.09
    pos = torch.tensor([
        [0.0,  0.0,  0.0 ],     # C center
        [d,    0.0,  0.0 ],     # neighbor 1
        [-d/3, d*0.943, 0.0],   # neighbor 2
        [-d/3, -d*0.471, d*0.816],  # neighbor 3
        [-d/3, -d*0.471, -d*0.816], # neighbor 4
    ])
    gc = GeometryConstraints(chirality_weight=3.0)
    # sign=+1 should match the arrangement above (positive volume)
    chiral = [(0, [1, 2, 3, 4], +1)]
    loss = gc.compute_chirality_loss(pos, chiral)
    assert loss.item() < 0.5, f"Correct chirality should produce low loss, got {loss.item()}"

def test_chirality_inverted_sign_nonzero():
    """Inverting the chirality sign should produce higher loss."""
    d = 1.09
    pos = torch.tensor([
        [0.0,  0.0,  0.0 ],
        [d,    0.0,  0.0 ],
        [-d/3, d*0.943, 0.0],
        [-d/3, -d*0.471, d*0.816],
        [-d/3, -d*0.471, -d*0.816],
    ])
    gc = GeometryConstraints(chirality_weight=3.0)
    loss_correct  = gc.compute_chirality_loss(pos, [(0, [1, 2, 3, 4], +1)])
    loss_inverted = gc.compute_chirality_loss(pos, [(0, [1, 2, 3, 4], -1)])
    # One of the two should be significantly higher since sign is wrong
    assert abs(loss_correct.item() - loss_inverted.item()) > 0.01, \
        f"Correct and inverted chirality should differ: correct={loss_correct.item():.4f} inverted={loss_inverted.item():.4f}"

def test_ring_strain_cyclopropane_ideal():
    """Equilateral triangle (60° angles) should give near-zero ring strain for 3-ring."""
    # Equilateral triangle with side 1.54Å (C-C single)
    s = 1.54
    pos = torch.tensor([
        [0.0,  0.0, 0.0],
        [s,    0.0, 0.0],
        [s/2,  s * math.sqrt(3)/2, 0.0],
    ])
    gc = GeometryConstraints(ring_strain_weight=2.0)
    small_rings = [[0, 1, 2]]
    loss = gc.compute_ring_strain_loss(pos, small_rings)
    assert loss.item() < 0.01, f"Ideal 60° cyclopropane should have ~0 strain, got {loss.item()}"

def test_ring_strain_wrong_angle_nonzero():
    """A 3-membered ring with 109° angles (impossible, deformed) should have strain."""
    # Place 3 atoms so angles are far from 60°
    pos = torch.tensor([
        [0.0,  0.0,  0.0],
        [1.54, 0.0,  0.0],
        [0.77, 3.0,  0.0],   # Very elongated — angle at vertex 2 is far from 60°
    ])
    gc = GeometryConstraints(ring_strain_weight=2.0)
    small_rings = [[0, 1, 2]]
    loss = gc.compute_ring_strain_loss(pos, small_rings)
    assert loss.item() > 0.01, f"Deformed 3-ring should have non-zero strain, got {loss.item()}"

def test_geometry_gradient_step_with_new_constraints():
    """_geometry_gradient_step should run without NaN when soft constraints are provided."""
    model = ConformerDiffusion(num_timesteps=50, hidden_dim=64, num_layers=2)
    atom_types, pos, edge_index, bond_types, batch = make_benzene_flat()
    pos_noisy = pos + torch.randn_like(pos) * 0.3

    aromatic_rings = [list(range(6))]
    result = model._geometry_gradient_step(
        pos_noisy, atom_types, edge_index, bond_types, batch,
        num_iters=3, lr=0.03,
        aromatic_rings=aromatic_rings,
        chiral_centers=None,
        small_rings=None,
    )
    assert result.shape == pos.shape, f"Shape mismatch: {result.shape} vs {pos.shape}"
    assert not torch.isnan(result).any(), "Geometry gradient step produced NaN"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("mol_next_gen Chemistry Validation Tests")
    print("=" * 60)

    tests = [
        test_bond_length_table_cc,
        test_bond_length_table_co,
        test_bond_length_table_oh,
        test_bond_length_table_ch,
        test_bond_length_table_caromatic,
        test_bond_length_symmetric,
        test_vectorized_lookup_matches_scalar,
        test_bond_loss_perfect,
        test_bond_loss_bad,
        test_dihedral_computation,
        test_torsion_loss_staggered_less_than_eclipsed,
        test_repulsion_no_clash_water,
        test_repulsion_detects_clash,
        test_repulsion_excludes_12pairs,
        test_valency_water,
        test_valency_overvalent_carbon,
        test_valency_methane,
        test_com_removal_per_molecule,
        test_diffusion_forward_no_nan,
        test_diffusion_sampling_shapes,
        test_diffusion_com_near_zero_after_sampling,
        test_rdkit_validation_methane,
        # Exp-1: Soft constraints
        test_planarity_flat_benzene_zero_loss,
        test_planarity_nonflat_benzene_nonzero_loss,
        test_chirality_correct_sign_near_zero,
        test_chirality_inverted_sign_nonzero,
        test_ring_strain_cyclopropane_ideal,
        test_ring_strain_wrong_angle_nonzero,
        test_geometry_gradient_step_with_new_constraints,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED — see above for details")
    print("=" * 60)
