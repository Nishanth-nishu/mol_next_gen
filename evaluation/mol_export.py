"""
mol_export.py — High-Quality Molecular Structure Export

Supports:
  - PDB  (HETATM records + CONECT for non-standard ligands, VMD/PyMOL compatible)
  - MOL2 (SYBYL atom types, AutoDock/MOE/Schrödinger compatible)
  - SDF  (MDL V2000, universal) — with RDKit enhancement when available

Research references:
  - SYBYL atom types: Tripos MOL2 format specification
  - PDB HETATM: PDB Exchange Dictionary v5
  - MDL Molfile: BIOVIA/Dassault MDL V2000 format specification
"""

import os
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import torch

# ─────────────────────────────────────────────────────────────────────────────
# ATOM TYPE TABLES
# ─────────────────────────────────────────────────────────────────────────────

ELEMENT_SYMBOLS = {
    1: 'H',  5: 'B',  6: 'C',  7: 'N',  8: 'O',  9: 'F',
    14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
}

# SYBYL atom type heuristics (element + hybridisation from bond types)
# bond_type codes: 1=single, 2=double, 3=triple, 4=aromatic
def _sybyl_type(atomic_num: int, neighbor_bond_types: List[int]) -> str:
    """Heuristic SYBYL atom type from atomic number and bond context."""
    el = ELEMENT_SYMBOLS.get(atomic_num, 'Du')
    has_double   = 2 in neighbor_bond_types
    has_triple   = 3 in neighbor_bond_types
    has_aromatic = 4 in neighbor_bond_types
    n_bonds      = len(neighbor_bond_types)

    if el == 'C':
        if has_triple:   return 'C.1'   # sp carbon
        if has_aromatic: return 'C.ar'  # aromatic
        if has_double:   return 'C.2'   # sp2
        return 'C.3'                    # sp3
    if el == 'N':
        if has_triple:   return 'N.1'
        if has_aromatic: return 'N.ar'
        if has_double:   return 'N.2'
        if n_bonds == 4: return 'N.4'   # quaternary
        return 'N.3'
    if el == 'O':
        if has_double:   return 'O.2'
        if has_aromatic: return 'O.ar'
        return 'O.3'
    if el == 'S':
        if has_double:   return 'S.2'
        if n_bonds == 4: return 'S.O'   # sulphoxide
        if n_bonds == 6: return 'S.O2'  # sulphone
        return 'S.3'
    if el == 'P':
        return 'P.3'
    if el == 'H':
        return 'H'
    # Halogens and residuals
    return el


# ─────────────────────────────────────────────────────────────────────────────
# PDB WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_pdb(
    pos: torch.Tensor,        # (N, 3) Ångström
    atom_types: torch.Tensor, # (N,)  atomic numbers
    edge_index: torch.Tensor, # (2, E) bond connectivity
    out_path: str,
    mol_name: str = 'LIG',
) -> None:
    """
    Write a molecule to PDB format.

    Uses HETATM records (correct for small-molecule ligands, not standard residues).
    Writes CONECT records so VMD/PyMOL build bonds correctly even for unusual elements.
    """
    N = pos.size(0)
    pos_np = pos.detach().cpu().numpy()
    types_np = atom_types.detach().cpu().numpy()

    lines = []
    lines.append(f"REMARK   mol_next_gen generated conformer: {mol_name}")
    lines.append(f"REMARK   {N} atoms")

    atom_serial = {}  # atom_idx -> serial (1-indexed)
    serial = 1
    for i in range(N):
        el = ELEMENT_SYMBOLS.get(int(types_np[i]), 'X')
        # PDB column positions (fixed-width):
        #   1-6   Record name   "HETATM"
        #   7-11  Serial        right-justified int
        #   13-16 Atom name     left-justified within field
        #   17    Alternate loc " "
        #   18-20 Res name      "LIG"
        #   22    Chain ID      "A"
        #   23-26 Res seq       1
        #   31-38 X (8.3f)
        #   39-46 Y (8.3f)
        #   47-54 Z (8.3f)
        #   55-60 Occupancy     1.00
        #   61-66 Temp factor   0.00
        #   77-78 Element symbol
        name_field = f"{el}{serial}".ljust(4)[:4]
        line = (
            f"HETATM{serial:5d} {name_field} {mol_name:3s} A   1    "
            f"{pos_np[i, 0]:8.3f}{pos_np[i, 1]:8.3f}{pos_np[i, 2]:8.3f}"
            f"  1.00  0.00          {el:>2s}  "
        )
        lines.append(line)
        atom_serial[i] = serial
        serial += 1

    # CONECT records (one per atom, lists all bonded atoms)
    # PDB spec: up to 4 bonded atoms per CONECT line
    from collections import defaultdict
    adj: Dict[int, List[int]] = defaultdict(list)
    ei = edge_index.detach().cpu().numpy()
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if j not in adj[i]:
            adj[i].append(j)

    for i in range(N):
        bonded = adj[i]
        # Write in chunks of 4
        for chunk_start in range(0, max(1, len(bonded)), 4):
            chunk = bonded[chunk_start:chunk_start + 4]
            serials = [f"{atom_serial[b]:5d}" for b in chunk]
            lines.append(f"CONECT{atom_serial[i]:5d}" + "".join(serials))

    lines.append("END")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MOL2 WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_mol2(
    pos: torch.Tensor,        # (N, 3)
    atom_types: torch.Tensor, # (N,)  atomic numbers
    edge_index: torch.Tensor, # (2, E)
    bond_types: torch.Tensor, # (E,)  bond orders (1,2,3,4)
    out_path: str,
    mol_name: str = 'LIG',
) -> None:
    """
    Write a molecule to Tripos MOL2 format.

    MOL2 is required by:
      - AutoDock Vina (docking)
      - Schrödinger Glide (docking)
      - MOE (molecular modelling)
      - UCSF Chimera / ChimeraX

    Uses SYBYL atom type heuristics for maximum software compatibility.
    """
    N = pos.size(0)
    pos_np = pos.detach().cpu().numpy()
    types_np = atom_types.detach().cpu().numpy()
    ei = edge_index.detach().cpu().numpy()
    bt = bond_types.detach().cpu().numpy()

    # Build per-atom bond context for SYBYL type inference
    from collections import defaultdict
    atom_bond_types: Dict[int, List[int]] = defaultdict(list)
    seen_edges = set()
    bond_list = []  # (i, j, bo) unique undirected bonds
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        bo = int(bt[k])
        atom_bond_types[i].append(bo)
        key = (min(i, j), max(i, j))
        if key not in seen_edges:
            seen_edges.add(key)
            bond_list.append((i, j, bo))

    # Derive SYBYL atom types
    sybyl_types = []
    for i in range(N):
        sybyl_types.append(_sybyl_type(int(types_np[i]), atom_bond_types[i]))

    n_bonds = len(bond_list)

    lines = []

    # @<TRIPOS>MOLECULE
    lines.append("@<TRIPOS>MOLECULE")
    lines.append(mol_name)
    lines.append(f"{N} {n_bonds} 0 0 0")
    lines.append("SMALL")
    lines.append("NO_CHARGES")
    lines.append("")

    # @<TRIPOS>ATOM
    # Fields: atom_id  atom_name  x  y  z  atom_type  subst_id  subst_name  charge
    lines.append("@<TRIPOS>ATOM")
    for i in range(N):
        el = ELEMENT_SYMBOLS.get(int(types_np[i]), 'Du')
        atom_name = f"{el}{i+1}"
        lines.append(
            f"{i+1:6d}  {atom_name:<6s}  "
            f"{pos_np[i, 0]:10.4f}  {pos_np[i, 1]:10.4f}  {pos_np[i, 2]:10.4f}  "
            f"{sybyl_types[i]:<8s}  1  {mol_name:<6s}  0.0000"
        )

    # @<TRIPOS>BOND
    # bond_type in MOL2: 1, 2, 3, ar (aromatic), am (amide), etc.
    _BO_TO_MOL2 = {1: '1', 2: '2', 3: '3', 4: 'ar'}
    lines.append("@<TRIPOS>BOND")
    for bid, (i, j, bo) in enumerate(bond_list, start=1):
        mol2_bt = _BO_TO_MOL2.get(bo, '1')
        lines.append(f"{bid:6d}  {i+1:5d}  {j+1:5d}  {mol2_bt}")

    lines.append("")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SDF WRITER (Pure Python — MDL V2000)
# ─────────────────────────────────────────────────────────────────────────────

def write_sdf(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    edge_index: torch.Tensor,
    bond_types: torch.Tensor,
    out_path: str,
    mol_name: str = 'LIG',
    append: bool = False,
) -> None:
    """
    Write a molecule to SDF (MDL V2000) format.

    This is a pure-Python fallback that does NOT require RDKit.
    For best results (valence checking, charge assignment) use RDKit when available.
    """
    N = pos.size(0)
    pos_np = pos.detach().cpu().numpy()
    types_np = atom_types.detach().cpu().numpy()
    ei = edge_index.detach().cpu().numpy()
    bt = bond_types.detach().cpu().numpy()

    seen_edges = set()
    bond_list = []
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        bo = int(bt[k])
        key = (min(i, j), max(i, j))
        if key not in seen_edges:
            seen_edges.add(key)
            bond_list.append((i, j, bo))

    n_bonds = len(bond_list)

    lines = []
    # Header block (3 lines)
    lines.append(mol_name)
    lines.append("  mol_next_gen 0D")
    lines.append("")
    # Counts line: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    lines.append(f"{N:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000")
    # Atom block
    for i in range(N):
        el = ELEMENT_SYMBOLS.get(int(types_np[i]), 'C')
        lines.append(
            f"{pos_np[i, 0]:10.4f}{pos_np[i, 1]:10.4f}{pos_np[i, 2]:10.4f} "
            f"{el:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    # Bond block: 111222tttsssxxxrrrccc  (1-indexed)
    _CLIP_BO = {1: 1, 2: 2, 3: 3, 4: 4}  # 4 = aromatic in V2000
    for i, j, bo in bond_list:
        mol_bo = _CLIP_BO.get(bo, 1)
        lines.append(f"{i+1:3d}{j+1:3d}{mol_bo:3d}  0  0  0  0")
    lines.append("M  END")
    lines.append("$$$$")

    mode = 'a' if append else 'w'
    with open(out_path, mode) as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# RDKIT-ENHANCED SDF (preferred when available)
# ─────────────────────────────────────────────────────────────────────────────

def write_sdf_rdkit(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    edge_index: torch.Tensor,
    bond_types: torch.Tensor,
    out_path: str,
    mol_name: str = 'LIG',
    append: bool = False,
) -> bool:
    """
    Write SDF using RDKit if available (better valence/sanitization).
    Returns True on success, False if RDKit unavailable or mol fails sanity check.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDetermineBonds
    except ImportError:
        return False

    N = pos.size(0)
    pos_np = pos.detach().cpu().numpy()
    types_np = atom_types.detach().cpu().numpy()
    ei = edge_index.detach().cpu().numpy()
    bt = bond_types.detach().cpu().numpy()

    try:
        em = Chem.RWMol()
        for i in range(N):
            at = Chem.Atom(int(types_np[i]))
            em.AddAtom(at)

        seen_edges = set()
        _BO_MAP = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
        }
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            key = (min(i, j), max(i, j))
            if key not in seen_edges:
                seen_edges.add(key)
                bo = int(bt[k])
                em.AddBond(i, j, _BO_MAP.get(bo, Chem.BondType.SINGLE))

        conf = Chem.Conformer(N)
        for i in range(N):
            conf.SetAtomPosition(i, pos_np[i].tolist())
        em.AddConformer(conf, assignId=True)

        mol = em.GetMol()
        Chem.SanitizeMol(mol)
        mol.SetProp('_Name', mol_name)
        mol.SetProp('SMILES', Chem.MolToSmiles(mol))

        mode = 'a' if append else 'w'
        writer = Chem.SDWriter(out_path) if not append else Chem.SDWriter.__new__(Chem.SDWriter)
        if append:
            writer = open(out_path, 'a')
            block = Chem.MolToMolBlock(mol)
            writer.write(block + "$$$$\n")
            writer.close()
        else:
            writer = Chem.SDWriter(out_path)
            writer.write(mol)
            writer.close()
        return True

    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EXPORT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def export_molecule(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    edge_index: torch.Tensor,
    bond_types: torch.Tensor,
    out_dir: str,
    mol_idx: int,
    formats: List[str] = ('sdf', 'pdb', 'mol2'),
    mol_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Export a single molecule in the requested formats.

    Args:
        formats: List of formats: any subset of ['sdf', 'pdb', 'mol2']
        mol_idx: Integer index for filename
        mol_name: HETATM/MOL2 residue name (max 3 chars, default 'LIG')

    Returns:
        Dict mapping format -> written file path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (mol_name or 'LIG')[:3].upper()
    written = {}

    if 'pdb' in formats:
        pdb_path = str(out_dir / f"mol_{mol_idx:04d}.pdb")
        write_pdb(pos, atom_types, edge_index, pdb_path, mol_name=name)
        written['pdb'] = pdb_path

    if 'mol2' in formats:
        mol2_path = str(out_dir / f"mol_{mol_idx:04d}.mol2")
        write_mol2(pos, atom_types, edge_index, bond_types, mol2_path, mol_name=name)
        written['mol2'] = mol2_path

    if 'sdf' in formats:
        sdf_path = str(out_dir / f"mol_{mol_idx:04d}.sdf")
        ok = write_sdf_rdkit(pos, atom_types, edge_index, bond_types, sdf_path, mol_name=name)
        if not ok:
            write_sdf(pos, atom_types, edge_index, bond_types, sdf_path, mol_name=name)
        written['sdf'] = sdf_path

    return written


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, os

    print("Testing mol_export.py…")

    # Ethanol: C(0)-C(1)-O(2) with one H on oxygen (4)
    #          C(0) has 3 H's implicitly — we represent heavy atoms only
    pos = torch.tensor([
        [0.000, 0.000, 0.000],
        [1.540, 0.000, 0.000],
        [2.000, 1.200, 0.000],
    ])
    atom_types  = torch.tensor([6, 6, 8])      # C, C, O
    edge_index  = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    bond_types  = torch.tensor([1, 1, 1, 1])   # all single

    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_molecule(pos, atom_types, edge_index, bond_types,
                                 out_dir=tmpdir, mol_idx=1,
                                 formats=['pdb', 'mol2', 'sdf'])
        for fmt, path in result.items():
            assert os.path.exists(path), f"Missing {path}"
            size = os.path.getsize(path)
            assert size > 0, f"Empty file: {path}"
            print(f"  {fmt.upper()} ✓  ({size} bytes)  → {path}")

    # Test benzene (aromatic bonds)
    benz_pos = torch.tensor([
        [1.4, 0.0, 0.0], [0.7, 1.21, 0.0], [-0.7, 1.21, 0.0],
        [-1.4, 0.0, 0.0], [-0.7, -1.21, 0.0], [0.7, -1.21, 0.0],
    ])
    benz_at = torch.tensor([6, 6, 6, 6, 6, 6])
    benz_ei = torch.tensor([[0,1,2,3,4,5,1,2,3,4,5,0],[1,2,3,4,5,0,0,1,2,3,4,5]])
    benz_bt = torch.tensor([4,4,4,4,4,4,4,4,4,4,4,4])  # aromatic

    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_molecule(benz_pos, benz_at, benz_ei, benz_bt,
                                 out_dir=tmpdir, mol_idx=1,
                                 formats=['mol2', 'sdf'])
        for fmt, path in result.items():
            assert os.path.exists(path), f"Missing {path}"
            content = open(path).read()
            if fmt == 'mol2':
                assert 'C.ar' in content, "Benzene should have C.ar in MOL2"
            print(f"  Benzene {fmt.upper()} ✓")

    print("\nAll mol_export tests passed!")
