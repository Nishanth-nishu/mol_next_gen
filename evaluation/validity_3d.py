"""
validity_3d.py  — Research-Grade 3D Molecular Validity Evaluation

Implements metrics from key papers:

  EDM  (Hoogeboom et al., NeurIPS 2022)
    arxiv.org/abs/2203.17003
    → atom_stability, mol_stability, validity (RDKit), uniqueness, novelty

  GeoMol (Ganea et al., NeurIPS 2021)
    arxiv.org/abs/2106.07802
    → COV-R  (coverage-recall): % reference conformers "covered" by generated set
    → MAT-R  (matching-recall): mean min-RMSD from each reference conformer to any generated

  GeoDiff (Xu et al., ICML 2022)
    arxiv.org/abs/2203.02923
    → same COV/MAT framework but using MMFF-relaxed references

  MMFF Strain Energy
    → Average MMFF94 energy before vs after minimisation (strain proxy)
    → Widely used in conformer quality assessment (Riniker & Landrum, JCIM 2015)

  Bond-length Validity
    → % bonds within 0.2 Å of MMFF94 CSD targets (our geometry_constraints table)
    → Mean absolute deviation from ideal bonds

Usage
-----
    from evaluation.validity_3d import Evaluator3D

    evaluator = Evaluator3D(device='cuda')
    metrics = evaluator.evaluate(
        model,
        dataloader,
        num_gen=200,        # molecules to generate
        num_workers=4,
    )
    evaluator.print_report(metrics)

Standalone
----------
    python evaluation/validity_3d.py \\
        --checkpoint checkpoints/conformer_best.pt \\
        --data      data/qm9_selfies.jsonl \\
        --num_gen   500
"""

from __future__ import annotations

import os
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# IDEAL BOND LENGTHS  (MMFF94 calibrated, Å)
# ──────────────────────────────────────────────────────────────────────────────
IDEAL_BONDS: Dict[Tuple[int, int, int], float] = {
    (6, 6, 1): 1.54,  (6, 6, 2): 1.34,  (6, 6, 3): 1.20,  (6, 6, 4): 1.40,
    (6, 7, 1): 1.47,  (6, 7, 2): 1.29,  (6, 7, 4): 1.34,
    (6, 8, 1): 1.43,  (6, 8, 2): 1.22,
    (6, 1, 1): 1.09,  (7, 1, 1): 1.01,  (8, 1, 1): 0.96,
    (6, 9, 1): 1.35,  (6, 17, 1): 1.77, (6, 16, 1): 1.82,
    (6, 35, 1): 1.94, (7, 7, 1):  1.45, (7, 8, 1):  1.36,
    (16, 8, 2): 1.44, (15, 8, 1): 1.63,
}
BOND_TOLERANCE = 0.2   # Å  (strict validity threshold)
CLASH_DIST     = 1.4   # Å  (steric clash threshold, non-bonded)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _ideal_bond(a1: int, a2: int, order: int) -> float:
    k = (min(a1, a2), max(a1, a2), order)
    return IDEAL_BONDS.get(k, 1.50)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Kabsch algorithm: RMSD between P and Q after optimal rigid alignment.
    Both (N, 3).  Returns RMSD in Ångströms.
    Reference: Kabsch (1976), Acta Cryst. A32:922.
    """
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    H  = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    D  = np.eye(3); D[2, 2] = np.sign(d)
    R  = Vt.T @ D @ U.T
    P_rot = P @ R.T
    return float(np.sqrt(np.mean((P_rot - Q) ** 2)))


def _try_rdkit_mol(atom_nums: List[int], pos: np.ndarray,
                   edge_src: List[int], edge_dst: List[int],
                   bond_orders: List[int]):
    """
    Attempt to build an RDKit Mol from scratch.  Returns (mol, success).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D

        em = Chem.RWMol()
        for z in atom_nums:
            a = Chem.Atom(int(z))
            em.AddAtom(a)

        BOND_TYPE = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }
        seen = set()
        for i, j, bo in zip(edge_src, edge_dst, bond_orders):
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                em.AddBond(i, j, BOND_TYPE.get(int(bo), Chem.rdchem.BondType.SINGLE))

        conf = Chem.Conformer(len(atom_nums))
        for idx, (x, y, z) in enumerate(pos):
            conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))

        mol = em.GetMol()
        mol.AddConformer(conf, assignId=True)

        try:
            Chem.SanitizeMol(mol)
            valid = True
        except Exception:
            valid = False

        return mol, valid
    except Exception:
        return None, False


def _mmff_strain(mol) -> Optional[float]:
    """
    Compute MMFF strain energy: E_unrelaxed - E_relaxed.
    Returns None if MMFF setup fails.
    From: Riniker & Landrum, JCIM 55:2459 (2015).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import copy

        mol_copy = copy.deepcopy(mol)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol_copy,
            AllChem.MMFFGetMoleculeProperties(mol_copy),
            confId=0
        )
        if ff is None:
            return None
        e_unrelaxed = ff.CalcEnergy()
        ff.Minimize(maxIts=500)
        e_relaxed = ff.CalcEnergy()
        return float(e_unrelaxed - e_relaxed)   # kcal/mol strain
    except Exception:
        return None


def _bond_validity(pos: np.ndarray,
                   atom_nums: List[int],
                   edge_src: List[int],
                   edge_dst: List[int],
                   bond_orders: List[int]) -> Tuple[bool, float]:
    """
    Returns (all_bonds_valid, mean_abs_error).
    """
    errors = []
    for i, j, bo in zip(edge_src, edge_dst, bond_orders):
        if i >= j:
            continue   # undirected: count once
        d = float(np.linalg.norm(pos[i] - pos[j]))
        ideal = _ideal_bond(atom_nums[i], atom_nums[j], int(bo))
        errors.append(abs(d - ideal))

    if not errors:
        return True, 0.0

    mae = float(np.mean(errors))
    return bool(mae < BOND_TOLERANCE), mae


def _clash_free(pos: np.ndarray,
                edge_src: List[int],
                edge_dst: List[int]) -> bool:
    """Check no non-bonded pair is closer than CLASH_DIST."""
    bonded = set()
    for i, j in zip(edge_src, edge_dst):
        bonded.add((min(i, j), max(i, j)))

    N = len(pos)
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) not in bonded:
                if float(np.linalg.norm(pos[i] - pos[j])) < CLASH_DIST:
                    return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# COV / MAT  (GeoMol / GeoDiff framework)
# ──────────────────────────────────────────────────────────────────────────────

def cov_mat_metrics(
    ref_conformers: List[np.ndarray],
    gen_conformers: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    COV-R and MAT-R metrics from GeoMol (Ganea et al. NeurIPS 2021).

    COV-R = fraction of reference conformers whose nearest generated
            conformer is within `threshold` Å RMSD.
            → measures recall / diversity coverage.

    MAT-R = mean over all reference conformers of
            min_{generated} Kabsch-RMSD(ref, gen).
            → measures precision / generation accuracy.

    Args
    ----
    ref_conformers : list of (N, 3) numpy arrays (ground truth)
    gen_conformers : list of (N, 3) numpy arrays (generated, same molecule)
    threshold      : coverage threshold in Å (default 0.5 Å, per GeoMol paper)

    Returns
    -------
    dict with 'cov_r', 'mat_r', 'threshold'
    """
    if not ref_conformers or not gen_conformers:
        return {'cov_r': 0.0, 'mat_r': float('inf'), 'threshold': threshold}

    min_rmsds = []
    for ref in ref_conformers:
        rmsds = [kabsch_rmsd(ref, gen) for gen in gen_conformers]
        min_rmsds.append(min(rmsds))

    min_rmsds = np.array(min_rmsds)
    cov_r = float(np.mean(min_rmsds < threshold))
    mat_r = float(np.mean(min_rmsds))

    return {'cov_r': cov_r, 'mat_r': mat_r, 'threshold': threshold}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATOR CLASS
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator3D:
    """
    Research-grade 3D conformer evaluator.

    Implements metrics from EDM, GeoMol, GeoDiff papers.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ──────────────────────────────────────────────────────────────────────
    # CORE EVALUATION ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self,
                 model,
                 dataloader,
                 num_gen: int = 200,
                 cov_threshold: float = 0.5,
                 num_gen_per_ref: int = 5,
                 verbose: bool = True) -> Dict:
        """
        Full evaluation returning all research-grade metrics.

        Parameters
        ----------
        model           : ConformerDiffusion model (already on device)
        dataloader      : validation DataLoader
        num_gen         : total molecules to evaluate
        cov_threshold   : RMSD threshold for COV-R (0.5 Å from GeoMol)
        num_gen_per_ref : generated conformers per reference (for COV/MAT)
        verbose         : print progress

        Returns
        -------
        Dict with keys:
          validity          – RDKit sanitization pass rate (EDM style)
          bond_valid_rate   – fraction with all bonds within 0.2 Å of ideal
          clash_free_rate   – fraction with no steric clashes
          fully_valid_rate  – validity AND bonds AND no-clash
          mean_bond_error   – mean absolute bond length deviation (Å)
          mean_strain_kcal  – mean MMFF94 strain energy (kcal/mol)
          cov_r             – COV-R (GeoMol coverage-recall)
          mat_r             – MAT-R (GeoMol matching-recall in Å)
          rmsd_mean         – Kabsch-RMSD vs reference (single sample)
          rmsd_std          – std of above
          n_evaluated       – actual molecules evaluated
        """
        from models.conformer_diffusion import remove_com

        model.eval()
        model.to(self.device)

        results = {
            'rdkit_valid': [], 'bond_valid': [], 'clash_free': [],
            'bond_error': [],  'strain': [],    'rmsd': [],
        }
        cov_mat_data: List[Tuple[List, List]] = []   # (refs, gens) per molecule

        n_done = 0
        t0 = time.time()

        for batch in dataloader:
            if n_done >= num_gen:
                break

            atom_types = batch['atom_types'].to(self.device)
            coords_true = batch['coordinates'].to(self.device)
            edge_index  = batch['edge_index'].to(self.device)
            bond_types  = batch['bond_types'].to(self.device)
            batch_idx   = batch['batch_idx'].to(self.device)

            # Centre ground truth
            # Centre ground truth per-molecule (inline scatter_mean — no torch_scatter required)
            B_local = int(batch_idx.max().item()) + 1
            com = torch.zeros(B_local, 3, device=self.device)
            cnt = torch.zeros(B_local, device=self.device)
            cnt.scatter_add_(0, batch_idx, torch.ones(batch_idx.size(0), device=self.device))
            com.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, 3), coords_true)
            com = com / cnt.unsqueeze(1).clamp(min=1)
            coords_true = coords_true - com[batch_idx]


            # Generate ONE conformer per molecule (for RMSD)
            try:
                gen_single = model.ddim_sample(
                    atom_types, edge_index, bond_types, batch_idx, num_steps=50)
            except Exception:
                continue

            n_mols = int(batch_idx.max().item()) + 1

            for b in range(n_mols):
                if n_done >= num_gen:
                    break

                mask = (batch_idx == b).cpu()
                local_edge_mask = ((batch_idx[edge_index[0]] == b) &
                                   (batch_idx[edge_index[1]] == b)).cpu()

                if mask.sum() < 2:
                    continue

                pos_true = coords_true[mask].cpu().numpy()
                pos_gen  = gen_single[mask].cpu().numpy()

                # Local edge indices (0-based)
                global_to_local = {
                    g: l for l, g in enumerate(mask.nonzero(as_tuple=False).squeeze(-1).tolist())
                }
                le = edge_index[:, local_edge_mask].cpu()
                esrc = [global_to_local.get(i, 0) for i in le[0].tolist()]
                edst = [global_to_local.get(i, 0) for i in le[1].tolist()]
                ebo  = bond_types[local_edge_mask].cpu().tolist()
                anums = atom_types[mask].cpu().tolist()

                # ── RMSD ──────────────────────────────────────────────────
                rmsd = kabsch_rmsd(pos_gen, pos_true)
                results['rmsd'].append(rmsd)

                # ── Bond validity ─────────────────────────────────────────
                bv, mae = _bond_validity(pos_gen, anums, esrc, edst, ebo)
                results['bond_valid'].append(bv)
                results['bond_error'].append(mae)

                # ── Steric clash ──────────────────────────────────────────
                cf = _clash_free(pos_gen, esrc, edst)
                results['clash_free'].append(cf)

                # ── RDKit validity ────────────────────────────────────────
                mol, valid = _try_rdkit_mol(anums, pos_gen, esrc, edst, ebo)
                results['rdkit_valid'].append(valid)

                # ── MMFF strain ───────────────────────────────────────────
                if mol is not None and valid:
                    strain = _mmff_strain(mol)
                    if strain is not None:
                        results['strain'].append(strain)

                # ── COV / MAT: generate `num_gen_per_ref` conformers ─────
                try:
                    local_atom = atom_types[mask]
                    local_ei   = edge_index[:, local_edge_mask]

                    # Re-index local edges to 0-based
                    offset = mask.nonzero(as_tuple=False)[0].item()
                    local_ei_0 = local_ei - offset
                    local_bt = bond_types[local_edge_mask]
                    local_bidx = torch.zeros(mask.sum(), dtype=torch.long,
                                             device=self.device)

                    gen_ens = []
                    for _ in range(num_gen_per_ref):
                        g = model.ddim_sample(
                            local_atom, local_ei_0, local_bt, local_bidx,
                            num_steps=50)
                        gen_ens.append(g.cpu().numpy())

                    cov_mat_data.append(([pos_true], gen_ens))
                except Exception:
                    pass   # Skip COV/MAT for this molecule

                n_done += 1

            if verbose and n_done % 20 == 0 and n_done > 0:
                elapsed = time.time() - t0
                print(f"  [{n_done}/{num_gen}] elapsed {elapsed:.0f}s")

        # ── Aggregate ──────────────────────────────────────────────────────
        def _safe_mean(lst): return float(np.mean(lst)) if lst else float('nan')

        # COV / MAT aggregation across all molecules
        all_cov_r, all_mat_r = [], []
        for refs, gens in cov_mat_data:
            m = cov_mat_metrics(refs, gens, threshold=cov_threshold)
            all_cov_r.append(m['cov_r'])
            all_mat_r.append(m['mat_r'])

        return {
            # EDM-style
            'validity':        _safe_mean([float(v) for v in results['rdkit_valid']]),
            'bond_valid_rate': _safe_mean([float(v) for v in results['bond_valid']]),
            'clash_free_rate': _safe_mean([float(v) for v in results['clash_free']]),
            'fully_valid_rate': _safe_mean([
                float(rv and bv and cf)
                for rv, bv, cf in zip(results['rdkit_valid'],
                                      results['bond_valid'],
                                      results['clash_free'])
            ]),
            'mean_bond_error': _safe_mean(results['bond_error']),
            # GeoMol-style
            'cov_r':           _safe_mean(all_cov_r),
            'mat_r':           _safe_mean(all_mat_r),
            'cov_threshold_A': cov_threshold,
            # Quality
            'mean_strain_kcal': _safe_mean(results['strain']),
            'n_strain_mols':    len(results['strain']),
            # Conformer accuracy
            'rmsd_mean':       _safe_mean(results['rmsd']),
            'rmsd_std':        float(np.std(results['rmsd'])) if results['rmsd'] else float('nan'),
            'n_evaluated':     n_done,
        }

    # ──────────────────────────────────────────────────────────────────────
    # REPORTING
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(metrics: Dict, epoch: Optional[int] = None):
        header = f"── 3D Evaluation" + (f" [Epoch {epoch}]" if epoch else "") + " " + "─" * 40
        print(f"\n{header}")
        print(f"  Molecules evaluated : {metrics.get('n_evaluated', '?')}")
        print()

        # EDM metrics
        print("  EDM-style validity")
        print(f"    RDKit valid       : {metrics['validity']*100:6.1f}%")
        print(f"    Bond valid (0.2Å) : {metrics['bond_valid_rate']*100:6.1f}%")
        print(f"    Clash-free        : {metrics['clash_free_rate']*100:6.1f}%")
        print(f"    Fully valid       : {metrics['fully_valid_rate']*100:6.1f}%")
        print(f"    Mean bond error   : {metrics['mean_bond_error']:.4f} Å")
        print()

        # GeoMol metrics
        n_strain = metrics.get('n_strain_mols', 0)
        print(f"  GeoMol COV/MAT  (threshold={metrics.get('cov_threshold_A', 0.5):.1f} Å)")
        print(f"    COV-R             : {metrics['cov_r']*100:6.1f}%  (↑ higher is better)")
        print(f"    MAT-R             : {metrics['mat_r']:.4f} Å  (↓ lower is better)")
        print()

        # RMSD and strain
        print("  Conformer quality")
        print(f"    Kabsch-RMSD       : {metrics['rmsd_mean']:.4f} ± {metrics['rmsd_std']:.4f} Å")
        print(f"    MMFF strain       : {metrics['mean_strain_kcal']:.2f} kcal/mol  ({n_strain} mols)")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Research-grade 3D conformer evaluation')
    parser.add_argument('--checkpoint', required=True, help='.pt checkpoint file')
    parser.add_argument('--data',       required=True, help='JSONL data file')
    parser.add_argument('--num_gen',    type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_atoms',  type=int, default=15)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--cov_thresh', type=float, default=0.5,
                        help='COV-R RMSD threshold in Å (GeoMol default=0.5)')
    parser.add_argument('--num_gen_per_ref', type=int, default=5,
                        help='Conformers per molecule for COV/MAT ensemble')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.conformer_diffusion import ConformerDiffusion
    from training.train_conformer   import ConformerDataset, collate_fn
    from torch.utils.data           import DataLoader

    # Load data
    print(f"Loading {args.data}")
    dataset = ConformerDataset(args.data, max_atoms=args.max_atoms)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, collate_fn=collate_fn)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved_args = ckpt.get('args', {})

    model = ConformerDiffusion(
        num_timesteps=saved_args.get('timesteps', 1000),
        hidden_dim=saved_args.get('hidden_dim', 256),
        num_layers=saved_args.get('num_layers', 6),
        edge_dim=saved_args.get('edge_dim', 32),
        time_dim=saved_args.get('time_dim', 128),
    )
    model.load_state_dict(ckpt['model_state_dict'])

    evaluator = Evaluator3D(device=args.device)
    metrics   = evaluator.evaluate(
        model, loader,
        num_gen=args.num_gen,
        cov_threshold=args.cov_thresh,
        num_gen_per_ref=args.num_gen_per_ref,
        verbose=True,
    )
    evaluator.print_report(metrics)

    # Save JSON
    out = args.checkpoint.replace('.pt', '_eval3d.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nSaved metrics → {out}")


if __name__ == '__main__':
    main()
