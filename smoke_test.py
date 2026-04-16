"""
smoke_test.py — End-to-end pipeline smoke test.

Verifies:
  1. Data loads correctly (heavy-atom only, no H)
  2. Collate_fn batches correctly
  3. Forward pass produces finite loss
  4. Backward pass produces finite gradients
  5. DDIM sample produces non-NaN coordinates
  6. Repulsion loss actually fires (was always skipped before)
  7. Geometry loss is active at all timesteps
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import tempfile
from torch.utils.data import DataLoader

from models.conformer_diffusion import ConformerDiffusion, remove_com
from models.geometry_constraints import GeometryConstraints
from training.train_v3 import ConformerDataset, collate_fn, _build_graph_from_smiles

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

errors = []


def check(name, condition, msg=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}  {msg}")
        errors.append(name)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Build a tiny synthetic JSONL dataset with known molecules
# ──────────────────────────────────────────────────────────────────────────────
print("\n[1] Building synthetic dataset…")

# Ethanol heavy atoms: C,C,O  (no H)
# SMILES: CCO
molecules = [
    {
        "smiles":     "CCO",
        "atom_types": [6, 6, 8],
        "coordinates": [
            [0.000,  0.000, 0.000],
            [1.540,  0.000, 0.000],
            [2.000,  1.200, 0.000],
        ],
        "num_atoms": 3,
    },
    {
        "smiles":     "CC",
        "atom_types": [6, 6],
        "coordinates": [
            [0.000, 0.000, 0.000],
            [1.540, 0.000, 0.000],
        ],
        "num_atoms": 2,
    },
    {
        "smiles":     "CN",
        "atom_types": [6, 7],
        "coordinates": [
            [0.000, 0.000, 0.000],
            [1.470, 0.000, 0.000],
        ],
        "num_atoms": 2,
    },
]

# Replicate to get a reasonable dataset size
molecules = molecules * 20

with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tf:
    for m in molecules:
        tf.write(json.dumps(m) + '\n')
    tmp_path = tf.name

# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset loading checks
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2] Dataset loading…")
dataset = ConformerDataset(tmp_path, max_atoms=50)
check("Dataset non-empty",      len(dataset) > 0, f"got {len(dataset)}")
check("Dataset loaded all mols", len(dataset) == len(molecules), f"got {len(dataset)}, expected {len(molecules)}")

sample = dataset[0]
check("atom_types is tensor",    isinstance(sample['atom_types'], torch.Tensor))
check("coordinates is tensor",   isinstance(sample['coordinates'], torch.Tensor))
check("edge_index is tensor",    isinstance(sample['edge_index'], torch.Tensor))
check("bond_types is tensor",    isinstance(sample['bond_types'], torch.Tensor))
check("No H atoms in ethanol",   (sample['atom_types'] != 1).all().item(),
      f"found H: {sample['atom_types'].tolist()}")
check("Atom/coord count match",  sample['atom_types'].size(0) == sample['coordinates'].size(0),
      f"{sample['atom_types'].size(0)} vs {sample['coordinates'].size(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Collate / DataLoader
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3] DataLoader / collate_fn…")
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
batch  = next(iter(loader))

check("batch has atom_types",    'atom_types'  in batch)
check("batch has batch_idx",     'batch_idx'   in batch)
check("batch has edge_index",    'edge_index'  in batch)
check("batch_idx within range",  batch['batch_idx'].max().item() < 4)
N = batch['atom_types'].size(0)
E = batch['edge_index'].size(1)
check("edge_index within N",     batch['edge_index'].max().item() < N,
      f"max edge {batch['edge_index'].max().item()} vs N={N}")
check("No H in batch",           (batch['atom_types'] != 1).all().item(),
      f"H atoms found: {(batch['atom_types'] == 1).sum().item()}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Model forward + loss
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4] Model forward pass &  loss…")
model = ConformerDiffusion(
    num_timesteps=100,
    hidden_dim=64,
    num_layers=3,
    num_rbf=20,
    time_dim=64,
).to(DEVICE)

at  = batch['atom_types'].to(DEVICE)
ei  = batch['edge_index'].to(DEVICE)
bt  = batch['bond_types'].to(DEVICE)
bi  = batch['batch_idx'].to(DEVICE)
x0  = batch['coordinates'].to(DEVICE)
x0  = remove_com(x0, bi)

loss_dict = model.get_loss(x0, at, ei, bt, bi, geometry_weight=1.0)

check("total loss is finite",    torch.isfinite(loss_dict['total']).item(),
      f"total={loss_dict['total'].item():.4f}")
check("mse loss is finite",      torch.isfinite(loss_dict['mse']).item())
check("geo loss is finite",      torch.isfinite(loss_dict['geo']).item())
check("geo loss > 0",            loss_dict['geo'].item() > 0,
      f"geo={loss_dict['geo'].item():.6f}  (was always 0 before fix)")

print(f"     total={loss_dict['total'].item():.4f}  "
      f"mse={loss_dict['mse'].item():.4f}  "
      f"geo={loss_dict['geo'].item():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Backward pass / gradient check
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5] Backward pass…")
loss_dict['total'].backward()

grad_norms = [
    p.grad.norm().item()
    for p in model.parameters()
    if p.grad is not None
]
has_grads   = len(grad_norms) > 0
has_nan_grad = any(not torch.isfinite(torch.tensor(g)) for g in grad_norms)
check("Gradients computed",      has_grads, f"grads on {len(grad_norms)} params")
check("No NaN gradients",        not has_nan_grad,
      f"NaN in {sum(not torch.isfinite(torch.tensor(g)) for g in grad_norms)} param grads")
print(f"     max_grad={max(grad_norms):.4f}  mean_grad={sum(grad_norms)/len(grad_norms):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. DDIM sampling — no NaN
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6] DDIM sampling (no NaN test)…")
model.eval()
with torch.no_grad():
    x_gen = model.ddim_sample(at, ei, bt, bi, num_steps=10)

check("Generated correct shape",  x_gen.shape == x0.shape, f"{x_gen.shape} vs {x0.shape}")
check("No NaN in generated coords", not torch.isnan(x_gen).any().item())
check("No Inf in generated coords", not torch.isinf(x_gen).any().item())

# CoM check per molecule
B = int(bi.max().item()) + 1
for b_idx in range(B):
    mask = bi == b_idx
    com  = x_gen[mask].mean(0)
    check(f"  CoM ~0 for mol {b_idx}",  com.abs().max().item() < 0.5,
          f"CoM={com.tolist()}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Repulsion loss fires on small molecules (per-mol fix check)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[7] Repulsion loss per-molecule (was always skipped before)…")
gc = GeometryConstraints(repulsion_weight=5.0)

# Artificially place two C atoms at 0.5 Å — guaranteed clash
clash_pos   = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], device=DEVICE)
clash_at    = torch.tensor([6, 6], dtype=torch.long, device=DEVICE)
clash_ei    = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)  # no bonds
clash_bi    = torch.tensor([0, 0], device=DEVICE)

rep_loss = gc.compute_repulsion_loss(clash_pos, clash_at, clash_ei, clash_bi)
check("Repulsion fires on small mol",  rep_loss.item() > 0,
      f"repulsion={rep_loss.item():.4f} (should be >0)")
print(f"     repulsion loss on 0.5Å C-C pair = {rep_loss.item():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
os.unlink(tmp_path)
print(f"\n{'='*55}")
if errors:
    print(f"  FAILED: {len(errors)} check(s) failed:")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print(f"  ALL CHECKS PASSED  ({PASS})")
print('='*55)
