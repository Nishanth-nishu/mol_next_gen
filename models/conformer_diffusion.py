"""
conformer_diffusion.py — E(3)-Equivariant Diffusion for 3D Conformer Generation

Research-based fixes (v5 — full audit corrections):
  EDM  (Hoogeboom et al. 2022)      : CoM removal, x_0 parameterization, SNR weighting
  GCDM (Morehead & Cheng 2023)      : Geometry loss at ALL timesteps, weight=1.0
  EGNN (Satorras et al. 2021)       : Correct degree normalization, RBF distances
  DDIM (Song et al. 2020)           : sigma clamp fix (prevent NaN)
  GeoDiff (Xu et al. 2022)          : x_0 prediction for stable geometry gradients

Critical bug-fixes over v4:
  FIX-1  EquivariantLayer: degree counts half what it should (bidirectional graph).
          Divide degree by 2 so normalization is correct.
  FIX-2  Add RBF (Radial Basis Function) distance features on edges — replaces raw
          scalar distance which gives the network almost no geometric information.
  FIX-3  Switch to x_0 parameterization. The denoiser now predicts clean coords x_0
          directly. Noise is derived from x_0_pred for the MSE loss.
          At high-noise timesteps the inverse formula x_0 = (x_t - √(1-ᾱ)·ε) / √ᾱ
          amplifies errors catastrophically when ᾱ≈0; x_0 prediction avoids this.
  FIX-4  Geometry loss: remove t-gating (was active only 20% of the time) and remove
          the extra ×0.1 multiplier. Weight = geometry_weight, applied every step.
  FIX-5  DDIM sigma: clamp (1 - αt/αnext) to ≥ 0 before sqrt → prevents NaN.
  FIX-6  SNR weighting: reduce per-molecule, not per-atom (avoids large-mol bias).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


# =============================================================================
# NOISE SCHEDULE
# =============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def remove_com(x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """
    Remove center-of-mass from coordinates, per molecule (EDM Eq. 4).
    FIX: uses scatter_add_ instead of Python loop for speed.
    """
    B = int(batch_idx.max().item()) + 1
    mol_sums = torch.zeros(B, 3, device=x.device, dtype=x.dtype)
    mol_counts = torch.zeros(B, device=x.device, dtype=x.dtype)
    mol_counts.scatter_add_(0, batch_idx, torch.ones(x.size(0), device=x.device))
    mol_sums.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, 3), x)
    mol_means = mol_sums / mol_counts.unsqueeze(1).clamp(min=1)
    return x - mol_means[batch_idx]


def rbf_features(dist: torch.Tensor,
                 num_rbf: int = 20,
                 d_min: float = 0.5,
                 d_max: float = 6.0) -> torch.Tensor:
    """
    Gaussian Radial Basis Function features for edge distances.
    Replaces raw scalar distance — gives the network interpretable geometric info.
    Used in SchNet, DimeNet, EGNN-good implementations.

    Args:
        dist: (E, 1) pairwise distances in Angstroms
        num_rbf: number of Gaussian centers
        d_min / d_max: range of distances (Angstroms)
    Returns:
        (E, num_rbf) RBF features
    """
    centers = torch.linspace(d_min, d_max, num_rbf, device=dist.device)  # (num_rbf,)
    gamma = 2.0 / (d_max - d_min) * (num_rbf - 1)  # width parameter
    return torch.exp(-gamma * (dist - centers.unsqueeze(0)) ** 2)  # (E, num_rbf)


# =============================================================================
# EQUIVARIANT LAYER  (FIX-1: degree normalization; FIX-2: RBF distances)
# =============================================================================

class EquivariantLayer(nn.Module):
    """
    E(3)-equivariant message passing layer (EGNN-style, Satorras et al. 2021).

    FIX-1: Degree normalization corrected.
      In a bidirectional graph, atom i appears in `col` once per neighbor j→i AND
      in `row` once per neighbor i→j.  We scatter coord_update into col (dest).
      The degree count from scatter_add into col double-counts relative to the
      physical number of neighbors → divide degree by 2.

    FIX-2: Replace raw scalar distance with RBF features (20 Gaussians, 0.5–6 Å).
      Raw distance loses geometric resolution; RBF gives the network interpretable
      distance information at every step.
    """

    def __init__(self, hidden_dim: int, num_rbf: int = 20):
        super().__init__()
        self.num_rbf = num_rbf

        # Edge MLP: node features + RBF distance features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update weight (scalar per edge, bounded)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                h: torch.Tensor,           # (N, hidden_dim)
                x: torch.Tensor,           # (N, 3)
                edge_index: torch.Tensor,  # (2, E)
                bond_embed: torch.Tensor   # (E, edge_dim) bond type embedding
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index          # row=source, col=destination

        # ── Pairwise geometry ────────────────────────────────────────────────
        diff = x[row] - x[col]                                            # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)    # (E, 1)
        unit_vec = diff / dist                                             # (E, 3)

        # FIX-2: RBF distance features instead of raw scalar
        rbf = rbf_features(dist, num_rbf=self.num_rbf)                   # (E, num_rbf)

        # ── Edge messages ────────────────────────────────────────────────────
        edge_input = torch.cat([h[row], h[col], rbf], dim=-1)
        m_ij = self.edge_mlp(edge_input)                                  # (E, hidden)

        # Bond embedding is added to message (residual)
        m_ij = m_ij + bond_embed

        # ── Coordinate update (equivariant) ──────────────────────────────────
        coord_weight = self.coord_mlp(m_ij)                               # (E, 1)
        coord_update = coord_weight * unit_vec                            # (E, 3)

        N = x.size(0)
        x_agg = torch.zeros_like(x)
        x_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, 3), coord_update)

        # FIX-1: bidirectional graph double-counts degree → divide by 2
        degree = torch.zeros(N, 1, device=x.device)
        degree.scatter_add_(0, col.unsqueeze(-1),
                            torch.ones(col.size(0), 1, device=x.device))
        degree = (degree / 2.0).clamp(min=1.0)

        x_new = x + x_agg / degree

        # ── Node feature update ───────────────────────────────────────────────
        m_agg = torch.zeros_like(h)
        m_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, h.size(-1)), m_ij)

        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))
        h_new = self.layer_norm(h + h_new)

        return h_new, x_new


# =============================================================================
# CONFORMER DENOISER  (FIX-3: x_0 parameterization)
# =============================================================================

class ConformerDenoiser(nn.Module):
    """
    Denoising network: predicts CLEAN coordinates x_0 from noisy x_t.

    FIX-3: x_0 parameterization (not ε-parameterization).
      - Network output = predicted clean coordinates x_0_pred  (N, 3)
      - Noise ε is DERIVED from x_0_pred: ε = (x_t - √ᾱ·x_0_pred) / √(1-ᾱ)
      - MSE loss on derived ε (same loss surface but MUCH more stable geometry gradients)
      - Geometry loss directly on x_0_pred (no amplification problem)
      - References: EDM Appendix B, GCDM Sec.3.2, GeoDiff Sec.3
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_atom_types: int = 10,
                 num_bond_types: int = 5,
                 num_rbf: int = 20,
                 time_dim: int = 128):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.time_dim = time_dim

        # Atom type embedding (covers H=1 to I=53 — heavy atoms only after data fix)
        self.atom_embed = nn.Embedding(54, hidden_dim)

        # Bond type embedding — now goes into edge messages directly
        self.bond_embed = nn.Embedding(num_bond_types + 1, hidden_dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Equivariant message passing layers
        self.layers = nn.ModuleList([
            EquivariantLayer(hidden_dim, num_rbf=num_rbf)
            for _ in range(num_layers)
        ])

        # Output: predict x_0 (clean coordinates) in angstroms
        # Final linear layer with no activation — output is unbounded coords
        self.coord_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self,
                x_noisy: torch.Tensor,     # (N, 3) noisy coordinates
                t: torch.Tensor,           # (B,) timesteps per molecule
                atom_types: torch.Tensor,  # (N,) atomic numbers
                edge_index: torch.Tensor,  # (2, E)
                bond_types: torch.Tensor,  # (E,) bond orders
                batch_idx: torch.Tensor    # (N,) batch assignment
                ) -> torch.Tensor:         # (N, 3) predicted x_0
        # Atom embeddings (positional info enters ONLY through distances in EquivariantLayer)
        h = self.atom_embed(atom_types.clamp(0, 53))

        # Time embedding (broadcast to per-atom via batch_idx)
        t_emb = sinusoidal_embedding(t.float(), self.time_dim)
        t_emb = self.time_mlp(t_emb)                   # (B, hidden_dim)
        h = h + t_emb[batch_idx]                        # (N, hidden_dim)

        # Bond embedding (feeds into edge MLP inside EquivariantLayer)
        bond_feat = self.bond_embed(bond_types.clamp(0, 5))  # (E, hidden_dim)

        x = x_noisy

        for layer in self.layers:
            h, x = layer(h, x, edge_index, bond_feat)

        # Predict clean coordinates x_0 (residual from noisy input)
        # The residual formulation helps: start from x_noisy and predict correction
        delta_x = self.coord_pred(h)               # (N, 3)
        x_0_pred = x + delta_x                     # (N, 3) predicted clean coords

        return x_0_pred


# =============================================================================
# CONFORMER DIFFUSION MODEL
# =============================================================================

class ConformerDiffusion(nn.Module):
    """
    E(3)-equivariant diffusion model for 3D conformer generation.
    Uses x_0 parameterization for stable geometry learning.
    """

    def __init__(self,
                 num_timesteps: int = 1000,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_rbf: int = 20,
                 time_dim: int = 128):
        super().__init__()

        self.num_timesteps = num_timesteps

        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1 - alphas_cumprod))

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance',
                             torch.log(posterior_variance.clamp(min=1e-20)))

        # SNR for loss weighting (Min-SNR, Hang et al. 2023)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        self.register_buffer('snr', snr)

        self.denoiser = ConformerDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            time_dim=time_dim,
        )

        # Geometry constraints (weights configured in training script)
        from models.geometry_constraints import GeometryConstraints
        self.geometry = GeometryConstraints(
            bond_weight=10.0,
            angle_weight=3.0,
            torsion_weight=1.0,
            repulsion_weight=5.0,
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """Extract per-atom schedule values using batch molecule timestep indices."""
        # a: (T,), t: (B,) → result: (N, 1)
        return a[t][batch_idx].unsqueeze(-1)

    def q_sample(self,
                 x_0: torch.Tensor,
                 t: torch.Tensor,
                 batch_idx: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(x_t | x_0) = √ᾱ·x_0 + √(1-ᾱ)·ε
        CoM of noise removed per-molecule (EDM Eq. 5).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # CoM-free noise (keeps diffusion in zero-CoM manifold)
        noise = remove_com(noise, batch_idx)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, batch_idx)           # (N,1)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, batch_idx)  # (N,1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        x_t = remove_com(x_t, batch_idx)

        return x_t, noise

    def p_sample(self,
                 x_t: torch.Tensor,
                 t: torch.Tensor,
                 atom_types: torch.Tensor,
                 edge_index: torch.Tensor,
                 bond_types: torch.Tensor,
                 batch_idx: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion step (DDPM)."""
        # FIX-3: denoiser now outputs x_0_pred directly
        x_0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
        x_0_pred = remove_com(x_0_pred, batch_idx)

        # Derive ε from x_0_pred
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, batch_idx)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, batch_idx)
        noise_pred = (x_t - sqrt_alpha * x_0_pred) / sqrt_one_minus.clamp(min=1e-6)

        # DDPM posterior mean
        beta = self._extract(self.betas, t, batch_idx)
        alpha = self._extract(self.alphas, t, batch_idx)
        mean = (x_t - beta * noise_pred / sqrt_one_minus.clamp(min=1e-6)) / torch.sqrt(alpha)
        mean = remove_com(mean, batch_idx)

        # Stochastic noise (zero at t=0)
        t_expanded = t[batch_idx]
        noise = torch.randn_like(x_t)
        noise = remove_com(noise, batch_idx)
        noise[t_expanded == 0] = 0.0

        posterior_var = self._extract(self.posterior_variance, t, batch_idx)
        return mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(self,
               atom_types: torch.Tensor,
               edge_index: torch.Tensor,
               bond_types: torch.Tensor,
               batch_idx: torch.Tensor,
               num_steps: Optional[int] = None) -> torch.Tensor:
        """Full DDPM sampling."""
        device = atom_types.device
        N = atom_types.size(0)
        B = int(batch_idx.max().item()) + 1
        num_steps = num_steps or self.num_timesteps

        x_t = remove_com(torch.randn(N, 3, device=device), batch_idx)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps,
                                   dtype=torch.long, device=device)

        for t_val in timesteps:
            t = torch.full((B,), t_val.item(), dtype=torch.long, device=device)
            x_t = self.p_sample(x_t, t, atom_types, edge_index, bond_types, batch_idx)

        return x_t

    @torch.no_grad()
    def ddim_sample(self,
                    atom_types: torch.Tensor,
                    edge_index: torch.Tensor,
                    bond_types: torch.Tensor,
                    batch_idx: torch.Tensor,
                    num_steps: int = 50,
                    eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling (Song et al. 2020).
        FIX-5: sigma clamped to prevent NaN from sqrt of negative.
        """
        device = atom_types.device
        N = atom_types.size(0)
        B = int(batch_idx.max().item()) + 1

        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        x_t = remove_com(torch.randn(N, 3, device=device), batch_idx)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val.item(), dtype=torch.long, device=device)

            # FIX-3: denoiser returns x_0_pred directly
            x_0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
            x_0_pred = remove_com(x_0_pred, batch_idx)

            alpha_t = self.alphas_cumprod[t][batch_idx].unsqueeze(-1)   # (N,1) ᾱ_t

            if i == len(timesteps) - 1:
                x_t = x_0_pred
            else:
                t_next_val = timesteps[i + 1].item()
                t_next = torch.full((B,), t_next_val, dtype=torch.long, device=device)
                alpha_next = self.alphas_cumprod[t_next][batch_idx].unsqueeze(-1)  # (N,1) ᾱ_{t-1}

                # Derive ε from x_0_pred
                sqrt_one_minus_at = torch.sqrt(1.0 - alpha_t).clamp(min=1e-6)
                noise_pred = (x_t - torch.sqrt(alpha_t) * x_0_pred) / sqrt_one_minus_at

                # FIX-5: clamp ratio before sqrt to avoid NaN
                ratio = (alpha_t / alpha_next.clamp(min=1e-8)).clamp(max=1.0)
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_next) / (1.0 - alpha_t).clamp(min=1e-8)
                ) * torch.sqrt((1.0 - ratio).clamp(min=0.0))  # ← FIX: clamp(min=0)

                direction = torch.sqrt((1.0 - alpha_next - sigma ** 2).clamp(min=0.0)) * noise_pred
                noise = remove_com(torch.randn_like(x_t), batch_idx) if eta > 0 else 0.0

                x_t = torch.sqrt(alpha_next) * x_0_pred + direction + sigma * noise

        return x_t

    @torch.no_grad()
    def guided_sample(self,
                      atom_types: torch.Tensor,
                      edge_index: torch.Tensor,
                      bond_types: torch.Tensor,
                      batch_idx: torch.Tensor,
                      num_steps: int = 50,
                      guidance_scale: float = 1.0,
                      aromatic_rings=None,
                      chiral_centers=None,
                      small_rings=None) -> torch.Tensor:
        """
        DDIM sampling with geometry gradient guidance.
        x_0_pred is refined by geometry gradient at each step.
        Using x_0 parameterization makes this guidance clean and meaningful.
        """
        device = atom_types.device
        N = atom_types.size(0)
        B = int(batch_idx.max().item()) + 1

        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        x_t = remove_com(torch.randn(N, 3, device=device), batch_idx)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val.item(), dtype=torch.long, device=device)

            x_0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
            x_0_pred = remove_com(x_0_pred, batch_idx)

            # Geometry guidance: gradient descent on x_0_pred
            if guidance_scale > 0:
                x_0_pred = self._geometry_gradient_step(
                    x_0_pred, atom_types, edge_index, bond_types, batch_idx,
                    num_iters=3, lr=guidance_scale * 0.02,
                    aromatic_rings=aromatic_rings,
                    chiral_centers=chiral_centers,
                    small_rings=small_rings,
                )

            alpha_t = self.alphas_cumprod[t][batch_idx].unsqueeze(-1)

            if i == len(timesteps) - 1:
                x_t = x_0_pred
            else:
                t_next_val = timesteps[i + 1].item()
                t_next = torch.full((B,), t_next_val, dtype=torch.long, device=device)
                alpha_next = self.alphas_cumprod[t_next][batch_idx].unsqueeze(-1)

                sqrt_one_minus_at = torch.sqrt(1.0 - alpha_t).clamp(min=1e-6)
                noise_pred = (x_t - torch.sqrt(alpha_t) * x_0_pred) / sqrt_one_minus_at

                direction = torch.sqrt((1.0 - alpha_next).clamp(min=0.0)) * noise_pred
                x_t = torch.sqrt(alpha_next) * x_0_pred + direction

        return x_t

    def _geometry_gradient_step(self,
                                 pos: torch.Tensor,
                                 atom_types: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 bond_types: torch.Tensor,
                                 batch_idx: torch.Tensor,
                                 num_iters: int = 3,
                                 lr: float = 0.02,
                                 aromatic_rings=None,
                                 chiral_centers=None,
                                 small_rings=None) -> torch.Tensor:
        """
        Gradient-based geometry correction on the predicted x_0.
        Geometry constraints object is created ONCE (not inside inner loop).
        """
        from models.geometry_constraints import GeometryConstraints

        pos = pos.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=lr)

        # Create constraints once, not per iteration
        _gc = GeometryConstraints(
            bond_weight=10.0,
            angle_weight=3.0,
            repulsion_weight=5.0,
            planarity_weight=5.0,
            chirality_weight=3.0,
            ring_strain_weight=2.0,
        )

        for _ in range(num_iters):
            optimizer.zero_grad()
            total_loss, _ = _gc.compute_total_loss(
                pos, atom_types, edge_index, bond_types, batch_idx,
                include_angles=True, include_torsions=False,
                aromatic_rings=aromatic_rings,
                chiral_centers=chiral_centers,
                small_rings=small_rings,
            )
            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()

        return pos.detach()

    def get_loss(self,
                 x_0: torch.Tensor,
                 atom_types: torch.Tensor,
                 edge_index: torch.Tensor,
                 bond_types: torch.Tensor,
                 batch_idx: torch.Tensor,
                 geometry_weight: float = 1.0,
                 epoch: int = 1,
                 max_epochs: int = 300,
                 min_snr_gamma: float = 5.0) -> Dict:
        """
        Training loss — v5 (complete fixes):

        FIX-3: x_0 parameterization
          - denoiser returns x_0_pred
          - ε derived: ε_pred = (x_t - √ᾱ · x_0_pred) / √(1-ᾱ)
          - MSE on ε_pred vs true ε → same loss surface, stable geometry gradients

        FIX-4: Geometry loss applied at ALL timesteps, weight = geometry_weight (not *0.1)
          - No t-gating (was active only 20% of time before)
          - Geometry supervised directly on x_0_pred (x_0 parameterization means
            x_0_pred is always explicitly available, not amplified from ε)

        FIX-6: SNR weighting is per-molecule (not per-atom) to avoid large-mol bias.
        """
        device = x_0.device
        B = int(batch_idx.max().item()) + 1

        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffusion
        x_t, noise = self.q_sample(x_0, t, batch_idx)

        # FIX-3: predict x_0 directly
        x_0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
        x_0_pred = remove_com(x_0_pred, batch_idx)

        # Derive ε from x_0_pred for MSE loss
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, batch_idx)        # (N,1)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, batch_idx)  # (N,1)
        noise_pred = (x_t - sqrt_alpha * x_0_pred) / sqrt_one_minus.clamp(min=1e-6)

        # Per-atom MSE on noise
        mse_per_atom = ((noise_pred - noise) ** 2).sum(-1)  # (N,)

        # FIX-6: reduce to per-molecule, then apply SNR weight per molecule
        # scatter mean: average mse within each molecule
        mse_per_mol = torch.zeros(B, device=device)
        mol_counts = torch.zeros(B, device=device)
        mse_per_mol.scatter_add_(0, batch_idx, mse_per_atom)
        mol_counts.scatter_add_(0, batch_idx, torch.ones(mse_per_atom.size(0), device=device))
        mse_per_mol = mse_per_mol / mol_counts.clamp(min=1)

        # Min-SNR weighting (Hang et al. 2023) — per molecule
        snr_t = self.snr[t]   # (B,)
        snr_weight = torch.minimum(snr_t, torch.full_like(snr_t, min_snr_gamma)) / snr_t.clamp(min=1e-8)
        mse_loss = (snr_weight * mse_per_mol).mean()

        # FIX-4: Geometry loss at ALL timesteps, direct on x_0_pred (no t-gating, no extra *0.1)
        geo_loss = self._compute_geometry_loss(
            x_0_pred, atom_types, edge_index, bond_types, batch_idx,
            include_angles=True, include_torsions=False
        )

        total_loss = mse_loss + geometry_weight * geo_loss

        return {
            'total': total_loss,
            'mse':   mse_loss.detach(),
            'geo':   geo_loss.detach() if isinstance(geo_loss, torch.Tensor) else torch.tensor(0.0),
        }

    def _compute_geometry_loss(self,
                                pos: torch.Tensor,
                                atom_types: torch.Tensor,
                                edge_index: torch.Tensor,
                                bond_types: torch.Tensor,
                                batch_idx: torch.Tensor,
                                include_angles: bool = True,
                                include_torsions: bool = False) -> torch.Tensor:
        """Geometry constraint loss using self.geometry instance."""
        total, _ = self.geometry.compute_total_loss(
            pos, atom_types, edge_index, bond_types, batch_idx,
            include_angles=include_angles,
            include_torsions=include_torsions
        )
        return total


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing ConformerDiffusion v5 (x_0 param + RBF + degree fix)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ConformerDiffusion(
        num_timesteps=100,
        hidden_dim=64,
        num_layers=3,
        num_rbf=20,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Heavy-atom-only molecules (as after data fix):
    # Mol 0: water (O + 2H → actually we use heavy-atom-only so just O for water,
    #         but let's use ethanol without H: C-C-O → 3 heavy atoms)
    # Mol 1: methane without H: just C  → 1 atom (minimal)
    atom_types = torch.tensor([6, 6, 8,   6], device=device)  # C,C,O  +  C
    edge_index  = torch.tensor([[0,1,1,2,  3],[1,0,2,1,  3]], device=device)
    bond_types  = torch.tensor([1,1,1,1,   0], dtype=torch.long, device=device)
    batch_idx   = torch.tensor([0,0,0,     1], device=device)

    # Ground truth coords (centered)
    x_0 = torch.randn(4, 3, device=device)
    x_0 = remove_com(x_0, batch_idx)

    # Test loss at different epochs
    for gw in [0.1, 1.0]:
        loss_dict = model.get_loss(
            x_0, atom_types, edge_index, bond_types, batch_idx,
            geometry_weight=gw, epoch=1, max_epochs=100
        )
        print(f"  geo_weight={gw}: total={loss_dict['total'].item():.4f} "
              f"mse={loss_dict['mse'].item():.4f} geo={loss_dict['geo'].item():.4f}")
        assert not torch.isnan(loss_dict['total']), f"NaN loss at geo_weight={gw}!"

    # Test DDIM sampling (no NaN check)
    print("Testing DDIM sampling...")
    x_gen = model.ddim_sample(atom_types, edge_index, bond_types, batch_idx, num_steps=10)
    print(f"  Generated shape: {x_gen.shape}")
    assert x_gen.shape == (4, 3), f"Wrong shape: {x_gen.shape}"
    assert not torch.isnan(x_gen).any(), "NaN in generated coordinates!"

    # CoM check
    for b in range(2):
        mask = batch_idx == b
        com = x_gen[mask].mean(0)
        print(f"  Mol {b} CoM: {com.tolist()} (should be ~0)")
        assert com.abs().max() < 0.5, f"CoM too large for mol {b}: {com}"

    print("\nAll ConformerDiffusion v5 tests passed!")
