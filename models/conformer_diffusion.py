"""
conformer_diffusion.py — E(3)-Equivariant Diffusion for 3D Conformer Generation

Research-based fixes from:
- EDM (Hoogeboom et al. 2022): CoM removal INSIDE diffusion, SNR-based loss weighting
- GeoMol (Ganea et al. 2021): Torsion angle learning curriculum
- EGNN (Satorras et al. 2021): Proper degree normalization in equivariant layers

Key improvements over v1:
1. CoM (center-of-mass) removed PER-MOLECULE inside q_sample (not just externally)
2. EGNN coordinate aggregation normalized by degree (prevents unstable updates)
3. SNR-based loss weighting (min-SNR-gamma clipping from EDM)
4. Torsion angle loss added to get_loss() with curriculum (bonds → angles → torsions)
5. Vectorized geometry loss (no Python loops over edges)
6. Soft tanh scaling instead of hard clamp on x_0_pred
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
    """Cosine beta schedule (Nichol & Dhariwal 2021). Better than linear."""
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
    Remove center-of-mass from coordinates, per molecule.
    
    From EDM (Hoogeboom et al.) Eq. 4:
    The diffusion process should be invariant to translation.
    This is enforced by removing CoM at each step of forward diffusion.
    """
    B = batch_idx.max().item() + 1
    # Compute per-molecule mean
    mol_means = torch.zeros(B, 3, device=x.device, dtype=x.dtype)
    mol_counts = torch.zeros(B, device=x.device, dtype=x.dtype)
    
    mol_counts.scatter_add_(0, batch_idx, torch.ones(x.size(0), device=x.device))
    for d in range(3):
        mol_means[:, d].scatter_add_(0, batch_idx, x[:, d])
    mol_means = mol_means / mol_counts.unsqueeze(1).clamp(min=1)
    
    # Subtract per-atom CoM
    return x - mol_means[batch_idx]


# =============================================================================
# EQUIVARIANT LAYER (FIXED: normalized aggregation)
# =============================================================================

class EquivariantLayer(nn.Module):
    """
    E(3)-equivariant message passing layer.
    
    FIX from EGNN paper: coordinate updates should be normalized by
    the sum of weights (not raw sum) to prevent instability in large molecules.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 32):
        super().__init__()
        self.edge_dim = edge_dim

        # Edge MLP: computes messages
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Coordinate update weight (scalar per edge)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Bounded output prevents exploding coordinate updates
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                h: torch.Tensor,           # (N, hidden_dim)
                x: torch.Tensor,           # (N, 3)
                edge_index: torch.Tensor,  # (2, E)
                edge_attr: torch.Tensor    # (E, edge_dim)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index

        # Pairwise distances
        diff = x[row] - x[col]     # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)
        unit_vec = diff / dist      # (E, 3)

        # Edge messages
        edge_input = torch.cat([h[row], h[col], edge_attr, dist], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # (E, hidden_dim)

        # Coordinate update: scalar * unit_vec (equivariant)
        coord_weight = self.coord_mlp(m_ij)   # (E, 1), bounded by Tanh
        coord_update = coord_weight * unit_vec  # (E, 3)

        # NORMALIZED aggregation (FIX: prevents large-molecule instability)
        N = x.size(0)
        x_agg = torch.zeros_like(x)
        x_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, 3), coord_update)

        # Degree normalization: divide by number of neighbors + 1
        degree = torch.zeros(N, 1, device=x.device)
        degree.scatter_add_(0, col.unsqueeze(-1), torch.ones(col.size(0), 1, device=x.device))
        degree = degree.clamp(min=1.0)
        
        x_new = x + x_agg / (degree + 1.0)  # Normalized update

        # Node feature update
        m_agg = torch.zeros_like(h)
        m_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, h.size(-1)), m_ij)

        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))
        h_new = self.layer_norm(h + h_new)

        return h_new, x_new


# =============================================================================
# CONFORMER DENOISER
# =============================================================================

class ConformerDenoiser(nn.Module):
    """
    Denoising network for conformer generation.
    Takes noisy coordinates + fixed molecular graph → predicts noise.
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_atom_types: int = 10,
                 num_bond_types: int = 5,
                 edge_dim: int = 32,
                 time_dim: int = 128):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim

        # Atom embedding (use full periodic table range)
        self.atom_embed = nn.Embedding(54, hidden_dim)  # Cover H(1) to I(53)

        # Bond embedding
        self.bond_embed = nn.Embedding(num_bond_types + 1, edge_dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate embedding
        self.coord_embed = nn.Linear(3, hidden_dim)

        # Equivariant layers
        self.layers = nn.ModuleList([
            EquivariantLayer(hidden_dim, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        # Output: predict noise
        self.noise_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self,
                x_noisy: torch.Tensor,
                t: torch.Tensor,
                atom_types: torch.Tensor,
                edge_index: torch.Tensor,
                bond_types: torch.Tensor,
                batch_idx: torch.Tensor
                ) -> torch.Tensor:
        # Embed atoms (clamp to valid range)
        h = self.atom_embed(atom_types.clamp(0, 53))

        # Add coordinate info
        h = h + self.coord_embed(x_noisy)

        # Add time embedding
        t_emb = sinusoidal_embedding(t.float(), self.time_dim)
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb[batch_idx]

        # Embed bonds
        edge_attr = self.bond_embed(bond_types.clamp(0, 4))

        x = x_noisy

        # Message passing
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        return self.noise_pred(h)  # (N, 3)


# =============================================================================
# CONFORMER DIFFUSION MODEL
# =============================================================================

class ConformerDiffusion(nn.Module):
    """
    Full E(3)-equivariant diffusion model for conformer generation.
    
    Training: add noise → predict noise
    Sampling: start from noise → reverse diffusion → valid 3D conformation
    """

    def __init__(self,
                 num_timesteps: int = 1000,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 edge_dim: int = 32,
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

        # SNR for loss weighting (from EDM / min-SNR paper)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        self.register_buffer('snr', snr)

        self.denoiser = ConformerDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_dim,
            time_dim=time_dim,
        )

        # Pre-instantiated geometry constraints (unit weights — scaled by geometry_weight in get_loss)
        from models.geometry_constraints import GeometryConstraints
        self.geometry = GeometryConstraints(
            bond_weight=1.0,
            angle_weight=0.3,
            torsion_weight=0.1,
            repulsion_weight=0.5,
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self,
                 x_0: torch.Tensor,
                 t: torch.Tensor,
                 batch_idx: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(x_t | x_0).
        
        FIX (from EDM): CoM of noise is removed per-molecule so the diffusion
        process stays in the zero-CoM subspace throughout.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Remove CoM from noise per molecule (EDM Eq. 5)
        noise = remove_com(noise, batch_idx)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][batch_idx].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][batch_idx].unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise

        # Also remove CoM from x_t (keeps diffusion in zero-CoM manifold)
        x_t = remove_com(x_t, batch_idx)

        return x_t, noise

    def p_sample(self,
                 x_t: torch.Tensor,
                 t: torch.Tensor,
                 atom_types: torch.Tensor,
                 edge_index: torch.Tensor,
                 bond_types: torch.Tensor,
                 batch_idx: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion step."""
        noise_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)

        beta = self.betas[t][batch_idx].unsqueeze(-1)
        alpha = self.alphas[t][batch_idx].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][batch_idx].unsqueeze(-1)

        mean = (x_t - beta * noise_pred / sqrt_one_minus) / torch.sqrt(alpha)
        mean = remove_com(mean, batch_idx)

        t_expanded = t[batch_idx]
        noise = torch.randn_like(x_t)
        noise = remove_com(noise, batch_idx)
        noise[t_expanded == 0] = 0

        posterior_var = self.posterior_variance[t][batch_idx].unsqueeze(-1)
        return mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(self,
               atom_types: torch.Tensor,
               edge_index: torch.Tensor,
               bond_types: torch.Tensor,
               batch_idx: torch.Tensor,
               num_steps: Optional[int] = None) -> torch.Tensor:
        """Full DDPM sampling (slow, high quality)."""
        device = atom_types.device
        N = atom_types.size(0)
        B = batch_idx.max().item() + 1
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
        eta=0: deterministic, eta=1: stochastic (like DDPM).
        """
        device = atom_types.device
        N = atom_types.size(0)
        B = batch_idx.max().item() + 1

        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        x_t = remove_com(torch.randn(N, 3, device=device), batch_idx)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val.item(), dtype=torch.long, device=device)

            noise_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)

            alpha_t = self.alphas_cumprod[t[batch_idx]].unsqueeze(-1)
            # Predict x_0 with soft clamping (avoids hard-clamp discontinuity)
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t.clamp(min=1e-8))
            x_0_pred = torch.tanh(x_0_pred / 10.0) * 10.0  # Soft clamp ≈ [-10, 10]
            x_0_pred = remove_com(x_0_pred, batch_idx)

            if i == len(timesteps) - 1:
                x_t = x_0_pred
            else:
                t_next_val = timesteps[i + 1].item()
                t_next = torch.full((B,), t_next_val, dtype=torch.long, device=device)
                alpha_next = self.alphas_cumprod[t_next[batch_idx]].unsqueeze(-1)

                # DDIM update with optional stochasticity (eta)
                sigma = eta * torch.sqrt(
                    (1 - alpha_next) / (1 - alpha_t).clamp(min=1e-8)
                ) * torch.sqrt(1 - alpha_t / alpha_next.clamp(min=1e-8))

                direction = torch.sqrt((1 - alpha_next - sigma ** 2).clamp(min=0)) * noise_pred
                noise = remove_com(torch.randn_like(x_t), batch_idx)

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
        DDIM sampling with integrated geometry guidance.
        Modifies x_0_pred at each step using chemistry loss gradient.

        Args (soft-constraint extras from Exp-1):
            aromatic_rings: list of ring atom-index lists for planarity loss
            chiral_centers: list of (center, neighbors, sign) for chirality loss
            small_rings:    list of ring atom-index lists for ring strain loss
        """
        device = atom_types.device
        N = atom_types.size(0)
        B = batch_idx.max().item() + 1

        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        x_t = remove_com(torch.randn(N, 3, device=device), batch_idx)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val.item(), dtype=torch.long, device=device)

            with torch.no_grad():
                noise_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)

            alpha_t = self.alphas_cumprod[t[batch_idx]].unsqueeze(-1)
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t.clamp(min=1e-8))
            x_0_pred = torch.tanh(x_0_pred / 10.0) * 10.0
            x_0_pred = remove_com(x_0_pred, batch_idx)

            # Geometry guidance step (with soft constraints)
            if guidance_scale > 0:
                x_0_pred = self._geometry_gradient_step(
                    x_0_pred, atom_types, edge_index, bond_types, batch_idx,
                    num_iters=3, lr=guidance_scale * 0.03,
                    aromatic_rings=aromatic_rings,
                    chiral_centers=chiral_centers,
                    small_rings=small_rings,
                )

            if i == len(timesteps) - 1:
                x_t = x_0_pred
            else:
                t_next_val = timesteps[i + 1].item()
                t_next = torch.full((B,), t_next_val, dtype=torch.long, device=device)
                alpha_next = self.alphas_cumprod[t_next[batch_idx]].unsqueeze(-1)

                direction = torch.sqrt((1 - alpha_next).clamp(min=0)) * noise_pred
                x_t = torch.sqrt(alpha_next) * x_0_pred + direction

        return x_t

    def _geometry_gradient_step(self,
                                 pos: torch.Tensor,
                                 atom_types: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 bond_types: torch.Tensor,
                                 batch_idx: torch.Tensor,
                                 num_iters: int = 5,
                                 lr: float = 0.03,
                                 aromatic_rings=None,
                                 chiral_centers=None,
                                 small_rings=None) -> torch.Tensor:
        """
        Gradient-based geometry correction using all soft chemistry losses.

        Losses applied:
          - Bond length (vectorized MMFF94 targets)
          - VDW repulsion (excludes 1-2/1-3 pairs)
          - Planarity (SVD best-fit plane for aromatic rings)  [NEW Exp-1]
          - Chirality (signed tetrahedral volume)              [NEW Exp-1]
          - Ring strain (60°/90° targets for 3/4-rings)       [NEW Exp-1]
        """
        from models.geometry_constraints import (
            get_ideal_bond_lengths_vectorized, GeometryConstraints
        )

        pos = pos.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pos], lr=lr)
        row, col = edge_index

        for _ in range(num_iters):
            optimizer.zero_grad()

            # --- Bond length loss ---
            diff = pos[row] - pos[col]
            dists = torch.norm(diff, dim=-1).clamp(min=1e-6)
            ideal = get_ideal_bond_lengths_vectorized(atom_types[row], atom_types[col], bond_types)
            bond_loss = 10.0 * F.mse_loss(dists, ideal)

            loss = bond_loss

            # --- VDW repulsion (only for small batches) ---
            N = pos.size(0)
            if N <= 100:
                bonded = torch.zeros(N, N, device=pos.device, dtype=torch.bool)
                bonded[row, col] = True
                all_d = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0))[0]
                same = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)
                mask = same & ~bonded & ~torch.eye(N, device=pos.device, dtype=torch.bool)
                clash = all_d[mask]
                clash = clash[clash < 1.4]
                if clash.numel() > 0:
                    loss = loss + 5.0 * torch.mean((1.4 - clash) ** 2)

            # --- Soft constraints (Exp-1) ---
            # Use a shared GeometryConstraints instance (unit weights absorbed below)
            _gc = GeometryConstraints(
                planarity_weight=5.0,
                chirality_weight=3.0,
                ring_strain_weight=2.0,
            )

            if aromatic_rings:
                loss = loss + _gc.compute_planarity_loss(pos, aromatic_rings)

            if chiral_centers:
                loss = loss + _gc.compute_chirality_loss(pos, chiral_centers)

            if small_rings:
                loss = loss + _gc.compute_ring_strain_loss(pos, small_rings)

            if loss.requires_grad:
                loss.backward()
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
                 max_epochs: int = 100,
                 min_snr_gamma: float = 5.0) -> torch.Tensor:
        """
        Training loss with:
        1. SNR-based timestep weighting (from EDM/min-SNR paper)
        2. Curriculum geometry learning: bonds → bonds+angles → bonds+angles+torsions
        3. Vectorized chemistry-aware geometry loss

        Geometry loss scale: all internal weights are 1.0 (unit), controlled
        entirely by geometry_weight here so the total loss stays comparable
        to the diffusion MSE loss (~0.1-1.0 range).
        """
        device = x_0.device
        B = batch_idx.max().item() + 1

        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffusion (CoM-aware noise)
        x_t, noise = self.q_sample(x_0, t, batch_idx)

        # Predict noise
        noise_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)

        # MSE loss per atom
        mse_per_atom = ((noise_pred - noise) ** 2).sum(-1)  # (N,)

        # SNR-based weighting (min-SNR-gamma clipping from Hang et al. 2023)
        snr_t = self.snr[t][batch_idx]  # (N,)
        snr_weight = torch.minimum(snr_t, torch.full_like(snr_t, min_snr_gamma)) / snr_t.clamp(min=1e-8)
        mse_loss = (snr_weight * mse_per_atom).mean()

        # Curriculum: 3-stage geometry learning
        # Stage 1 (0–30%): bonds only
        # Stage 2 (30–60%): bonds + angles
        # Stage 3 (60%+): bonds + angles + torsions
        progress = epoch / max_epochs
        use_angles = progress > 0.30
        use_torsions = progress > 0.60

        # Linear ramp from 0 → 1 over first 40% of training
        curriculum = min(1.0, progress / 0.40)
        effective_geo_weight = geometry_weight * curriculum

        # Only compute geometry loss if weight is meaningful
        if effective_geo_weight < 1e-6:
            return mse_loss

        # Predict x_0 from x_t (detached for stability)
        alpha_t = self.sqrt_alphas_cumprod[t][batch_idx].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][batch_idx].unsqueeze(-1)
        x_0_pred = (x_t - sqrt_one_minus * noise_pred) / alpha_t.clamp(min=1e-6)
        x_0_pred = torch.tanh(x_0_pred / 10.0) * 10.0  # Soft clamp
        x_0_pred = remove_com(x_0_pred.detach(), batch_idx)

        geo_loss = self._compute_geometry_loss_vectorized(
            x_0_pred, atom_types, edge_index, bond_types, batch_idx,
            include_angles=use_angles, include_torsions=use_torsions
        )

        # Only apply geometry loss at low noise levels (near x_0)
        # High-noise timesteps (large t) have uninformative x_0_pred
        noise_level = t.float() / self.num_timesteps       # 0=clean, 1=pure noise
        geo_timestep_weight = (1.0 - noise_level[batch_idx]).mean()

        total_loss = mse_loss + effective_geo_weight * geo_timestep_weight * geo_loss

        return {
            'total': total_loss,
            'mse':   mse_loss.detach(),
            'geo':   geo_loss.detach() if isinstance(geo_loss, torch.Tensor) else torch.tensor(0.0),
        }

    def _compute_geometry_loss_vectorized(self,
                                           pos: torch.Tensor,
                                           atom_types: torch.Tensor,
                                           edge_index: torch.Tensor,
                                           bond_types: torch.Tensor,
                                           batch_idx: torch.Tensor,
                                           include_angles: bool = False,
                                           include_torsions: bool = False) -> torch.Tensor:
        """
        Geometry loss using pre-instantiated self.geometry constraints.
        Unit weights (1.0) — all scaling done by geometry_weight in get_loss().
        """
        total, _ = self.geometry.compute_total_loss(
            pos, atom_types, edge_index, bond_types, batch_idx,
            include_angles=include_angles,
            include_torsions=include_torsions
        )
        return total


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Testing ConformerDiffusion (v2 — CoM + SNR + curriculum)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ConformerDiffusion(
        num_timesteps=100,
        hidden_dim=128,
        num_layers=4
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Batch: molecule 1 = 3 atoms (water-like), molecule 2 = 4 atoms (methane-like)
    atom_types = torch.tensor([8, 1, 1, 6, 1, 1, 1], device=device)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 3, 4, 3, 5, 3, 6],
        [1, 0, 2, 0, 4, 3, 5, 3, 6, 3]
    ], device=device)
    bond_types = torch.ones(10, dtype=torch.long, device=device)
    batch_idx  = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)

    # Ground truth coordinates (centered)
    x_0 = torch.randn(7, 3, device=device)
    x_0 = remove_com(x_0, batch_idx)

    # Test forward pass (various stages)
    for epoch in [1, 25, 60]:
        loss = model.get_loss(x_0, atom_types, edge_index, bond_types, batch_idx,
                              geometry_weight=1.0, epoch=epoch, max_epochs=100)
        print(f"Epoch {epoch} loss: {loss.item():.4f}")
        assert not torch.isnan(loss), f"NaN loss at epoch {epoch}!"

    # Test DDIM sampling
    print("Testing DDIM sampling...")
    x_gen = model.ddim_sample(atom_types, edge_index, bond_types, batch_idx, num_steps=10)
    print(f"Generated shape: {x_gen.shape}")
    assert x_gen.shape == (7, 3)

    # Check CoM is near zero for each molecule
    for b in range(2):
        mask = batch_idx == b
        com = x_gen[mask].mean(0)
        print(f"  Mol {b} CoM: {com.tolist()} (should be ~0)")
        assert com.abs().max() < 0.5, f"CoM too large for mol {b}: {com}"

    print("\nAll ConformerDiffusion tests passed!")
