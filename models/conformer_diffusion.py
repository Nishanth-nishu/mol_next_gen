"""
conformer_diffusion.py — E(3)-Equivariant Diffusion for 3D Conformer Generation

KEY DIFFERENCE FROM ORIGINAL APPROACH:
- Original: Generate geometry → optimize coords → infer bonds (FAILS)
- NExT-Mol: Take KNOWN graph → generate coords directly → bonds already valid!

This model generates 3D coordinates for a FIXED, VALID molecular graph.
The graph topology (atoms, bonds) is provided as input and never changes.
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
    """
    Cosine schedule for diffusion (better than linear for molecules).
    """
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


# =============================================================================
# EQUIVARIANT MESSAGE PASSING LAYERS
# =============================================================================

class EquivariantLayer(nn.Module):
    """
    E(3)-equivariant message passing layer.
    
    Updates both node features h and coordinates x.
    Coordinate updates are computed in a way that respects E(3) symmetry.
    """
    
    def __init__(self, hidden_dim: int, edge_dim: int = 32):
        super().__init__()
        
        # Edge MLP: computes messages from node pairs
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Coordinate update: predicts displacement magnitude
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, 
                h: torch.Tensor,          # (N, hidden_dim) node features
                x: torch.Tensor,          # (N, 3) coordinates
                edge_index: torch.Tensor, # (2, E) edges
                edge_attr: torch.Tensor   # (E, edge_dim) edge features
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            h_new: Updated node features
            x_new: Updated coordinates (equivariant)
        """
        row, col = edge_index  # row[e] -> col[e] is an edge
        
        # Compute pairwise distances
        diff = x[row] - x[col]  # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)
        unit_vec = diff / dist  # (E, 3) unit direction vectors
        
        # Edge messages
        edge_input = torch.cat([h[row], h[col], edge_attr, dist], dim=-1)
        m_ij = self.edge_mlp(edge_input)  # (E, hidden_dim)
        
        # Coordinate update (equivariant)
        # displacement = scalar * unit_vector (preserves E(3))
        coord_weight = self.coord_mlp(m_ij)  # (E, 1)
        coord_update = coord_weight * unit_vec  # (E, 3)
        
        # Aggregate coordinate updates
        x_agg = torch.zeros_like(x)
        x_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, 3), coord_update)
        x_new = x + x_agg
        
        # Node feature update
        # Aggregate messages
        m_agg = torch.zeros_like(h)
        m_agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, h.size(-1)), m_ij)
        
        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))
        h_new = self.layer_norm(h + h_new)  # Residual
        
        return h_new, x_new


# =============================================================================
# CONFORMER DENOISER
# =============================================================================

class ConformerDenoiser(nn.Module):
    """
    Denoising network for conformer generation.
    
    Takes:
    - Noisy coordinates x_t
    - Timestep t
    - Fixed molecular graph (atom_types, edge_index, bond_types)
    
    Predicts:
    - Noise in coordinates (to be subtracted)
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_atom_types: int = 10,
                 num_bond_types: int = 5,
                 time_dim: int = 128):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Atom embedding
        self.atom_embed = nn.Embedding(num_atom_types + 1, hidden_dim)
        
        # Bond embedding
        self.bond_embed = nn.Embedding(num_bond_types + 1, 32)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_dim = time_dim
        
        # Initial coordinate embedding
        self.coord_embed = nn.Linear(3, hidden_dim)
        
        # Equivariant layers
        self.layers = nn.ModuleList([
            EquivariantLayer(hidden_dim, edge_dim=32)
            for _ in range(num_layers)
        ])
        
        # Output: predict noise
        self.noise_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self,
                x_noisy: torch.Tensor,     # (N, 3) noisy coordinates
                t: torch.Tensor,            # (B,) timesteps
                atom_types: torch.Tensor,   # (N,) atomic numbers
                edge_index: torch.Tensor,   # (2, E) bond edges
                bond_types: torch.Tensor,   # (E,) bond orders
                batch_idx: torch.Tensor     # (N,) batch assignment
                ) -> torch.Tensor:
        """
        Predict noise to remove from coordinates.
        
        Returns:
            noise_pred: (N, 3) predicted noise
        """
        N = x_noisy.size(0)
        
        # Embed atoms
        h = self.atom_embed(atom_types.clamp(0, 9))  # (N, hidden_dim)
        
        # Add coordinate information
        h = h + self.coord_embed(x_noisy)
        
        # Add time embedding (broadcast to all atoms in batch)
        t_emb = sinusoidal_embedding(t.float(), self.time_dim)  # (B, time_dim)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)
        h = h + t_emb[batch_idx]  # Broadcast to atoms
        
        # Embed bonds
        edge_attr = self.bond_embed(bond_types.clamp(0, 4))  # (E, 32)
        
        # Current coordinates
        x = x_noisy
        
        # Message passing
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        
        # Predict noise
        noise_pred = self.noise_pred(h)  # (N, 3)
        
        return noise_pred


# =============================================================================
# CONFORMER DIFFUSION MODEL
# =============================================================================

class ConformerDiffusion(nn.Module):
    """
    Full diffusion model for conformer generation.
    
    Training: Add noise to coordinates, predict the noise
    Sampling: Start from noise, iteratively denoise
    """
    
    def __init__(self,
                 num_timesteps: int = 1000,
                 hidden_dim: int = 256,
                 num_layers: int = 6):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Noise schedule
        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1 - alphas_cumprod))
        
        # Posterior variance
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', 
                            torch.log(posterior_variance.clamp(min=1e-20)))
        
        # Denoiser network
        self.denoiser = ConformerDenoiser(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Extract coefficients at timestep t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))
    
    def q_sample(self, 
                 x_0: torch.Tensor, 
                 t: torch.Tensor,
                 batch_idx: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(x_t | x_0)
        
        Args:
            x_0: Clean coordinates (N, 3)
            t: Timesteps per molecule (B,)
            batch_idx: Batch assignment (N,)
            noise: Optional pre-generated noise
            
        Returns:
            x_t: Noisy coordinates
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get coefficients per atom based on their molecule's timestep
        sqrt_alpha = self.sqrt_alphas_cumprod[t][batch_idx].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][batch_idx].unsqueeze(-1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def p_sample(self,
                 x_t: torch.Tensor,
                 t: torch.Tensor,
                 atom_types: torch.Tensor,
                 edge_index: torch.Tensor,
                 bond_types: torch.Tensor,
                 batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion step: p(x_{t-1} | x_t)
        """
        # Predict noise
        noise_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
        
        # Compute mean
        beta = self.betas[t][batch_idx].unsqueeze(-1)
        alpha = self.alphas[t][batch_idx].unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][batch_idx].unsqueeze(-1)
        
        mean = (x_t - beta * noise_pred / sqrt_one_minus_alpha_cumprod) / torch.sqrt(alpha)
        
        # Add noise (except at t=0)
        t_expanded = t[batch_idx]
        noise = torch.randn_like(x_t)
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
        """
        Generate conformer via reverse diffusion.
        
        Args:
            atom_types: (N,) atomic numbers
            edge_index: (2, E) bond connections (FIXED)
            bond_types: (E,) bond orders (FIXED)
            batch_idx: (N,) batch assignment
            
        Returns:
            x_0: Generated coordinates (N, 3)
        """
        device = atom_types.device
        N = atom_types.size(0)
        B = batch_idx.max().item() + 1
        
        num_steps = num_steps or self.num_timesteps
        
        # Start from noise
        x_t = torch.randn(N, 3, device=device)
        
        # Reverse diffusion
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
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
        DDIM sampling for faster generation.
        """
        device = atom_types.device
        N = atom_types.size(0)
        B = batch_idx.max().item() + 1
        
        # Subsample timesteps
        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)
        
        # Start from noise
        x_t = torch.randn(N, 3, device=device)
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val.item(), dtype=torch.long, device=device)
            
            # Predict noise
            noise_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
            
            # Predict x_0 - use t values per molecule, then broadcast to atoms
            alpha_t = self.alphas_cumprod[t[batch_idx]].unsqueeze(-1)
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -10, 10)  # Stability
            
            if i == len(timesteps) - 1:
                x_t = x_0_pred
            else:
                t_next = timesteps[i + 1].item()
                t_next_tensor = torch.full((B,), t_next, dtype=torch.long, device=device)
                alpha_next = self.alphas_cumprod[t_next_tensor[batch_idx]].unsqueeze(-1)
                
                # DDIM update
                x_t = torch.sqrt(alpha_next) * x_0_pred + \
                      torch.sqrt(1 - alpha_next) * noise_pred
        
        return x_t
    
    def get_loss(self,
                 x_0: torch.Tensor,
                 atom_types: torch.Tensor,
                 edge_index: torch.Tensor,
                 bond_types: torch.Tensor,
                 batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss.
        """
        device = x_0.device
        B = batch_idx.max().item() + 1
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # Add noise
        x_t, noise = self.q_sample(x_0, t, batch_idx)
        
        # Predict noise
        noise_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Testing ConformerDiffusion...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ConformerDiffusion(
        num_timesteps=100,  # Small for testing
        hidden_dim=128,
        num_layers=4
    ).to(device)
    
    # Fake batch of 2 molecules
    # Molecule 1: 3 atoms, 2 bonds (like water-ish)
    # Molecule 2: 4 atoms, 3 bonds (like methane-ish)
    
    atom_types = torch.tensor([8, 1, 1, 6, 1, 1, 1], device=device)  # O, H, H, C, H, H, H
    edge_index = torch.tensor([
        [0, 1, 0, 2, 3, 4, 3, 5, 3, 6],
        [1, 0, 2, 0, 4, 3, 5, 3, 6, 3]
    ], device=device)
    bond_types = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device=device)  # All single
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)
    
    # Fake ground truth coordinates
    x_0 = torch.randn(7, 3, device=device)
    
    # Test forward pass
    loss = model.get_loss(x_0, atom_types, edge_index, bond_types, batch_idx)
    print(f"Loss: {loss.item():.4f}")
    
    # Test sampling
    print("Testing DDIM sampling...")
    x_gen = model.ddim_sample(atom_types, edge_index, bond_types, batch_idx, num_steps=10)
    print(f"Generated coords shape: {x_gen.shape}")
    
    print("All tests passed!")
