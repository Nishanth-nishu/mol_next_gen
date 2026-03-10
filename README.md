# NExT-Mol Gen — E(3)-Equivariant Diffusion for 3D Molecular Conformer Generation

> **Experiment:** `09-03-2026-Exp-3 (stable_full_constraints)`  
> **Status:** Active training · SLURM job `nextmol_v3`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Research Contributions](#2-core-research-contributions)
3. [Mathematical Foundations](#3-mathematical-foundations)
   - [3.1 Denoising Diffusion Probabilistic Models (DDPM)](#31-denoising-diffusion-probabilistic-models-ddpm)
   - [3.2 E(3) Equivariance — Why It Matters for Molecules](#32-e3-equivariance--why-it-matters-for-molecules)
   - [3.3 Cosine Noise Schedule](#33-cosine-noise-schedule)
   - [3.4 Center-of-Mass Removal (EDM Fix)](#34-center-of-mass-removal-edm-fix)
   - [3.5 Min-SNR Loss Weighting](#35-min-snr-loss-weighting)
   - [3.6 DDIM Accelerated Sampling](#36-ddim-accelerated-sampling)
4. [Model Architecture — Deep Dive](#4-model-architecture--deep-dive)
   - [4.1 Input Representation](#41-input-representation)
   - [4.2 Equivariant Message-Passing Layer (EGNN)](#42-equivariant-message-passing-layer-egnn)
   - [4.3 ConformerDenoiser](#43-conformerdenoiser)
   - [4.4 ConformerDiffusion (Full Model)](#44-conformerdiffusion-full-model)
5. [Geometry Constraint System](#5-geometry-constraint-system)
   - [5.1 Bond Length Loss (MMFF94)](#51-bond-length-loss-mmff94)
   - [5.2 Bond Angle Loss (Hybridization-Aware)](#52-bond-angle-loss-hybridization-aware)
   - [5.3 Torsion (Dihedral) Loss (OPLS-AA)](#53-torsion-dihedral-loss-opls-aa)
   - [5.4 VDW Repulsion Loss](#54-vdw-repulsion-loss)
   - [5.5 Planarity Loss (Aromatic Rings)](#55-planarity-loss-aromatic-rings)
   - [5.6 Chirality Loss (Tetrahedral Volume)](#56-chirality-loss-tetrahedral-volume)
   - [5.7 Ring Strain Loss](#57-ring-strain-loss)
6. [Training Pipeline — Data Flow](#6-training-pipeline--data-flow)
   - [6.1 Dataset Format (QM9-100K)](#61-dataset-format-qm9-100k)
   - [6.2 Data Loading and Graph Construction](#62-data-loading-and-graph-construction)
   - [6.3 Training Loop (v3)](#63-training-loop-v3)
   - [6.4 Loss Computation (Full Decomposition)](#64-loss-computation-full-decomposition)
   - [6.5 LR Schedule: Warmup + Cosine Annealing](#65-lr-schedule-warmup--cosine-annealing)
7. [Inference — Sampling Pipeline](#7-inference--sampling-pipeline)
8. [Validity Evaluation](#8-validity-evaluation)
9. [Research Papers Referenced](#9-research-papers-referenced)
10. [Bug History and Lessons Learned](#10-bug-history-and-lessons-learned)
11. [Project Structure](#11-project-structure)
12. [Running the Experiment](#12-running-the-experiment)

---

## 1. Project Overview

This project implements a **text-conditioned 3D molecular conformer generation** system
using E(3)-equivariant diffusion. Given a molecular graph (atom types + bond topology),
the model learns to generate a valid 3D conformation of that molecule.

**Why is 3D conformation important?**  
A molecule's biological function (binding to a receptor, crossing membranes) depends not
just on its atomic composition (SMILES) but on its precise 3D shape. Two molecules with
the same graph topology but different 3D conformations can have completely different
pharmacological effects.

**What makes this hard?**
- The output lives in 3D Euclidean space, which must be **translation, rotation, and
  reflection invariant** (E(3) symmetry group).
- Small violations of bond lengths/angles result in chemically implausible molecules.
- Torsion angles (rotatable bonds) encode the major conformational degrees of freedom.

---

## 2. Core Research Contributions

Our implementation distils and combines techniques from 5 key papers:

| Technique | Paper | What we use |
|---|---|---|
| E(3) equivariant GNN | [EGNN (Satorras et al. 2021)](#9-research-papers-referenced) | EquivariantLayer message-passing |
| DDPM framework | [DDPM (Ho et al. 2020)](#9-research-papers-referenced) | Forward/reverse diffusion on 3D coords |
| CoM subspace + SNR loss | [EDM (Hoogeboom et al. 2022)](#9-research-papers-referenced) | CoM removal, SNR-weighted MSE |
| Fast inference | [DDIM (Song et al. 2020)](#9-research-papers-referenced) | 50-step deterministic sampling |
| Min-SNR weighting | [Min-SNR (Hang et al. 2023)](#9-research-papers-referenced) | Clipped SNR loss weights |
| Immediate geometry supervision | [EQGAT-diff (Le et al. 2024)](#9-research-papers-referenced) | Geometry loss from epoch 1 |
| Geometry constraints active early | [GCDM (Morehead et al. 2023)](#9-research-papers-referenced) | No curriculum ramp |
| MMFF94 force-field targets | [Halgren 1996](#9-research-papers-referenced) | Bond/angle ideal values |
| OPLS-AA torsion parameters | [Jorgensen et al. 1996](#9-research-papers-referenced) | Torsion potential V1,V2,V3 |
| Torsion angle diffusion | [TorDiff (Jing et al. 2022)](#9-research-papers-referenced) | Torsion loss motivation |

---

## 3. Mathematical Foundations

### 3.1 Denoising Diffusion Probabilistic Models (DDPM)

**Reference:** *Ho et al. "Denoising Diffusion Probabilistic Models," NeurIPS 2020.*

DDPM defines two Markov chains:

#### Forward Process (Data → Noise)

Given clean coordinates $\mathbf{x}_0 \in \mathbb{R}^{N \times 3}$ (one entry per atom),
the forward process gradually adds Gaussian noise over $T=1000$ steps:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\right)$$

Using the reparametrisation trick, we can sample $\mathbf{x}_t$ directly from $\mathbf{x}_0$
in one step (no Markov chain needed at training time):

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I}\right)$$

where:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s \quad$ (cumulative product)

**In code** (`q_sample`):
```python
x_t = sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise
```

This is implemented for a batch of molecules, where `t` is per-molecule but `noise` is
per-atom. The noise term has its **center-of-mass removed** (see §3.4).

#### Reverse Process (Noise → Data)

The model learns to reverse this process. The key insight from DDPM is that the reverse
transition $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ is also Gaussian when the noise
schedule is small, and its mean can be parameterised by predicting the **noise** $\epsilon_t$:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\, \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\, \sigma_t^2 \mathbf{I}\right)$$

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$

where $\hat{\epsilon}_\theta$ is our neural network (ConformerDenoiser).

The **training objective** is simply:

$$\mathcal{L}_\text{MSE} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}\left[\|\epsilon - \hat{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

---

### 3.2 E(3) Equivariance — Why It Matters for Molecules

The E(3) group consists of translations, rotations, and reflections in 3D space.
A function $f$ is **E(3)-equivariant** if:

$$f(R\mathbf{x} + \mathbf{t}) = R\,f(\mathbf{x}) + \mathbf{t} \quad \forall R \in O(3),\; \mathbf{t} \in \mathbb{R}^3$$

**Why this is crucial:** If we rotate a molecule and then denoise it, we should get the
same result as denoising first and then rotating. Without this, the model must memorise all
possible orientations of each molecule — an impossible task.

**How EGNN achieves it:**  
*(Reference: Satorras et al. "E(n) Equivariant Graph Neural Networks," ICML 2021)*

The equivariant layer updates positions as a weighted sum of **unit displacement vectors**:

$$\mathbf{x}_i^{(l+1)} = \mathbf{x}_i^{(l)} + \frac{1}{\text{deg}(i)+1}\sum_{j \in \mathcal{N}(i)} \phi_x\!\left(\mathbf{m}_{ij}\right) \cdot \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\| + \epsilon}$$

This update is equivariant because:
1. Unit vectors transform as $R(\mathbf{x}_i - \mathbf{x}_j)/\|\cdot\| \to R \cdot (\cdots)$
2. Scalar weights $\phi_x(\mathbf{m}_{ij})$ are invariant (distances don't change under rotation)

Node features $\mathbf{h}_i$ are **invariant** (scalars), updated via:

$$\mathbf{m}_{ij} = \phi_e\!\left(\mathbf{h}_i, \mathbf{h}_j, \|\mathbf{x}_i - \mathbf{x}_j\|^2, \mathbf{e}_{ij}\right)$$
$$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\!\left(\mathbf{h}_i + \phi_h\!\left(\mathbf{h}_i, \textstyle\sum_{j} \mathbf{m}_{ij}\right)\right)$$

where $\mathbf{e}_{ij}$ is the bond-type embedding (invariant feature on the edge).

---

### 3.3 Cosine Noise Schedule

**Reference:** *Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models," ICML 2021.*

The linear noise schedule from the original DDPM paper causes near-zero noise at early
timesteps and near-complete noise at late timesteps, wasting training capacity.

The **cosine schedule** spreads the information more evenly:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

where $s = 0.008$ prevents $\beta_t$ from becoming too small near $t=0$.

Betas are derived as:
$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$

clamped to $[0.0001, 0.9999]$.

**In code** (`cosine_beta_schedule`):
```python
alphas_cumprod = cos((t/T + s) / (1+s) * π/2)²
betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
```

---

### 3.4 Center-of-Mass Removal (EDM Fix)

**Reference:** *Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D," ICML 2022.*

**The problem:** Standard DDPM diffuses in the full 3D space $\mathbb{R}^{3N}$.
But molecular conformations are **translation-invariant**: moving all atoms by the same
vector $\mathbf{t}$ gives the same molecule.This creates an ill-posed problem — the model
must also learn to fix the global position, which consumes model capacity and causes instability.

**The fix:** Constrain all diffusion to the **zero-CoM subspace**:

$$\mathcal{M}_0 = \left\{\mathbf{x} \in \mathbb{R}^{N\times 3} : \sum_{i=1}^{N} \mathbf{x}_i = \mathbf{0}\right\}$$

This is enforced by removing the center-of-mass at **every step**:

$$\text{CoM-remove}(\mathbf{x}) = \mathbf{x} - \frac{1}{N}\sum_i \mathbf{x}_i \cdot \mathbf{1}^\top$$

We apply this:
1. To the **noise** $\epsilon$ before forward diffusion (so $\epsilon \in \mathcal{M}_0$)
2. To $\mathbf{x}_t$ after the forward step
3. To the predicted mean $\boldsymbol\mu_\theta$ in the reverse step
4. To $\mathbf{x}_0^\text{pred}$ during inference

**Why this helps:** The model now only needs to learn the *shape* of the molecule, not its
absolute position in space. The loss landscape becomes much smoother.

**In code** — applied per-molecule in a batch:
```python
def remove_com(x, batch_idx):
    mol_means = scatter_mean(x, batch_idx)   # (B, 3)
    return x - mol_means[batch_idx]          # subtract per-atom CoM
```

---

### 3.5 Min-SNR Loss Weighting

**Reference:** *Hang et al. "Efficient Diffusion Training via Min-SNR Weighting Strategy," ICCV 2023.*

**The problem:** At different timesteps, the signal-to-noise ratio (SNR) varies enormously:

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

- At $t \approx 0$: SNR is huge (mostly signal) → easy, well-supervised
- At $t \approx T$: SNR ≈ 0 (pure noise) → very hard, high-loss timesteps

Without reweighting, the standard MSE loss over-emphasises high-noise timesteps which
carry little useful information, causing **instability and slow convergence**.

**Min-SNR weighting** clips the SNR weight to prevent high-noise timesteps from dominating:

$$w(t) = \min\!\left(\text{SNR}(t),\; \gamma\right) / \text{SNR}(t) = \min\!\left(1,\; \frac{\gamma}{\text{SNR}(t)}\right)$$

where $\gamma = 5$ (empirically optimal across models). This weight:
- $= 1$ when $\text{SNR}(t) \leq \gamma$ (standard weight at most timesteps)
- $< 1$ when $\text{SNR}(t) > \gamma$ (down-weights clean/easy timesteps)

**In code:**
```python
snr_t = self.snr[t][batch_idx]         # SNR per atom
w_t   = torch.minimum(snr_t, γ) / snr_t   # Min-SNR weight
mse_loss = (w_t * per_atom_mse).mean()
```

**Geometry loss gating via SNR-mean:**  
We also gate the geometry loss by the batch-mean SNR weight. When SNR is high (low noise),
the predicted $\mathbf{x}_0^\text{pred}$ is accurate, so geometry supervision is informative.
When SNR is low (high noise), $\mathbf{x}_0^\text{pred}$ is inaccurate and geometry supervision
would penalise meaningless predictions:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{MSE} + \underbrace{\bar{w}}_{\text{SNR-mean gate}} \cdot \lambda_\text{geo} \cdot \mathcal{L}_\text{geo}$$

---

### 3.6 DDIM Accelerated Sampling

**Reference:** *Song et al. "Denoising Diffusion Implicit Models," ICLR 2021.*

Full DDPM sampling requires $T = 1000$ forward passes of the denoiser — slow at inference.
DDIM reformulates the reverse process as a **non-Markovian** deterministic process that
can be integrated with large steps:

Given a subsequence of timesteps $\tau_1 > \tau_2 > \cdots > \tau_S$ with $S \ll T$:

1. Predict $\mathbf{x}_0$ from $\mathbf{x}_{\tau_i}$:
$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_{\tau_i} - \sqrt{1-\bar\alpha_{\tau_i}}\,\hat\epsilon_\theta}{\sqrt{\bar\alpha_{\tau_i}}}$$

Soft-clamped to $[-10, 10]$ via $\tanh$ to prevent extreme values.

2. Compute the "direction" pointing to $\mathbf{x}_{\tau_i}$:
$$\text{dir} = \sqrt{1-\bar\alpha_{\tau_{i+1}} - \sigma^2} \cdot \hat\epsilon_\theta$$

3. Update:
$$\mathbf{x}_{\tau_{i+1}} = \sqrt{\bar\alpha_{\tau_{i+1}}}\,\hat{\mathbf{x}}_0 + \text{dir} + \sigma \cdot \epsilon$$

where $\sigma = \eta\sqrt{\frac{1-\bar\alpha_{\tau_{i+1}}}{1-\bar\alpha_{\tau_i}}} \cdot \sqrt{1 - \bar\alpha_{\tau_i}/\bar\alpha_{\tau_{i+1}}}$

**We use $\eta = 0$ (deterministic)**, giving exact samples in just **50 steps** (20× speedup).

**Geometry-guided variant** (`guided_sample`): After each DDIM step, we run 3 gradient
descent iterations on $\hat{\mathbf{x}}_0$ using the geometry loss as an energy function,
nudging the prediction towards chemically valid geometry before re-encoding into the
next diffusion step.

---

## 4. Model Architecture — Deep Dive

### 4.1 Input Representation

Each molecule in a batch is represented as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

| Component | Type | Description |
|---|---|---|
| Atom types $z_i$ | `Long[N]` | Atomic number (1=H, 6=C, 7=N, 8=O, ...) |
| Coordinates $\mathbf{x}_i$ | `Float[N, 3]` | 3D position in Ångströms |
| Edge index | `Long[2, E]` | Undirected bonds as directed pairs (both directions) |
| Bond types $b_{ij}$ | `Long[E]` | 1=single, 2=double, 3=triple, 4=aromatic |
| Batch index | `Long[N]` | Which molecule each atom belongs to |

For a batch of $B$ molecules, atoms from all molecules are concatenated; the batch index
tracks molecule membership.

**Atom embedding:** $\mathbf{h}_i^{(0)} = \text{Embed}(z_i; 54 \to d_h)$ — a learnable
lookup table covering the full periodic table from H(1) to I(53). The embedding is a
learnable matrix $W_\text{atom} \in \mathbb{R}^{54 \times d_h}$.

---

### 4.2 Equivariant Message-Passing Layer (EGNN)

Each `EquivariantLayer` performs one round of E(3)-equivariant message passing.

**Inputs:** Node features $\mathbf{H}^{(l)} \in \mathbb{R}^{N \times d_h}$,
coordinates $\mathbf{X}^{(l)} \in \mathbb{R}^{N \times 3}$, edge attributes $\mathbf{E} \in \mathbb{R}^{E \times d_e}$.

**Step 1 — Edge messages** (invariant MLP):
$$\mathbf{m}_{ij} = \phi_e\!\left(\mathbf{h}_i \| \mathbf{h}_j \| \|\mathbf{x}_i - \mathbf{x}_j\|^2 \| \mathbf{e}_{ij}\right)$$

where $\phi_e$ is a 3-layer MLP with SiLU activations and **Dropout(0.1)**. The input size is
$2d_h + 1 + d_e$ (features + squared distance + bond embedding).

**Step 2 — Coordinate update** (equivariant):
$$\Delta\mathbf{x}_i = \frac{1}{\text{deg}(i)+1}\sum_{j \in \mathcal{N}(i)} \phi_x(\mathbf{m}_{ij}) \cdot \hat{\mathbf{u}}_{ij}$$

where $\hat{\mathbf{u}}_{ij} = (\mathbf{x}_i - \mathbf{x}_j)/\|\mathbf{x}_i - \mathbf{x}_j\|$ is the unit displacement
vector (equivariant), and $\phi_x: \mathbb{R}^{d_h} \to \mathbb{R}$ is a scalar MLP with Tanh output.

The degree normalization $(\text{deg}(i)+1)$ **prevents instability in dense molecules**
— without it, coordinate updates grow with molecule size.

$$\mathbf{x}_i^{(l+1)} = \mathbf{x}_i^{(l)} + \Delta\mathbf{x}_i$$

**Step 3 — Node feature update** (invariant MLP with residual):
$$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\!\left(\mathbf{h}_i^{(l)} + \phi_h\!\left(\mathbf{h}_i^{(l)} \| \textstyle\sum_j \mathbf{m}_{ij}\right)\right)$$

where $\phi_h$ is a 2-layer MLP with SiLU activations and **Dropout(0.1)**.

The **Dropout(0.1)** was added in v3 to fix overfitting (the train/val MSE gap widened after
epoch 150 in v1/v2 — classic sign of overfitting). Dropout breaks co-adaptation between
neurons and significantly reduced the gap.

---

### 4.3 ConformerDenoiser

The full denoiser is a stack of $L$ equivariant layers with shared-time conditioning.

**Architecture:**

```
Input:
  z_i (atom type)  → Embedding(54, d_h)    → h_i^atom           [N, d_h]
  x_t (noisy xyz)  → Linear(3, d_h)        → h_i^coord          [N, d_h]
  t (timestep)     → SinEmbed(d_t)         [B, d_t]
                   → MLP(d_t → d_h)        → t_emb              [B, d_h]
  b_ij (bond type) → Embedding(5, d_e)     → e_ij               [E, d_e]

Initial node features:
  h_i = h_i^atom + h_i^coord + t_emb[batch_idx[i]]

Equivariant Layers (repeated L times):
  h^(l), x^(l) → EquivariantLayer → h^(l+1), x^(l+1)

Output head:
  h_i^(L) → Linear(d_h, d_h) → SiLU → Linear(d_h, 3) → ε̂_i    [N, 3]
```

**Key design choices:**
- Time conditioning is **additive** to node features (not FiLM/AdaLN), keeping the architecture simple
- The coordinate stream $\mathbf{x}^{(l)}$ carries pure positional information and is updated equivariantly
- The feature stream $\mathbf{h}^{(l)}$ is invariant and carries chemistry information

**Default hyperparameters (Exp-3):**

| Parameter | Value |
|---|---|
| Hidden dim $d_h$ | 512 |
| Num layers $L$ | 10 |
| Edge dim $d_e$ | 64 |
| Time dim $d_t$ | 256 |
| Timesteps $T$ | 1000 |
| Dropout | 0.1 |
| **Total params** | **~22M** |

---

### 4.4 ConformerDiffusion (Full Model)

The `ConformerDiffusion` class wraps the denoiser with the DDPM framework:

```
ConformerDiffusion
├── Buffers (precomputed, registered with model state):
│   ├── betas              [T]          β_t (cosine schedule)
│   ├── alphas             [T]          α_t = 1 - β_t
│   ├── alphas_cumprod     [T]          ᾱ_t = ∏α
│   ├── sqrt_alphas_cumprod[T]          √ᾱ_t
│   ├── sqrt_one_minus_*   [T]          √(1-ᾱ_t)
│   ├── posterior_variance [T]          σ²_t = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)
│   └── snr                [T]          ᾱ_t / (1-ᾱ_t)
│
├── ConformerDenoiser      (ε̂_θ)
└── GeometryConstraints    (differentiable chemistry constraints)
```

**Three sampling modes:**
1. `sample()` — Full DDPM (1000 steps, slow, highest quality)
2. `ddim_sample()` — DDIM with 50 steps, `η=0` (deterministic, 20× faster)
3. `guided_sample()` — DDIM + geometry gradient guidance at each step

---

## 5. Geometry Constraint System

The file `models/geometry_constraints.py` implements differentiable chemistry-aware
constraints. All constraints are active from **epoch 1** in v3.

### 5.1 Bond Length Loss (MMFF94)

**Reference:** *Halgren, "Merck molecular force field," J. Comput. Chem. 1996.*

For each bond $(i, j)$ with bond order $b$, we look up the ideal length $d^*_{ij}$ from
the MMFF94 parameter table (pre-built as a $54 \times 54 \times 5$ tensor, enabling O(1)
vectorized lookup):

$$\mathcal{L}_\text{bond} = \lambda_\text{bond} \cdot \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left(\|\mathbf{x}_i - \mathbf{x}_j\| - d^*_{z_i, z_j, b_{ij}}\right)^2$$

Example ideal lengths from MMFF94:

| Bond | Single | Double | Triple | Aromatic |
|---|---|---|---|---|
| C-C | 1.54 Å | 1.34 Å | 1.20 Å | 1.40 Å |
| C-N | 1.47 Å | 1.29 Å | 1.16 Å | 1.34 Å |
| C-O | 1.43 Å | 1.23 Å | — | 1.36 Å |
| C-H | 1.09 Å | — | — | — |
| O-H | 0.96 Å | — | — | — |

---

### 5.2 Bond Angle Loss (Hybridization-Aware)

For each atom $j$ with neighbors $i$ and $k$, we compute the ideal bond angle based on
its **hybridization** (inferred from atom type + bond types):

| Hybridization | Ideal angle | Example |
|---|---|---|
| $\text{sp}^3$ | 109.5° | methane C, water O |
| $\text{sp}^2$ | 120.0° | ethylene C, amide N |
| $\text{sp}$ | 180.0° | acetylene C |
| aromatic | 120.0° | benzene C |

$$\mathcal{L}_\text{angle} = \lambda_\text{angle} \cdot \frac{1}{|\mathcal{A}|} \sum_{(i,j,k) \in \mathcal{A}} \left(\angle(\mathbf{x}_i, \mathbf{x}_j, \mathbf{x}_k) - \theta^*_j\right)^2$$

where $\angle(\mathbf{x}_i, \mathbf{x}_j, \mathbf{x}_k) = \arccos\!\left(\hat{\mathbf{v}}_{ji} \cdot \hat{\mathbf{v}}_{jk}\right)$.

---

### 5.3 Torsion (Dihedral) Loss (OPLS-AA)

**Reference:** *Jorgensen et al. "Development and Testing of the OPLS All-Atom Force Field," 1996.*

For each rotatable bond $j$-$k$, we define a torsion quadruple $(i, j, k, l)$ and compute
the dihedral angle $\phi$ as the angle between the planes $(i,j,k)$ and $(j,k,l)$:

$$\phi = \text{atan2}\!\left(\sin\phi, \cos\phi\right)$$

where $\cos\phi = (\hat{\mathbf{n}}_1 \cdot \hat{\mathbf{n}}_2)$ and
$\hat{\mathbf{n}}_1, \hat{\mathbf{n}}_2$ are the normals to the respective planes.

The torsion energy is the **OPLS cosine series**:

$$E_\text{tors}(\phi) = \frac{V_1}{2}(1+\cos\phi) + \frac{V_2}{2}(1-\cos 2\phi) + \frac{V_3}{2}(1+\cos 3\phi)$$

Parameters depend on hybridization of the central bond atoms:

| Central bond hybridization | $V_1$ | $V_2$ | $V_3$ | Notes |
|---|---|---|---|---|
| sp³–sp³ | 0 | 0 | 1.0 | 3-fold rotation (C-C single) |
| sp³–sp² | 0 | 2.0 | 0 | 2-fold, prefers 180° |
| sp²–sp² | 0 | 6.0 | 0 | Conjugated, strong barrier |
| aromatic–aromatic | 0 | 10.0 | 0 | Planarity enforced |

$$\mathcal{L}_\text{tors} = \lambda_\text{tors} \cdot \frac{1}{|\mathcal{T}|} \sum_{(i,j,k,l) \in \mathcal{T}} E_\text{tors}(\phi_{ijkl})$$

---

### 5.4 VDW Repulsion Loss

Prevents atomic clashes using Van der Waals radii from Bondi (1964):

$$r_\text{clash}(i, j) = 0.70 \cdot (r_\text{vdw}(z_i) + r_\text{vdw}(z_j))$$

**1-2 pairs** (directly bonded) and **1-3 pairs** (bonded through one atom) are
**excluded** from repulsion — these are naturally at their correct distance via the
bond/angle losses.

For all other (1-4+) same-molecule atom pairs:

$$\mathcal{L}_\text{rep} = \lambda_\text{rep} \cdot \frac{1}{N_\text{clash}} \sum_{\substack{i < j \\ d_{ij} < r_\text{clash}}} \left(r_\text{clash}(i,j) - d_{ij}\right)^2$$

---

### 5.5 Planarity Loss (Aromatic Rings)

**Novel contribution for v1's soft-constraint experiment.**

Aromatic rings must be planar (all atoms lie in one plane). We enforce this via **SVD**
of the centred ring atom positions to find the best-fit plane normal:

Given ring atoms $\{\mathbf{r}_k\}_{k=1}^{R}$ with centroid $\bar{\mathbf{r}}$:

$$\mathbf{C} = [\mathbf{r}_1 - \bar{\mathbf{r}},\; \ldots,\; \mathbf{r}_R - \bar{\mathbf{r}}]^\top \in \mathbb{R}^{R \times 3}$$
$$\mathbf{C} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top \quad \Rightarrow \quad \hat{\mathbf{n}} = V_{-1} \quad \text{(last row of } V^\top \text{, smallest singular value)}$$

Perpendicular distance of each atom from the plane:

$$\mathcal{L}_\text{plan} = \lambda_\text{plan} \cdot \text{mean}_k\left[(\hat{\mathbf{n}} \cdot (\mathbf{r}_k - \bar{\mathbf{r}}))^2\right]$$

---

### 5.6 Chirality Loss (Tetrahedral Volume)

Enforces **R/S stereochemistry** at sp³ carbon centres using the **signed scalar triple product**:

$$V_\text{signed} = (\mathbf{v}_1 - \mathbf{c}) \cdot \left[(\mathbf{v}_2 - \mathbf{c}) \times (\mathbf{v}_3 - \mathbf{c})\right]$$

where $\mathbf{c}$ is the chiral centre position and $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$
are three of its four neighbours. The sign $(+1\;\text{or}\;-1)$ encodes the handedness (R or S).

$$\mathcal{L}_\text{chir} = \lambda_\text{chir} \cdot \text{mean}\left[\text{ReLU}\!\left(-s \cdot V_\text{signed} + m\right)\right]$$

where $s \in \{+1,-1\}$ is the target sign and $m = 0.1\;\text{Å}^3$ is a margin.
This is a **margin hinge loss**: zero when the signed volume has the correct sign with margin,
penalises wrong handedness proportionally.

---

### 5.7 Ring Strain Loss

Small rings (3- and 4-membered) have tighter bond angles than sp³ VSEPR predicts:

| Ring size | Ideal internal angle | Example |
|---|---|---|
| Cyclopropane (3) | 60° | Epoxide |
| Cyclobutane (4) | 90° | Azetidine |

$$\mathcal{L}_\text{ring} = \lambda_\text{ring} \cdot \text{mean}_{a \in \text{ring}}\left[\left(\angle(\text{neighbours of } a) - \theta^*_\text{ring}\right)^2\right]$$

---

## 6. Training Pipeline — Data Flow

### 6.1 Dataset Format (QM9-100K)

The file `data/qm9_100k.jsonl` contains ~98,571 molecules from the QM9 dataset in JSONL
format (one JSON object per line). Each entry is **padded to 50 atoms**:

```json
{
  "smiles":     "[H]O[H]",
  "selfies":    "[H][O][H]",
  "coords":     [[x0,y0,z0], ..., [0,0,0]],   // 50 entries, padded with zeros
  "atom_types": [8, 1, 1, -1, -1, ...],       // 50 entries, -1 = padding
  "coord_mask": [1, 1, 1,  0,  0, ...],       // 50 entries, 1 = real atom
  "short_caption": "H2O with 1 heavy atoms...",
  "long_caption":  "Molecule H2O (3 atoms, ...) ..."
}
```

QM9 atom types (atomic numbers): H(1), C(6), N(7), O(8), F(9)  
All explicit H atoms are included, so molecule sizes range from 3 to 50 atoms.

---

### 6.2 Data Loading and Graph Construction

```
qm9_100k.jsonl
      │
      ▼
ConformerDataset.__init__()
  For each line:
    1. Parse JSON
    2. Unpad: use coord_mask to extract real atoms
    3. Validate: 0 < n_atoms ≤ max_atoms (50)
    4. Store {atom_types, coords, smiles, num_atoms}
  ──────────────────────────────────────────────
  ~98,000 molecules loaded

      │
      ▼ (on __getitem__)
_build_graph_from_smiles(smiles)
  RDKit: MolFromSmiles(smiles)   ← SMILES has explicit H, so no AddHs needed
  For each bond:
    bond.GetBondType() → {SINGLE:1, DOUBLE:2, TRIPLE:3, AROMATIC:4}
    Add both directions (undirected → directed pairs)
  Returns edge_index [2, E], bond_types [E]
  ──────────────────────────────────────────────
  Fallback: fully-connected graph if SMILES fails

      │
      ▼ collate_fn()
  Batches N molecules into one large graph by:
    1. Concatenating atom_types, coords, bond_types
    2. Offsetting edge_index by cumulative atom counts
    3. Building batch_idx tensor
  ──────────────────────────────────────────────
  Output: one batched graph tensor dict

      │
      ▼ Training/Validation
  DataLoader  (num_workers=4, shuffle=True for train)
  90/10 train/val split
```

---

### 6.3 Training Loop (v3)

```
for epoch in 1..300:
  ┌─────────────────── train_epoch() ──────────────────┐
  │ for each batch:                                     │
  │   1. remove_com(coords, batch_idx)                  │
  │   2. optimizer.zero_grad()                          │
  │   3. loss_dict = model.get_loss(...)                │
  │      ├── q_sample: add noise (CoM-aware)            │
  │      ├── denoiser forward pass                      │
  │      ├── MSE loss with Min-SNR weighting            │
  │      └── geometry loss (all 7 types, epoch 1+)     │
  │   4. loss_dict['total'].backward()                  │
  │   5. clip_grad_norm_(params, 1.0)                   │
  │   6. optimizer.step()                               │
  └─────────────────────────────────────────────────────┘
  
  ┌─────────────────── LR Schedule ────────────────────┐
  │ Epochs 1-5:   Linear warmup  0 → 3e-4              │
  │ Epochs 6-300: CosineAnnealingLR  3e-4 → 0          │
  └─────────────────────────────────────────────────────┘
  
  ┌─────────────────── validate() ─────────────────────┐
  │ Same as train_epoch but no grad, no optimizer step  │
  └─────────────────────────────────────────────────────┘
  
  ┌─────────────────── Checkpointing ──────────────────┐
  │ if val_mse < best_val_mse:                          │
  │   save conformer_best_mse.pt                        │ ← primary checkpoint
  │ every 25 epochs:                                    │
  │   save conformer_epochNNNN.pt                       │
  └─────────────────────────────────────────────────────┘
  
  ┌─────────────────── Validity Eval (every 25 ep) ────┐
  │ evaluate_validity(model, val_loader, 200 samples)   │
  │   → validity_rate, uniqueness, novelty              │
  │ if validity_rate > best_validity:                   │
  │   save conformer_best_validity.pt                   │ ← separate checkpoint
  └─────────────────────────────────────────────────────┘
```

---

### 6.4 Loss Computation (Full Decomposition)

The total training loss for one batch is:

$$\boxed{\mathcal{L}_\text{total} = \mathcal{L}_\text{MSE} + \bar{w} \cdot \lambda_\text{geo} \cdot \mathcal{L}_\text{geo}}$$

Where:

$$\mathcal{L}_\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} w(t_{m(i)}) \cdot \|\hat{\epsilon}_i - \epsilon_i\|^2$$

$$w(t) = \min\!\left(1,\; \frac{\gamma}{\text{SNR}(t)}\right),\quad \gamma=5,\quad \text{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$$

$$\mathcal{L}_\text{geo} = \mathcal{L}_\text{bond} + \mathcal{L}_\text{angle} + \mathcal{L}_\text{tors} + \mathcal{L}_\text{rep} + \mathcal{L}_\text{plan} + \mathcal{L}_\text{chir} + \mathcal{L}_\text{ring}$$

$$\lambda_\text{geo} = 0.01 \quad \text{(geometry\_weight=0.1 × 0.1 fixed scale)}$$

$$\bar{w} = \text{mean}_{i}[w(t_{m(i)})].\text{clamp}(\max=1.0) \quad \text{(SNR-mean gate)}$$

Geometry loss weight breakdown:

| Loss term | Weight $\lambda$ |
|---|---|
| Bond length | 1.0 |
| Angle | 0.5 |
| Torsion | 0.2 |
| Repulsion | 0.5 |
| Planarity | 0.5 |
| Chirality | 0.3 |
| Ring strain | 0.2 |

---

### 6.5 LR Schedule: Warmup + Cosine Annealing

**Reference:** *Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017.*

Cold-start with a high learning rate causes large, destabilising gradient steps in early
training when the model's predictions are worst.

**Linear warmup** (5 epochs): LR increases linearly from 0 to $\eta_\text{max}=3\times10^{-4}$:

$$\eta_\text{epoch} = \eta_\text{max} \cdot \frac{\text{epoch}}{W}, \quad \text{epoch} \leq W=5$$

**Cosine annealing** (epochs 6–300): LR decays smoothly to 0:

$$\eta_\text{epoch} = \frac{\eta_\text{max}}{2}\!\left(1 + \cos\!\left(\pi \cdot \frac{\text{epoch} - W}{T - W}\right)\right)$$

**Gradient clipping:** $\|\nabla \theta\|_2 \leq 1.0$ at every step (prevents gradient explosions
which were a primary source of divergence after epoch 150 in earlier experiments).

---

## 7. Inference — Sampling Pipeline

```
Input: molecular graph (atom_types, edge_index, bond_types)
       from a SMILES string (e.g., from val set or novel SMILES)

DDIM Sampling (50 steps, η=0):
  x_T ~ N(0, I)    # Gaussian noise in zero-CoM subspace

  for t in [T, T-20, T-40, ..., 0]:
    1. ε̂ = denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)
    2. x̂_0 = (x_t - √(1-ᾱ_t) * ε̂) / √ᾱ_t
    3. x̂_0 = tanh(x̂_0 / 10) * 10     # soft clamp
    4. x̂_0 = remove_com(x̂_0)

    [Geometry guidance, optional]:
    5. For 3 iterations:
       L = L_bond + L_repulsion + L_plan + L_chir + L_ring
       x̂_0 = x̂_0 - lr * ∇_{x̂_0} L

    6. x_{t-20} = √ᾱ_{t-20} * x̂_0
                + √(1 - ᾱ_{t-20}) * ε̂   # DDIM update

Output: x̂_0 ∈ R^{N×3}    # Generated 3D conformation
```

**Post-processing:**
1. Build RDKit molecule from (atom_types, edge_index, bond_types, x̂_0)
2. Call `Chem.SanitizeMol(mol)` — checks valence, aromaticity, etc.
3. Valid molecules → export as individual `.pdb` files in `pdb_files/` subdirectory
4. Generate `load_all_vmd.tcl` for batch visualisation in VMD

---

## 8. Validity Evaluation

**Why 200 samples?** With $n$ samples, validity estimation has standard error:

$$\sigma = \sqrt{\frac{p(1-p)}{n}}$$

At $p = 0.8$, $n=50$: $\sigma = 5.7\%$ → measurements 14% apart are indistinguishable.  
At $p = 0.8$, $n=200$: $\sigma = 2.8\%$ → 3.5× more reliable signal.

**Metrics computed:**

| Metric | Definition |
|---|---|
| **Validity rate** | Fraction that passes `Chem.SanitizeMol()` |
| **Uniqueness** | Fraction of valid molecules with unique canonical SMILES |
| **Novelty** | Fraction of unique molecules not in the training set |
| **Clash-free rate** | Fraction with no atom pair closer than 1.4 Å (VDW threshold) |

**Fallback (no RDKit):** Validity is approximated by checking that all bond lengths are
within ±0.20 Å of MMFF94 ideal values (geometry check).

---

## 9. Research Papers Referenced

1. **Ho et al. (2020).** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*  
   [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)  
   → DDPM framework, noise schedule, ε-prediction parameterisation.

2. **Satorras et al. (2021).** "E(n) Equivariant Graph Neural Networks." *ICML 2021.*  
   [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)  
   → EGNN: equivariant GNN with coordinate updates via unit displacement vectors.

3. **Hoogeboom et al. (2022).** "Equivariant Diffusion for Molecule Generation in 3D." *ICML 2022.*  
   [arXiv:2203.17003](https://arxiv.org/abs/2203.17003)  
   → EDM: CoM-subspace diffusion, SNR-weighted loss, molecular graph conditioning.

4. **Song et al. (2021).** "Denoising Diffusion Implicit Models." *ICLR 2021.*  
   [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)  
   → DDIM: deterministic sampling in 50 steps instead of 1000.

5. **Nichol & Dhariwal (2021).** "Improved Denoising Diffusion Probabilistic Models." *ICML 2021.*  
   [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)  
   → Cosine noise schedule that distributes diffusion capacity more evenly.

6. **Hang et al. (2023).** "Efficient Diffusion Training via Min-SNR Weighting Strategy." *ICCV 2023.*  
   [arXiv:2303.09556](https://arxiv.org/abs/2303.09556)  
   → Min-SNR-γ loss weighting to balance easy vs. hard timesteps.

7. **Le et al. (2024).** "EQGAT-diff: a Novel SE(3)-Equivariant Graph Attention Diffusion Model for Molecular Generation." *ICLR 2024.*  
   [arXiv:2306.01473](https://arxiv.org/abs/2306.01473)  
   → Immediate geometry supervision from epoch 1: no curriculum ramp needed.

8. **Morehead & Cheng (2023).** "Geometry-Complete Diffusion for 3D Molecule Generation." *arXiv 2023.*  
   [arXiv:2302.04313](https://arxiv.org/abs/2302.04313)  
   → GCDM: geometry constraints active from the start are strictly better.

9. **Ganea et al. (2021).** "GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles." *NeurIPS 2021.*  
   [arXiv:2106.07802](https://arxiv.org/abs/2106.07802)  
   → Torsion angles as primary 3D conformational DOF; torsion loss motivation.

10. **Jing et al. (2022).** "Torsional Diffusion for Molecular Conformer Generation." *NeurIPS 2022.*  
    [arXiv:2206.01729](https://arxiv.org/abs/2206.01729)  
    → TorDiff: diffusion directly over torsion angles. Inspired our torsion loss.

11. **Halgren, T.A. (1996).** "Merck Molecular Force Field. I–V." *J. Comput. Chem.*  
    DOI: 10.1002/jcc.540170082  
    → MMFF94: bond length, angle, and torsion parameters used as supervision targets.

12. **Jorgensen et al. (1996).** "Development and Testing of the OPLS All-Atom Force Field."  
    *J. Am. Chem. Soc. 118, 11225–11236.*  
    → OPLS-AA torsion potential V1/V2/V3 for dihedral angle supervision.

13. **Loshchilov & Hutter (2017).** "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR 2017.*  
    [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)  
    → Cosine annealing LR schedule with warmup to stabilise early training.

---

## 10. Bug History and Lessons Learned

The path to a stable training run required fixing **7 compounding bugs**:

| # | Bug | Root Cause | Fix |
|---|---|---|---|
| 1 | Geometry loss curriculum | Staged activation caused val_total spikes, froze checkpoint at ep.12 | Remove curriculum — all geometry from epoch 1 |
| 2 | Wrong checkpoint criterion | Saved on `val_total` which was inflated by curriculum | Checkpoint on `val_mse` only |
| 3 | Train/val MSE gap after ep.150 | No regularisation → overfitting in EGNN MLPs | Add `Dropout(0.1)` to edge_mlp + node_mlp |
| 4 | Noisy validity estimate | Only 50 samples → 14% noise floor | Increase to 200 samples |
| 5 | Soft constraints active late | Destabilising spike when planarity/chirality turned on at 60% | Active from epoch 1 with small weight |
| 6 | Wrong data keys | Code read `coordinates`, `num_atoms`; data has `coords`, `coord_mask` | Rewrite dataset loader |
| 7 | `max_atoms=15` too small | QM9 includes explicit H atoms; all 98K mols filtered out (0 samples!) | Raise to `max_atoms=50` |

**Key insight from paper survey:** EQGAT-diff and GCDM both report that immediate geometry
supervision is strictly better than staged activation — the geometry signal is *weakened*
by the SNR gate at high-noise timesteps anyway, so there is no benefit to delaying it.

---

## 11. Project Structure

```
mol_next_gen/
├── models/
│   ├── conformer_diffusion.py    # Core model: EGNN + DDPM + DDIM
│   └── geometry_constraints.py   # Differentiable chemistry constraints
│
├── training/
│   └── train_v3.py               # v3 training script (all 7 fixes)
│
├── data/
│   └── qm9_100k.jsonl            # 98,571 QM9 molecules (explicit H, padded to 50)
│
├── experiments/
│   └── 09-03-2026-Exp-3(stable_full_constraints)/
│       ├── checkpoints/
│       │   ├── conformer_best_mse.pt       # primary checkpoint (val_mse)
│       │   ├── conformer_best_validity.pt  # best RDKit validity
│       │   └── conformer_epochNNNN.pt      # periodic saves
│       ├── evaluation/
│       │   ├── epoch_NNNN_validity.json
│       │   └── generation_metrics.json
│       ├── molecules/
│       │   ├── generated_valid_molecules.sdf
│       │   ├── load_all_vmd.tcl            # VMD batch loader
│       │   └── pdb_files/
│       │       ├── mol_0001.pdb            # one clean PDB per molecule
│       │       └── ...
│       ├── plots/
│       │   └── loss_curves.png
│       └── logs/
│           └── training_YYYYMMDD_HHMMSS.log
│
├── logs/                          # SLURM stdout/stderr
│   └── slurm_v3_JOBID.out
│
├── slurm_job_v3.sh                # SLURM submission script
└── README.md                      # This file
```

---

## 12. Running the Experiment

### Prerequisites

```bash
# Activate the virtual environment (created automatically by SLURM script)
source /scratch/nishanth.r/nextmol_venv/bin/activate

# Verify environment
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -c "from rdkit import Chem; print('RDKit OK')"
```

### Submit SLURM Job

```bash
cd /scratch/nishanth.r/nextmol_experiment/mol_next_gen

# Submit
sbatch slurm_job_v3.sh

# Monitor
squeue -u nishanth.r
tail -f logs/slurm_v3_<JOBID>.out
```

### Manual Run (for debugging / smoke test)

```bash
cd /scratch/nishanth.r/nextmol_experiment/mol_next_gen

# Quick 2-epoch CPU test
python3 training/train_v3.py \
    --data data/qm9_100k.jsonl \
    --max_atoms 50 \
    --epochs 2 \
    --batch_size 4 \
    --hidden_dim 64 \
    --num_layers 2 \
    --geometry_weight 0.1
```

### Expected Startup Log

```
Loading data from data/qm9_100k.jsonl...
100%|████████| 98571/98571 [...]
Loaded 98XXX molecules (N skipped)
Train: ~88000, Val: ~9800
Parameters: ~22,000,000
Epoch    1: loss=1.xxxx, mse=1.xxxx
```

### Key Hyperparameters (Exp-3)

| Argument | Value | Rationale |
|---|---|---|
| `--epochs 300` | 300 | Sufficient for cosine schedule to converge |
| `--batch_size 64` | 64 | Good GPU utilisation on ~32GB VRAM |
| `--lr 3e-4` | 3×10⁻⁴ | Adam default; peak LR after warmup |
| `--warmup 5` | 5 epochs | Avoids cold-start instability |
| `--hidden_dim 512` | 512 | 2× bigger than baseline |
| `--num_layers 10` | 10 | Deeper = more expressive |
| `--timesteps 1000` | 1000 | Cosine schedule, 50-step DDIM at inference |
| `--geometry_weight 0.1` | 0.1 | Small, fixed, stable — no curriculum |
| `--max_atoms 50` | 50 | Covers full QM9 with explicit H |
| `--num_generate 500` | 500 | PDB export at evaluation epochs |

### Visualisation in VMD

```tcl
# Load all generated molecules as separate entities:
vmd -e experiments/09-03-2026-Exp-3.../molecules/load_all_vmd.tcl

# Load a single molecule:
vmd experiments/09-03-2026-Exp-3.../molecules/pdb_files/mol_0001.pdb
```

---

*Generated: 2026-03-10 · Experiment: 09-03-2026-Exp-3 (stable_full_constraints)*
