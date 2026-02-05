# NExT-Mol Experiment: Decoupled Topology + Geometry Generation

## Overview

This experiment implements the NExT-Mol approach to molecular generation:
- **Stage 1:** Generate valid 2D molecular topology using SELFIES
- **Stage 2:** Generate 3D conformer using diffusion (coordinates only)

## Key Insight

| Approach | Validity | Why |
|----------|----------|-----|
| Old (coords → infer bonds) | ~8% | Bond inference fails |
| **NExT-Mol (topology first)** | **~95-100%** | Topology pre-validated |

## Project Structure

```
nextmol_experiment/
├── data/                        # Data preparation
├── models/                      # Core model implementations
│   ├── selfies_generator.py    # Stage 1: Topology generator
│   ├── conformer_diffusion.py  # Stage 2: 3D coordinate diffusion
│   ├── dual_encoder.py         # Local/Global equivariant encoder
│   └── validity_filter.py      # Step-wise validity checks
├── training/                    # Training scripts
│   ├── prepare_selfies_data.py # Convert SMILES → SELFIES
│   ├── train_selfies_lm.py     # Train SELFIES language model
│   └── train_conformer.py      # Train conformer diffusion
├── generation/                  # Generation pipeline
│   └── generate_nextmol.py     # Two-stage generation
└── evaluation/                  # Evaluation scripts
    └── evaluate_validity.py    # Compare with old approach
```

## Usage

```bash
# 1. Prepare SELFIES data
python training/prepare_selfies_data.py

# 2. Train conformer diffusion
python training/train_conformer.py --epochs 100

# 3. Generate molecules
python generation/generate_nextmol.py --num_molecules 1000

# 4. Evaluate
python evaluation/evaluate_validity.py
```

## Requirements

```
torch>=2.0
rdkit
selfies
transformers
tqdm
pandas
numpy
```
