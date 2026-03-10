Experiment: 09-03-2026-Exp-3(stable_full_constraints)
Script:     training/train_v3.py
SLURM:      slurm_job_v3.sh

Key improvements over Exp-1 and Exp-2:
  - All geometry losses active from epoch 1 (no curriculum)
  - Checkpoint saved on val_mse + separate best-validity checkpoint
  - 200-sample validity evaluation (was 50)
  - LR warmup 5 epochs then CosineAnnealing
  - Per-molecule PDB files in pdb_files/ (VMD-safe, no MODEL/ENDMDL)
  - Dropout(0.1) in EquivariantLayer to prevent train/val divergence

References:
  - EQGAT-diff (ICLR 2024): time-dep loss + immediate geometry training
  - GCDM (arXiv 2023): geometry supervision from epoch 1
  - Min-SNR (Hang et al. 2023): SNR-gated geometry weight
