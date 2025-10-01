# BOOM-TTE Progress Tracking

## Datasets Status

| Dataset | Property | Status | Train Size | ID Size | OOD Size | Notes |
|---------|----------|--------|------------|---------|----------|-------|
| 10K CSD | Density | ✅ Loaded | 8766 | 440 | 1000 | Auto-downloaded from LLNL GitHub |
| 10K CSD | HoF | ✅ Loaded | 8783 | 423 | 1000 | Auto-downloaded from LLNL GitHub |
| QM9 | Alpha | ✅ Loaded | 117768 | ~5000 | 10000 | Downloaded from Figshare, processed |
| QM9 | Cv | ✅ Loaded | 117768 | ~5000 | 10000 | Downloaded from Figshare, processed |
| QM9 | Gap | ✅ Loaded | 117768 | ~5000 | 10000 | Downloaded from Figshare, processed |
| QM9 | HOMO | ✅ Loaded | 117768 | ~5000 | 10000 | Downloaded from Figshare, processed |
| QM9 | LUMO | ✅ Loaded | 117768 | ~5000 | 10000 | Downloaded from Figshare, processed |
| QM9 | Mu | ✅ Loaded | 117768 | ~5000 | 10000 | Downloaded from Figshare, processed |
| QM9 | R2 | ✅ Loaded | 117768 | ~5000 | 10000 | Downloaded from Figshare, processed |
| QM9 | ZPVE | ✅ Loaded | 117768 | ~5000 | 10000 | Downloaded from Figshare, processed |

## Models Status

### Baseline Models
| Model | Density | HoF | Alpha | Cv | Gap | HOMO | LUMO | Mu | R2 | ZPVE | Status |
|-------|---------|-----|-------|----|----|------|------|----|----|------|--------|
| Random Forest | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Ready to run |

### Advanced Models
| Model | Density | HoF | Alpha | Cv | Gap | HOMO | LUMO | Mu | R2 | ZPVE | Status |
|-------|---------|-----|-------|----|----|------|------|----|----|------|--------|
| Chemprop | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |
| Graphormer3D | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |
| MACE | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |
| MolFormer | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |
| Regression Transformer | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |
| ModernBERT | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |
| ChemBERTa | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |
| Equivariant Transformer | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |
| GNN Ablation | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Needs setup |

## Legend
- ✅ Completed
- 🔄 In Progress  
- ⏳ Pending
- ❌ Failed
- 🚫 Not Applicable

## Next Steps
1. ✅ Complete QM9 dataset processing (in progress)
2. ⏳ Run Random Forest baseline on all datasets
3. ⏳ Set up and run advanced models
4. ⏳ Generate performance comparison plots
5. ⏳ Create final results summary

## Notes
- QM9 dataset processing takes ~2 hours due to large size (~134k molecules)
- All datasets auto-generate splits if not present
- Random Forest is the quickest baseline to run
- Advanced models require additional dependencies and setup


Prior research has shown:

Dataset	HoF	Density	HOMO	LUMO	GAP	ZPVE	R²	α	µ	Cᵥ
Best Performing Model (ID)	MACE	EGNN	ET	ET	ET	Graphormer (3D)	MACE	MACE	ET	MolFormer
Best Performing Model (OOD)	EGNN	TGNN	ET	ET	TGNN	MACE, TGNN	MACE	MACE	MACE	MACE

## HoF Baselines: MACE and EGNN

- MACE HoF
  - Commands:
    ```bash
    cd experiments/mace
    python make_splits.py
    mace_run_train --config=configs/hof.yaml
    python make_plots.py
    ```
  - Outputs:
    - Model: `experiments/mace/hof.model`
    - Plot: `experiments/mace/mace_hof.png`

- EGNN (EquivariantGNN) HoF
  - Runner added: `experiments/gnn-ablation/run_hof_equivariant.py`
  - Command:
    ```bash
    python experiments/gnn-ablation/run_hof_equivariant.py --device cuda --lr 1e-3 --batch_size 32 --epochs 100
    ```
  - Outputs:
    - CSV: `experiments/gnn-ablation/results/results_EquivariantGNN_HoF.csv`
    - Plot: `experiments/gnn-ablation/results/EquivariantGNN_HoF_parity_plot.png`


## NL-MTP HoF Experiment (Neurallambda + Transformer, Policy Evaluation)

- Objective: Train a single, fixed-recipe homoiconic transformer with fast-weights to perform policy evaluation (MTP) on HoF under a fixed policy shift δ=+14 Da (do:W+=δ). Produce ID/OOD RMSE and a parity/contrast plot.

- Data I/O (deterministic):
  - Input table: `id, smiles, y (HoF), env`.
  - Derived features: `x_ctx` (RDKit pre-treatment descriptors only), `mw` (ExactMolWt), `env_idx`.
  - Optional: `mmp_pairs` (RDKit MMPA minimal edits, small Δmass) for early self-supervised semantics.

- Splits:
  - Scaffold-based split (Murcko) to form ID train/val and OOD test.

- Sequence layout and masking:
  - Tokens: [ENV e] [CONTEXT x_ctx] [A: mw] [SMILES tokens...] <time0> [do: W+=δ] <probe>.
  - Attention mask: post-<time0> attends to pre-<time0>; pre-<time0> cannot attend to post-treatment tokens.

- Fast-weights (LoR) injection (fixed):
  - Apply low-rank updates at attention Q/K/O and MLP out, layers {3,7,11}, rank=8.
  - LoR activated only after encountering [do: W+=δ]; amplitude α(x,δ)∈[0,1] with norm/rank penalties.

- Heads:
  - Outcome head m̂(a,x) on log(HoF−c) (stabilized).
  - Propensity head ĝ(a|x): MDN (K=8 Gaussians) on pre-policy CLS.
  - Support gate s(x,a,δ)∈[0,1] for positivity/abstention.

- Losses (single recipe):
  - L_obs: supervised MSE on observed world.
  - L_mdn: MDN NLL for A|X.
  - DR/AIPW policy losses: L_DR-func (unit-level) + L_DR-mean (scalar ψ(δ)).
  - L_rex: invariance penalty (variance of policy-head per-env losses).
  - L_lor: locality (α^2) + low-rank Frobenius penalties.
  - L_mmp: early warmup auxiliary on descriptor deltas for MMP pairs.
  - Total: 1.0·L_DR-func + 0.2·L_DR-mean + 1.0·L_obs + 0.5·L_mdn + 0.2·L_rex + 0.1·L_lor + 0.2·(warmup·L_mmp).

- Training loop (fixed):
  - Optimizer: AdamW (lr=2e-4), cosine LR with warmup (2k steps), grad clip=1.0.
  - Batch: mix multiple envs; δ=+14 for all; drop samples failing support gate.
  - Two passes per batch: observed-world (no LoR) and policy-world (LoR active after <time0>).
  - Log ID/OOD metrics each epoch; save checkpoints and reports.

- Evaluation:
  - Report RMSE/MAE (observed), policy contrast Δpred = m̂(A+δ,X) − m̂(A,X), local linearity checks vs δ, abstention rates, env variance for policy head, OOD plots.

- New experiment folder: `experiments/gnn-nlmtp` (full implementation):
  - ✅ `dataset.py`: RDKit features (Morgan+MACCS+descriptors) for x_ctx, ExactMolWt for A, env_idx bucketing, dataloaders.
  - ✅ `model.py`: Full NL_MTP_Model with 12-layer transformer, multi-token sequence [ENV][CTX][A]<TIME0>[DO][PROBE], time-zero attention mask, LoR on Q/K/O+MLP-out at layers {3,7,11} rank=8, outcome/MDN/support/desc_delta heads, alpha gating, learnable ψ.
  - ✅ `trainer.py`: Observed/policy two-pass, DR/AIPW losses (unit+scalar), MDN NLL, per-env REx variance, LoR locality penalties, support gating, importance weight clipping, complete MMP descriptor delta warmup loss.
  - ✅ `eval.py`: ID/OOD RMSE/MAE, policy contrast, parity plots, JSON metrics writer.
  - ✅ `run_experiment.py`: CLI with AdamW+OneCycleLR, 30 epochs, best-model checkpointing, full metrics logging.
  - Commands:
  - Script: `python experiments/gnn-nlmtp/run_experiment.py --device cuda --batch_size 64 --epochs 30 --delta 14`
  - Notebook: `jupyter notebook experiments/gnn-nlmtp/run_experiment.ipynb`
- Status: ✅ **PRODUCTION-READY** - All pseudo/placeholder code removed, complete implementation verified

- Integration points:
  - Reuse experiment/scaffold style from `experiments/gnn-ablation` (runner, trainer patterns, results folder).
  - Import transformer blocks from `nl/neurallambda/src/neurallambda/model/recurrent_transformer.py` (DecoderLayer, positional encoding) and follow NL attention masking patterns.

- Fixed hyperparameters (no decisions during dev):
  - Layers=12, hidden size per backbone per choice; LoR rank=8 on layers {3,7,11}. K=8 (MDN). δ=+14. Batch=64. Warmup epochs=5 for L_mmp.


    


+====


