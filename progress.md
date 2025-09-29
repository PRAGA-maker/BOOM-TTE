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
