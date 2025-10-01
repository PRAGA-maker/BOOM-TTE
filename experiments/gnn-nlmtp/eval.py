"""
Evaluation utilities for NL-MTP HoF experiment.

Includes:
- Metrics computation (wraps trainer.evaluate)
- JSON metrics writer
- Parity plot generation
"""

import json
from typing import Dict, Any
import matplotlib.pyplot as plt
import torch


def compute_metrics(model, loaders, delta: float, device: str = "cuda") -> Dict[str, Any]:
    """Compute ID/OOD metrics. Wraps trainer.evaluate."""
    from .trainer import evaluate
    return evaluate(model, loaders, delta, device)


def save_metrics_json(metrics: Dict[str, Any], path: str):
    """Save metrics dict to JSON, excluding tensor values."""
    # Filter out tensor values for JSON serialization
    json_metrics = {k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)}
    with open(path, "w") as f:
        json.dump(json_metrics, f, indent=2)


def make_parity_plot(y_true: torch.Tensor, y_pred: torch.Tensor, title: str, out_path: str):
    """Generate and save a parity plot."""
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Parity line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', alpha=0.75, linewidth=1, label='Parity')
    
    # Scatter
    ax.scatter(y_true, y_pred, s=8, alpha=0.5, edgecolors='none')
    
    ax.set_xlabel('True HoF', fontsize=12)
    ax.set_ylabel('Predicted HoF', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_all_plots(metrics: Dict[str, Any], out_dir: str):
    """Generate ID and OOD parity plots from metrics dict."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    # ID plot
    if "id_y_true" in metrics and "id_y_pred" in metrics:
        make_parity_plot(
            metrics["id_y_true"],
            metrics["id_y_pred"],
            f'ID: RMSE={metrics["id_rmse"]:.3f}, MAE={metrics["id_mae"]:.3f}',
            os.path.join(out_dir, "NL_MTP_HoF_ID_parity.png"),
        )
    
    # OOD plot
    if "ood_y_true" in metrics and "ood_y_pred" in metrics:
        make_parity_plot(
            metrics["ood_y_true"],
            metrics["ood_y_pred"],
            f'OOD: RMSE={metrics["ood_rmse"]:.3f}, MAE={metrics["ood_mae"]:.3f}',
            os.path.join(out_dir, "NL_MTP_HoF_OOD_parity.png"),
        )
