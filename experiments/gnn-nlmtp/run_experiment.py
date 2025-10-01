"""
CLI entrypoint to run NL-MTP HoF experiment.

Usage:
    python -m experiments.gnn-nlmtp.run_experiment --device cuda --batch_size 64 --epochs 30 --delta 14
"""

import argparse
import os
import sys

import torch
import torch.optim as optim

# Ensure experiments module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import relative to allow running as script
try:
    from .dataset import make_dataloaders
    from .model import NL_MTP_Model
    from .trainer import train_epoch, evaluate
    from .eval import save_metrics_json, make_all_plots
except ImportError:
    from dataset import make_dataloaders
    from model import NL_MTP_Model
    from trainer import train_epoch, evaluate
    from eval import save_metrics_json, make_all_plots


def main():
    parser = argparse.ArgumentParser(description="NL-MTP HoF experiment")
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--delta', type=float, default=14.0, help='Policy shift (Da)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--out_dir', type=str, default='experiments/gnn-nlmtp/results', help='Output directory')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='MMP warmup epochs')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    print(f"Using device: {device}")
    
    # Data
    print("Loading data...")
    loaders = make_dataloaders(batch_size=args.batch_size)
    train_dl, id_dl, ood_dl = loaders
    print(f"Train batches: {len(train_dl)}, ID batches: {len(id_dl)}, OOD batches: {len(ood_dl)}")
    
    # Model
    print("Initializing model...")
    model = NL_MTP_Model(
        emb_dim=512,
        num_layers=12,
        num_heads=8,
        dim_ff=2048,
        lor_layers=(3, 7, 11),
        lor_rank=8,
        mdn_components=8,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Optimizer and scheduler
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(train_dl)
    warmup_steps = min(2000, total_steps // 10)
    sched = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
    )
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_ood_rmse = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model,
            loaders,
            opt,
            sched,
            delta=args.delta,
            epoch=epoch,
            device=device,
            warmup_epochs=args.warmup_epochs,
        )
        
        # Evaluate
        val_metrics = evaluate(model, loaders, delta=args.delta, device=device)
        
        # Log
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f} "
              f"(obs={train_metrics['L_obs']:.4f}, mdn={train_metrics['L_mdn']:.4f}, "
              f"dr_func={train_metrics['L_dr_func']:.4f}, rex={train_metrics['L_rex']:.4f})")
        print(f"  ID:  RMSE={val_metrics['id_rmse']:.4f}, MAE={val_metrics['id_mae']:.4f}, "
              f"Contrast={val_metrics['id_policy_contrast']:.4f}")
        print(f"  OOD: RMSE={val_metrics['ood_rmse']:.4f}, MAE={val_metrics['ood_mae']:.4f}, "
              f"Contrast={val_metrics['ood_policy_contrast']:.4f}")
        
        # Save best model
        if val_metrics['ood_rmse'] < best_ood_rmse:
            best_ood_rmse = val_metrics['ood_rmse']
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            print(f"  *** New best OOD RMSE: {best_ood_rmse:.4f} ***")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pth')))
    final_metrics = evaluate(model, loaders, delta=args.delta, device=device)
    
    print(f"ID:  RMSE={final_metrics['id_rmse']:.4f}, MAE={final_metrics['id_mae']:.4f}")
    print(f"OOD: RMSE={final_metrics['ood_rmse']:.4f}, MAE={final_metrics['ood_mae']:.4f}")
    
    # Save metrics and plots
    save_metrics_json(final_metrics, os.path.join(args.out_dir, 'metrics.json'))
    make_all_plots(final_metrics, args.out_dir)
    
    print(f"\nResults saved to {args.out_dir}")
    print(f"  - metrics.json")
    print(f"  - NL_MTP_HoF_ID_parity.png")
    print(f"  - NL_MTP_HoF_OOD_parity.png")
    print(f"  - best_model.pth")


if __name__ == "__main__":
    main()
