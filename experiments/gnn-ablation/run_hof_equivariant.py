################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from functools import partial
import argparse

from boom.viz.ParityPlot import OODParityPlot
from dataset import Torch3DDataset, batched_3D_collator
from models import EquivariantGNN
from run_experiment import run_experiment, process_results, get_norm_constants


def main():
    parser = argparse.ArgumentParser(description="Train EquivariantGNN on HoF only")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--noise", action="store_true", help="Add coordinate noise during training")
    parser.add_argument("--noise_amount", type=float, default=0.05)
    parser.add_argument("--show", action="store_true", help="Call plt.show() after saving plot")
    args = parser.parse_args()

    property = "HoF"

    train_dataset = Torch3DDataset(property, "train", add_noise=args.noise, noise_amount=args.noise_amount)
    iid_test_dataset = Torch3DDataset(property, "id")
    ood_test_dataset = Torch3DDataset(property, "ood")

    prop_mean, prop_std = get_norm_constants(train_dataset)

    model = EquivariantGNN(
        in_node_nf=5,
        in_edge_nf=4,
        hidden_nf=64,
        include_charge=True,
        device=args.device,
        reduce="sum",
    )

    results = run_experiment(
        target=property,
        model=model,
        train_dataset=train_dataset,
        iid_test_dataset=iid_test_dataset,
        ood_test_dataset=ood_test_dataset,
        collator=batched_3D_collator,
        model_type="EquivariantGNN",
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        prop_mean=prop_mean,
        prop_std=prop_std,
    )

    plot_func = partial(OODParityPlot, target_value=property.lower())
    process_results(results, "EquivariantGNN", plot_func)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()


