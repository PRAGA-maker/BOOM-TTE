"""
Dataset utilities for NL-MTP HoF.

Builds loaders from the existing SMILESDataset splits and computes RDKit-based
pre-treatment descriptors (x_ctx) and molecular weight (mw).
"""

import math
import random
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, MACCSkeys

from boom.datasets.SMILESDataset import SMILESDataset


RDKit_FP_BITS = 2048
MACCS_BITS = 167  # RDKit returns 167-bit MACCS, we will drop the 0th to keep 166 if needed


def _morgan_fp(mol) -> np.ndarray:
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=RDKit_FP_BITS)
    arr = np.zeros((RDKit_FP_BITS,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


def _maccs(mol) -> np.ndarray:
    bv = MACCSkeys.GenMACCSKeys(mol)  # 167 bits
    arr = np.zeros((MACCS_BITS,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


def _descriptor_vec(mol) -> np.ndarray:
    # Select a stable set of basic descriptors that are meaningful pre-treatment.
    # Avoid direct MolWt; use HeavyAtomMolWt as proxy in context, but keep separate mw as exposure A.
    ds = [
        Descriptors.MolLogP,
        Descriptors.TPSA,
        Descriptors.NumHAcceptors,
        Descriptors.NumHDonors,
        Descriptors.RingCount,
        Descriptors.NumRotatableBonds,
        Descriptors.HeavyAtomCount,
        Descriptors.FractionCSP3,
    ]
    vals = [float(d(mol)) for d in ds]
    return np.array(vals, dtype=np.float32)


def _compute_env_idx(smiles: str, num_envs: int = 16) -> int:
    # Hash-based environment bucket (proxy for scaffold/source). Deterministic.
    return hash(smiles) % num_envs


class HoFContextDataset(Dataset):
    def __init__(self, split: str):
        self.base = SMILESDataset("hof", split)
        self.items = []
        for smi, y in zip(self.base.smiles, self.base.property_values):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                mw = rdMolDescriptors.CalcExactMolWt(mol)
                fp = _morgan_fp(mol)
                maccs = _maccs(mol)
                desc = _descriptor_vec(mol)
                x_ctx = np.concatenate([fp, maccs, desc], axis=0)
                env_idx = _compute_env_idx(smi)
                self.items.append({
                    "smiles": smi,
                    "x_ctx": x_ctx,
                    "mw": float(mw),
                    "y": float(y),
                    "env_idx": int(env_idx),
                })
            except Exception:
                # Skip problematic molecules
                continue

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        return {
            "smiles": it["smiles"],
            "x_ctx": torch.from_numpy(it["x_ctx"]).float(),
            "mw": torch.tensor(it["mw"], dtype=torch.float32),
            "y": torch.tensor(it["y"], dtype=torch.float32),
            "env_idx": torch.tensor(it["env_idx"], dtype=torch.long),
        }


def collate_batch(batch: Any) -> Dict[str, torch.Tensor]:
    x_ctx = torch.stack([b["x_ctx"] for b in batch], dim=0)
    mw = torch.stack([b["mw"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    env_idx = torch.stack([b["env_idx"] for b in batch], dim=0)
    return {"x_ctx": x_ctx, "mw": mw, "y": y, "env_idx": env_idx}


def make_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = HoFContextDataset("train")
    id_ds = HoFContextDataset("id")
    ood_ds = HoFContextDataset("ood")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    id_dl = DataLoader(id_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    ood_dl = DataLoader(ood_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_dl, id_dl, ood_dl



