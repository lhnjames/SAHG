"""
data/loader.py
==============
Data loading utilities for SAH Bot Detector.

Supports:
  - TwiBot-20: 229,580-node Twitter graph, 11,826 labeled nodes
  - MGTAB:     10,199-node Twitter graph, fully labeled
  - Fox8-23:   2,280-user Twitter dataset (non-graph, preprocessed CSV)
  - BotSim-24: 2,907-user Reddit-simulation dataset (non-graph, preprocessed CSV)

Usage:
    from data.loader import load_twibot20, load_mgtab, load_fox8, load_botsim
"""
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


# ── MGTAB ─────────────────────────────────────────────────────────────────────

def load_mgtab(
    data_dir: str,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load and preprocess MGTAB dataset.

    Applies StandardScaler → clip(-3, 3) on the 788-dim features.
    Uses 70/20/10 stratified random split (controlled by seed).

    Args:
        data_dir: Path to MGTAB directory containing:
                  edge_index.pt, features.pt, labels_bot.pt
        seed:     Random seed for the 70/20/10 split

    Returns:
        x:          (10199, 788) normalized feature tensor
        edge_index: (2, E)       graph edges
        labels_np:  (10199,)     bot labels (0=human, 1=bot)
        split:      dict with 'train'/'val'/'test' numpy index arrays
    """
    # Resolve data directory
    for candidate in [data_dir,
                      os.path.join(data_dir, "Dataset", "MGTAB"),
                      os.path.join(data_dir, "MGTAB")]:
        if os.path.isdir(candidate) and \
           all(os.path.exists(os.path.join(candidate, f))
               for f in ["edge_index.pt", "features.pt", "labels_bot.pt"]):
            data_dir = candidate
            break
    else:
        raise FileNotFoundError(
            f"MGTAB data not found at {data_dir}. "
            "Needs: edge_index.pt, features.pt, labels_bot.pt"
        )

    edge_index = torch.load(os.path.join(data_dir, "edge_index.pt")).long().contiguous()
    x_raw      = torch.load(os.path.join(data_dir, "features.pt")).to(torch.float32)
    labels     = torch.load(os.path.join(data_dir, "labels_bot.pt")).long()

    x_np = np.clip(
        StandardScaler().fit_transform(x_raw.cpu().numpy()), -3, 3
    ).astype(np.float32)
    x = torch.from_numpy(x_np)

    N = x.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(N); rng.shuffle(indices)
    n_train = int(0.7 * N); n_val = int(0.2 * N)
    split = dict(
        train=indices[:n_train],
        val  =indices[n_train:n_train + n_val],
        test =indices[n_train + n_val:],
    )

    labels_np = labels.cpu().numpy()
    print(f"MGTAB: N={N}, edges={edge_index.shape[1]:,}, "
          f"bot={(labels_np==1).sum()}, human={(labels_np==0).sum()}, seed={seed}")
    return x, edge_index, labels_np, split


# ── Fox8-23 ───────────────────────────────────────────────────────────────────

def load_fox8(
    data_dir: str,
    seed: int = 0,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load preprocessed Fox8-23 dataset.

    Expects preprocess_fox8.py to have been run first, producing:
      features.csv, labels.csv

    Uses 70/15/15 stratified random split (controlled by seed).

    Args:
        data_dir: Path to fox8_23/ directory containing features.csv, labels.csv
        seed:     Random seed for the split

    Returns:
        x:          (N, D)  normalized feature tensor
        labels_np:  (N,)    bot labels (0=human, 1=bot)
        split:      dict with 'train'/'val'/'test' numpy index arrays
    """
    feat_path  = os.path.join(data_dir, "features.csv")
    label_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Fox8-23 processed data not found at {data_dir}.\n"
            "Run: python data/preprocess_fox8.py --ndjson /path/to/fox8_23_dataset.ndjson "
            "--out_dir data/fox8_23/"
        )

    x_raw_np = pd.read_csv(feat_path).drop(columns=["user_id"], errors="ignore").fillna(0.0).values.astype(np.float32)
    labels_np = pd.read_csv(label_path)["label"].values.astype(np.int64)

    N = x_raw_np.shape[0]
    rng = np.random.default_rng(seed)
    # Stratified split: keep bot/human balance in each split
    bot_idx   = np.where(labels_np == 1)[0].copy()
    human_idx = np.where(labels_np == 0)[0].copy()
    rng.shuffle(bot_idx);   rng.shuffle(human_idx)

    def _split70_15_15(idx):
        n = len(idx); n_tr = int(0.70 * n); n_va = int(0.15 * n)
        return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

    btr, bva, bte = _split70_15_15(bot_idx)
    htr, hva, hte = _split70_15_15(human_idx)
    split = dict(
        train=np.concatenate([btr, htr]),
        val  =np.concatenate([bva, hva]),
        test =np.concatenate([bte, hte]),
    )

    # ✅ Fit scaler ONLY on training split to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(x_raw_np[split["train"]])
    x_np = np.clip(scaler.transform(x_raw_np), -3, 3).astype(np.float32)
    x = torch.from_numpy(x_np)

    print(f"Fox8-23: N={N}, D={x.shape[1]}, "
          f"bot={( labels_np==1).sum()}, human={(labels_np==0).sum()}, seed={seed}")
    print(f"  train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    print(f"  train_bot={( labels_np[split['train']]==1).sum()}, "
          f"test_bot={(labels_np[split['test']]==1).sum()}")
    return x, labels_np, split


# ── BotSim-24 ─────────────────────────────────────────────────────────────────

def load_botsim(
    data_dir: str,
    seed: int = 0,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load preprocessed BotSim-24 dataset.

    Expects preprocess_botsim.py to have been run first, producing:
      features.csv, labels.csv

    Uses 70/15/15 stratified random split (controlled by seed).

    Args:
        data_dir: Path to botsim_24/ directory containing features.csv, labels.csv
        seed:     Random seed for the split

    Returns:
        x:          (N, D)  normalized feature tensor
        labels_np:  (N,)    bot labels (0=human, 1=bot)
        split:      dict with 'train'/'val'/'test' numpy index arrays
    """
    feat_path  = os.path.join(data_dir, "features.csv")
    label_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        raise FileNotFoundError(
            f"BotSim-24 processed data not found at {data_dir}.\n"
            "Run: python data/preprocess_botsim.py "
            "--csv /path/to/BotSim-24_user.csv "
            "--json /path/to/BotSim-24_user_post_comment.json "
            "--out_dir data/botsim_24/"
        )

    x_raw_np  = pd.read_csv(feat_path).fillna(0.0).values.astype(np.float32)
    labels_np = pd.read_csv(label_path)["label"].values.astype(np.int64)

    N = x_raw_np.shape[0]
    rng = np.random.default_rng(seed)
    bot_idx   = np.where(labels_np == 1)[0].copy()
    human_idx = np.where(labels_np == 0)[0].copy()
    rng.shuffle(bot_idx);   rng.shuffle(human_idx)

    def _split70_15_15(idx):
        n = len(idx); n_tr = int(0.70 * n); n_va = int(0.15 * n)
        return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

    btr, bva, bte = _split70_15_15(bot_idx)
    htr, hva, hte = _split70_15_15(human_idx)
    split = dict(
        train=np.concatenate([btr, htr]),
        val  =np.concatenate([bva, hva]),
        test =np.concatenate([bte, hte]),
    )

    # ✅ Fit scaler ONLY on training split to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(x_raw_np[split["train"]])
    x_np = np.clip(scaler.transform(x_raw_np), -3, 3).astype(np.float32)
    x = torch.from_numpy(x_np)

    print(f"BotSim-24: N={N}, D={x.shape[1]}, "
          f"bot={( labels_np==1).sum()}, human={(labels_np==0).sum()}, seed={seed}")
    print(f"  train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    print(f"  train_bot={( labels_np[split['train']]==1).sum()}, "
          f"test_bot={(labels_np[split['test']]==1).sum()}")
    return x, labels_np, split
