"""
eval.py
=======
SAHG Bot Detector — Evaluation script for MGTAB, Fox8-23, and BotSim-24.

Usage:
    # Evaluate all three datasets:
    python eval.py \
        --mgtab_data  /path/to/MGTAB \
        --fox8_data   data/fox8_23 \
        --botsim_data data/botsim_24

    # Skip specific datasets:
    python eval.py --skip_mgtab
    python eval.py --skip_fox8 --skip_botsim
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from algorithm.sahg_model import (
    SAHGSingleDetector, build_knn_graph, train_single_graph,
    set_seed, _TrainingConfig,
)
from data.loader import load_mgtab, load_fox8, load_botsim


def _compute_metrics(te_prob, te_y):
    """Compute ACC (optimal threshold), AUC, F1, Precision, Recall."""
    auc = roc_auc_score(te_y, te_prob)
    ba, bt = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        a = accuracy_score(te_y, (te_prob >= t).astype(int))
        if a > ba: ba, bt = a, t
    pred = (te_prob >= bt).astype(int)
    f1   = f1_score(te_y, pred, average="macro")
    prec = precision_score(te_y, pred, average="macro", zero_division=0)
    rec  = recall_score(te_y, pred, average="macro", zero_division=0)
    return ba, prec, rec, f1, auc, bt
def _print_seed_row(seed, bt, acc, prec, rec, f1, auc):
    print(f"  seed={seed}(thr={bt:.3f}): "
          f"ACC={acc*100:.2f}%  PRE={prec*100:.2f}%  "
          f"REC={rec*100:.2f}%  F1={f1*100:.2f}%  AUC={auc*100:.2f}%")
def _print_summary(results, dataset_name):
    n = len(results)
    print(f"\n{'='*60}")
    print(f"  {dataset_name} Summary ({n} seeds) [SAHG]")
    print(f"{'='*60}")
    for k, label in [("acc", "ACC"), ("prec", "PRE"), ("rec", "REC"), ("f1", "F1"), ("auc", "AUC")]:
        vs = [r[k] * 100 for r in results]
        print(f"  {label:4s}   {np.mean(vs):7.2f}%   ±{np.std(vs):.2f}%")
def run_mgtab(data_dir: str, seeds: tuple = (0, 1, 2)) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[MGTAB] seeds={list(seeds)} | device={device}")
    results = []
    for seed in seeds:
        set_seed(seed)
        x, ei, labels_np, split = load_mgtab(data_dir, seed=seed)
        x_dev = x.to(device)
        ei_dev = ei.to(device)
        y_t = torch.LongTensor(labels_np).to(device)
        tr_t = torch.LongTensor(split["train"])
        va_t = torch.LongTensor(split["val"])

        model = SAHGSingleDetector(in_dim=x.shape[1], d_proj=64, d_hidden=256,
                                   K=2, dropout=0.3).to(device)
        cfg = _TrainingConfig()
        cfg.lr = 2e-4; cfg.batch_size = 512; cfg.epochs = 80; cfg.patience = 15
        cfg.focal_alpha = 0.85; cfg.focal_gamma = 0.5; cfg.entropy_lambda = 0.0; cfg.weight_decay = 1e-4

        model = train_single_graph(model, x_dev, y_t, labels_np, ei_dev, tr_t, va_t, device, cfg)
        model.eval()
        with torch.no_grad():
            te_prob = model.predict_prob(x_dev, ei_dev).cpu().numpy()[split["test"]]

        acc, prec, rec, f1, auc, bt = _compute_metrics(te_prob, labels_np[split["test"]])
        _print_seed_row(seed, bt, acc, prec, rec, f1, auc)
        results.append(dict(acc=acc, auc=auc, f1=f1, prec=prec, rec=rec))

    _print_summary(results, "MGTAB")
    return results
def run_fox8(data_dir: str, seeds: tuple = (0, 1, 2), knn_k: int = 10) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Fox8-23] seeds={list(seeds)} | knn_k={knn_k} | device={device}")
    results = []
    for seed in seeds:
        set_seed(seed)
        x, labels_np, split = load_fox8(data_dir, seed=seed)
        x_dev = x.to(device)
        y_t = torch.LongTensor(labels_np).to(device)

        print(f"  [seed={seed}] Building k-NN graph (k={knn_k}) ...", end=" ", flush=True)
        ei = build_knn_graph(x_dev, k=knn_k)
        print(f"edges={ei.shape[1]:,}")

        tr_t = torch.LongTensor(split["train"])
        va_t = torch.LongTensor(split["val"])
        model = SAHGSingleDetector(in_dim=x.shape[1], d_proj=64, d_hidden=128,
                                   K=2, dropout=0.25).to(device)
        cfg = _TrainingConfig()
        cfg.lr = 3e-4; cfg.batch_size = 128; cfg.epochs = 120; cfg.patience = 15
        cfg.entropy_lambda = 0.03; cfg.entropy_warmup = 20

        model = train_single_graph(model, x_dev, y_t, labels_np, ei, tr_t, va_t, device, cfg)
        model.eval()
        with torch.no_grad():
            te_prob = model.predict_prob(x_dev, ei).cpu().numpy()[split["test"]]

        acc, prec, rec, f1, auc, bt = _compute_metrics(te_prob, labels_np[split["test"]])
        _print_seed_row(seed, bt, acc, prec, rec, f1, auc)
        results.append(dict(acc=acc, auc=auc, f1=f1, prec=prec, rec=rec))

    _print_summary(results, "Fox8-23")
    return results
def run_botsim(data_dir: str, seeds: tuple = (0, 1, 2), knn_k: int = 10) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[BotSim-24] seeds={list(seeds)} | knn_k={knn_k} | device={device}")
    results = []
    for seed in seeds:
        set_seed(seed)
        x, labels_np, split = load_botsim(data_dir, seed=seed)
        x_dev = x.to(device)
        y_t = torch.LongTensor(labels_np).to(device)

        print(f"  [seed={seed}] Building k-NN graph (k={knn_k}) ...", end=" ", flush=True)
        ei = build_knn_graph(x_dev, k=knn_k)
        print(f"edges={ei.shape[1]:,}")

        tr_t = torch.LongTensor(split["train"])
        va_t = torch.LongTensor(split["val"])
        model = SAHGSingleDetector(in_dim=x.shape[1], d_proj=32, d_hidden=64,
                                   K=2, dropout=0.30).to(device)
        cfg = _TrainingConfig()
        cfg.lr = 3e-4; cfg.focal_alpha = 0.80; cfg.batch_size = 256
        cfg.epochs = 120; cfg.patience = 15
        cfg.entropy_lambda = 0.03; cfg.entropy_warmup = 20

        model = train_single_graph(model, x_dev, y_t, labels_np, ei, tr_t, va_t, device, cfg)
        model.eval()
        with torch.no_grad():
            te_prob = model.predict_prob(x_dev, ei).cpu().numpy()[split["test"]]

        acc, prec, rec, f1, auc, bt = _compute_metrics(te_prob, labels_np[split["test"]])
        _print_seed_row(seed, bt, acc, prec, rec, f1, auc)
        results.append(dict(acc=acc, auc=auc, f1=f1, prec=prec, rec=rec))

    _print_summary(results, "BotSim-24")
    return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHG — Social Bot Detection Evaluation")
    parser.add_argument("--mgtab_data",  type=str, default="data/mgtab",
                        help="Path to MGTAB dataset directory")
    parser.add_argument("--fox8_data",   type=str, default="data/fox8_23")
    parser.add_argument("--botsim_data", type=str, default="data/botsim_24")
    parser.add_argument("--mgtab_seeds",  type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fox8_seeds",   type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--botsim_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--knn_k",        type=int, default=10)
    parser.add_argument("--skip_mgtab",   action="store_true")
    parser.add_argument("--skip_fox8",    action="store_true")
    parser.add_argument("--skip_botsim",  action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  SAHG — Social Bot Detection Evaluation")
    print("=" * 60)

    m_mg = m_f8 = m_bs = None

    if not args.skip_mgtab:
        m_mg = run_mgtab(args.mgtab_data, seeds=tuple(args.mgtab_seeds))
    if not args.skip_fox8:
        m_f8 = run_fox8(args.fox8_data, seeds=tuple(args.fox8_seeds), knn_k=args.knn_k)
    if not args.skip_botsim:
        m_bs = run_botsim(args.botsim_data, seeds=tuple(args.botsim_seeds), knn_k=args.knn_k)

    print("\n" + "=" * 70)
    print("  Final Summary [SAHG]")
    print("=" * 70)
    print(f"  {'Dataset':12s}  {'ACC':>10s}  {'PRE':>10s}  {'REC':>10s}  {'F1':>10s}  {'AUC':>10s}")
    print(f"  {'-'*66}")
    for label, results in [("MGTAB", m_mg), ("Fox8-23", m_f8), ("BotSim-24", m_bs)]:
        if results:
            accs  = [r["acc"]  * 100 for r in results]
            pres  = [r["prec"] * 100 for r in results]
            recs  = [r["rec"]  * 100 for r in results]
            f1s   = [r["f1"]   * 100 for r in results]
            aucs  = [r["auc"]  * 100 for r in results]
            print(f"  {label:12s}  "
                  f"{np.mean(accs):6.2f}±{np.std(accs):.1f}%  "
                  f"{np.mean(pres):6.2f}±{np.std(pres):.1f}%  "
                  f"{np.mean(recs):6.2f}±{np.std(recs):.1f}%  "
                  f"{np.mean(f1s):6.2f}±{np.std(f1s):.1f}%  "
                  f"{np.mean(aucs):6.2f}±{np.std(aucs):.1f}%")
