"""
algorithm/sahg_model.py
========================
SAHG: Sector-Anisotropic Hyperbolic + Graph Augmentation



  x_orig   ──→ SAHEncoder ──→ [r, H, A, γ] (5D)  ─┐
                                                    ├─→ head(10D) → logit
  x_agg    ──→ SAHEncoder ──→ [r, H, A, γ] (5D)  ─┘
  (GraphSAGE)
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sah import SAHEncoder, LocalWarpNet, SectorPrototypes
def set_seed(s: int = 42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class FocalBCE(nn.Module):
    """Focal Loss for binary classification"""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma

    def forward(self, logit: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        y_f = y.float()
        p = torch.sigmoid(logit).clamp(1e-6, 1-1e-6)
        a = self.alpha * y_f + (1 - self.alpha) * (1 - y_f)
        pt = p * y_f + (1 - p) * (1 - y_f)
        return (-a * (1 - pt) ** self.gamma * torch.log(pt)).mean()


class _TrainingConfig:
    """Training hyperparameters"""
    lr: float = 5e-4
    weight_decay: float = 1e-4
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    entropy_lambda: float = 0.05
    entropy_warmup: int = 12
    batch_size: int = 256
    epochs: int = 40
    patience: int = 8
    val_auc_w: float = 0.6
    hard_auc_w: float = 0.4



@torch.no_grad()
def build_knn_graph(x: torch.Tensor, k: int = 10) -> torch.Tensor:
    """


    Args:

    Returns:
    """
    device = x.device
    x_norm = F.normalize(x.float(), dim=-1)   # (N, D)

    N = x_norm.shape[0]
    chunk = max(1, min(512, N // 4))
    rows, cols = [], []

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sim = x_norm[start:end] @ x_norm.T   # (chunk, N)
        sim[:, start:end].fill_diagonal_(-2.0)
        topk_idx = sim.topk(k, dim=-1).indices  # (chunk, k)
        src = torch.arange(start, end, device=device).unsqueeze(1).expand_as(topk_idx)
        rows.append(src.reshape(-1))
        cols.append(topk_idx.reshape(-1))

    edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
    return edge_index






class SAHGSingleDetector(nn.Module):
    """
    SAHG for single-feature datasets (MGTAB / Fox8-23 / BotSim-24)

    """
    def __init__(self, in_dim: int, d_proj: int = 64,
                 d_hidden: int = 128, K: int = 2, dropout: float = 0.25):
        super().__init__()
        self.sage1     = SAGEConv(in_dim, in_dim)
        self.sage1_act = nn.Sequential(nn.LayerNorm(in_dim), nn.GELU(), nn.Dropout(dropout))
        self.sage2     = SAGEConv(in_dim, in_dim)
        self.sage2_act = nn.Sequential(nn.LayerNorm(in_dim), nn.GELU(), nn.Dropout(dropout))

        self.enc_orig     = SAHEncoder(in_dim, d_proj, d_hidden, dropout)
        self.warp_orig    = LocalWarpNet(d_proj, 32)
        self.sectors_orig = SectorPrototypes(K, d_proj, tau_init=5.0)

        self.enc_graph     = SAHEncoder(in_dim, d_proj, d_hidden, dropout)
        self.warp_graph    = LocalWarpNet(d_proj, 32)
        self.sectors_graph = SectorPrototypes(K, d_proj, tau_init=5.0)

        self.K = K

        self.head = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16), nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward_full(self, x: torch.Tensor, x_agg: torch.Tensor,
                     return_geo: bool = False):
        """
        """
        r_o, u_o, _ = self.enc_orig(x)
        g_o = self.warp_orig(u_o)
        H_o, A_o, _ = self.sectors_orig(u_o, g_o)
        r_o_n = r_o / (r_o.detach().std() + 1e-6)
        H_o_n = H_o / (math.log(self.K) + 1e-6)
        f_o = torch.stack([r_o_n, H_o_n, A_o, r_o_n * A_o, g_o * A_o], dim=-1)

        r_g, u_g, _ = self.enc_graph(x_agg)
        g_g = self.warp_graph(u_g)
        H_g, A_g, _ = self.sectors_graph(u_g, g_g)
        r_g_n = r_g / (r_g.detach().std() + 1e-6)
        H_g_n = H_g / (math.log(self.K) + 1e-6)
        f_g = torch.stack([r_g_n, H_g_n, A_g, r_g_n * A_g, g_g * A_g], dim=-1)

        f = torch.cat([f_o, f_g], dim=-1)   # (batch, 10)
        logit = self.head(f).squeeze(-1)

        if return_geo:
            return logit, dict(H_o=H_o, A_o=A_o, H_g=H_g, A_g=A_g)
        return logit

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_geo: bool = False):
        """Full forward: run SAGEConv on full graph (x must be full-graph tensor)"""
        h1 = self.sage1_act(self.sage1(x, edge_index))
        x_agg = self.sage2_act(self.sage2(h1, edge_index))
        return self.forward_full(x, x_agg, return_geo=return_geo)

    @torch.no_grad()
    def predict_prob(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, edge_index))

    def sage_aggregate(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run 2-hop SAGEConv on the full graph and return aggregated embeddings.
        NOTE: No @no_grad — gradients must flow here during training."""
        h1 = self.sage1_act(self.sage1(x, edge_index))
        return self.sage2_act(self.sage2(h1, edge_index))






def train_single_graph(
    model: SAHGSingleDetector,
    x_t, y_t, y_np: np.ndarray,
    edge_index,
    tr_idx, va_idx,
    device,
    cfg: _TrainingConfig = None,
    verbose: bool = False,
) -> SAHGSingleDetector:
    """
    """
    if cfg is None:
        cfg = _TrainingConfig()
        cfg.batch_size = 512; cfg.epochs = 120; cfg.patience = 15
        cfg.entropy_lambda = 0.03; cfg.entropy_warmup = 20

    ei = edge_index.to(device)
    focal = FocalBCE(cfg.focal_alpha, cfg.focal_gamma)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)

    va_np = va_idx.cpu().numpy() if isinstance(va_idx, torch.Tensor) else np.asarray(va_idx)
    best_val_auc = 0.0; best_state = None; no_imp = 0; best_ep = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        lam = cfg.entropy_lambda * max(0.0, 1.0 - (ep-1) / cfg.entropy_warmup)

        perm  = torch.randperm(len(tr_idx))
        tr_sh = tr_idx[perm]
        for i in range(0, len(tr_sh), cfg.batch_size):
            bi = tr_sh[i:i+cfg.batch_size].to(device)
            # Run SAGEConv per batch on full graph — fresh graph per step, no retain_graph
            x_agg_bi = model.sage_aggregate(x_t, ei)[bi]
            logit, geo = model.forward_full(x_t[bi], x_agg_bi, return_geo=True)
            loss = focal(logit, y_t[bi])
            if lam > 0:
                bm = (y_t[bi] == 1).float()
                H_n = geo['H_o'] / (math.log(model.K) + 1e-6)
                loss = loss + lam * (bm * H_n).sum() / (bm.sum() + 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = model.predict_prob(x_t, ei)[va_idx.to(device)].cpu().numpy()
            val_auc = roc_auc_score(y_np[va_np], vp)

        if verbose and ep % 10 == 0:
            print(f"  ep={ep:3d} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc; best_ep = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= cfg.patience: break

    model.load_state_dict(best_state)
    return model
