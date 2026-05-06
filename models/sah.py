"""
SAH: Sector-Anisotropic Hyperbolic Space Bot Detector
=======================================================

=========================================================

-----------


  ds² = a(r)² dr² + b(r,u)² dΣ²_A(u)



------------------------------


  q_k(u) ∝ exp(τ_k · γ(u) · φ_k(u))


--------------

   H(u) = -Σ_k q_k · log(q_k)

   A(z) = max_k φ_k(u)

   R(z) = σ(α(u) · r + β(u))

------------------------
  logit(z) = f_cls([R(z), A(z), H(z), r·A(z), γ(u)·A(z)])


--------------------------

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def safe_normalize(x, eps=1e-10):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return x / norm



class LocalWarpNet(nn.Module):
    """
    
    
    """
    def __init__(self, d_proj: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_proj, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=0.05) # DO NOT use exactly zero
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        return: (N,) γ > 0
        """
        return F.softplus(self.net(u)).squeeze(-1) + 0.1  # γ ∈ (0.1, ∞)


class SectorPrototypes(nn.Module):
    """
    
    
    """
    def __init__(self, K: int, d_proj: int, tau_init: float = 5.0):
        super().__init__()
        raw = torch.randn(K, d_proj)
        self.prototypes = nn.Parameter(F.normalize(raw, dim=-1))
        self.log_tau = nn.Parameter(torch.full((K,), math.log(tau_init)))

    def forward(self, u: torch.Tensor, gamma: torch.Tensor):
        """
        
        return:
        """
        protos = F.normalize(self.prototypes, dim=-1)  # (K, d_proj)
        tau    = self.log_tau.exp().clamp(1.0, 50.0)   # (K,)
        
        phi = u @ protos.T                              # (N, K)  ∈ [-1, 1]
        
        weighted = gamma.unsqueeze(-1) * tau.unsqueeze(0) * phi  # (N, K)
        
        weighted = weighted - weighted.max(dim=-1, keepdim=True).values
        q = F.softmax(weighted, dim=-1)                  # (N, K)
        
        H     = -(q * (q + 1e-12).log()).sum(-1)         # (N,) ∈ [0, log K]
        max_a = phi.max(dim=-1).values                   # (N,)
        
        return H, max_a, q


class SAHEncoder(nn.Module):
    """
    
    """
    def __init__(self, d_in: int = 768, d_proj: int = 128,
                 d_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_proj),
        )
        with torch.no_grad():
            nn.init.orthogonal_(self.proj[0].weight)
            nn.init.orthogonal_(self.proj[-1].weight)

    def forward(self, x: torch.Tensor):
        """
        x: (N, d_in) tweet BERT
        return:
        """
        z = self.proj(x)                          # (N, d_proj)
        r = z.norm(dim=-1)                        # (N,)
        u = safe_normalize(z)                     # (N, d_proj)
        return r, u, z

