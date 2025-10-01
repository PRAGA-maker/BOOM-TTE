"""
NL_MTP_Model: Full-spec homoiconic transformer for MTP policy evaluation.

Implements:
- Multi-token sequence: [ENV] [CONTEXT] [A:mw] <TIME0> [DO:W+=δ] <PROBE>
- Time-zero attention barrier (pre-time0 cannot see post-time0)
- LoR fast-weights on Q/K/O attention + MLP-out at layers {3,7,11}, rank=8
- Outcome, MDN propensity, support, and descriptor-delta heads
- Alpha gating for LoR amplitude
"""

import os
import sys
import math
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure neurallambda is importable
_nl_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'nl', 'neurallambda', 'src'))
if _nl_src not in sys.path:
    sys.path.insert(0, _nl_src)


def positional_encoding(emb_dim: int, max_len: int = 5000) -> torch.Tensor:
    """Sin/cos positional encoding."""
    pos_enc = torch.zeros(max_len, emb_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim
        
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B = q.size(0)
        
        # Project and split heads
        q = self.q_proj(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product
        scores = torch.einsum('bhse, bhte -> bhst', q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhst, bhtd -> bshd', attn, v)
        out = out.contiguous().view(B, -1, self.emb_dim)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention + FFN."""
    def __init__(self, emb_dim: int, num_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, num_heads)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_dim),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_out = self.self_attn(q, k, v, mask)
        attn_out = self.dropout(attn_out)
        xs = self.norm1(q + attn_out)
        
        # FFN
        ffn_out = self.ffnn(xs)
        ffn_out = self.dropout(ffn_out)
        xs = self.norm2(xs + ffn_out)
        return xs


class LowRankAdapter(nn.Module):
    """Low-rank adapter U @ V for LoR fast-weights."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.V = nn.Parameter(torch.zeros(rank, dim))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
    
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """Return W + U @ V."""
        return W + self.U @ self.V


class MLPHead(nn.Module):
    """2-layer MLP head."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MDNHead(nn.Module):
    """Mixture Density Network with K Gaussian components."""
    def __init__(self, in_dim: int, hidden: int, n_components: int):
        super().__init__()
        self.n_components = n_components
        self.pi_head = MLPHead(in_dim, hidden, n_components)
        self.mu_head = MLPHead(in_dim, hidden, n_components)
        self.log_sigma_head = MLPHead(in_dim, hidden, n_components)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-6, 6)
        return pi, mu, log_sigma
    
    def log_prob(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute log p(a|h) under mixture."""
        pi, mu, log_sigma = self.forward(h)
        sigma = torch.exp(log_sigma)
        a = a.unsqueeze(-1)  # [B, 1]
        # Gaussian log-likelihood per component
        comp = -0.5 * (((a - mu) / (sigma + 1e-8)) ** 2 + 2 * log_sigma + math.log(2 * math.pi))
        # Log-sum-exp with mixture weights
        log_prob = torch.logsumexp(torch.log(pi + 1e-8) + comp, dim=-1)
        return log_prob


class NL_MTP_Model(nn.Module):
    """
    Full-spec homoiconic transformer for MTP policy evaluation on HoF.
    
    Sequence: [ENV] [CONTEXT] [A:mw] <TIME0> [DO:W+=δ] <PROBE>
    LoR applied Q/K/O attention + MLP-out at layers {3,7,11} only for positions >= TIME0.
    """
    def __init__(
        self,
        emb_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        dim_ff: int = 2048,
        lor_layers: Tuple[int, ...] = (3, 7, 11),
        lor_rank: int = 8,
        mdn_components: int = 8,
        ctx_dim: int = 2048 + 167 + 8,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.lor_layers = set(lor_layers)
        self.lor_rank = lor_rank
        
        # Token embeddings
        self.env_proj = nn.Linear(1, emb_dim)
        self.ctx_proj = nn.Linear(ctx_dim, emb_dim)
        self.mw_proj = nn.Linear(1, emb_dim)
        self.time0_emb = nn.Parameter(torch.randn(1, emb_dim))
        self.do_emb = nn.Parameter(torch.randn(1, emb_dim))
        self.probe_emb = nn.Parameter(torch.randn(1, emb_dim))
        
        # Positional encoding
        self.register_buffer('pos_enc', positional_encoding(emb_dim, max_len=10))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderLayer(emb_dim, num_heads, dim_ff, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        
        # LoR adapters for Q/K/O attention + MLP-out at selected layers
        self.adapters_q = nn.ModuleDict({str(i): LowRankAdapter(emb_dim, lor_rank) for i in self.lor_layers})
        self.adapters_k = nn.ModuleDict({str(i): LowRankAdapter(emb_dim, lor_rank) for i in self.lor_layers})
        self.adapters_o = nn.ModuleDict({str(i): LowRankAdapter(emb_dim, lor_rank) for i in self.lor_layers})
        # MLP adapters need custom class since they have different dims [emb_dim, dim_ff]
        self.adapters_mlp = nn.ModuleDict({
            str(i): self._make_mlp_adapter(emb_dim, dim_ff, lor_rank) for i in self.lor_layers
        })
        
        # Heads
        self.outcome_head = MLPHead(emb_dim, emb_dim, 1)
        self.prop_head = MDNHead(emb_dim, emb_dim, mdn_components)
        self.support_head = MLPHead(emb_dim, emb_dim, 1)
        self.desc_delta_head = MLPHead(emb_dim, emb_dim, 8)  # For MMP surrogate
        
        # Alpha gate for LoR amplitude
        self.alpha_gate = MLPHead(emb_dim, emb_dim, 1)
        
        # Learnable scalar ψ for DR-mean
        self.psi_scalar = nn.Parameter(torch.tensor(0.0))
    
    def _make_mlp_adapter(self, out_dim: int, in_dim: int, rank: int) -> nn.Module:
        """Create MLP adapter with correct dimensions."""
        class MLPAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.U = nn.Parameter(torch.zeros(out_dim, rank))
                self.V = nn.Parameter(torch.zeros(rank, in_dim))
                nn.init.xavier_uniform_(self.U)
                nn.init.xavier_uniform_(self.V)
            
            def forward(self, W):
                return W + self.U @ self.V
        
        return MLPAdapter()
    
    def _build_tokens(
        self,
        env_idx: torch.Tensor,
        x_ctx: torch.Tensor,
        mw: torch.Tensor,
        delta: float,
        apply_lor: bool,
    ) -> torch.Tensor:
        """
        Build token sequence: [ENV] [CONTEXT] [A:mw] <TIME0> [DO:W+=δ] <PROBE>
        Returns: [B, 6, D]
        """
        B = x_ctx.size(0)
        device = x_ctx.device
        
        env_tok = self.env_proj(env_idx.float().unsqueeze(-1))  # [B, D]
        ctx_tok = self.ctx_proj(x_ctx)  # [B, D]
        mw_tok = self.mw_proj(mw.unsqueeze(-1))  # [B, D]
        time0_tok = self.time0_emb.expand(B, -1)  # [B, D]
        
        if apply_lor:
            # Encode delta into do-token
            do_tok = self.do_emb.expand(B, -1) + self.mw_proj(torch.full((B, 1), delta, device=device))
        else:
            # Null in observed world
            do_tok = torch.zeros(B, self.emb_dim, device=device)
        
        probe_tok = self.probe_emb.expand(B, -1)  # [B, D]
        
        # Stack: [ENV, CONTEXT, A, TIME0, DO, PROBE]
        tokens = torch.stack([env_tok, ctx_tok, mw_tok, time0_tok, do_tok, probe_tok], dim=1)  # [B, 6, D]
        return tokens
    
    def _build_time0_mask(self, seq_len: int, time0_pos: int, device: str) -> torch.Tensor:
        """
        Attention mask: pre-time0 cannot see post-time0.
        Returns: [S, S] with -inf for blocked positions.
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        # Block positions before time0 from seeing positions >= time0
        mask[:time0_pos, time0_pos:] = float('-inf')
        return mask
    
    def _apply_lor(
        self,
        layer: DecoderLayer,
        layer_ix: int,
        alpha: float,
    ):
        """Apply LoR deltas to Q/K/O attention + MLP-out (second FFN linear)."""
        if layer_ix not in self.lor_layers:
            return
        
        # Attention projections
        attn = layer.self_attn
        attn.q_proj.weight.data = self.adapters_q[str(layer_ix)](attn.q_proj.weight.data).lerp(
            attn.q_proj.weight.data, 1 - alpha
        )
        attn.k_proj.weight.data = self.adapters_k[str(layer_ix)](attn.k_proj.weight.data).lerp(
            attn.k_proj.weight.data, 1 - alpha
        )
        attn.out_proj.weight.data = self.adapters_o[str(layer_ix)](attn.out_proj.weight.data).lerp(
            attn.out_proj.weight.data, 1 - alpha
        )
        
        # MLP output projection (layer.ffnn[3])
        mlp_out = layer.ffnn[3]
        mlp_out.weight.data = self.adapters_mlp[str(layer_ix)](mlp_out.weight.data).lerp(
            mlp_out.weight.data, 1 - alpha
        )
    
    def forward(
        self,
        env_idx: torch.Tensor,
        x_ctx: torch.Tensor,
        mw: torch.Tensor,
        delta: float,
        apply_lor: bool,
    ) -> Dict[str, Any]:
        """
        Forward pass with or without LoR active.
        
        Args:
            env_idx: [B] environment bucket
            x_ctx: [B, ctx_dim] pre-treatment context
            mw: [B] molecular weight (exposure A)
            delta: policy shift (e.g., +14)
            apply_lor: whether LoR is active (policy world)
        
        Returns:
            Dict with m_obs, mdn_params, support, desc_delta, alpha
        """
        B = x_ctx.size(0)
        device = x_ctx.device
        
        # Build token sequence
        tokens = self._build_tokens(env_idx, x_ctx, mw, delta, apply_lor)  # [B, 6, D]
        seq_len = tokens.size(1)
        time0_pos = 3  # Position of <TIME0> token
        
        # Add positional encoding
        tokens = tokens + self.pos_enc[:seq_len, :].unsqueeze(0).to(device)
        
        # Build attention mask
        mask = self._build_time0_mask(seq_len, time0_pos, device)
        
        # Compute alpha gate from pre-time0 representation
        pre_time0_repr = tokens[:, :time0_pos, :].mean(dim=1)  # [B, D]
        alpha = torch.sigmoid(self.alpha_gate(pre_time0_repr)).squeeze(-1)  # [B]
        alpha_scalar = alpha.mean().clamp(0, 1).item()
        
        # Apply LoR if requested (in-place weight modification)
        if apply_lor:
            for i, layer in enumerate(self.layers, start=1):
                self._apply_lor(layer, i, alpha_scalar)
        
        # Forward through transformer
        xs = tokens
        for layer in self.layers:
            xs = layer(q=xs, k=xs, v=xs, mask=mask)
        
        xs = self.norm(xs)
        
        # Use PROBE token (last position) for readout
        h_probe = xs[:, -1, :]
        
        return {
            "m_obs": self.outcome_head(h_probe).squeeze(-1),
            "mdn_params": self.prop_head(h_probe),
            "support": torch.sigmoid(self.support_head(h_probe).squeeze(-1)),
            "desc_delta": self.desc_delta_head(h_probe),
            "alpha": alpha,
        }
