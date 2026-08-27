"""TTT-MLP 证据定位头：隐藏状态为一个可在线更新的两层 MLP。

忠实移植官方 ttt-lm-pytorch 的 TTTMLP 内核（隐藏状态=权重、自监督重建、
内环一步梯度、可学习 lr 门控 η、mini-batch 顺序扫描），裁剪语言建模专用的
RoPE/Conv/QK共享；patch 特征天然自含位置，直接 XK=XV=特征投影。

机制（对一张图，序列=576 个 patch 按光栅序）：
  共享初始化 Θ0（可学习）→ 顺序扫过 mini-batch：
    每步用重建损失 ∥Θ(XK)−(XV−XK)∥² 对 Θ 做一步内环梯度（η 门控步长）
    读出 s_i = LN(Θ(XQ_i))，即该 patch 的证据分数
训练时外环梯度经内环反传（用官方 dual form 的等价扫描实现）。

参考文献: "Learning to (Learn at Test Time): RNNs with Expressive Hidden
States" (arXiv 2407.04620), 官方实现 test-time-training/ttt-lm-pytorch。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ln_fused_l2_bwd(x: torch.Tensor, target: torch.Tensor,
                     gamma: torch.Tensor, beta: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """LayerNorm 前向与 L2 损失梯度的融合计算（官方同款，返回 ∂L/∂x）。"""
    d = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_out = y - target
    grad_x_hat = grad_out * gamma
    return ((d * grad_x_hat - grad_x_hat.sum(dim=-1, keepdim=True)
             - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True))
            / d / std)


class TTTMLPLocalizer(nn.Module):
    """输入 (B,576,in_dim) patch 特征 -> (B,576) 证据分数 logits。

    hidden: TTT 隐藏 MLP 宽度；mini_batch: 内环扫描粒度（官方默认 16）。
    """

    def __init__(self, in_dim: int = 1024, hidden: int = 128,
                 expand: int = 4, num_heads: int = 4,
                 mini_batch: int = 16):
        super().__init__()
        self.mini_batch = mini_batch
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        f = self.head_dim * expand

        self.in_proj = nn.Linear(in_dim, hidden, bias=False)
        # TTT 隐藏状态 Θ0 = (W1,b1,W2,b2)，逐头、全局共享初始值
        self.W1 = nn.Parameter(torch.normal(0, 0.02, (num_heads, self.head_dim, f)))
        self.b1 = nn.Parameter(torch.zeros(num_heads, 1, f))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, (num_heads, f, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(num_heads, 1, self.head_dim))
        # 逐 patch 学习率门控 η（官方 learnable_ttt_lr）
        self.lr_w = nn.Parameter(torch.zeros(self.head_dim, 1))
        self.lr_b = nn.Parameter(torch.zeros(1, 1))
        # 读出归一化与投影
        self.norm_w = nn.Parameter(torch.ones(self.head_dim))
        self.norm_b = nn.Parameter(torch.zeros(self.head_dim))
        self.out_proj = nn.Linear(hidden, 1, bias=True)
        # 位置编码保留：光栅序之外的绝对位置先验
        self.pos = nn.Parameter(torch.zeros(1, 576, in_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, feats: torch.Tensor,
                cls_score: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_proj(feats + self.pos)                     # (B,N,hid)
        x = self._split_heads(x)                               # (B,H,N,d)
        b, h, n, d = x.shape

        theta = {k: v.expand(b, *v.shape).contiguous()
                 for k, v in (("W1", self.W1), ("b1", self.b1),
                              ("W2", self.W2), ("b2", self.b2))}
        # η = 门控的逐 patch 步长（官方 learnable_ttt_lr 语义），逐头独立
        eta = torch.sigmoid(x @ self.lr_w + self.lr_b) * 2.0   # (B,H,N,1)∈(0,2)

        outs = []
        for s in range(0, n, self.mini_batch):
            e = min(s + self.mini_batch, n)
            xk, xv, xq = x[:, :, s:e], x[:, :, s:e], x[:, :, s:e]
            # --- 内环一步梯度（官方 dual form, 因果: mini-batch 内 tril）---
            z1 = xk @ theta["W1"] + theta["b1"]
            z2 = F.gelu(z1, approximate="tanh") @ theta["W2"] + theta["b2"]
            g2 = _ln_fused_l2_bwd(z2, xv - xk, self.norm_w, self.norm_b)
            g1 = g2 @ theta["W2"].transpose(-1, -2) * F.gelu(z1, approximate="tanh")
            # 读出（含 mini-batch 内因果更新）
            k = e - s
            causal = torch.tril(torch.ones(k, k, device=x.device, dtype=x.dtype))
            attn1 = xq @ xk.transpose(-1, -2) * causal
            z1_q = xq @ theta["W1"] + theta["b1"] \
                - (eta[:, :, s:e] * attn1) @ (g1 * eta[:, :, s:e])
            x2_q = F.gelu(z1_q, approximate="tanh")
            attn2 = x2_q @ F.gelu(z1, approximate="tanh").transpose(-1, -2) * causal
            z2_q = x2_q @ theta["W2"] + theta["b2"] \
                - (eta[:, :, s:e] * attn2) @ (g2 * eta[:, :, s:e])
            outs.append(z2_q * self.norm_w + self.norm_b)
            # --- 更新隐藏状态（mini-batch 末尾一步，供下一段使用）---
            eta_last = eta[:, :, e - 1:e]
            theta["W1"] = theta["W1"] - (eta_last * xk).transpose(-1, -2) @ (g1 * eta[:, :, s:e])
            theta["b1"] = theta["b1"] - (g1 * eta[:, :, s:e]).sum(dim=2, keepdim=True)
            theta["W2"] = theta["W2"] - (eta_last * F.gelu(z1, approximate="tanh")).transpose(-1, -2) @ (g2 * eta[:, :, s:e])
            theta["b2"] = theta["b2"] - (g2 * eta[:, :, s:e]).sum(dim=2, keepdim=True)

        y = torch.cat(outs, dim=2)                             # (B,H,N,d)
        y = y.transpose(1, 2).reshape(b, n, -1)
        return self.out_proj(y).squeeze(-1)                    # (B,N)
