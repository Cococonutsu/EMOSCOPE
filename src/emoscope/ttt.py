"""TTT 证据定位头（ViT³ 风格）：对每张图做一次全图内环适应。

依据 ViT³ (arXiv 2512.01643, LeapLabTHU/ViTTT) 的视觉 TTT 设计：
  - 图像无因果序，不做语言式 mini-batch 因果扫描，全图一次内环梯度步；
  - 内环梯度逐头归一化 g/(||g||+1)（官方注释明写 for stability）；
  - 内部模块用简化 SwiGLU，闭式梯度 e = -v/N·scale 直接给出；
  - 读出 = 更新后的内模块作用于 q：(q@W1')·silu(q@W2')。
适应是逐图的（transductive）：W' 由这张图的 k,v 现场算出，推理即适应。

参考文献: ViT³: Unlocking Test-Time Training in Vision (官方 ttt_block.py)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTMLPLocalizer(nn.Module):
    """输入 (B,576,in_dim) patch 特征 -> (B,576) 证据分数 logits。"""

    def __init__(self, in_dim: int = 1024, hidden: int = 128, num_heads: int = 4,
                 inner_lr: float = 1.0):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.inner_lr = inner_lr
        self.scale = self.head_dim ** -0.5

        self.in_proj = nn.Linear(in_dim, hidden, bias=False)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=True)
        # 内环模块初始权重（ViT³ 同款形状与初始化）
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, self.head_dim, self.head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, self.head_dim, self.head_dim))
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        self.post_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.out_head = nn.Linear(hidden, 1, bias=True)
        # 场景性位置编码（构造时创建，保证进入优化器参数表）
        self.pos = nn.Parameter(torch.zeros(1, 576, in_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, feats: torch.Tensor,
                cls_score: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = feats.shape
        x = self.in_proj(feats + self.pos)                       # (B,N,hid)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(t):
            return t.view(b, n, self.num_heads, -1).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)                   # (B,H,N,d)

        # ---- ViT³ 简化 SwiGLU 内环：全图一步，闭式梯度 ----
        z1 = k @ self.w1                                         # (B,H,N,d)
        z2 = k @ self.w2
        sig = torch.sigmoid(z2)
        a = z2 * sig
        e = -v / float(n) * self.scale                          # dl/dv_hat
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))
        # 逐头梯度归一化（ViT³ 稳定性关键）
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        w1 = self.w1 - self.inner_lr * g1
        w2 = self.w2 - self.inner_lr * g2

        # ---- 读出：适应后的模块作用于 q ----
        y = (q @ w1) * F.silu(q @ w2)                            # (B,H,N,d)
        y = y.transpose(1, 2).reshape(b, n, -1)
        return self.out_head(self.post_norm(y)).squeeze(-1)     # (B,N)
