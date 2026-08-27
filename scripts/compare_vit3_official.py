#!/usr/bin/env python3
"""ViT³ 风格 TTT 头与官方 ViTTT 代码的逐位对照。

官方 TTT 块含 SwiGLU 与 3x3 卷积两条支路；我们只移植 SwiGLU 支路，
故对照对象 = 官方 inner_train_simplified_swiglu + 读出 (q@W1')·silu(q@W2')。
scale 对齐为官方值 9^-0.5（官方注释：用 head_dim^-0.5 性能相近；
且内环梯度归一化后 scale 影响基本被抵消）。
"""

import sys

sys.path.insert(0, "/tmp/ViTTT")                      # 官方仓
sys.path.insert(0, "/home/ubuntu2404/SSD/EMO-SCOPE/src")

import torch
import torch.nn.functional as F

from ttt_block import TTT as OfficialTTT              # 官方
from emoscope.ttt import TTTMLPLocalizer              # 我们的

torch.manual_seed(0)
DIM, H, B, N = 128, 4, 2, 576
DTYPE = torch.float64

off = OfficialTTT(dim=DIM, num_heads=H).to(DTYPE).eval()
mine = TTTMLPLocalizer(in_dim=DIM, hidden=DIM, num_heads=H).to(DTYPE).eval()

with torch.no_grad():
    # qkv 对齐：官方 qkv 输出 [q1,k1,v1 | q2,k2,v2]，前 3*dim 列是 SwiGLU 支路
    mine.qkv.weight.copy_(off.qkv.weight[: 3 * DIM])
    mine.qkv.bias.copy_(off.qkv.bias[: 3 * DIM])
    mine.in_proj.weight.copy_(torch.eye(DIM, dtype=DTYPE))   # 恒等，直接喂同特征
    mine.pos.zero_()
    mine.w1.copy_(off.w1)
    mine.w2.copy_(off.w2)
    mine.scale = off.scale                                   # 对齐官方 9^-0.5

x = torch.randn(B, N, DIM, dtype=DTYPE)

with torch.no_grad():
    # ---- 官方路径：用官方模块自身的投影与内环函数 ----
    q1, k1, v1 = off.qkv(x)[..., : 3 * DIM].chunk(3, dim=-1)
    hs = lambda t: t.reshape(B, N, H, -1).transpose(1, 2)
    q, k, v = hs(q1), hs(k1), hs(v1)
    w1n, w2n = off.inner_train_simplified_swiglu(k, v, off.w1, off.w2)
    x1_off = (q @ w1n) * F.silu(q @ w2n)                      # 官方读出

    # ---- 我们的路径：手动执行 forward 内核（避开 out_head 维度）----
    xx = mine.in_proj(x + mine.pos)
    qq, kk, vv = mine.qkv(xx).chunk(3, dim=-1)
    qm, km, vm = hs(qq), hs(kk), hs(vv)
    z1 = km @ mine.w1
    z2 = km @ mine.w2
    sig = torch.sigmoid(z2)
    e = -vm / float(N) * mine.scale
    g1 = km.transpose(-2, -1) @ (e * z2 * sig)
    g2 = km.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))
    g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
    g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
    w1m = mine.w1 - mine.inner_lr * g1
    w2m = mine.w2 - mine.inner_lr * g2
    x1_mine = (qm @ w1m) * F.silu(qm @ w2m)

    d_w1 = (w1n - w1m).abs().max().item()
    d_w2 = (w2n - w2m).abs().max().item()
    d_out = (x1_off - x1_mine).abs().max().item()
    rel = d_out / x1_off.abs().max().item()
    print(f"[内环W1'] 最大差 = {d_w1:.2e}")
    print(f"[内环W2'] 最大差 = {d_w2:.2e}")
    print(f"[读出]   最大差 = {d_out:.2e}  相对误差 = {rel:.2e}")
    print("✓ 与官方完全对齐" if rel < 1e-10 else "✗ 不一致，需排查")

# 梯度完整性：所有参数（含 pos/qkv/w1/w2）都应收到梯度
head = TTTMLPLocalizer()
s = head(torch.randn(2, 576, 1024))
s.square().mean().backward()
no_grad = [n for n, p in head.named_parameters() if p.grad is None]
print(f"梯度覆盖: {len(no_grad) == 0 and '全部参数有梯度 ✓' or f'缺梯度: {no_grad} ✗'}")
