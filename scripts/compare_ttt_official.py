#!/usr/bin/env python3
"""TTT 头与官方 TTTMLP 的逐位对照：同一输入、对齐权重、比输出与中间量。

官方中性化（使其只剩 TTT 内核，与我们的裁剪对齐）：
  - q/k/v 投影置单位阵（等价 XQ=XK=XV=x，即我们的场景性裁剪）
  - RoPE 置恒等（我们无 RoPE 的声明性裁剪）
  - use_gate=False, share_qk=False（绕开 Conv）
对比链：eta → Z2_bar(dual读出) → 末端输出 → 整序列输出。
"""

import sys

sys.path.insert(0, "/tmp/ttt-lm-pytorch")   # 官方仓
sys.path.insert(0, "/home/ubuntu2404/SSD/EMO-SCOPE/src")

import torch
import torch.nn as nn

import ttt as official
from emoscope.ttt import TTTMLPLocalizer

torch.manual_seed(0)
H, WIDTH, MINI = 4, 128, 16
B, N = 2, 576
DTYPE = torch.float64

# ---- 官方实例（eval、no_grad，中性化）----
cfg = official.TTTConfig(hidden_size=WIDTH, num_attention_heads=H,
                         mini_batch_size=MINI, share_qk=False, use_gate=False)
off = official.TTTMLP(cfg, layer_idx=0).to(DTYPE).eval()
eye = torch.eye(WIDTH, dtype=DTYPE)
with torch.no_grad():
    off.q_proj.weight.copy_(eye)
    off.k_proj.weight.copy_(eye)
    off.v_proj.weight.copy_(eye)
# RoPE 恒等化：模块内 apply_rotary_pos_emb 直接返回原值
official.apply_rotary_pos_emb = (lambda q, k, cos, sin,
                                 position_ids=None, unsqueeze_dim=1: (q, k))

# ---- 我的实例，权重逐个从官方拷入 ----
mine = TTTMLPLocalizer(in_dim=WIDTH).to(DTYPE).eval()
with torch.no_grad():
    mine.in_proj.weight.copy_(eye)
    mine.pos = torch.nn.Parameter(torch.zeros(1, N, WIDTH, dtype=DTYPE))
    for p in ["W1", "b1", "W2", "b2"]:
        getattr(mine, p).copy_(getattr(off, p))
    mine.learnable_ttt_lr_weight.copy_(off.learnable_ttt_lr_weight)
    mine.learnable_ttt_lr_bias.copy_(off.learnable_ttt_lr_bias)
    mine.learnable_token_idx.copy_(off.learnable_token_idx)
    # 官方 ttt_norm 为 (H,d) 逐头；我的 (d,) 广播——逐头相同才可压
    assert torch.allclose(off.ttt_norm_weight, off.ttt_norm_weight[0].expand(H, -1))
    mine.ttt_norm_weight.copy_(off.ttt_norm_weight[0])
    mine.ttt_norm_bias.copy_(off.ttt_norm_bias[0])
    mine.post_norm.weight.copy_(off.post_norm.weight)
    mine.post_norm.bias.copy_(off.post_norm.bias)
    mine.o_proj.weight.copy_(off.o_proj.weight)
    mine.out_head = nn.Identity()          # 对齐官方输出维度 (B,N,width)

x = torch.randn(B, N, WIDTH, dtype=DTYPE)

# ---- 中间量1：eta（官方 get_eta vs 我的 _get_eta，同为 (B,H,K,K)）----
with torch.no_grad():
    x_mb = x.view(B, N // MINI, MINI, WIDTH)      # 官方期望的重排形态
    te_off, lr_off = off.get_eta(x_mb, 0, MINI)  # (B,H,nmb,K,1) 与 (B,H,nmb,1,K)
    eta_off = (te_off * lr_off)[:, :, 0]         # 取第一个 mini-batch (B,H,K,K)
    eta_mine = mine._get_eta(x[:, :MINI, :], 0)
    print(f"[eta]     最大差 = {(eta_off - eta_mine).abs().max().item():.2e}")

# ---- 最终输出 ----
with torch.no_grad():
    y_off = off(x, position_ids=torch.arange(N).unsqueeze(0).expand(B, N))
    y_mine = mine(x)
    diff = (y_off - y_mine).abs()
    print(f"[输出]    最大差 = {diff.max().item():.2e}  均值 = {diff.mean().item():.2e}")
    print(f"[输出]    相对误差 = {diff.max().item() / y_off.abs().max().item():.2e}")
    if diff.max().item() < 1e-6:
        print("✓ 完全对齐（机器精度内）")
        bad = -1
    else:
        per_tok = diff.amax(dim=(0, 2))
        bad = next(i for i in range(N) if per_tok[i] > 1e-6)
        print(f"首个不一致 token = {bad}（mini-batch {bad // MINI} 内第 {bad % MINI} 位）")
        print("逐 token 差(首个mini-batch):",
              [f"{v:.0e}" for v in per_tok[:MINI].tolist()])
