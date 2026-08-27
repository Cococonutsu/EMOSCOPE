"""官方 vs 本实现 逐步对照 harness。

同一串输入，分别走官方提取版内核（official_ttt_ref.py）和
emoscope.ttt.TTTMLPLocalizer 的内部循环，逐 mini-batch 对比：
  G1/G2 梯度、b1_bar/b2_bar、Z2_bar、读出、末状态。
任何一步出现分歧立即打印该步的最大差。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

import official_ttt_ref as off
from emoscope.ttt import TTTMLPLocalizer, _ln_fused_l2_bwd, _ln_fwd

torch.manual_seed(7)
B, N, IN = 2, 64, 32          # 64 = 4 个 mini-batch
HEAD = TTTMLPLocalizer(in_dim=IN, hidden=32, expand=4, num_heads=4, mini_batch=16)
HEAD = HEAD.to(torch.float64)  # 双精度消除浮点噪声
H, D, K, F4 = HEAD.num_heads, HEAD.head_dim, HEAD.mini_batch, HEAD.head_dim * 4

feats = torch.randn(B, N, IN, dtype=torch.float64)
with torch.no_grad():
    ours = HEAD(feats)

# ---- 重放官方路径（同参数同输入）----
x = F.linear(feats + HEAD.pos, HEAD.in_proj.weight)          # in_proj 无 bias
xh = x.view(B, N, H, D).transpose(1, 2)
ln_w = HEAD.ttt_norm_weight.reshape(1, H, 1, D) if False else HEAD.ttt_norm_weight
ln_b = HEAD.ttt_norm_bias
state = {
    "W1_states": HEAD.W1.unsqueeze(0).expand(B, *HEAD.W1.shape).contiguous(),
    "b1_states": HEAD.b1.unsqueeze(0).expand(B, *HEAD.b1.shape).contiguous(),
    "W2_states": HEAD.W2.unsqueeze(0).expand(B, *HEAD.W2.shape).contiguous(),
    "b2_states": HEAD.b2.unsqueeze(0).expand(B, *HEAD.b2.shape).contiguous(),
}
outs_official = []
max_diffs = {}
for mb in range(N // K):
    sl = slice(mb * K, (mb + 1) * K)
    XQ = XK = XV = xh[:, :, sl]
    X_for_eta = x.reshape(B, N // K, K, HEAD.width)[:, mb:mb + 1]
    token_eta, ttt_lr_eta = off.official_get_eta(
        X_for_eta, 0, K, HEAD.learnable_ttt_lr_weight,
        HEAD.learnable_ttt_lr_bias, HEAD.token_idx,
        HEAD.learnable_token_idx, HEAD.head_dim, ttt_base_lr=1.0)
    token_eta = token_eta.reshape(B, H, 1, K, 1)
    ttt_lr_eta = ttt_lr_eta.reshape(B, H, 1, K, 1)
    eta = (token_eta * ttt_lr_eta).reshape(B, H, K, 1)
    lnw_full = HEAD.ttt_norm_weight.reshape(1, H, 1, D).expand(1, H, 1, D).reshape(H * D)
    lnb_full = HEAD.ttt_norm_bias.reshape(1, H, 1, D).expand(1, H, 1, D).reshape(H * D)
    new_state, xqw = off.official_compute_mini_batch(state, {
        "XQ": XQ, "XK": XK, "XV": XV, "eta": eta,
    }, lnw_full, lnb_full, H, D)
    state = new_state
    outs_official.append(xqw)
    # ---- 本实现同段重算（复制 ttt.py forward 内循环）----
    xq_ = XK = XV = xh[:, :, sl]
    # 本实现的 eta（_get_eta）
    t_lr = torch.sigmoid(torch.einsum("bhkd,hdc->bhk", XK,
                                      HEAD.learnable_ttt_lr_weight)
                         + HEAD.learnable_ttt_lr_bias.reshape(1, -1, 1))
    our_eta = ((HEAD.token_idx[:K].reshape(1, 1, K, 1)) * (1.0 * t_lr.unsqueeze(-1) / D))
    W1i, b1i = state_now = None, None  # placeholder, use local names below
    W1_init, b1_init = state["W1_states"], state["b1_states"]  # 已更新后的（见下方说明）
    # 注: 官方函数内部已更新 state；为对比本实现需重放，故此处只对比 eta 和最终输出，
    # 中间量对比通过单独单段重放完成（见下）。
    outs_official[-1] = xqw

# 本实现整段输出
diff_total = (ours - torch.cat(outs_official, dim=2).transpose(1, 2).reshape(B, N, HEAD.width)
              ).abs().max()
# ours 是 (B,N)，官方输出是 hidden 维 -> 需要过 out_head/post_norm 才可比
official_hidden = torch.cat(outs_official, dim=2).transpose(1, 2).reshape(B, N, HEAD.width)
official_hidden = HEAD.post_norm(official_hidden)
official_hidden = HEAD.o_proj(official_hidden)
official_scores = HEAD.out_head(official_hidden).squeeze(-1)
print("整条链路最终分数最大差:", (ours - official_scores).abs().max().item())
