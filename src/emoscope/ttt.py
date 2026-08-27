"""TTT-MLP 证据定位头：隐藏状态为一个可在线更新的两层 MLP。

严格对齐官方 ttt-lm-pytorch 的 TTTMLP 前向（dual form 分支逐行核对）：
  - η = token_idx(1/i 递减) * (base_lr/head_dim) * sigmoid(lr_gate)，base_lr=1.0
  - mini-batch 内：Attn1 = tril(XQ @ X1^T)，Z1_bar = XQ@W1 - (η*Attn1)@G1 + b1_bar
  - mini-batch 间：传递完整 8 元组（4 权重 + 4 梯度残量），末 token 更新
  - 读出 = XQ + ln_fwd(Z2_bar)（XQ 残差），末端 post_norm + o_proj
裁剪仅两处（场景性，已在注释声明）：无 RoPE/Conv（patch 特征自含），
in_proj 单投影替代 QKV 三投影（XK=XV=XQ），位置编码在 in_proj 前注入。

参考文献: "Learning to (Learn at Test Time)" (arXiv 2407.04620)，
官方实现 test-time-training/ttt-lm-pytorch ttt.py::TTTMLP。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TTT_BASE_LR = 1.0  # 官方 TTTConfig.ttt_base_lr 默认值


def _gelu_bwd(x):
    """tanh 近似 gelu 的导数（官方 ttt.py::gelu_bwd 逐系数同款）。"""
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out)
                      * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)


def _ln_fwd(x, gamma, beta, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + eps)
    return gamma * (x - mu) / std + beta


def _ln_fused_l2_bwd(x, target, gamma, beta, eps=1e-6):
    """LayerNorm 前向与 L2 损失梯度的融合（官方同款，返回 ∂L/∂x）。"""
    d = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + eps)
    x_hat = (x - mu) / std
    grad_out = gamma * x_hat + beta - target
    grad_x_hat = grad_out * gamma
    return ((d * grad_x_hat - grad_x_hat.sum(dim=-1, keepdim=True)
             - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)) / d / std)


class TTTMLPLocalizer(nn.Module):
    """输入 (B,576,in_dim) patch 特征 -> (B,576) 证据分数 logits。

    hidden: TTT 宽度；mini_batch: 官方默认 16；expand: TTT MLP 隐层扩张 4 倍。
    """

    def __init__(self, in_dim: int = 1024, hidden: int = 128,
                 expand: int = 4, num_heads: int = 4, mini_batch: int = 16):
        super().__init__()
        assert hidden % num_heads == 0
        self.n_patches = None  # 首次 forward 时按输入长度建位置编码
        self.width = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.mini_batch = mini_batch
        f = self.head_dim * expand

        self.in_proj = nn.Linear(in_dim, hidden, bias=False)
        # TTT 隐藏状态初始 Θ0（官方 8 元组中的 4 权重）
        self.W1 = nn.Parameter(torch.normal(0, 0.02, (num_heads, self.head_dim, f)))
        self.b1 = nn.Parameter(torch.zeros(num_heads, 1, f))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, (num_heads, f, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(num_heads, 1, self.head_dim))
        # 官方 learnable_ttt_lr（官方源码 _init_ttt_lr_gate 实测形状）：
        #   weight = stack(Linear(width,1).weight) -> [H, 1, width]
        #   bias   = stack(Linear(width,1).bias)   -> [H, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.normal(0, 0.02, (num_heads, 1, hidden)))
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.zeros(num_heads, 1))
        # 官方 token_idx（1/i 递减）与可学习修正
        self.register_buffer(
            "token_idx", 1.0 / torch.arange(1, mini_batch + 1), persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros(mini_batch))
        # 官方 ttt_ln 与末端 post_norm/o_proj
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.head_dim))
        self.post_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_head = nn.Linear(hidden, 1, bias=True)
        # 场景性位置编码（官方无，patch 打分需要绝对位置先验）。
        # 必须在构造时创建：优化器先于首次前向构建，惰性创建会漏出参数表，
        # 导致 pos 永远冻结在随机初始化上。
        self.pos = nn.Parameter(torch.zeros(1, 576, in_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _get_eta(self, x_mb: torch.Tensor, offset: int) -> torch.Tensor:
        """官方 get_eta 同构：返回 eta (B,H,K,width)。

        ttt_lr = sigmoid(einsum) -> (B,H,K,width)；token_eta = 1/i (B,H,K,1)；
        官方实测 eta = token_eta * ttt_lr_eta 形状 (B,H,nmb,K,K)。
        tril(eta) 在最后两维 (K,width) 上做下三角（width≥K 时等效前 K 列），
        (eta*Attn1)@G1: (B,H,K,width)@(B,H,K,head_dim) -> (B,H,K,head_dim)。
        """
        ttt_lr = torch.sigmoid(
            torch.einsum("bkc,hoc->bhko", x_mb, self.learnable_ttt_lr_weight)
            + self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1))       # (B,H,K,1)
        ttt_lr_eta = TTT_BASE_LR * ttt_lr.permute(0, 1, 3, 2) / self.head_dim  # (B,H,1,K) 列门控
        token_idx = (self.token_idx + self.learnable_token_idx)[:x_mb.shape[1]]
        token_idx = token_idx.clamp_min(0.0)
        token_eta = token_idx.reshape(1, 1, x_mb.shape[1], 1)
        return token_eta * ttt_lr_eta                                # (B,H,K,K)

    def forward(self, feats: torch.Tensor,
                cls_score: torch.Tensor | None = None) -> torch.Tensor:
        assert feats.shape[1] == self.pos.shape[1], "TTT头固定576 patch输入"
        x = self.in_proj(feats + self.pos)                           # (B,N,hid)
        b, n, _ = x.shape
        xh = x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # 官方 8 元组初始：4 权重 tile 到 batch + 4 零梯度残量
        theta = {
            "W1_states": self.W1.unsqueeze(0).expand(b, *self.W1.shape).contiguous(),
            "b1_states": self.b1.unsqueeze(0).expand(b, *self.b1.shape).contiguous(),
            "W2_states": self.W2.unsqueeze(0).expand(b, *self.W2.shape).contiguous(),
            "b2_states": self.b2.unsqueeze(0).expand(b, *self.b2.shape).contiguous(),
            "W1_grad": torch.zeros(b, *self.W1.shape, device=x.device),
            "b1_grad": torch.zeros(b, *self.b1.shape, device=x.device),
            "W2_grad": torch.zeros(b, *self.W2.shape, device=x.device),
            "b2_grad": torch.zeros(b, *self.b2.shape, device=x.device),
        }

        outs = []
        for s in range(0, n, self.mini_batch):
            e = min(s + self.mini_batch, n)
            xq = xk = xv = xh[:, :, s:e]                             # 场景性裁剪: XK=XV=XQ
            eta = self._get_eta(x[:, s:e, :], s % self.mini_batch)  # (B,H,K,K)
            k = e - s

            # ---- 官方 dual form（use_dual_form=True 分支）逐行对应 ----
            W1_init, b1_init = theta["W1_states"], theta["b1_states"]
            W2_init, b2_init = theta["W2_states"], theta["b2_states"]
            X1 = xk
            Z1 = X1 @ W1_init + b1_init                              # (B,H,K,f)
            X2 = F.gelu(Z1, approximate="tanh")
            Z2 = X2 @ W2_init + b2_init                              # (B,H,K,d)
            reconstruction_target = xv - xk
            G2 = _ln_fused_l2_bwd(Z2, reconstruction_target,
                                  self.ttt_norm_weight, self.ttt_norm_bias)
            G1 = G2 @ W2_init.transpose(-2, -1) * _gelu_bwd(Z1)  # 链式法则用导数

            Attn1 = torch.tril(xq @ X1.transpose(-2, -1))            # (B,H,K,K)
            b1_bar = b1_init - torch.tril(eta) @ G1
            Z1_bar = xq @ W1_init - (eta * Attn1) @ G1 + b1_bar
            X2_bar = F.gelu(Z1_bar, approximate="tanh")
            Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
            b2_bar = b2_init - torch.tril(eta) @ G2
            Z2_bar = X2_bar @ W2_init - (eta * Attn2) @ G2 + b2_bar

            eta_last = eta[:, :, -1, :, None]                        # (B,H,1,1)
            W1_last = W1_init - (eta_last * X1).transpose(-1, -2) @ G1
            b1_last = b1_init - (eta_last * G1).sum(dim=2, keepdim=True)
            W2_last = W2_init - (eta_last * X2).transpose(-1, -2) @ G2
            b2_last = b2_init - (eta_last * G2).sum(dim=2, keepdim=True)
            theta["W1_states"], theta["b1_states"] = W1_last, b1_last
            theta["W2_states"], theta["b2_states"] = W2_last, b2_last
            # 官方 dual form 分支：梯度残量 = 末参数（primal form 的累积等价物）
            theta["W1_grad"], theta["b1_grad"] = W1_last, b1_last
            theta["W2_grad"], theta["b2_grad"] = W2_last, b2_last

            # 读出 = XQ 残差 + ln_fwd(Z2_bar)（官方 XQW_mini_batch）
            z2n = _ln_fwd(Z2_bar, self.ttt_norm_weight, self.ttt_norm_bias)
            outs.append(xq + z2n)

        y = torch.cat(outs, dim=2)                                   # (B,H,N,d)
        y = y.transpose(1, 2).reshape(b, n, self.width)
        y = self.post_norm(y)
        y = self.o_proj(y)
        return self.out_head(y).squeeze(-1)                          # (B,N)
