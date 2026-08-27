"""官方 TTTMLP dual-form 内核的独立提取版（从 ttt-lm-pytorch ttt.py 逐行拷贝，
仅去掉 cache/scan 外壳，用于与 emoscope.ttt 逐步对照）。

来源: test-time-training/ttt-lm-pytorch commit 2025-08 版 ttt.py
      TTTMLP.ttt -> compute_mini_batch (use_dual_form=True 分支)
"""

import torch
import torch.nn.functional as F


def ln_fwd(x, gamma, beta, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    return gamma * x_hat + beta


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = ((1.0 / D) * (D * grad_x_hat - grad_x_hat.sum(dim=-1, keepdim=True)
         - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)) / std)
    return z


def official_compute_mini_batch(params_dict, inputs, ln_weight, ln_bias,
                                num_heads, head_dim):
    """官方 TTTMLP.compute_mini_batch, use_dual_form=True 分支，逐行对应。"""
    W1_init = params_dict["W1_states"]
    b1_init = params_dict["b1_states"]
    W2_init = params_dict["W2_states"]
    b2_init = params_dict["b2_states"]

    XQ_mini_batch = inputs["XQ"]
    XV_mini_batch = inputs["XV"]
    XK_mini_batch = inputs["XK"]
    eta_mini_batch = inputs["eta"]

    X1 = XK_mini_batch
    Z1 = X1 @ W1_init + b1_init
    X2 = F.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2_init + b2_init
    reconstruction_target = XV_mini_batch - XK_mini_batch

    ln_w = ln_weight.reshape(num_heads, 1, head_dim)
    ln_b = ln_bias.reshape(num_heads, 1, head_dim)
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_w, ln_b)
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * F.gelu(Z1, approximate="tanh")

    Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
    b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
    Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
    X2_bar = F.gelu(Z1_bar, approximate="tanh")
    Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
    b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
    Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

    last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
    W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
    b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
    W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
    b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

    Z2_bar = ln_fwd(Z2_bar, ln_w, ln_b)
    XQW_mini_batch = XQ_mini_batch + Z2_bar

    last_param_dict = {
        "W1_states": W1_last, "b1_states": b1_last,
        "W2_states": W2_last, "b2_states": b2_last,
    }
    return last_param_dict, XQW_mini_batch


def official_get_eta(X, mini_batch_step_offset, mini_batch_size,
                     learnable_ttt_lr_weight, learnable_ttt_lr_bias,
                     token_idx_buf, learnable_token_idx, head_dim,
                     ttt_base_lr=1.0):
    """官方 TTTBase.get_eta 逐行对应。"""
    ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, learnable_ttt_lr_weight) \
        + learnable_ttt_lr_bias.reshape(1, -1, 1, 1, 1)
    ttt_lr = torch.sigmoid(ttt_lr)
    ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
    ttt_lr_eta = ttt_base_lr * ttt_lr / head_dim

    token_idx = token_idx_buf + learnable_token_idx
    token_idx = token_idx[mini_batch_step_offset: mini_batch_step_offset + mini_batch_size]
    token_idx = torch.clamp_min(token_idx, 0.0)

    B, nmb = X.shape[0], X.shape[1]
    token_eta = torch.broadcast_to(
        token_idx.reshape(1, 1, 1, mini_batch_size, 1),
        (B, learnable_ttt_lr_weight.shape[0], nmb, mini_batch_size, 1))
    return token_eta, ttt_lr_eta
