"""视觉 token 剪枝：用 CLIP 自身的 [CLS] 注意力给 576 个 patch 打分，只保留
分数最高的前若干个送入 LLM，训练免费。思路参考 arXiv 2412.01818。

本模块基于 HF transformers 栈独立实现（官方仓库基于其自有 LLaVA 框架）：
  - 特征取视觉塔倒数第二层（v1.5 的特征层），完整 576 个 patch 过 projector；
  - 重要性 = 同层注意力先按头平均，再取 [CLS] 行、丢 [CLS] 列（对 576 个 patch）；
  - 按分数选前 k 个，并把下标升序排好（等价于保持 token 原始位置顺序）；
  - 文本里 <image> 只留一个 token，k 个视觉嵌入在拼接阶段替换它。
sdpa 后端不返回注意力权重，视觉塔须切 eager（LLM 仍为 sdpa）。
"""

from __future__ import annotations

import torch


def enable_vision_attention(model) -> None:
    """视觉塔切 eager，使其 forward 能返回注意力权重。"""
    model.model.vision_tower.set_attn_implementation("eager")


def vision_feats_scores(model, pixel_values) -> tuple[torch.Tensor, torch.Tensor]:
    """视觉塔一次前向，返回 (投影前 patch 特征, 每个 patch 的重要性)。

    特征取倒数第二层；重要性 = 同层注意力按头平均后的 [CLS] 行（丢 CLS 列）。
    """
    vis = model.model.vision_tower(pixel_values, output_hidden_states=True,
                                   output_attentions=True)
    feats = vis.hidden_states[-2][:, 1:]                  # (B,576,1024)
    scores = vis.attentions[-2].mean(dim=1)[:, 0, 1:]     # (B,576)
    return feats, scores


@torch.no_grad()
def generate_pruned(model, processor, images: list, prompt: str, keep_pct: float,
                    max_new_tokens: int = 16, localizer=None,
                    random: bool = False) -> list[str]:
    """对一批同模板图片剪枝视觉 token 后生成，返回每张图的回答文本。

    keep_pct 为保留 token 的百分比（0-100），内部换算成个数。
    分数来源三选一：random=均匀随机（零信息对照）；localizer=证据定位头；
    默认 CLS 注意力（须先 enable_vision_attention）。
    """
    device = model.device
    bsz = len(images)
    n_keep = max(1, round(576 * min(keep_pct, 100.0) / 100.0))

    # 视觉侧：全部 576 个 patch 投影 + 分数来源
    px = processor.image_processor(images=images,
                                   return_tensors="pt")["pixel_values"]
    px = px.to(device, torch.bfloat16)
    if random:
        feats = model.model.vision_tower(px, output_hidden_states=True)\
                  .hidden_states[-2][:, 1:]                 # (B,576,1024)
        scores = torch.rand(bsz, 576, device=device)        # 零信息对照
    elif localizer is not None:
        feats = model.model.vision_tower(px, output_hidden_states=True)\
                  .hidden_states[-2][:, 1:]                 # (B,576,1024)
        scores = localizer(feats.float())                   # (B,576)
    else:
        feats, scores = vision_feats_scores(model, px)
    feats = model.model.multi_modal_projector(feats)         # (B,576,4096)

    # 选前 n_keep 个，下标升序 = 保留 token 的原始位置顺序
    keep_idx = scores.topk(n_keep, dim=1).indices.sort(dim=1).values
    gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
    picked = torch.gather(feats, 1, gather_idx)             # (B,n_keep,4096)

    # 文本侧：<image> 单 token 编码，k 个视觉嵌入替换它
    tok = processor.tokenizer
    enc = tok([f"USER: <image>\n{prompt} ASSISTANT:"] * bsz,
              return_tensors="pt", padding=True)
    ids, am = enc.input_ids.to(device), enc.attention_mask.to(device)
    word = model.model.language_model.embed_tokens(ids)     # (B,L,4096)
    cut = int((ids[0] == model.config.image_token_index).nonzero().item())

    embeds = torch.cat([word[:, :cut], picked, word[:, cut + 1:]], dim=1)
    am = torch.cat([am[:, :cut], am.new_ones(bsz, n_keep), am[:, cut + 1:]], dim=1)
    gen = model.generate(inputs_embeds=embeds, attention_mask=am,
                         max_new_tokens=max_new_tokens, do_sample=False)
    return [tok.decode(g, skip_special_tokens=True) for g in gen]
