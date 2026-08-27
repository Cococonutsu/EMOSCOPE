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

import copy

import torch


def enable_vision_attention(model) -> None:
    """整塔切 eager（旧法）：只为取一层注意力付出全塔代价，仅作对照保留。"""
    model.model.vision_tower.set_attn_implementation("eager")


def enable_layer_attention(model, layer_idx: int = 22) -> None:
    """仅倒数第二层切 eager，其余 23 层保持 sdpa。

    5.x 的 CLIPAttention 每次前向动态读 self_attn.config._attn_implementation，
    故单层替换副本即可；该层返回的注意力是 attentions 里唯一非空项。
    """
    attn = model.model.vision_tower.encoder.layers[layer_idx].self_attn
    attn.config = copy.deepcopy(attn.config)
    attn.config._attn_implementation = "eager"


def vision_feats_scores(model, pixel_values) -> tuple[torch.Tensor, torch.Tensor]:
    """视觉塔一次前向，返回 (投影前 patch 特征, 每个 patch 的重要性)。

    特征取倒数第二层；重要性 = 同层注意力按头平均后的 [CLS] 行（丢 CLS 列）。
    兼容两种模式：整塔 eager（attentions 全满，取 [-2]）与单层 eager
    （只有一项非空，取该项）。
    """
    vis = model.model.vision_tower(pixel_values, output_hidden_states=True,
                                   output_attentions=True)
    feats = vis.hidden_states[-2][:, 1:]                  # (B,576,1024)
    attns = [a for a in vis.attentions if a is not None]
    a = attns[0] if len(attns) == 1 else vis.attentions[-2]
    scores = a.mean(dim=1)[:, 0, 1:]                      # (B,576)
    return feats, scores


def _z(s: torch.Tensor) -> torch.Tensor:
    """图内标准化：头 logits 与 CLS 注意力（和为1）尺度对齐后才能加权。"""
    return (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-6)


@torch.no_grad()
def generate_pruned(model, processor, images: list, prompt: str, keep_pct: float,
                    max_new_tokens: int = 16, localizer=None,
                    random: bool = False, ensemble_alpha: float | None = None
                    ) -> list[str]:
    """对一批同模板图片剪枝视觉 token 后生成，返回每张图的回答文本。

    keep_pct 为保留 token 的百分比（0-100），内部换算成个数。
    分数来源：random=均匀随机（零信息对照）；localizer=证据定位头；
    localizer+ensemble_alpha=头与CLS按图内标准化后加权（alpha 归头）；
    默认 CLS 注意力。除默认外均须先 enable_layer_attention 或不需要注意力。
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
    elif localizer is not None and ensemble_alpha is not None:
        # 集成：单层 eager 已开启，一次前向同取特征与 CLS 注意力
        vis = model.model.vision_tower(px, output_hidden_states=True,
                                       output_attentions=True)
        feats = vis.hidden_states[-2][:, 1:]
        a = next(a for a in vis.attentions if a is not None)
        s_cls = a.mean(dim=1)[:, 0, 1:]                     # (B,576)
        s_head = localizer(feats.float())
        scores = ensemble_alpha * _z(s_head) + (1 - ensemble_alpha) * _z(s_cls)
    elif localizer is not None:
        if getattr(localizer, "blend_alpha", None) is not None:
            # 分数级加权推理：alpha 取训练学到的系数（标量）
            vis = model.model.vision_tower(px, output_hidden_states=True,
                                           output_attentions=True)
            feats = vis.hidden_states[-2][:, 1:]
            a = next(a for a in vis.attentions if a is not None)
            s_cls = a.mean(dim=1)[:, 0, 1:].float()
            s_head = localizer(feats.float())
            al = float(localizer.blend_alpha)
            scores = al * _z(s_head) + (1 - al) * _z(s_cls)
        elif getattr(localizer, "use_cls", False):
            # 融合头：一次前向同取特征与 CLS 注意力（单层 eager 已开启）
            vis = model.model.vision_tower(px, output_hidden_states=True,
                                           output_attentions=True)
            feats = vis.hidden_states[-2][:, 1:]
            a = next(a for a in vis.attentions if a is not None)
            s_cls = a.mean(dim=1)[:, 0, 1:]
            scores = localizer(feats.float(), s_cls.float())
        else:
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
