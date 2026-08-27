"""LLM 定位头：LLaVA 内部 text→visual 注意力作为定位信号（参照 CVPR25 Loc-Head）。

定位头只能当"老师"：注意力是全量 prefill 的副产品，推理时用它剪枝省不了算力。
两个用途：
  1. calibrate(): 带框小样本标定哪些 (层,头) 是定位头（热图与框掩码 IoU 排序）
  2. teacher_maps(): 指定头集合的平均热图，作为蒸馏老师信号
"""

from __future__ import annotations

import torch
from PIL import Image

from .localizer import patch_targets


def _enable_llm_attention(model) -> None:
    """LLM 切 eager 才能返回注意力权重（仅标定/提老师时用，评测不用）。"""
    model.model.language_model.set_attn_implementation("eager")


@torch.no_grad()
def _head_maps(model, processor, image: Image.Image, question: str) -> torch.Tensor:
    """单图一次前向，返回所有头的定位热图，形状 (层数, 头数, 576)。

    每个头的图 = 该头 (层,头) 在「最后一个文本 token 行」上对视觉段的注意力。
    行选择与 Loc-Head 官方实现一致：只用紧邻生成位的单行回看，
    不对全部文本行取平均（BOS/指令行含 attention sink，平均会稀释信号）。
    """
    text = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(images=image, text=text, return_tensors="pt").to(
        model.device, torch.bfloat16)
    out = model(**inputs, output_attentions=True)
    ids = inputs["input_ids"][0]
    vis = ids == model.config.image_token_index          # 576 个视觉位置
    txt = ~vis
    last_text = int(txt.nonzero()[-1].item())            # 生成前最后的文本位
    maps = []
    for attn in out.attentions:                         # 每层 (1,头数,L,L)
        a = attn[0].float()[:, last_text]               # (头数, L)
        maps.append(a[:, vis])                          # (头数,576)
    return torch.stack(maps)                            # (层数,头数,576)


@torch.no_grad()
def calibrate(model, processor, samples: list[dict], question: str,
              top_k: int = 8) -> list[dict]:
    """标定定位头：头热图与证据框掩码的 IoU 排序，返回 top_k 个头。

    samples 来自 emoset_boxes.load()，取多少条就是多少条（论文用 ~1k）。
    """
    _enable_llm_attention(model)
    scores = {}  # (层,头) -> IoU 累计
    for s in samples:
        img = Image.open(s["path"]).convert("RGB")
        maps = _head_maps(model, processor, img, question)
        target = patch_targets(s["boxes"], img.width, img.height).to(maps.device)
        pos_rate = target.mean()                        # 按正例率二值化，IoU 才公平
        n_pos = max(1, int(pos_rate * 576))
        nL, nH = maps.shape[:2]
        for l in range(nL):
            for h in range(nH):
                top = maps[l, h].topk(n_pos).indices
                pred = torch.zeros(576, device=maps.device)
                pred[top] = 1
                inter = (pred * target).sum()
                union = ((pred + target) > 0).sum()
                scores[(l, h)] = scores.get((l, h), 0) + (inter / union).item()
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [{"layer": l, "head": h, "iou": v / len(samples)} for (l, h), v in ranked]


@torch.no_grad()
def teacher_maps(model, processor, samples: list[dict], question: str,
                 heads: list[dict]) -> dict[str, torch.Tensor]:
    """提取蒸馏老师信号：指定头集合的平均热图（softmax 归一化）。

    返回 {样本路径: (576,) 目标分布}，训练脚本落盘缓存后离线使用，
    避免训练循环里反复跑全量注意力的 LLM 前向。
    """
    _enable_llm_attention(model)
    out: dict[str, torch.Tensor] = {}
    for s in samples:
        img = Image.open(s["path"]).convert("RGB")
        maps = _head_maps(model, processor, img, question)
        picked = torch.stack([maps[h["layer"], h["head"]] for h in heads])
        mean = picked.mean(dim=0)
        out[s["path"]] = torch.softmax(mean, dim=-1).cpu()
    return out
