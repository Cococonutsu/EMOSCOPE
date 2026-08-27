#!/usr/bin/env python3
"""可视化视觉 token 剪枝：左=原图，右=模型实际输入图（剪掉的 patch 涂灰）。

用法：.venv/bin/python scripts/visualize_prune.py --image path.jpg --keep 25 [-o out.png]
默认输出到项目 viz/ 目录。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image, ImageDraw

from emoscope.llava import load_llava
from emoscope.prune import enable_vision_attention, vision_feats_scores

GRID = 24   # 336 / 14 = 24，576 个 patch 的网格
SIZE = 336  # CLIP 输入边长
KNOWN_DATASETS = {"artphoto", "emotion6", "fi", "emoset", "artemis"}


def detect_dataset(path: str) -> str:
    """从图片路径中识别数据集名（按路径组件匹配，如 .../Artphoto/xxx）。"""
    for part in Path(path).parts:
        if part.lower() in KNOWN_DATASETS:
            return part.lower()
    return "unknown"


def fit_square(img: Image.Image) -> Image.Image:
    """原图等比缩放后贴到 336×336 白底上，与右侧对比图等高。"""
    canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    thumb = img.copy()
    thumb.thumbnail((SIZE, SIZE))
    canvas.paste(thumb, ((SIZE - thumb.width) // 2, (SIZE - thumb.height) // 2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--keep", type=float, required=True,
                    help="保留视觉 token 的百分比（0-100）")
    ap.add_argument("-o", "--out", default=None, help="输出 png 路径")
    args = ap.parse_args()

    model, processor = load_llava()
    enable_vision_attention(model)
    img = Image.open(args.image).convert("RGB")

    # 与评测同一份打分逻辑
    px = processor.image_processor(images=[img], return_tensors="pt")["pixel_values"]
    px = px.to(model.device, torch.bfloat16)
    _, scores = vision_feats_scores(model, px)               # (1,576)
    n_keep = max(1, round(GRID * GRID * args.keep / 100.0))
    keep_mask = torch.zeros(GRID * GRID, dtype=torch.bool)
    keep_mask[scores[0].topk(n_keep).indices.cpu()] = True
    keep_mask = keep_mask.reshape(GRID, GRID)

    # 反归一化还原模型实际看到的 336×336 输入图
    ip = processor.image_processor
    mean = torch.tensor(ip.image_mean).view(3, 1, 1)
    std = torch.tensor(ip.image_std).view(3, 1, 1)
    seen = (px[0].float().cpu() * std + mean).clamp(0, 1)
    seen = (seen.permute(1, 2, 0).numpy() * 255).astype("uint8")
    seen_img = Image.fromarray(seen)

    # 剪掉的 patch 涂灰
    draw = ImageDraw.Draw(seen_img)
    cell = SIZE // GRID
    for i in range(GRID):
        for j in range(GRID):
            if not keep_mask[i, j]:
                draw.rectangle([j * cell, i * cell,
                                (j + 1) * cell - 1, (i + 1) * cell - 1],
                               fill=(110, 110, 110))

    out = Image.new("RGB", (SIZE * 2 + 4, SIZE))
    out.paste(fit_square(img), (0, 0))
    out.paste(seen_img, (SIZE + 4, 0))
    # 与评测结果同构：viz/{模型名}/{数据集}/{图片名}.png，比例编码在模型名里
    model_dir = f"llava1.5-prune-keep{args.keep:g}pct"
    out_path = args.out or str(Path(__file__).resolve().parent.parent / "viz"
                               / model_dir / detect_dataset(args.image)
                               / f"{Path(args.image).stem}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"保留 {n_keep}/{GRID * GRID} 个 token -> {out_path}")


if __name__ == "__main__":
    main()
