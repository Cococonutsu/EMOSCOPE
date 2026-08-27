#!/usr/bin/env python3
"""构建项目统一训练数据集（EMO-SCOPE/dataset/）。

两个来源统一为同一种格式（无思维链，只留监督必需字段）：
  train_emoset.jsonl / val_emoset.jsonl   —— 46k 带框池头部 2100 条
      （seed=0 洗牌，与历史所有训练版本同批，保证可比）
  train_emotionroi.jsonl                  —— EmotionROI 训练切分 1386 条
      （MD5 验证与 Emotion6 测试集零重叠）

行格式（统一）：
  {"path": 图片绝对路径, "label": 标签, "qset": "emoset8"|"e6_6",
   "target": [576 个 0/1]  —— 证据 patch 目标网格（24x24 光栅序）}

emoset：证据框并集 -> patch_targets 网格（与训练代码同一函数，防漂移）。
EmotionROI：显著掩码 -> 双线性缩到 24x24 -> 逐图 top-20% 秩阈值二值化
（对齐 emoset 框的典型覆盖率；掩码是软显著图，固定阈值会圈进 40% 面积）。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image
from tqdm import tqdm

from emoscope.datasets import emoset_boxes
from emoscope.localizer import patch_targets

ROI_ROOT = Path("/home/ubuntu2404/SSD/EMO/Dataset/EmotionROI")
OUT = Path(__file__).resolve().parent.parent / "dataset"
GRID = 24
RANK_FRAC = 0.20  # ROI 网格正例比例（与 emoset 框典型覆盖对齐）


def mask_to_grid(mask: Image.Image) -> list[int]:
    """显著掩码 -> 24x24 0/1 网格（top-20% 亮度秩阈值）。"""
    m = np.array(mask.convert("L").resize((GRID, GRID), Image.BILINEAR),
                 dtype=np.float32).reshape(-1)
    k = max(1, int(len(m) * RANK_FRAC))
    thr = np.partition(m, -k)[-k]                       # 第 k 大的值
    return (m >= thr).astype(int).tolist()


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # ---- emoset：同历史版本的采样（seed=0 洗牌取头部） ----
    rows = emoset_boxes.load(limit=2100)
    random.Random(0).shuffle(rows)
    train_rows, val_rows = rows[:2000], rows[2000:]
    for name, seg in [("train_emoset.jsonl", train_rows), ("val_emoset.jsonl", val_rows)]:
        with open(OUT / name, "w", encoding="utf-8") as f:
            for r in tqdm(seg, desc=name):
                with Image.open(r["path"]) as im:
                    w, h = im.size
                rec = {"path": r["path"], "label": r["label"], "qset": "emoset8",
                       "target": patch_targets(r["boxes"], w, h).int().tolist()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- EmotionROI：训练切分，掩码->网格 ----
    out_path = OUT / "train_emotionroi.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        lines = [l.strip() for l in
                 open(ROI_ROOT / "training_testing_split/training.txt") if l.strip()]
        for rel in tqdm(lines, desc="emotionroi"):
            img = ROI_ROOT / "images" / rel
            cls, fname = rel.split("/")
            mask = ROI_ROOT / "ground_truth" / cls / fname
            if not (img.exists() and mask.exists()):
                continue
            rec = {"path": str(img), "label": cls, "qset": "e6_6",
                   "target": mask_to_grid(Image.open(mask))}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"\n完成: dataset/train_emoset.jsonl(2000) val_emoset.jsonl(100) "
          f"train_emotionroi.jsonl({n})")


if __name__ == "__main__":
    main()
