#!/usr/bin/env python3
"""定位头标定 + 老师信号缓存（Gate 0 机器，纯推理不训练）。

产出两个文件：
  heads.json   top-K 定位头（层,头,IoU）
  teacher.pt   {样本路径: (576,) 老师热图分布}，训练时 --teacher 读入

用法：
  .venv/bin/python scripts/calibrate_loc_heads.py --n-calib 200 --n-teacher 2000 \
      --heads heads.json --teacher teacher.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tqdm import tqdm

from emoscope.datasets import emoset_boxes
from emoscope.distill import calibrate, teacher_maps
from emoscope.llava import load_llava

QUESTION = ("You are given an image and must infer its emotional category.\n"
            "Answer with one word from: amusement, anger, awe, contentment, "
            "disgust, excitement, fear, sadness.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=200,
                    help="标定用带框样本数（每样本一次全注意力前向，较慢）")
    ap.add_argument("--n-teacher", type=int, default=2000,
                    help="提取老师热图的样本数")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--heads", default="heads.json")
    ap.add_argument("--teacher", default="teacher.pt")
    args = ap.parse_args()

    rows = emoset_boxes.load(limit=args.n_calib + args.n_teacher)
    calib_rows, teacher_rows = rows[:args.n_calib], rows[args.n_calib:]

    model, processor = load_llava()
    heads = calibrate(model, processor, calib_rows, QUESTION, top_k=args.top_k)
    Path(args.heads).write_text(json.dumps(heads, indent=2))
    print(f"定位头 -> {args.heads}")
    for h in heads:
        print(f"  layer {h['layer']:2d} head {h['head']:2d}  IoU={h['iou']:.3f}")

    maps = teacher_maps(model, processor, teacher_rows, QUESTION, heads)
    torch.save(maps, args.teacher)
    print(f"老师热图 {len(maps)} 条 -> {args.teacher}")


if __name__ == "__main__":
    main()
