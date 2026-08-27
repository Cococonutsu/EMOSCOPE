#!/usr/bin/env python3
"""清洗后的标注 -> 统一数据集格式（576 网格目标，与其他源同构）。

用法:
  .venv/bin/python scripts/annot_to_dataset.py \
      dataset/manifests/annot_artemis_clean.jsonl dataset/train_artemis.jsonl emoset8 \
      dataset/manifests/annot_emotionroi_clean.jsonl dataset/train_emotionroi.jsonl e6_6
（参数三元组：输入 输出 qset，可多组）
兜底补充：同目录下若有 *_fallback.jsonl（子代理标注），先行合并。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from PIL import Image
from tqdm import tqdm
from emoscope.localizer import patch_targets

EMO_ROOT = "/home/ubuntu2404/SSD/EMO/"


def main() -> None:
    args = sys.argv[1:]
    assert len(args) % 3 == 0, "参数须为 (输入, 输出, qset) 三元组"
    for i in range(0, len(args), 3):
        inp, out, qset = Path(args[i]), Path(args[i + 1]), args[i + 2]
        rows = {}
        for line in open(inp, encoding="utf-8"):
            r = json.loads(line)
            rows[r["row_index"]] = r
        fb = inp.with_name(inp.stem.replace("_clean", "_fallback") + ".jsonl")
        if fb.exists():  # 子代理兜底结果并入
            for line in open(fb, encoding="utf-8"):
                r = json.loads(line)
                rows[r["row_index"]] = r
        with open(out, "w", encoding="utf-8") as f:
            for r in tqdm(sorted(rows.values(), key=lambda x: x["row_index"]),
                          desc=out.name):
                path = r["image_path"]
                if not path.startswith("/"):
                    path = EMO_ROOT + path
                with Image.open(path) as im:
                    w, h = im.size
                rec = {"path": path, "label": r["label"], "qset": qset,
                       "target": patch_targets(r["boxes"], w, h).int().tolist()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{out.name}: {len(rows)} 条（含兜底 {len(rows) - sum(1 for _ in open(inp))}）")


if __name__ == "__main__":
    main()
