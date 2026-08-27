"""带证据框的训练图加载（46k SFT 池，rel1000 归一化坐标，全部 emoset118k）。

来源 MyTraningData/SFT/Train/rel1000/train.jsonl：每条含图片路径、
assistant 回复中的 {"bbox_2d":[x1,y1,x2,y2]} 证据框和 <answer> 标签。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/home/ubuntu2404/SSD/EMO/Dataset/MyTraningData/SFT/Train/rel1000")
BOX_RE = re.compile(r'"bbox_2d":\s*\[([\d.,\s]+)\]')
ANS_RE = re.compile(r"<answer>\s*(\w+)\s*</answer>")


def load(limit: int | None = None) -> list[dict]:
    """返回 [{"path", "label", "boxes"}, ...]，按文件顺序稳定。

    boxes: [[x1,y1,x2,y2], ...]，rel1000 坐标（0-1000）。
    """
    rows = []
    with open(ROOT / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user, assistant = r["messages"][0], r["messages"][1]
            path = next(p["image"] for p in user["content"] if p.get("type") == "image")
            text = str(assistant["content"])
            boxes = [[float(v) for v in b.split(",")] for b in BOX_RE.findall(text)]
            m = ANS_RE.search(text)
            if not path or not boxes or not m:
                continue  # 缺路径/缺框/缺标签的样本不要
            rows.append({"path": path, "label": m.group(1).lower(), "boxes": boxes})
            if limit and len(rows) >= limit:
                break
    return rows
