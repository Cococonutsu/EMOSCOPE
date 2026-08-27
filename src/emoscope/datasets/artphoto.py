"""ArtPhoto 数据集加载（806 张艺术图，8 类 EmoSet 标签空间）。"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/ubuntu2404/SSD/EMO/Dataset/Artphoto")


def load() -> list[dict]:
    """返回 [{"path": 绝对路径, "label": 标签}, ...]，按文件名稳定排序。"""
    rows = []
    with open(ROOT / "metadata.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = ROOT / "testImages_artphoto" / Path(r["image_path"]).name
            if p.exists():
                rows.append({"path": str(p), "label": r["label"].strip().lower()})
    rows.sort(key=lambda x: Path(x["path"]).name)
    return rows
