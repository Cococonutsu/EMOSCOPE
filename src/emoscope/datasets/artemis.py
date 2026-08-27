"""ArtEmis 数据集加载（49k 幅 WikiArt 画作，8 类）。

dataset_final.csv 为行级标注（同一画作多行情感语句，无官方 split），
图像级分类按 ArtEmis 惯例取每幅画的多数票标签，平票时按标签名字典序破平。
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path("/home/ubuntu2404/SSD/EMO/Dataset/ArtEmis")


def load() -> list[dict]:
    """返回 [{"path": 绝对路径, "label": 多数票标签}, ...]，按文件名稳定排序。"""
    votes: dict[str, Counter] = {}
    with open(ROOT / "dataset_final.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            votes.setdefault(r["filename"], Counter())[r["emotion"]] += 1

    rows = []
    for fname, counter in votes.items():
        p = ROOT / "images" / fname
        if p.exists():
            label = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]
            rows.append({"path": str(p), "label": label})
    rows.sort(key=lambda x: Path(x["path"]).name)
    return rows
