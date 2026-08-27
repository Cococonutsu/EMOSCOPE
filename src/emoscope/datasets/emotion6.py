"""Emotion6 数据集加载（594 张测试图，6 类：anger/disgust/fear/joy/sadness/surprise）。"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/ubuntu2404/SSD/EMO/Dataset/Emotion6")


def load() -> list[dict]:
    """返回 [{"path": 绝对路径, "label": 标签}, ...]，只取 test 划分，按文件名稳定排序。"""
    rows = []
    with open(ROOT / "test.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            p = ROOT / r["image_path"]
            if p.exists():
                rows.append({"path": str(p), "label": r["label"].strip().lower()})
    rows.sort(key=lambda x: Path(x["path"]).name)
    return rows
