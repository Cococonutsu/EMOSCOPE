"""FI 数据集加载（8 类 EmoSet 标签空间），只取 test 划分。"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/ubuntu2404/SSD/EMO/Dataset/FI")


def load() -> list[dict]:
    """返回 [{"path": 绝对路径, "label": 标签}, ...]，按文件名稳定排序。

    metadata_w_split.csv 的 image_path 形如 images/FI/emotion_dataset/.../x.jpg，
    前缀 images/FI/ 相对于 EMO/Dataset 根，此处剥掉后接到 FI 目录下。
    """
    rows = []
    with open(ROOT / "metadata_w_split.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            rel = r["image_path"]
            if rel.startswith("images/FI/"):
                rel = rel[len("images/FI/"):]
            p = ROOT / rel
            if p.exists():
                rows.append({"path": str(p), "label": r["label"].strip().lower()})
    rows.sort(key=lambda x: Path(x["path"]).name)
    return rows
