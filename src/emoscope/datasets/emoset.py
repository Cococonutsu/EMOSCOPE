"""EmoSet-118K 数据集加载（test 划分 17716 张，8 类）。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/ubuntu2404/SSD/EMO/Dataset/EmoSet-118K")


def load() -> list[dict]:
    """返回 [{"path": 绝对路径, "label": 标签}, ...]，按文件名稳定排序。

    test.json 每行为 [label, image_path, annotation_path] 三元组。
    """
    rows = []
    for label, img, _ann in json.load(open(ROOT / "test.json")):
        p = ROOT / img
        if p.exists():
            rows.append({"path": str(p), "label": label})
    rows.sort(key=lambda x: Path(x["path"]).name)
    return rows
