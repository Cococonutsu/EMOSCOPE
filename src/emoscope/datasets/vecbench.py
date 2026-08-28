"""VECBench 外部泛化测试集加载：UnbiasedEmo-6 与 WebEmo-25。

来源 VECBench/test_metadata.csv 的 split==test 行（与我们的训练池无同源
交集，UnbiasedEmo/WebEmo 为独立数据集；图片根在 VECBench/ 下）。
标签空间与我们训练词表不同（含 love/confusion 等），是干净的域外考场。
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/ubuntu2404/SSD/EMO/Dataset/VECBench")  # images/ 在仓库根下
META = ROOT / "VECBench" / "test_metadata.csv"

LABEL_SETS = {
    "unbiasedemo": ["anger", "fear", "joy", "love", "sadness", "surprise"],
    "webemo25": ["affection", "cheerfullness", "confusion", "contentment",
                 "disappointment", "disgust", "enthrallment", "envy",
                 "exasperation", "gratitude", "horror", "irritabilty",
                 "lust", "neglect", "nervousness", "optimism", "pride",
                 "rage", "relief", "sadness", "shame", "suffering",
                 "surprise", "sympathy", "zest"],
}
TASK_MAP = {"unbiasedemo": "UnbiasedEmo-6", "webemo25": "WebEmo-25"}


def load(name: str) -> list[dict]:
    """返回 [{"path": 绝对路径, "label": 标签}, ...]，按文件路径稳定排序。"""
    task = TASK_MAP[name]
    rows = []
    with open(META, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["task"] != task or r["split"] != "test":
                continue
            p = ROOT / r["image_path"]
            if p.exists():
                rows.append({"path": str(p), "label": r["label"].strip().lower()})
    rows.sort(key=lambda x: x["path"])
    return rows
