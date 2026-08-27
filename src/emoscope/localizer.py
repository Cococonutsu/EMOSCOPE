"""证据定位头：冻结大模型外唯一的可训练模块（约 0.9M 参数）。

输入 CLIP 倒数第二层的 576 个 patch 特征，加可学习位置编码，
3 层 MLP 输出每个 patch 的情感证据分数 logits。
结构档位参照 VisionSelector（3 层 MLP 打分器）与 TRIM（极简预测器）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

GRID = 24    # 336/14，576 patch 的网格
CLIP_SIZE = 336  # CLIP 输入边长


class EvidenceLocalizer(nn.Module):
    """patch 特征 (B,576,1024) -> 证据分数 logits (B,576)。"""

    def __init__(self, in_dim: int = 1024, hid: int = 256):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, GRID * GRID, in_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.GELU(),
            nn.Linear(hid, hid // 4), nn.GELU(),
            nn.Linear(hid // 4, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(feats + self.pos).squeeze(-1)


def patch_targets(boxes: list[list[float]], orig_w: int, orig_h: int) -> torch.Tensor:
    """证据框 -> 576 patch 的 0/1 监督掩码。

    boxes 为 rel1000 坐标：原图分辨率下 x 除以宽、y 除以高归一化到
    0-1000（数据管线语义，经画框目检实锤：框精确落在情感主体上）。
    反算即 x/1000*原图宽。再按 CLIPImageProcessor 的 resize（短边 336）
    + 中心裁剪几何映射到 24x24 网格；patch 中心落在任一框内记 1。
    """
    scale = CLIP_SIZE / min(orig_w, orig_h)
    rw, rh = orig_w * scale, orig_h * scale          # resize 后尺寸
    ox, oy = (rw - CLIP_SIZE) / 2, (rh - CLIP_SIZE) / 2  # 裁剪偏移
    cell = CLIP_SIZE / GRID
    mask = torch.zeros(GRID, GRID, dtype=torch.float32)
    for i in range(GRID):
        for j in range(GRID):
            # patch 中心在裁剪图坐标系的位置，反推回原图
            cx = (j * cell + cell / 2 + ox) / scale
            cy = (i * cell + cell / 2 + oy) / scale
            for x1, y1, x2, y2 in boxes:
                bx1, by1 = x1 / 1000 * orig_w, y1 / 1000 * orig_h
                bx2, by2 = x2 / 1000 * orig_w, y2 / 1000 * orig_h
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    mask[i, j] = 1.0
                    break
    return mask.reshape(-1)
