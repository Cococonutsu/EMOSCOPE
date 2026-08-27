# EMO-SCOPE

多模态大模型情感识别的证据定位头（实验代码）。

当前最优用法：纯头训练 + 推理时与 CLS 注意力按 α 加权（`--scorer ensemble --alpha 0.5`）。

## 结构

- `main.py` — 评测入口（基线 / CLS / 随机 / 定位头 / 集成五种打分模式，断点续跑）
- `train_localizer.py` — 证据定位头训练（框监督 + 分类探针 + 预算正则）
- `scripts/` — 定位头标定、剪枝可视化
- `src/emoscope/` — 核心模块与数据集加载器

## 模块

| 模块 | 说明 |
|---|---|
| `localizer.py` | 0.87M 证据定位头（3层MLP+位置编码） |
| `prune.py` | 视觉token剪枝管线（CLS/随机/定位头/集成打分） |
| `distill.py` | LLM 定位头标定与蒸馏信号提取 |
| `llava.py` | LLaVA-1.5-7B 加载 |

## 依赖

LLaVA-1.5-7B-HF 本地权重 + `requirements.txt`（transformers 5.x, torch 2.x）。
