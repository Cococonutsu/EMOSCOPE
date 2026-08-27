# EMO-SCOPE — 面向多模态大模型情感推理的自驱视觉分配

**EmoScope: Self-Driven Visual Allocation for Emotion Reasoning in MLLMs**

一句话：情感问句是"证据盲"的（不告诉模型看哪里），我们让模型在**视觉侧**自己决定
每张图分配多少全局观测与哪些局部裁剪——双流编码 + 证据驱动的定位头路由，纯 SFT。

## 快速导航

| 文件 | 内容 |
|---|---|
| `main.py` | **唯一入口**：加载 LLaVA 基座做情感推理（训练/评测后续在此扩展） |
| `docs/REQUIREMENTS.md` | **任务要求（本项目的 ground truth）**：动机、方法、约束、指标 |
| `docs/ROADMAP.md` | 阶段计划与 go/no-go 判据（小规模验证纪律） |
| `docs/RELATED.md` | 竞品格局与差异化（GapSight/EMO-Echo/MS-Resampler 等） |
| `src/emoscope/llava.py` | 模型加载与视觉塔结构探查（1.5/1.6 自动选择） |
| `experiments/` | 实验记录（每次跑的结论落档） |

运行：
```bash
.venv/bin/python main.py --image <图片路径>          # 情感推理
.venv/bin/python main.py --image <图> --info         # 附视觉塔结构
```

## 外部资产（不在本仓库，均为现成可用）

| 资产 | 位置 |
|---|---|
| 训练数据 46k（23k 分类带证据框 + 23k 排序无框，rel1000 CoT） | `~/SSD/EMO/Dataset/MyTraningData/SFT/Train/rel1000/train.jsonl` |
| 情感基准 | `~/SSD/EMO/Dataset/{EmoSet-118K, FI, Artphoto, Emotion6, EmotionROI}` |
| 基座模型 | `~/SSD/Models/llava-1.5-7b-hf`（官方 llava-hf 版） |
| Python 环境（unsloth/torch/peft） | `~/SSD/EMO/Unsloth/.venv` |
| 旧仓库（SFT 主线、TTT 归档、评测参考） | `~/SSD/EMO/` |

Windows 访问：`\\wsl.localhost\Ubuntu-24.04\home\ubuntu2404\SSD\EMO-SCOPE`
