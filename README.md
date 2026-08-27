# EMO-SCOPE

多模态大模型情感识别的证据定位头（实验代码）。

## 方法

冻结 LLaVA-1.5-7B，外挂 0.87M 证据定位头（3层MLP+位置编码），
用带证据框的图像训练（框监督+分类探针+预算正则），推理时按分数
保留 top-K 视觉 token。最优用法为纯头训练 + 推理时与 CLS 注意力加权：

```bash
# 训练（2k 小规模，约50分钟）
.venv/bin/python train_localizer.py --n 2000 --epochs 2 \
  --out checkpoints/localizer_emoset2k.pt

# 评测（最优配置：头+CLS 集成打分）
.venv/bin/python main.py --dataset emoset --bs 64 --keep 10 \
  --scorer ensemble --alpha 0.5 --localizer checkpoints/localizer_emoset2k.pt
```

打分器选项：`cls`（CLS注意力，默认）/ `random`（随机对照）/
`ensemble`（头+CLS加权，需 `--localizer` 和 `--alpha`）。
结果按 `results/{模型名}/{数据集}.jsonl` 落盘，逐批写入支持断点续跑。

## 结构

- `main.py` — 评测入口
- `train_localizer.py` — 定位头训练
- `scripts/` — 定位头标定（蒸馏用）、剪枝可视化
- `src/emoscope/` — localizer(头) / prune(剪枝管线) / distill(LLM定位头蒸馏) / llava(模型加载) / 数据集加载器

## 依赖

LLaVA-1.5-7B-HF 本地权重 + `requirements.txt`（transformers 5.x, torch 2.x）。
