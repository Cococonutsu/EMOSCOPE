#!/usr/bin/env python3
"""EmoScope 入口。

用法：
  python main.py --image /path/to/img.jpg [--prompt "自定义问题"]   # 单图推理
  python main.py --dataset artphoto|emotion6|fi [--limit N] [--bs 32]  # 测试集评测
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from PIL import Image
from tqdm import tqdm

from emoscope.llava import load_llava
from emoscope.prune import enable_layer_attention, generate_pruned
from emoscope.datasets import artemis, artphoto, emotion6, emoset, fi

# 各数据集的标签空间（Emotion6 为 6 类，其余为 EmoSet 8 类）
LABEL_SETS = {
    "artphoto": ["amusement", "anger", "awe", "contentment",
                 "disgust", "excitement", "fear", "sad"],  # ArtPhoto 金标用 sad 非 sadness
    "fi":       ["amusement", "anger", "awe", "contentment",
                 "disgust", "excitement", "fear", "sadness"],
    "emoset":   ["amusement", "anger", "awe", "contentment",
                 "disgust", "excitement", "fear", "sadness"],
    "artemis":  ["amusement", "anger", "awe", "contentment",
                 "disgust", "excitement", "fear", "sadness"],
    "emotion6": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
}
DATASET_LOADERS = {"artphoto": artphoto, "emotion6": emotion6, "fi": fi,
                    "emoset": emoset, "artemis": artemis}


def classify_prompt(labels: list[str]) -> str:
    """测试集评测用的分类指令，标签集合随数据集切换。"""
    return ("You are given an image and must infer its emotional category.\n"
            "Answer with one word from: " + ", ".join(labels) + ".")


def parse_pred(text: str, labels: list[str]) -> str | None:
    """从生成文本解析标签：先按分词精确匹配，退化为子串扫描。"""
    words = re.findall(r"[a-z]+", text.lower())
    for w in words:
        if w in labels:
            return w
    for lab in labels:
        if lab in text.lower():
            return lab
    return None


@torch.no_grad()
def run_dataset(model, processor, name: str, limit: int | None, bs: int,
                 model_name: str, keep: float | None = None,
                 localizer=None, random: bool = False) -> None:
    labels = LABEL_SETS[name]
    rows = DATASET_LOADERS[name].load()
    if limit:
        rows = rows[:limit]
    question = classify_prompt(labels)
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    if keep and localizer is None and not random:
        enable_layer_attention(model)  # CLS 打分：仅倒数第二层 eager，其余层保持 sdpa

    # 断点续跑：读已有记录跳过完成样本，结果逐批追加落盘（中断后重跑不重来）
    out_dir = Path(__file__).parent / "results" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"
    records, done = [], set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 上次中断写坏的残行丢弃
            records.append(rec)
            done.add(rec["path"])
        if done:
            print(f"断点续跑：已有 {len(done)} 条，继续剩余样本")
    rows = [r for r in rows if r["path"] not in done]

    prev_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "left"  # decoder-only 批量生成的标准做法
    with open(out_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(rows), bs), desc=name):
            chunk = rows[i : i + bs]
            images = [Image.open(r["path"]).convert("RGB") for r in chunk]
            if keep:  # 剪枝生成（随机/定位头/CLS 三种打分；传裸问题，模板在内拼）
                gens = generate_pruned(model, processor, images, question, keep,
                                       localizer=localizer, random=random)
            else:
                inputs = processor(images=images, text=[prompt] * len(chunk),
                                   return_tensors="pt", padding=True).to(
                    model.device, torch.bfloat16)
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                in_len = inputs["input_ids"].shape[1]
                gens = [processor.decode(out[b][in_len:], skip_special_tokens=True)
                        for b in range(len(chunk))]
            batch_recs = []
            for gen, r in zip(gens, chunk):
                pred = parse_pred(gen, labels)
                batch_recs.append({"path": r["path"],      # 原始数据路径
                                   "raw": gen,             # 模型原始回答
                                   "pred": pred,           # 抽取出的回答
                                   "label": r["label"],    # 真实答案
                                   "match": pred == r["label"],
                                   "prompt": question})    # 本次评测用的指令
            records.extend(batch_recs)
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in batch_recs))
            f.flush()  # 每批立即落盘
    processor.tokenizer.padding_side = prev_side

    acc = sum(r["match"] for r in records) / len(records)
    per_label = defaultdict(lambda: [0, 0])
    for r in records:
        per_label[r["label"]][0] += r["match"]
        per_label[r["label"]][1] += 1
    print(f"\n{name}: acc = {acc:.4f}（{len(records)} 样本）")
    for lab in sorted(per_label):
        ok, n = per_label[lab]
        print(f"  {lab:12s} {ok}/{n} = {ok / n:.3f}")
    print(f"predictions -> {out_path}")


@torch.no_grad()
def run_single(model, processor, image_path: str, prompt: str,
               keep: float | None = None) -> None:
    img = Image.open(image_path).convert("RGB")
    text = f"USER: <image>\n{prompt} ASSISTANT:"
    if keep:
        enable_vision_attention(model)
        print(generate_pruned(model, processor, [img], prompt, keep,
                              max_new_tokens=256)[0].strip())
    else:
        inputs = processor(images=img, text=text,
                           return_tensors="pt").to(model.device, torch.bfloat16)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        print(processor.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="单图推理模式")
    ap.add_argument("--prompt", default=classify_prompt(LABEL_SETS["artphoto"]),
                    help="单图模式自定义问题")
    ap.add_argument("--dataset", choices=sorted(DATASET_LOADERS),
                    help="数据集测试集模式")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--model-name", default="llava1.5-base",
                    help="结果目录名，训练出变体后用不同名字区分")
    ap.add_argument("--keep", type=float, default=None, metavar="PCT",
                    help="剪枝：保留视觉 token 的百分比（0-100，100=不剪）")
    ap.add_argument("--localizer", default=None, metavar="CKPT",
                    help="证据定位头 checkpoint，替代 CLS 注意力打分")
    ap.add_argument("--scorer", choices=["cls", "random"], default="cls",
                    help="打分器：cls=CLS注意力（默认），random=随机保留（对照）")
    ap.add_argument("--seed", type=int, default=0,
                    help="random 打分器的抽样种子（多seed检验方差用）")
    args = ap.parse_args()
    if not args.image and not args.dataset:
        ap.error("需要 --image 或 --dataset 之一")

    model, processor = load_llava()
    if args.scorer == "random":
        torch.manual_seed(args.seed)  # 随机保留可复现，seed 入目录名防覆盖
    localizer = None
    if args.localizer:
        from emoscope.localizer import EvidenceLocalizer
        localizer = EvidenceLocalizer().to(model.device)
        localizer.load_state_dict(torch.load(args.localizer,
                                             map_location=model.device))
        localizer.eval()
    model_name = args.model_name
    if args.keep and args.model_name == "llava1.5-base":  # 剪枝结果另存目录
        base = (f"llava1.5-{Path(args.localizer).stem}" if localizer else
                "llava1.5-random" if args.scorer == "random" else "llava1.5-prune")
        if args.scorer == "random":
            base += f"-keep{args.keep:g}pct_{args.seed}"  # seed 后缀必带
            model_name = base
        else:
            model_name = f"{base}-keep{args.keep:g}pct"
    if args.dataset:
        run_dataset(model, processor, args.dataset, args.limit, args.bs,
                   model_name, keep=args.keep, localizer=localizer,
                   random=args.scorer == "random")
    else:
        run_single(model, processor, args.image, args.prompt, keep=args.keep)


if __name__ == "__main__":
    main()
