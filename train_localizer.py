#!/usr/bin/env python3
"""证据定位头训练：框监督 BCE + 任务 CE（软掩码直通）+ 预算正则（+可选蒸馏 KL）。

CLIP/projector/LLM 全部冻结，只更新 EvidenceLocalizer（约 0.9M 参数）。
软掩码乘在投影后特征上，梯度经掩码回传到头；训练保持全部 576 token，
硬剪枝只发生在推理（prune.py 管线）。

用法（小规模先行）：
  .venv/bin/python train_localizer.py --n 2000 --epochs 2 --out checkpoints/localizer_2k.pt
  带蒸馏：先 scripts/calibrate_loc_heads.py 产 heads.json + teacher.pt，再
  --heads heads.json --teacher teacher.pt --lambda-distill 1.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from emoscope.datasets import emoset_boxes
from emoscope.llava import load_llava
from emoscope.localizer import GRID, EvidenceLocalizer, patch_targets

QUESTION = ("You are given an image and must infer its emotional category.\n"
            "Answer with one word from: amusement, anger, awe, contentment, "
            "disgust, excitement, fear, sadness.")  # 与 emoset 测试集同款指令


def build_batch(model, processor, localizer, chunk, device):
    """一个批的前向数据：软掩码后的嵌入序列 + 监督目标。"""
    images = [Image.open(r["path"]).convert("RGB") for r in chunk]
    px = processor.image_processor(images=images,
                                   return_tensors="pt")["pixel_values"]
    px = px.to(device, torch.bfloat16)
    with torch.no_grad():
        feats = model.model.vision_tower(px, output_hidden_states=True)\
                  .hidden_states[-2][:, 1:]                     # (B,576,1024)
    s = localizer(feats.float())                                # 证据分数 logits
    m = torch.sigmoid(s)                                        # 软掩码
    with torch.no_grad():
        emb = model.model.multi_modal_projector(feats)          # (B,576,4096)
    emb = emb * m.unsqueeze(-1).to(emb.dtype)                   # 梯度经 m 回传

    # 文本：单 <image> 编码，视觉嵌入替换它；答案 token 位置算 CE
    tok = processor.tokenizer
    tok.padding_side = "right"  # 左填充会移动 <image> 位置并破坏标签对齐
    texts = [f"USER: <image>\n{QUESTION} ASSISTANT: {r['label']}" for r in chunk]
    enc = tok(texts, return_tensors="pt", padding=True)
    ids, text_am = enc.input_ids.to(device), enc.attention_mask.to(device)
    word = model.model.language_model.embed_tokens(ids).detach()
    img_col = int((ids[0] == model.config.image_token_index).nonzero().item())
    embeds = torch.cat([word[:, :img_col], emb, word[:, img_col + 1:]], 1)
    am = torch.cat([text_am[:, :img_col], text_am.new_ones(len(chunk), GRID * GRID),
                    text_am[:, img_col + 1:]], 1)
    # 只监督答案 token：prompt 部分按无答案版本长度截掉（须满足前缀性质）
    prompt_ids = tok(f"USER: <image>\n{QUESTION} ASSISTANT:").input_ids
    n_prompt = len(prompt_ids)
    assert all(ids[b, :n_prompt].tolist() == prompt_ids for b in range(len(chunk)))
    pos = torch.arange(ids.shape[1], device=device)[None, :]
    text_labels = ids.masked_fill((pos < n_prompt) | (text_am == 0), -100)
    # 标签与嵌入同步拼接：视觉段全 -100，只留答案 token
    labels = torch.cat([text_labels[:, :img_col],
                        text_labels.new_full((len(chunk), GRID * GRID), -100),
                        text_labels[:, img_col + 1:]], 1)

    # 框 -> patch 0/1 目标（B,576）
    tgt = torch.stack([patch_targets(r["boxes"], im.width, im.height)
                       for r, im in zip(chunk, images)])
    return embeds, am, labels, s, m, tgt.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="训练样本数（小规模先行）")
    ap.add_argument("--val", type=int, default=100, help="验证样本数")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4,
                    help="梯度累积步数，有效batch=bs*accum")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k-target", type=float, default=144, help="软掩码保留量目标")
    ap.add_argument("--lambda-box", type=float, default=1.0)
    ap.add_argument("--lambda-u", type=float, default=0.1,
                    help="框外软负例权重（仅正例监督的折中）")
    ap.add_argument("--lambda-budget", type=float, default=1.0,
                    help="预算正则权重（损失已按576归一，量级与其他项相当）")
    ap.add_argument("--heads", default=None, help="定位头标定结果 json（可选）")
    ap.add_argument("--teacher", default=None,
                    help="老师热图缓存 pt（可选，配合 --heads）")
    ap.add_argument("--lambda-distill", type=float, default=0.0)
    ap.add_argument("--out", default="checkpoints/localizer.pt")
    args = ap.parse_args()

    rows = emoset_boxes.load(limit=args.n + args.val)
    random.Random(0).shuffle(rows)
    train_rows, val_rows = rows[:args.n], rows[args.n:args.n + args.val]

    teacher = torch.load(args.teacher) if args.teacher else None  # {path: (576,)}

    model, processor = load_llava()
    model.requires_grad_(False)
    device = model.device
    localizer = EvidenceLocalizer().to(device)
    opt = torch.optim.AdamW(localizer.parameters(), lr=args.lr)

    def run_batches(data, train, epoch, log_f):
        localizer.train(train)
        tot = {"ce": 0.0, "box": 0.0, "bud": 0.0, "kl": 0.0,
               "gn": 0.0, "iou": 0.0, "kept": 0.0, "n": 0}
        pbar = tqdm(range(0, len(data), args.bs), desc="train" if train else "val")
        for i in pbar:
            chunk = data[i:i + args.bs]
            embeds, am, labels, s, m, tgt = build_batch(
                model, processor, localizer, chunk, device)
            out = model(inputs_embeds=embeds, attention_mask=am, labels=labels)
            loss = out.loss  # 任务 CE（软掩码直通）

            # 框监督：框内正例 + 框外小权重软负例
            w = tgt + args.lambda_u * (1 - tgt)
            l_box = (F.binary_cross_entropy_with_logits(s, tgt, reduction="none")
                     * w).mean()
            # 预算正则：软掩码总量拴在目标附近（按576归一，防压过其他损失）
            l_bud = ((m.sum(-1).mean() - args.k_target) / 576.0) ** 2
            l_kl = torch.zeros((), device=device)
            total = out.loss + args.lambda_box * l_box + args.lambda_budget * l_bud

            if teacher is not None:
                q = torch.stack([teacher[r["path"]] for r in chunk]).to(device)
                l_kl = (q * (q.clamp_min(1e-8).log()
                             - F.log_softmax(s, dim=-1))).sum(-1).mean()
                total = total + args.lambda_distill * l_kl

            grad_norm = 0.0
            if train:
                total = total / args.accum  # 梯度累积：损失按步数均摊
                total.backward()
                step = (i // args.bs + 1) % args.accum == 0 or i + args.bs >= len(data)
                if step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        localizer.parameters(), 1.0).item()
                    opt.step()
                    opt.zero_grad()
            # 监控指标：框覆盖率（top-|框数| 命中率）与软保留量
            n_pos = tgt.sum(-1).clamp_min(1).long()
            hit = 0.0
            for b in range(len(chunk)):
                top = s[b].topk(n_pos[b]).indices
                hit += tgt[b][top].mean().item()
            tot["ce"] += out.loss.item(); tot["box"] += l_box.item()
            tot["bud"] += l_bud.item(); tot["kl"] += l_kl.item()
            tot["gn"] += grad_norm; tot["iou"] += hit / len(chunk)
            tot["kept"] += m.sum(-1).mean().item(); tot["n"] += 1
            pbar.set_postfix(ce=f"{out.loss.item():.3f}",
                             box=f"{l_box.item():.3f}",
                             bud=f"{l_bud.item():.1f}",
                             gn=f"{grad_norm:.2f}",
                             iou=f"{hit / len(chunk):.3f}",
                             kept=f"{m.sum(-1).mean().item():.0f}")
            # 每10步落一行训练日志，供事后检查训练是否健康
            step = tot["n"]
            if train and step % 10 == 0:
                log_f.write(json.dumps({
                    "phase": "train", "epoch": epoch, "step": step,
                    "ce": round(out.loss.item(), 4), "box": round(l_box.item(), 4),
                    "bud": round(l_bud.item(), 5), "gn": round(grad_norm, 3),
                    "iou": round(hit / len(chunk), 3),
                    "kept": round(m.sum(-1).mean().item(), 1)}) + "\n")
                log_f.flush()
        means = {k: v / tot["n"] for k, v in tot.items() if k != "n"}
        if not train:
            log_f.write(json.dumps({"phase": "val", "epoch": epoch,
                                    **{k: round(v, 4) for k, v in means.items()}})
                        + "\n")
            log_f.flush()
        return means

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log_f:
        for ep in range(args.epochs):
            tr = run_batches(train_rows, train=True, epoch=ep, log_f=log_f)
            va = run_batches(val_rows, train=False, epoch=ep, log_f=log_f)
            print(f"epoch {ep}: train ce={tr['ce']:.4f} box={tr['box']:.4f} "
                  f"bud={tr['bud']:.1f} kl={tr['kl']:.3f} gn={tr['gn']:.2f} "
                  f"iou={tr['iou']:.3f} kept={tr['kept']:.0f}")
            print(f"          val   ce={va['ce']:.4f} box={va['box']:.4f} "
                  f"iou={va['iou']:.3f} kept={va['kept']:.0f}")
    torch.save(localizer.state_dict(), out_path)
    print(f"checkpoint -> {out_path}")
    print(f"训练日志   -> {log_path}")


if __name__ == "__main__":
    main()
