"""LLaVA-1.5 基座加载。"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import (AutoTokenizer, CLIPImageProcessor,
                          LlavaForConditionalGeneration, LlavaProcessor)

MODEL_PATH = "/home/ubuntu2404/SSD/Models/llava-1.5-7b-hf"


def load_llava(model_path: str | Path = MODEL_PATH):
    """加载官方 llava-hf 版 LLaVA-1.5-7B（bf16 / sdpa / 单卡）。

    官方仓库是 transformers 4.x 布局，缺 processor_config.json，5.x 的
    LlavaProcessor 需要 patch_size —— 故在此显式构造，不动官方目录。
    """
    model = LlavaForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = LlavaProcessor(
        image_processor=CLIPImageProcessor.from_pretrained(str(model_path)),
        tokenizer=AutoTokenizer.from_pretrained(str(model_path)),
        patch_size=14,  # CLIP ViT-L/14
    )
    model.eval()
    return model, processor
