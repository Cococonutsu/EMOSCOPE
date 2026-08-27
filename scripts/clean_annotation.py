#!/usr/bin/env python3
"""标注结果清洗：剥掉思维链等冗余字段，只留训练需要的骨架。

输入 generate_external 产出的 annot_*.jsonl（含完整 response/描述/推理链），
输出 *_clean.jsonl，每行：
  {"row_index": int, "image_path": str, "label": str,
   "boxes": [[x1,y1,x2,y2], ...]}   # rel1000，1-3 个

打捞逻辑：严格校验(parse_ok=False)失败的行，只要 response 里能解出
合法框（坐标在值域内、x1<x2、y1<y2），照样保留——我们只用框，
文本字段不合格不等于框不能用。真正无框的行写入 *_failed.jsonl 供兜底。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def salvage_boxes(response) -> list | None:
    """从原始 response 宽松提取合法框（不做任何字段校验）。"""
    if not isinstance(response, str):
        return None
    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    boxes = obj.get("local_evidence_boxes")
    if not isinstance(boxes, list) or not boxes:
        return None
    good = []
    for b in boxes:
        if not isinstance(b, dict):
            continue
        v = b.get("box_2d")
        if (isinstance(v, list) and len(v) == 4
                and all(isinstance(x, (int, float)) for x in v)
                and 0 <= v[0] < v[2] <= 1000 and 0 <= v[1] < v[3] <= 1000):
            good.append([int(x) for x in v])
    return good or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="annot_*.jsonl 文件列表")
    args = ap.parse_args()
    for inp in args.inputs:
        p = Path(inp)
        raw = [json.loads(line) for line in open(p, encoding="utf-8")]
        # 断点重试会产生重复行：按 row_index 去重，保留最后一次（重试结果覆盖旧失败）
        rows = {}
        for r in raw:
            rows[r["row_index"]] = r
        rows = list(rows.values())
        ok = salvaged = failed = 0
        out = p.with_name(p.stem + "_clean.jsonl")
        fail = p.with_name(p.stem + "_failed.jsonl")
        with open(out, "w", encoding="utf-8") as fo, \
                open(fail, "w", encoding="utf-8") as ff:
            for r in rows:
                if r.get("parse_ok"):
                    boxes = [b["box_2d"] for b in r["local_evidence_boxes"]]
                    ok += 1
                else:  # 打捞：response 里能解出合法框就用
                    boxes = salvage_boxes(r.get("response"))
                    if boxes is None:
                        ff.write(json.dumps({
                            "row_index": r["row_index"],
                            "image_path": r["image_path"],
                            "label": r["label"]}, ensure_ascii=False) + "\n")
                        failed += 1
                        continue
                    salvaged += 1
                fo.write(json.dumps({
                    "row_index": r["row_index"],
                    "image_path": r["image_path"],
                    "label": r["label"],
                    "boxes": boxes}, ensure_ascii=False) + "\n")
        print(f"{p.name}: 严格ok {ok} + 打捞 {salvaged} = {ok + salvaged} 条，"
              f"真无框 {failed} 条 -> {out.name} / {fail.name}")


if __name__ == "__main__":
    main()
