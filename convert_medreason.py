"""
Convert MedReason dataset to Alpaca format for Llama-Factory.

Supports json / jsonl input with configurable field names.
"""

import argparse
import json
import pathlib
from typing import List, Dict, Any


def load_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # common patterns: {"data": [...]}, {"examples": [...]}
            for key in ("data", "examples"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        raise ValueError(f"Unsupported JSON structure in {path}")
    raise ValueError(f"Unsupported file extension: {path}")


def convert(
    records: List[Dict[str, Any]],
    question_key: str,
    answer_key: str,
    cot_key: str = None,
    prefix: str = "",
) -> List[Dict[str, str]]:
    out = []
    for r in records:
        q = r.get(question_key, "")
        ans = r.get(answer_key, "")
        cot = r.get(cot_key, "") if cot_key else ""
        instr = f"{prefix}{q}".strip()
        if cot:
            output = f"{cot}\n\n最终答案：{ans}"
        else:
            output = ans
        out.append(
            {
                "instruction": instr,
                "input": "",
                "output": output,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始 json/jsonl 路径")
    ap.add_argument("--output", required=True, help="输出 Alpaca json 路径")
    ap.add_argument("--question-key", default="question")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument("--cot-key", default="reasoning", help="链式推理字段名，可为空")
    ap.add_argument("--prefix", default="", help="instruction 前缀，可选")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(in_path)
    alpaca = convert(
        records,
        question_key=args.question_key,
        answer_key=args.answer_key,
        cot_key=args.cot_key if args.cot_key else None,
        prefix=args.prefix,
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(alpaca, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(alpaca)} samples -> {out_path}")


if __name__ == "__main__":
    main()

