#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    return {
        "file": path.name,
        "mode": summary.get("mode", ""),
        "accuracy": summary.get("accuracy", 0.0),
        "tokens_per_sec_end_to_end": summary.get("tokens_per_sec_end_to_end", 0.0),
        "correct": summary.get("correct", 0),
        "n_cases": summary.get("n_cases", 0),
    }


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = ["label", "file", "mode", "accuracy", "correct", "n_cases", "tokens_per_sec_end_to_end"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        vals = [row.get(h, "") for h in headers]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare core experiment results")
    parser.add_argument("--target_baseline", required=True)
    parser.add_argument("--draft_only", required=True)
    parser.add_argument("--standard_speculative", required=True)
    parser.add_argument("--divergence_v2", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = []
    rows.append({"label": "target_baseline", **read_summary(Path(args.target_baseline))})
    rows.append({"label": "draft_only", **read_summary(Path(args.draft_only))})
    rows.append({"label": "standard_speculative", **read_summary(Path(args.standard_speculative))})
    rows.append({"label": "divergence_v2", **read_summary(Path(args.divergence_v2))})

    md = to_markdown(rows)
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
