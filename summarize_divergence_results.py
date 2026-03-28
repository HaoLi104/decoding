#!/usr/bin/env python3
"""Summarize divergence override experiment JSON outputs into a single table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _collect_json_files(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("*.json") if p.is_file()])


def _read_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    return {
        "file": path.name,
        "mode": summary.get("mode", ""),
        "accuracy": summary.get("accuracy", 0.0),
        "tokens_per_sec_end_to_end": summary.get("tokens_per_sec_end_to_end", 0.0),
        "override_rate": summary.get("override_rate", 0.0),
        "accepted_override": summary.get("accepted_override", 0),
        "override_calls": summary.get("override_calls", 0),
        "small_base_calls": summary.get("small_base_calls", 0),
        "small_base_call_rate_per_override_call": summary.get("small_base_call_rate_per_override_call", 0.0),
        "v2_precheck_skips": summary.get("v2_precheck_skips", 0),
        "tau_delta": summary.get("tau_delta", None),
        "tau_target_opp": summary.get("tau_target_opp", None),
    }


def _to_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "file",
        "mode",
        "accuracy",
        "tokens_per_sec_end_to_end",
        "override_rate",
        "accepted_override",
        "override_calls",
        "small_base_calls",
        "small_base_call_rate_per_override_call",
        "v2_precheck_skips",
        "tau_delta",
        "tau_target_opp",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        vals = [row.get(h, "") for h in headers]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize divergence override outputs")
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_md", default=None)
    args = parser.parse_args()

    root = Path(args.in_dir)
    files = _collect_json_files(root)
    if not files:
        raise FileNotFoundError(f"No json files found in: {root}")

    rows = [_read_summary(path) for path in files]
    rows.sort(key=lambda x: (x["mode"], -float(x["accuracy"])))

    md = _to_markdown(rows)
    print(md)

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
