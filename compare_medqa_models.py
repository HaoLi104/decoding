import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _load_results(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized format in {path}")


def _index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    keyed: Dict[str, Dict[str, Any]] = {}
    for i, row in enumerate(rows):
        key = str(row.get("id", i))
        keyed[key] = row
    return keyed


def collect_diffs(big_rows: List[Dict[str, Any]], small_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    big_map = _index_by_id(big_rows)
    small_map = _index_by_id(small_rows)

    common_keys = big_map.keys() & small_map.keys()
    diffs: List[Dict[str, Any]] = []

    for k in common_keys:
        b = big_map[k]
        s = small_map[k]
        b_correct = bool(b.get("is_correct"))
        s_correct = bool(s.get("is_correct"))
        if (not b_correct) and s_correct:
            diffs.append(
                {
                    "id": k,
                    "question": b.get("question") or s.get("question"),
                    "options": b.get("options") or s.get("options"),
                    "ground_truth": b.get("ground_truth") or s.get("ground_truth"),
                    "big_model_pred": b.get("predicted_answer"),
                    "small_model_pred": s.get("predicted_answer"),
                    "big_model_is_correct": b_correct,
                    "small_model_is_correct": s_correct,
                    "big_model_response": b.get("response"),
                    "small_model_response": s.get("response"),
                }
            )
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Find cases where big model is wrong and small model is correct.")
    parser.add_argument("--big", required=True, help="Path to big model results JSON")
    parser.add_argument("--small", required=True, help="Path to small model results JSON")
    parser.add_argument("--output", default="logs/medqa_big_wrong_small_right.json", help="Path to write filtered JSON")
    args = parser.parse_args()

    big_rows = _load_results(Path(args.big))
    small_rows = _load_results(Path(args.small))

    diffs = collect_diffs(big_rows, small_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diffs, ensure_ascii=False, indent=2))
    print(f"Saved {len(diffs)} diff cases to {out_path}")


if __name__ == "__main__":
    main()
