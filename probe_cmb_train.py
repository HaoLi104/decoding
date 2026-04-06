"""
直接探测 CMB-Exam train 数据的实际格式与规模。

绕过 datasets 的 config 验证（列名不匹配问题），直接从本地 HF 缓存
或通过 hf-mirror.com 下载 CMB-train-merge.json，分析字段与样本。

运行：
  cd /data/ocean/decoding && conda activate kvner
  export HF_ENDPOINT=https://hf-mirror.com
  python probe_cmb_train.py 2>&1 | tee results/probe_cmb_train.txt
"""

import os
import json
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SEP = "=" * 60

# ---------------------------------------------------------------------------
# Step 1：在 HuggingFace 本地缓存中搜索 CMB train 文件
# ---------------------------------------------------------------------------
print(SEP)
print("Step 1: 搜索本地 HF 缓存中的 CMB train 文件")

CACHE_ROOTS = [
    Path.home() / ".cache/huggingface",
    Path("/home/ocean/.cache/huggingface"),
    Path("/root/.cache/huggingface"),
    Path("/data/ocean/.cache/huggingface"),
]

found_files: list[Path] = []
for cache_root in CACHE_ROOTS:
    try:
        if not cache_root.exists():
            continue
        # 搜索 CMB train merge JSON（可能在 datasets/downloads/ 下）
        for pattern in [
            "**/CMB-train-merge.json",
            "**/CMB*train*.json",
            "**/FreedomIntelligence*CMB*train*.json",
        ]:
            found_files.extend(cache_root.glob(pattern))
    except PermissionError:
        print(f"  跳过（无权限）: {cache_root}")

# 去重并过滤超小文件（< 10KB 可能是索引文件）
found_files = sorted(set(f for f in found_files if f.stat().st_size > 10_000))
print(f"  发现 {len(found_files)} 个候选文件：")
for f in found_files:
    print(f"    {f}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

# ---------------------------------------------------------------------------
# Step 2：直接加载 JSON 文件分析字段
# ---------------------------------------------------------------------------

def analyze_json_file(path: Path) -> None:
    print(f"\n{SEP}")
    print(f"  分析文件: {path.name}  ({path.stat().st_size/1024/1024:.1f} MB)")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"  !! 非 list 格式，实际类型: {type(data)}")
        return

    print(f"  总条数: {len(data)}")
    if not data:
        print("  !! 空文件")
        return

    sample = data[0]
    print(f"  字段名: {list(sample.keys())}")
    print(f"  示例[0]:")
    for k, v in sample.items():
        v_str = str(v)
        print(f"    {k}: {v_str[:200]}{'...' if len(v_str) > 200 else ''}")

    # 判断是否为 MCQ 格式（含答案）
    answer_field = next((k for k in sample if "answer" in k.lower()), None)
    option_fields = [k for k in sample if k.lower() in {"option", "options", "choices"} or
                     any(letter in k.lower() for letter in ["_a", "_b", "_c", "_d", "opta", "optb"])]
    print(f"\n  ⚙ 答案字段: {answer_field}")
    print(f"  ⚙ 选项字段: {option_fields}")

    if answer_field:
        # 统计有效答案的条数
        valid = sum(1 for row in data if str(row.get(answer_field, "")).strip())
        print(f"  ✅ 含有效答案的条数: {valid}/{len(data)}")
        # 采样答案分布
        answers = [str(row.get(answer_field, "")).strip().upper()[:1] for row in data[:500]]
        dist = {k: answers.count(k) for k in set(answers) if k}
        print(f"  答案分布（前500条）: {dist}")
    else:
        print("  ❌ 未发现答案字段，不适合用于微调")


if found_files:
    for f in found_files[:3]:  # 最多分析前3个
        try:
            analyze_json_file(f)
        except Exception as e:
            print(f"  分析失败: {e}")
else:
    print("  本地缓存未命中，尝试从 hf-mirror.com 下载...")

# ---------------------------------------------------------------------------
# Step 3：若本地缓存无文件，用 huggingface_hub 下载
# ---------------------------------------------------------------------------
if not found_files:
    print(f"\n{SEP}")
    print("Step 3: 通过 huggingface_hub 下载 CMB train 文件")
    try:
        from huggingface_hub import hf_hub_download
        file_path = hf_hub_download(
            repo_id="FreedomIntelligence/CMB",
            filename="CMB-Exam/CMB-train/CMB-train-merge.json",
            repo_type="dataset",
        )
        print(f"  下载成功: {file_path}")
        analyze_json_file(Path(file_path))
    except Exception as e:
        print(f"  下载失败: {e}")
        print(f"\n  !! 建议改用 wget 直接下载：")
        print(f"  wget -O /data/ocean/decoding/data/cmb_train.json \\")
        print(f"    'https://hf-mirror.com/datasets/FreedomIntelligence/CMB/resolve/main/CMB-Exam/CMB-train/CMB-train-merge.json'")

# ---------------------------------------------------------------------------
# Step 4：探测 CMB-Exam val split（用 json type 直接加载，绕过 config 验证）
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("Step 4: 尝试直接加载 CMB-Exam val split（绕过 config 验证）")
try:
    from datasets import load_dataset
    # 使用 'json' type 直接加载，绕过 FreedomIntelligence/CMB 的 config 验证
    ds_val = load_dataset(
        "FreedomIntelligence/CMB",
        "CMB-Exam",
        split="val",
        trust_remote_code=True,
    )
    print(f"  val split: {len(ds_val)} 条  字段: {ds_val.column_names}")
    print(f"  val[0]: {ds_val[0]}")
except Exception as e:
    print(f"  val 加载失败: {e}")

print(f"\n{SEP}")
print("探测完成。")
