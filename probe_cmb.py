"""
补充探测 FreedomIntelligence/CMB 的两个 config：CMB-Clin 与 CMB-Exam。
上次 probe_datasets.py 因缺少 config_name 参数而失败。

运行：
  cd /data/ocean/decoding && conda activate kvner
  export HF_ENDPOINT=https://hf-mirror.com
  python probe_cmb.py 2>&1 | tee results/probe_cmb.txt
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset, get_dataset_split_names

SEP = "=" * 60

for config in ["CMB-Clin", "CMB-Exam"]:
    print(f"\n{SEP}")
    print(f"=== FreedomIntelligence/CMB  config={config} ===")
    try:
        splits = get_dataset_split_names("FreedomIntelligence/CMB", config_name=config)
        print(f"  splits: {splits}")
        for sp in splits:
            ds = load_dataset("FreedomIntelligence/CMB", config, split=sp)
            print(f"  {sp}: {len(ds)} 条  字段: {ds.column_names}")
        ds0 = load_dataset("FreedomIntelligence/CMB", config, split=splits[0])
        print(f"  示例 ({splits[0]}[0]):", ds0[0])
    except Exception as e:
        print(f"  失败: {e}")

print(f"\n{SEP}")
print("CMB 探测完成。")
