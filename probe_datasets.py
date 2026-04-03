"""
数据集规模与结构探测脚本。

用途：在选定下一个实验领域前，探测各候选数据集的 split 规模、字段格式与样本示例。
覆盖：FreedomIntelligence/CMB、CMB-Exam、C-Eval 医学子集、MMLU 医学子集、MedMCQA。

运行：
  cd /data/ocean/decoding && conda activate kvner
  export HF_ENDPOINT=https://hf-mirror.com
  python probe_datasets.py 2>&1 | tee results/probe_datasets.txt
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset, get_dataset_split_names

SEP = "=" * 60


def probe_dataset(name: str, config: str = None, splits_override=None, cache_dir=None):
    """通用探测函数：打印 split 规模、字段名和第一条样本。"""
    label = f"{name}" + (f" ({config})" if config else "")
    print(f"\n{SEP}")
    print(f"=== {label} ===")
    try:
        kwargs = {}
        if config:
            kwargs["name"] = config
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        splits = splits_override or get_dataset_split_names(name, **({"config_name": config} if config else {}))
        print(f"  splits: {splits}")

        for sp in splits:
            try:
                ds = load_dataset(name, split=sp, **kwargs)
                print(f"  {sp}: {len(ds)} 条  字段: {ds.column_names}")
            except Exception as e:
                print(f"  {sp}: 加载失败 -> {e}")

        # 打印第一个可用 split 的第一条样本
        try:
            first_split = splits[0]
            ds0 = load_dataset(name, split=first_split, **kwargs)
            print(f"  示例 ({first_split}[0]):", ds0[0])
        except Exception as e:
            print(f"  示例加载失败: {e}")

    except Exception as e:
        print(f"  失败: {e}")


def probe_ceval_subjects():
    """探测 C-Eval 医学相关子集的 val/test 规模。"""
    subjects = [
        "physician",
        "veterinary_medicine",
        "traditional_chinese_medicine",
        "basic_medicine",
        "clinical_medicine",
    ]
    print(f"\n{SEP}")
    print("=== C-Eval 医学相关子集 ===")
    for subj in subjects:
        try:
            dv = load_dataset("ceval/ceval-exam", subj, split="val")
            dt = load_dataset("ceval/ceval-exam", subj, split="test")
            print(f"  {subj}: val={len(dv)}, test={len(dt)}")
            print(f"    val[0]: {dv[0]}")
        except Exception as e:
            print(f"  {subj}: 失败 -> {e}")


def probe_mmlu_medical():
    """探测 MMLU 医学 + 专业子集的 test 和 auxiliary_train 规模。"""
    subjects = [
        "professional_medicine",
        "clinical_knowledge",
        "medical_genetics",
        "anatomy",
        "college_medicine",
        "college_biology",
    ]
    print(f"\n{SEP}")
    print("=== MMLU 医学专业子集 ===")
    for subj in subjects:
        try:
            dt = load_dataset("cais/mmlu", subj, split="test")
            try:
                tr = load_dataset("cais/mmlu", subj, split="auxiliary_train")
                print(f"  {subj}: test={len(dt)}, auxiliary_train={len(tr)}")
            except Exception:
                print(f"  {subj}: test={len(dt)}, auxiliary_train=不可用")
            print(f"    test[0]: {dt[0]}")
        except Exception as e:
            print(f"  {subj}: 失败 -> {e}")


if __name__ == "__main__":
    # 1. FreedomIntelligence/CMB（通用中文医学基准）
    probe_dataset("FreedomIntelligence/CMB")

    # 2. FreedomIntelligence/CMB-Exam（执照考试 MCQ，重点关注 train 规模）
    probe_dataset("FreedomIntelligence/CMB-Exam")

    # 3. C-Eval 医学相关子集
    probe_ceval_subjects()

    # 4. MMLU 医学子集（含 auxiliary_train）
    probe_mmlu_medical()

    # 5. MedMCQA（印度执照考试，MCQ，train/val/test 全有）
    probe_dataset("medmcqa", splits_override=["train", "validation", "test"])

    print(f"\n{SEP}")
    print("探测完成。结果已输出，请根据以下标准选域：")
    print("  1. train 集 >= 3000 条 MCQ 格式")
    print("  2. test 集 >= 100 条（含答案）")
    print("  3. 优先 Target-32B 预估 acc <= 0.70 的领域")
