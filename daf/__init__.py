"""DAF — Domain Absorption Flywheel
毕业论文第二点的飞轮 PoC 工具包：

模块约定：
  flip_definition     —— flip 事件统一口径（accepted ∧ draft != target_top1）
  run_flip_logger     —— Round k 解码 + flip 事件采集（产出 flip_events_round{k}.jsonl）
  fdlp_score          —— Flip-Driven Layer Placement 反向传播打分（含 4 套对照）
  build_flip_sft_data —— flip 事件 → LLaMA-Factory Alpaca SFT 数据
  gen_lora_yaml       —— 根据 layer_scores 自动生成 LLaMA-Factory LoRA yaml
  merge_lora          —— peft merge_and_unload 合并 LoRA → base
  convergence_check   —— 跨轮 flip rate 收敛判定 (ρ_k)
  hotspot_stability   —— 热点 Top-K 模块 Jaccard / Spearman
  run_eval_round      —— 纯 Target 评测 + MMLU 通用域守护
"""
