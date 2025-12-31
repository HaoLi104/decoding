项目架构设计文档：领域引导投机解码 (Domain-Steered Decoding)

1. 项目背景与目标

本项目旨在实现一种无需训练 (Training-free) 的大模型领域增强方法。
通过引入一对“专家小模型”和“普通小模型”，计算它们在 Logits 层面的差异（$\Delta z$），并利用该差异动态引导通用大模型，使其在不经过微调的情况下获得领域专业能力。

当前阶段目标：

搭建三模型协同推理框架。

在 MedQA (USMLE) 数据集上进行 QA 问答测试。

对比 "Baseline (Target Model Only)" 与 "Ours (Steered Decoding)" 的准确率。

注意： 本阶段暂不实现投机解码（Speculative Decoding）的加速循环，仅实现自回归引导（Autoregressive Steering）以验证效果。

2. 模型配置 (Model Zoo)

所有模型均基于 Llama-3 架构，共享相同的 Tokenizer，无需词表对齐。

Target Model (通用大模型): meta-llama/Llama-3.1-8B-Instruct

作用: 负责最终的逻辑推理、格式遵循和上下文连贯性。

Draft Base (普通底座): meta-llama/Llama-3.2-1B-Instruct

作用: 作为减数，消除通用语料中的高频词噪音。

Draft Expert (领域专家): ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025

作用: 提供医学领域的 Logits 分布。

特性: 该模型经过 CoT 训练，倾向于逐步推理。

3. 核心算法逻辑 (Steering Mechanism)

3.1 引导公式

对于每一个生成步 $t$：


$$z_{final} = z_{target} + \alpha(t) \cdot (z_{expert} - z_{base})$$

3.2 自适应 Alpha (Adaptive Alpha via ALFAR)

介入强度 $\alpha$ 不是常数，而是根据 Target 和 Expert 的分歧程度动态计算：

计算 Target 和 Expert 输出概率分布的欧氏距离 (L2 Norm) 或 Jensen-Shannon 散度。

逻辑：

差异小 -> $\alpha \approx 1.0$ (保持大模型原意)。

差异大 -> $\alpha \to \text{Max\_Scale}$ (强行修正，听专家的)。

实现公式参考：
alpha = base_alpha + tanh(distance * sensitivity_scale) * max_boost

4. 模块划分与文件结构

请生成以下文件结构：

config.py (配置管理)

定义模型 ID 常量。

定义超参数：BASE_ALPHA, ADAPTIVE_SCALE。

定义硬件参数：LOAD_IN_4BIT = True (必须支持量化加载以适配单卡)。

model_loader.py (模型加载)

使用 transformers 和 bitsandbytes。

实现 load_models()：同时加载 3 个模型，显存优化。

实现 load_tokenizer()：加载 Target 的 Tokenizer（三者通用）。

steering_utils.py (核心算法)

calculate_adaptive_alpha(logits_target, logits_expert): 实现自适应强度计算。

compute_steered_logits(logits_t, logits_e, logits_b): 实现 Logits 加减融合。

注意： 保持 PyTorch 的张量运算高效性。

data_loader.py (数据处理)

使用 datasets 库加载 GBaker/MedQA-USMLE-4-options (HuggingFace 数据集)。

实现 format_prompt(question, options)：

将题目和选项格式化为 Llama-3 的 Chat Template。

关键点： System Prompt 必须包含 "You are a medical expert. Think step by step." 以激活 Expert 模型的 CoT 能力。

evaluator.py (评测逻辑)

实现 QA 推理循环。

Baseline 模式： 仅使用 Target Model 生成。

Steered 模式： 在 generate 循环中，每一步调用 compute_steered_logits 修改 Logits。

提示： 由于 HuggingFace generate() 接口修改 Logits 比较复杂，可以写一个简单的 greedy_search 或 sample 的 Python while 循环来实现逐 Token 生成。

答案提取： 从生成的文本中提取最终选项（A/B/C/D）。MedQA 通常需要正则匹配。

main.py (入口)

加载模型。

加载数据（取测试集的前 50-100 条做 Demo）。

运行 Baseline 评测 -> 记录准确率。

运行 Steered 评测 -> 记录准确率。

输出对比报告。

5. 编码要求 (Coding Guidelines)

中文注释： 所有关键函数、类和复杂逻辑块必须包含详细的中文注释。

类型提示： 使用 Python Type Hints (e.g., def func(x: torch.Tensor) -> float:).

异常处理： 考虑显存不足的情况，代码应尽量节省显存（使用 torch.no_grad()）。

复现性： 设置随机种子。

请开始编写代码。