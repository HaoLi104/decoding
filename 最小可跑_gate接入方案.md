# 最小可跑 gate 接入方案（student_mlp / student_logistic）

## 目标

在现有三模型设定下，把已训练的分类器接入“有损投机解码”流程，先验证两件事：

1. 领域题正确率是否高于 target 原始解码。
2. 速度是否优于纯 target 解码（至少不显著变慢，理想是提升）。

---

## 结论先行：推荐分两阶段做

- 阶段 A（最小可跑，1-2 天）：先在你当前仓库实现可控的 gate 解码实验脚本，快速拿到准确率与速度曲线。
- 阶段 B（工程化接入）：再把 gate 逻辑接入 vLLM + speculators 线上路径。

原因：
- speculators 主要提供 draft/speculator 训练与标准化格式；
- 真正 token 接受/拒绝执行在 vLLM 推理链路中完成；
- 直接改 vLLM 内核成本较高，先用阶段 A 验证策略正确性更稳。

---

## 阶段 A：最小可跑闭环（不改 vLLM 内核）

## A1. 需要新增的模块（在当前仓库）

1. `lossy_gate_model.py`
- 作用：加载 `train_token_classifiers.py` 产出的 checkpoint（teacher/student 都可，先用 student）。
- 功能：
  - 读取 `model_state` 与 `standardizer`；
  - 输入特征向量返回 gate 概率 `p_accept`；
  - 支持阈值 `tau`。

2. `lossy_spec_decode_eval.py`
- 作用：实现“draft 提案 + target 校验 + gate 放行”的有损投机解码主循环，并在 MedQA 上评估。
- 输入：
  - target / draft 模型路径；
  - classifier ckpt（student_mlp 或 student_logistic）；
  - 评测集与阈值。
- 输出：
  - 正确率、平均生成长度、tokens/s、平均 accepted tokens、gate 触发率。

3. `scripts/run_lossy_gate_sweep.sh`
- 作用：扫一组阈值（例如 0.30/0.40/0.50/0.60/0.70），产出对比表。

## A2. 算法（最小版本）

对每一步：

1. 用 draft 生成候选 token（先做 1-step，最稳）。
2. 用 target 计算同一前缀下的 next-token logits。
3. 若 draft token 与 target greedy token 一致：直接接受（等价无损 accept）。
4. 若不一致：
   - 提取 gate 特征（先用 student 需要的 `[H_target, H_draft]`）；
   - 分类器输出 `p_accept`；
   - 若 `p_accept >= tau`，放行 draft token（有损 accept）；否则拒绝并回退 target token。
5. 重复直到结束。

说明：
- 这就是最小可跑的“classifier-gated lossy speculative decoding”。
- 先做 1-step 可以避免复杂的多 token 回滚逻辑，先验证方向。

## A3. 第一版实验命令模板

### Baseline 1：target 原始解码

```bash
python medqa_test_single_model_transformers(1).py \
  --model_path /data/ocean/decoding/model/Qwen/Qwen3-14B \
  --split test \
  --limit 300 \
  --out logs/baseline_target_test300.json
```

### Baseline 2：target + draft（无 gate，严格拒绝分歧）

```bash
python lossy_spec_decode_eval.py \
  --mode strict \
  --target_model /data/ocean/decoding/model/Qwen/Qwen3-14B \
  --draft_model /data/ocean/decoding/model/II-Medical-8B \
  --tokenizer /data/ocean/decoding/model/Qwen/Qwen3-14B \
  --split test \
  --limit 300 \
  --out logs/spec_strict_test300.json
```

### Ours：有损 gate（student_mlp）

```bash
python lossy_spec_decode_eval.py \
  --mode gate \
  --target_model /data/ocean/decoding/model/Qwen/Qwen3-14B \
  --draft_model /data/ocean/decoding/model/II-Medical-8B \
  --tokenizer /data/ocean/decoding/model/Qwen/Qwen3-14B \
  --gate_ckpt checkpoints/token_classifier/student_mlp_mined_full_test_qwen_dp.pt \
  --tau 0.50 \
  --split test \
  --limit 300 \
  --out logs/spec_gate_tau050_test300.json
```

### 阈值扫描

```bash
bash scripts/run_lossy_gate_sweep.sh
```

建议扫：`tau in {0.30, 0.40, 0.50, 0.60, 0.70}`。

## A4. 阶段 A 验收标准

满足以下至少两条即可进入阶段 B：

1. 在至少一个 `tau` 上，正确率高于 target baseline。
2. 在至少一个 `tau` 上，tokens/s 高于 target baseline。
3. 能观察到清晰 trade-off：`tau` 越低更激进，速度更快但可能更不稳；`tau` 越高更保守。

---

## 阶段 B：接入 speculators + vLLM（工程化）

## B1. 需要改哪些模块（speculators-main）

以下路径相对 `speculators-main/` 根目录。

1. `src/speculators/proposals/greedy.py`
- 新增 gate 配置字段（先放在 Greedy 配置里，最小改动）：
  - `gate_enabled: bool = False`
  - `gate_ckpt_path: str | None = None`
  - `gate_threshold: float = 0.5`
  - `gate_feature_mode: str = "student_target_draft"`
  - `gate_max_overrides_per_step: int = 1`

2. `src/speculators/proposals/base.py`
- 保持基类不动或只补注释；核心是让 proposal config 可携带 gate 字段。

3. `src/speculators/models/eagle3/core.py`
- 在生成 draft proposal 与 verifier 比对处增加 hook：
  - 若原本可接受：按原逻辑接受。
  - 若原本不可接受且 `gate_enabled=True`：
    - 构造 gate 特征；
    - 调用 gate scorer；
    - `p_accept >= threshold` 时允许放行。

4. 新增 `src/speculators/proposals/gate_runtime.py`
- 放 gate 推理运行时：
  - 加载你已有 classifier checkpoint；
  - 做标准化；
  - 输出 `p_accept`。

5. 新增 `src/speculators/proposals/feature_extract.py`
- 从 runtime 状态提取 gate 特征（最小先支持 student 两路特征）。

备注：
- 真正线上 vLLM 路径也会消费 speculator config；如果 vLLM 对额外字段严格校验，需要在 vLLM 对应 spec decode 插件处同步放开字段并调用 gate。这个是阶段 B 的主要工程点。

## B2. 改动顺序（必须按顺序）

1. 先完成阶段 A 脚本，验证 gate 策略有效。
2. 在 speculators 增加 gate config 字段（不改行为）。
3. 增加 gate runtime（可独立单测）。
4. 在 eagle3 core 插入最小 gate hook（仅 1-step override）。
5. 小规模回归测试（10~50 条样本）。
6. 再扩大到 300 / 全 test。

---

## 第一版实验矩阵（建议）

固定数据：MedQA test，先 `limit=300`。

1. target baseline
2. strict speculative（无 gate）
3. gate + student_logistic，`tau` 扫描
4. gate + student_mlp，`tau` 扫描

输出统一指标：
- accuracy
- tokens/s
- avg accepted tokens / step
- gate override rate
- 平均每题时延

---

## 建议你现在立刻执行的命令（最短路径）

1. 先做阶段 A（不依赖 speculators 内核改造）。
2. 在 `tau=0.5` 跑通后，再做 sweep。
3. 若 sweep 中出现“accuracy 上升且速度不降”，就进入阶段 B。

---

## 风险与规避

1. 风险：classifier 输入特征与在线提取不一致。
- 规避：第一版只用 student 特征切片（与训练脚本一致）。

2. 风险：过度放行导致正确率下降。
- 规避：用 `tau` 扫描 + 上限 `gate_max_overrides_per_step=1`。

3. 风险：`delta_p` 加权导致不稳定。
- 规避：上线 gate 先不用 `delta_p` 作为在线规则，仅作为离线分析项。

---

## 最终判断标准

当你看到以下结果时，说明方案成立：

- 存在某个 `tau*`，使得
  - accuracy(target+gate) > accuracy(target baseline)
  - 且 tokens/s(target+gate) >= tokens/s(target baseline) 或接近不降

这就达成你当前阶段目标：
“在加速大模型的同时，让大模型在领域内做对原本做不对的题目”。
