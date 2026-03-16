# 新Idea实验计划：基于 `draft-small_base` 分歧的拒绝后直接放行策略

## 1. 背景与目标

### 1.1 背景
在当前投机解码框架中，`target` 负责最终验收，`draft` 负责提案。新想法是引入 `small_base` 作为“是否由领域知识驱动”的参照：

- 若 `draft` 与 `small_base` 在同前缀下预测不一致，说明 `draft` 中注入的领域知识在发挥作用。
- 当 `target` 拒绝 `draft` token 时，若同时观察到 `draft != small_base`，则将该 token 直接放行（override accept）。

### 1.2 实验目标
在不微调 `target` 的前提下，验证该策略是否可以：

1. 提升医疗领域任务准确率（重点：MedQA）。
2. 保持通用能力不显著下降。
3. 在可接受速度损失下，获得更高的领域收益。

---

## 2. 核心假设与可证伪标准

### 2.1 核心假设
- **H1（有效性）**：`draft != small_base` 事件中，包含更高比例的“可修复 `target` 错误”的 token。
- **H2（收益）**：在 `target` 拒绝时启用分歧放行，可提高领域任务最终准确率。
- **H3（风险）**：无阈值直接放行会引入噪声，导致部分错误放行与通用能力回退。

### 2.2 可证伪标准（失败判据）
任一条件满足即判该方案当前不可用：
- 相比 baseline，MedQA 准确率提升 < 0.5%。
- 通用集准确率下降 > 1.0%。
- TPS 下降 > 10%。
- 错误放行率（Override Harm Rate）高于 30%。

---

## 3. 策略定义（从激进到保守）

### 3.1 V0：硬规则（最小可跑）
当且仅当以下条件同时满足时直接放行：

- `target` 拒绝 `draft` token。
- `draft` top-1 token 与 `small_base` top-1 token 不一致。

形式化：

$$
\text{accept} = [\text{target\_reject}] \land [y_d \neq y_b]
$$

### 3.2 V1：分歧强度阈值
在 V0 基础上增加概率差阈值：

$$
\Delta\log P = \log P_d(y_d) - \log P_b(y_d)
$$

仅当 $\Delta\log P > \tau_\Delta$ 才放行。

### 3.3 V2：双阈值安全版（推荐上线候选）
在 V1 基础上增加 `target` 反对度约束（例如 logit margin / log prob gap）：

$$
\text{accept} = [\text{target\_reject}] \land [y_d \neq y_b] \land [\Delta\log P > \tau_\Delta] \land [\text{opp}_t < \tau_t]
$$

---

## 4. 对照组与实验矩阵

## 4.1 对照组
- **Baseline-A**：原生 speculative decoding（无 gate、无 override）。
- **Baseline-B**：当前已有 lossy gate 方案（你仓库中的现有最佳配置）。

### 4.2 实验组
- **Exp-V0**：硬规则直接放行。
- **Exp-V1**：分歧 + $\Delta\log P$ 阈值。
- **Exp-V2**：分歧 + $\Delta\log P$ + `target` 反对度阈值。

### 4.3 参数网格（首轮建议）
- $\tau_\Delta \in \{0.0, 0.2, 0.5, 1.0, 1.5\}$
- $\tau_t \in \{0.5, 1.0, 1.5, 2.0\}$（仅 V2）

---

## 5. 数据集与评测设置

### 5.1 数据集
- **领域集**：MedQA（主评测集，固定 seed）。
- **通用集**：从现有通用 QA/推理集抽样一份固定子集（用于监控退化）。

### 5.2 统一评测约束
- 统一 decoding 参数（temperature、top_p、max_new_tokens）。
- 统一 batch size、硬件、并发设置。
- 每个配置至少跑 3 次不同 seed，报告均值与标准差。

---

## 6. 关键指标与日志字段

### 6.1 主指标
- **Task Accuracy**（MedQA / 通用集）
- **TPS**（tokens per second）
- **Acceptance Rate**（总体接受率）
- **Override Rate**（触发直接放行比例）

### 6.2 新增质量指标（必须记录）
- **Override Precision**：被 override 的 token 中，最终带来正确答案修复的比例。
- **Override Harm Rate**：override 后导致结果变差的比例。
- **Domain Gain per 1% Speed Loss**：每 1% 速度损失带来的领域准确率提升。

### 6.3 建议日志字段
每条样本至少记录：
- `sample_id`, `prefix_len`, `target_reject`。
- `draft_token`, `base_token`, `target_token`。
- `logp_draft(draft_token)`, `logp_base(draft_token)`, `delta_logp`。
- `target_opposition_score`。
- `override_triggered`, `final_correct`。
- `answer_changed_by_override`（布尔）。

---

## 7. 落地实施步骤（可执行）

### Step 1：在解码路径加开关（最小改造）
在现有评测/解码入口新增配置项：
- `enable_divergence_override: bool`
- `override_mode: v0|v1|v2`
- `tau_delta: float`
- `tau_target_opp: float`

### Step 2：实现 override 判定函数
新增统一函数（示意）：

- 输入：`target_reject`, `draft_top1`, `base_top1`, `delta_logp`, `target_opp`, `mode`。
- 输出：`should_override: bool`, `override_reason: str`。

### Step 3：补齐日志
在每次触发 target reject 的分支中记录第 6 节字段，输出到 `jsonl`。

### Step 4：运行实验矩阵
按“对照组 + Exp-V0/V1/V2 + 网格参数”批量跑，统一写入 `logs/`。

### Step 5：汇总分析
离线脚本计算：
- 速度-精度 Pareto 前沿。
- 不同阈值下的 `Override Precision/Harm` 曲线。
- 领域词 vs 通用词分桶表现。

---

## 8. 建议执行命令模板

> 说明：以下为命令模板，请按你当前脚本参数名替换。

```bash
# 1) Baseline-A（无 override）
python lossy_spec_decode_eval.py \
  --config train_med_1b.yaml \
  --enable_divergence_override false \
  --run_name baseline_native

# 2) Exp-V0
python lossy_spec_decode_eval.py \
  --config train_med_1b.yaml \
  --enable_divergence_override true \
  --override_mode v0 \
  --run_name exp_v0

# 3) Exp-V1（示例阈值）
python lossy_spec_decode_eval.py \
  --config train_med_1b.yaml \
  --enable_divergence_override true \
  --override_mode v1 \
  --tau_delta 0.5 \
  --run_name exp_v1_d05

# 4) Exp-V2（示例阈值）
python lossy_spec_decode_eval.py \
  --config train_med_1b.yaml \
  --enable_divergence_override true \
  --override_mode v2 \
  --tau_delta 0.5 \
  --tau_target_opp 1.0 \
  --run_name exp_v2_d05_t10
```

---

## 9. 结果汇报模板（建议）

| 方案 | MedQA Acc | 通用 Acc | TPS | Override Rate | Override Precision | Override Harm | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline-A |  |  |  |  |  |  |  |
| Baseline-B |  |  |  |  |  |  |  |
| Exp-V0 |  |  |  |  |  |  |  |
| Exp-V1(best) |  |  |  |  |  |  |  |
| Exp-V2(best) |  |  |  |  |  |  |  |

---

## 10. 里程碑与时间安排（建议）

- **D1**：完成开关接入 + 日志字段打通（可跑单配置）。
- **D2**：跑完 Baseline + V0，验证 idea 是否成立。
- **D3-D4**：完成 V1/V2 阈值网格。
- **D5**：输出最终对比报告（含推荐上线阈值）。

---

## 11. 决策规则（最终）

若满足以下条件，进入下一阶段（集成到主流程）：
- `Exp-V2(best)` 相比 `Baseline-B`：MedQA 提升显著（建议 ≥1.0%）。
- 通用集下降可控（建议 ≤0.5%）。
- TPS 下降可控（建议 ≤5%）。
- `Override Harm Rate` 明显低于 V0，且稳定。

否则：
- 若 V0 有收益但风险高，保留为“数据挖掘信号”，回流到 gate 训练；
- 若 V1/V2 无稳定收益，终止该方向或仅在特定题型触发。

---

## 12. 附：最小实现优先级（MVP）

优先只做三件事，快速验证核心想法：
1. 实现 **V0** 硬规则；
2. 打通 **override 日志** 与最终正确性标记；
3. 跑 **Baseline-A vs V0**，先看 MedQA 是否有可见提升。

若 MVP 不成立，不建议立刻投入 V1/V2 网格。