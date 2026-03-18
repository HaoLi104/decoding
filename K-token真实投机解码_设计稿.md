# K-token 真实投机解码设计稿（基于 draft-small_base 分歧放行）

## 0. 结论先行

- **可以用 `speculators-main`**，它已经具备 K-token proposal + verifier 的基础抽象（`proposal_type=greedy`、`speculative_tokens`、`verifier_accept_k`、`accept_tolerance`）。
- 但你的三模型规则（`target` 拒绝后，再用 `draft vs small_base` 决策是否放行）**不是现成能力**，需要新增一个自定义 proposal/accept 策略或自定义算法实现。
- 结合你现有结果（300题：`baseline 0.29`，`v0 0.3167`，`v2 0.3133`，但速度约-50%），下一阶段目标应是：
  1) 保留精度增益；
  2) 把当前逐 token 串行三模型开销改成“**K-token 批验证 + 拒绝点稀疏触发 small_base**”。

---

## 1. 当前实现与问题复盘

## 1.1 当前实现（你现在跑通的版本）
- 每步只生成 1 token。
- 每步串行：`target` forward -> `draft` forward -> （仅不一致时）`small_base` forward。
- 这属于“逐 token 判决框架”，不是标准“draft 一次提 K 个 token、target 一次性 verify K 个位置”的真实 speculative decoding。

### 1.2 现有实验信号（300 题）
- 准确率：
  - baseline: 0.2900
  - divergence_v0: 0.3167（+2.67pp）
  - divergence_v2: 0.3133（+2.33pp）
- 速度：
  - baseline TPS: 27.1
  - v0/v2 TPS: 13.35 左右（约 -50%）

### 1.3 根因判断
- 精度增益来自规则有效；
- 速度损失来自“逐 token + 多模型串行调用”，而不是规则本身不可用。

---

## 2. `speculators-main` 可复用性评估

根据已检查代码：
- `src/speculators/proposals/greedy.py`：支持 `speculative_tokens`（一次提 K 个 token）与 verifier 接受规则。
- `src/speculators/proposals/base.py`：proposal 方法可注册扩展。
- `src/speculators/config.py`：支持 algorithm/proposal/verifier 的配置化装配。

### 2.1 能直接复用的部分
- K-token proposal 骨架。
- verifier 侧接受/拒绝流程框架。
- 配置系统（便于跑参数网格）。

### 2.2 不能直接复用的部分
- 你的 `draft-small_base` 分歧放行逻辑（尤其 v1/v2 里的 `delta_logp` 和 `target_opp` 条件）需要自定义。
- `small_base` 作为第三模型参与“拒绝点复判”的策略需要新增接口。

### 2.3 结论
- **可用，但需要二次开发**。
- 最优路径：以 `greedy proposal` 为骨架，新增 `medical_divergence_verify` 策略。

---

## 3. 目标架构（真实 K-token verify）

## 3.1 模型角色
- `draft`：一次生成长度 K 的候选链 `y_1...y_K`。
- `target`：一次 forward 输出对应 K 位置 logits，做主验证。
- `small_base`：仅在“首个拒绝点”触发复判（默认只查 1 个位置，控开销）。

### 3.2 核心流程
1. 给定前缀 `x`，`draft` 提案 K token：`y_1...y_K`。
2. `target` 对 `x + y_1...y_K` 一次验证，找到第一个不满足 target 验收的位置 `j`。
3. 处理规则：
   - 若 `j` 不存在：接受 K 个 token。
   - 若 `j` 存在：
     - 对位置 `j`，触发你的规则：
       - v0: `y_j != y_base_j` 即放行；
       - v1: 额外要求 `delta_logp_j > tau_delta`；
       - v2: 再要求 `target_opp_j < tau_target_opp`。
     - 若放行：接受到 `j`（含 `j`），然后继续下一轮。
     - 若不放行：回退到 target token（标准 verify fallback）。

> 关键：`small_base` 只在拒绝点调用，而非每个位置都调用。

---

## 4. 验收判定定义（与现有实验一致）

### 4.1 target 验收
- 默认与现在一致：top-1 一致视为通过。
- 可选扩展：支持 `verifier_accept_k` + `accept_tolerance`（借鉴 speculators）。

### 4.2 拒绝点复判（你的新 idea）
在首拒绝点 `j`：

- `delta_logp_j = log P_draft(y_j) - log P_base(y_j)`
- `target_opp_j = log P_target(t_j^*) - log P_target(y_j)`

放行条件：

$$
\text{override}_j = [y_j \neq y^{base}_j] \land [\Delta\log P_j > \tau_\Delta] \land [\text{opp}_{t,j} < \tau_t]
$$

其中 v0/v1/v2 分别是上式的子集。

---

## 5. 与现有脚本的对接策略

## 5.1 新建实现（不改旧代码）
建议新增：
- `k_spec_decode_divergence_eval.py`：真实 K-token 验证主程序。
- `k_spec_kernels.py`：proposal + verify + override 逻辑。
- `scripts/run_k_spec_300.sh`：300题一键串行脚本。

### 5.2 运行参数（首版）
- `--speculative_tokens`（K，默认 4 或 5）
- `--mode strict|divergence_v0|divergence_v1|divergence_v2`
- `--tau_delta`
- `--tau_target_opp`
- `--max_reject_rechecks`（默认 1）

---

## 6. 实验计划（基于当前结果收敛）

### Phase A：验证速度回收
对比：
- baseline（target-only）
- 当前逐 token v2（已有）
- **新 K-token v2**（本设计）

目标：
- 在维持 +2pp 左右精度增益时，TPS 相对 baseline 从 -50% 改善到可接受区间（先争取 -20% ~ -30%，再优化）。

### Phase B：K 与阈值联调
- K ∈ {2,4,6}
- `tau_delta` ∈ {0.2, 0.5, 1.0}
- `tau_target_opp` ∈ {1.0, 1.5, 2.0}

输出 Pareto：`Accuracy` vs `TPS`。

### Phase C：稳健性
- 每个最佳配置至少 3 次 seed。
- 记录 override 触发率、拒绝点位置分布、拒绝复判成功率。

---

## 7. 日志字段（新增）

除现有字段外，新增 K-token 专用字段：
- `speculative_tokens`（K）
- `proposed_len`
- `verified_prefix_len`
- `first_reject_pos`（-1 表示全通过）
- `reject_recheck_called`
- `reject_recheck_override`
- `accepted_tokens_this_round`

这些字段用于回答：速度瓶颈在 proposal 还是 verify，规则增益来自哪里。

---

## 8. 性能优化建议（实现时必须做）

1. `small_base` 仅在首拒绝点调用，默认每轮最多一次。
2. 尽量让 `draft` 与 `target` 分布在不同 GPU，减少互相抢占。
3. 预留 `small_base_device_map=cpu` 兜底开关（可跑通优先）。
4. 使用 KV cache，避免重复 prefix 计算。

---

## 9. 风险与回退

- 风险1：K 过大导致误拒绝传播，反而拖慢。
  - 回退：限制 K 到 2/4，先看有效吞吐。
- 风险2：v2 过滤过严，override 几乎不触发。
  - 回退：先调宽 `tau_target_opp`。
- 风险3：三模型显存压力大。
  - 回退：`small_base` 上 CPU，仅在拒绝点调用。

---

## 10. 与 `speculators-main` 的集成建议（落地顺序）

### 路线 A（建议先做）
在你当前仓库先落地 `k_spec_decode_divergence_eval.py`，验证算法本身是否成立。
- 优点：迭代快，日志和判定可控。
- 缺点：工程复用有限。

### 路线 B（第二步）
迁移到 `speculators-main`：
- 新增自定义 proposal/config（例如 `proposal_type=medical_divergence_greedy`）
- 在 verifier accept 流程加入“拒绝点复判 hook”
- 复用其配置系统和测试框架。

---

## 11. 里程碑

- M1：完成 K-token strict（无 override）并跑通 300 题。
- M2：接入 v0/v2 拒绝点复判，跑 300 题。
- M3：完成 K 与阈值网格，输出 Pareto 与推荐配置。
- M4：决定是否迁移到 `speculators-main` 的正式实现。

---

## 12. 一句话结论

你现在的实验已经证明“规则有用”，下一步不是继续堆逐 token 实验，而是切到“**K-token verify + 拒绝点稀疏复判**”；`speculators-main` 可以作为工程化底座，但需要为三模型复判策略做定制扩展。
