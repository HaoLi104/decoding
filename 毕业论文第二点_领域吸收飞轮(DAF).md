# 毕业论文第二点：领域吸收飞轮（Domain Absorption Flywheel, DAF）

**英文题名（草拟）：** Domain Absorption Flywheel: A Co-Evolving Decoding–Training Loop for Speculative-Decoding-Aware PEFT Placement

**与第一点关系：**
第一点（基于对比置信度的领域知识挟持与软引导投机解码，以下简称 **DSSD**）证明了一件事——在 **不更新任何权重** 的前提下，仅依赖 Draft−Base 的对比信号即可在投机解码的验收链路上注入领域知识，使 32B Target 的 Surgery 子集 acc 从 0.650 提升至 ~0.690（C1/C5/C6 多策略一致），但代价是每次推理都必须挂载 Draft + Base 双模型，**部署仍然贵**。

本第二点要回答的真正工程问题是：

> **能否把 DSSD 在解码期"在线注入"的领域信号，沉淀为 Target 自身权重里的最小 PEFT 增量？并且，能否让"沉淀"这一过程本身被 DSSD 解码事件持续监督，形成自驱动的协同进化？**

我们提出 **Domain Absorption Flywheel (DAF)**：把 DSSD 解码过程中产生的 **token flip 事件**同时用作两个角色——

1. **作为稀疏监督锚点**，反推 LoRA 应该挂在 Target 的哪些层／模块（"在哪里植入");
2. **作为收敛可观测量**，判断该轮 LoRA 训练是否已经"消化"了相应的领域信号（"何时停止");

并把"解码 → 选层 → 微调 → 再解码"嵌套成 $K$ 轮迭代，直到 flip rate 收敛到噪声底。

---

## 0. 审稿人视角：核心创新与可证伪门槛

**严格审稿人会立刻指出，下列子组件都不新：**

- 用梯度／敏感度做层重要性排序 → AdaLoRA / IGU-LoRA / GoRA / LISA 等已成熟；
- 用 token 级权重做 SFT → GIFT / ProFit / ssToken 已成熟；
- 用 activation patching 做机制验证 → MI 文献标准工具；
- 用迭代 SFT / 自蒸馏 → 已有成熟 pipeline。

**因此本工作的"创新锚点"必须收敛到一处独有组合：**

> **将 speculative decoding 内生的 token flip 事件赋予"双重角色"**——同时作为 PEFT placement 的 *event-conditioned supervision* 与训练收敛的 *self-consistent stopping criterion*——并据此形成"解码事件 ↔ 训练放置"的闭环飞轮。这一组合在公开文献检索中未见。

**可证伪门槛（必须写进 limitations 章）：**

- **门槛 A**：若用 entropy / disagreement / all-token 等替代事件跑同样的飞轮也能稳定收敛、acc 接近 → flip 事件的**不可替代性**主张应收缩；
- **门槛 B**：若飞轮 1 轮内即收敛（$K^\ast=1$）→ "飞轮"叙事降级为"FDLP 单轮 + 残值微调"，应回到 FDLP 单轮版本；
- **门槛 C**：若飞轮收敛到 $\text{KL}(\text{Target}_{v_K}\,\|\,\text{Draft})\approx 0$ 且通用域显著退化 → 飞轮只是把 Target 退化成 Draft 的复制品，应承认"飞轮存在但没解决根本问题"。

> 主动把这三条门槛写进论文，反而会让答辩委员认为方案严谨而非脆弱。

---

## 1. 论文故事：为什么重要、为什么难、创新是什么

### 1.1 为什么重要（Important）

1. **DSSD 的部署成本不可持续**：第一点的最优策略需要同时驻留 32B Target + 3B Draft + 3B Base，单卡 H200（141 GB）虽勉强容纳，但生产环境推广困难；
2. **"领域知识 → 参数"的转化是工业刚需**：能否用极小 LoRA 增量把"在线引导带来的 +4 acc"沉淀为离线的 Target 能力，是检验"DSSD 是否值得做"的最强落地证据；
3. **传统 PEFT placement 缺机制锚点**：现有 LoRA 选层方法（Flexora / Act-LoRA / IGU-LoRA）都依赖 *任务级* loss 或 *层级* 激活统计，与"具体哪些 token 决策需要被改写"没有直接绑定。DSSD 内生的 flip 事件恰好提供了这一绑定。

**故事一句话：**
我们让"DSSD 解码期事件"指导 Target 的 PEFT 训练，再让"训练后的 Target"反过来减少 DSSD 解码期事件，迭代直至两者达到动态平衡。

### 1.2 为什么难（Hard）

1. **Flip 是稀疏事件**：在 C9 等强引导策略下 flip 率约 5–15%，跨题样本量需要足够大才能稳定统计层重要性；
2. **飞轮可能收敛到"Target = Draft"**（伪进步）：必须用通用域守护集监控，证明 Target 不是简单地变成 Draft 的拷贝；
3. **Flip 热点可能在轮间漂移**：Round k 加固的层未必是 Round k+1 的热层 → 需要"热点稳定性"分析作为机制证据；
4. **算力代价**：32B Target 上的 LoRA 训练 + 多轮迭代，每轮约小时级；必须严格预算 $K\leq 3$；
5. **训练-推理动力学不匹配**：训练用标准 SFT，推理用 DSSD；FDLP 选出的"对推理 flip 最敏感的层"是否同样是"对 SFT loss 最敏感的层"？这本身需要实验验证。

### 1.3 我们到底哪里新（创新边界 + 可证伪条件）

为避免被审稿人扣"换皮重要性分配"的帽子，本工作把创新点严格收敛为三层：

**创新点 1（核心）：Flip 事件的"双重角色"**

> Flip 事件既是 placement 的稀疏监督锚点（$F_t=1$ 时计算 flip-weighted 梯度 → 选层），又是飞轮收敛的可观测指标（$\overline{F}^{(k)}$ 单调下降 → 早停）。监督信号与终止信号同源，规避了"用 acc 早停带来的过拟合争议"。

**创新点 2（系统）：解码-训练协同进化的闭环**

> "Decode → Place → Train → Re-decode" 形成 $K$ 轮迭代飞轮。第一点产生 flip 事件，第二点把它转为参数更新；更新后的 Target 又改变下一轮 flip 分布。该闭环在公开文献中未见。

**创新点 3（防御性）：飞轮事件必要性的迭代级对照**

> 不仅在单轮 placement 上对比 flip vs entropy / disagreement，还在 **整个飞轮迭代轨迹上**对比：用非 flip 事件能否同样形成稳定收敛的飞轮。若只有 flip 飞轮收敛、其他事件类型不会 → 这是"flip 事件不可替代"的强证据。

---

## 2. 方法（Method）

### 2.1 准备：复用第一点的 flip 事件定义与日志字段

**Flip 事件定义**（与第一点 `TelemetryLogger` 字段逐字段对齐，避免口径漂移）：

第 $t$ 解码步发生 flip 当且仅当：

$$
F_t = 1 \iff \underbrace{\arg\max_x P_{\mathrm{target}}(x\mid p_t) \neq x^{\mathrm{final}}_t}_{\text{Target 原本不会输出 final token}} \;\land\; \underbrace{x^{\mathrm{final}}_t = x^{\mathrm{draft}}_t}_{\text{final token 来自 Draft 提案}} \;\land\; \underbrace{\text{accept}_t = \text{True}}_{\text{且被验收通过}}
$$

记号：

- $A_t = \arg\max_x P_{\mathrm{target}}(x\mid p_t)$：Target 在该步本来要输出的 token；
- $B_t = x^{\mathrm{final}}_t$：实际进入序列的 token（即 flip 后的目标 token）；
- $p_t$：第 $t$ 步的 prefix；
- 触发策略：本工作主线固定使用第一点的 **C9 策略**（token 级二值门控 + 线性 ΔP，PPT 主推），所有飞轮实验在该策略下产生 flip 事件，避免飞轮结果被引导策略选择污染。

**日志 schema（Round $k$ 解码阶段输出）：**

```json
{"qid": ..., "step": t, "prefix_ids": [...], "A": ..., "B": ...,
 "F": 1, "delta_P": ..., "delta_logit_x": ..., "H_t": ...,
 "draft_top1": ..., "base_top1": ..., "round": k}
```

> 实现增量小：第一点的 `TelemetryLogger` 已记录 accept、draft_token、target 概率，只需扩展 `A_t` 与 `B_t` 字段及 `F_t` 二值标签。

### 2.2 飞轮主算法（伪代码）

```
输入：Target_v0 = Qwen2.5-32B-Instruct (冻结基座)
      Draft = Qwen2.5-3B-Instruct-Surgery
      Base = Qwen2.5-3B-Instruct
      训练集 D_train（MedMCQA Surgery 训练子集）
      总 LoRA rank 预算 R_total
      最大轮数 K_max = 3
      flip rate 收敛阈值 ε = 0.02

输出：Target_vK + 累积 LoRA 适配器

flip_rate_history = []
for k = 0 to K_max - 1:
    # 阶段 1: 引导解码 + flip 事件采集
    E_k, flip_rate_k = run_DSSD_with_C9(
        target = Target_v_k, draft = Draft, base = Base,
        dataset = D_train, log_flip = True
    )
    flip_rate_history.append(flip_rate_k)

    # 收敛判定
    if k > 0 and abs(flip_rate_history[k-1] - flip_rate_k) < ε:
        break  # 飞轮收敛
    if flip_rate_k < flip_rate_history[0] * 0.1:
        break  # 已吸收 90% 信号

    # 阶段 2: FDLP 选层（详见 2.3）
    L_k, r_k = FDLP(Target_v_k, E_k, R_total)

    # 阶段 3: 在选定层／模块上挂 LoRA + 标准 SFT
    LoRA_k = train_LoRA(
        target = Target_v_k, layers = L_k, ranks = r_k,
        data = build_sft_data_from_flips(E_k),  # 详见 2.4
        max_steps = 1000  # 单轮训练步数预算
    )
    Target_v_{k+1} = Target_v_k + LoRA_k

return Target_v_K, flip_rate_history
```

### 2.3 子模块 A：FDLP（Flip-driven LoRA Placement，单轮选层）

> 这是上一版第二点的核心方法，作为飞轮 **每一轮内部** 的"选层"步骤被复用。完整推导参见档案文件 `毕业论文第二点_Flip驱动LoRA层位选择.md`，本节仅给出最小必要定义。

#### 2.3.1 Flip-weighted 梯度敏感度打分

对当前 Target $\theta$，定义"把概率推向 flip 目标 token $B_t$"的损失：

$$
\mathcal{L}(p_t, B_t) = -\log P_\theta(B_t \mid p_t)
$$

层／模块敏感度分数（按参数量归一化，避免大矩阵天然梯度更大）：

$$
s_{\ell, m} = \mathbb{E}_{t : F_t = 1}\!\left[\frac{\lVert \nabla_{W_{\ell, m}} \mathcal{L}(p_t, B_t) \rVert_F}{\sqrt{\lVert W_{\ell, m} \rVert_0} + \epsilon}\right]
$$

层分数：$S_\ell = \sum_{m \in \mathcal{M}} s_{\ell, m}$（$\mathcal{M}$ = `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj`）。

#### 2.3.2 层选择与 rank 分配

- 取 $S_\ell$ 排序的 Top-$K$ 层为 $\mathcal{L}_K$；
- 在固定总预算 $R_{\text{total}}$ 下做 rank 分配：$r_\ell \propto \mathrm{clip}(S_\ell)$，$\sum_{\ell \in \mathcal{L}_K} r_\ell = R_{\text{total}}$。

#### 2.3.3 因果验证（小样本，机制章节用）

- **Activation patching**：$\Delta_\ell = \mathbb{E}\big[\log P(B \mid \text{patch at }\ell) - \log P(B)\big]$；
- 仅在 50–200 个 flip 样本上做，验证 FDLP 选出的层与 patching 峰值层一致性。

### 2.4 子模块 B：飞轮迭代调度

#### 2.4.1 训练数据构造（按 flip 事件采样）

每一轮的 SFT 数据来自该轮采集的 flip 集合 $E_k$：

- **正样本**：以 flip 发生的 prefix $p_t$ 为输入，目标 token 为 $B_t$（即"教 Target 在这个 prefix 下产出 Draft 提议的领域 token"）；
- **平衡样本**（防止过拟合到 flip 子分布）：按 1:1 加入相同数量的 **非 flip 步**（$F_t=0$）作为负对照，目标即 Target 自身的 argmax token，相当于"复习不需要改的位置"；
- **可选**：保留 25% 的 Alpaca 通用数据作为格式锚点（与第一点 Draft 训练保持一致）。

#### 2.4.2 LoRA 累积策略

两种实现方式，论文里固定其一：

- **方式 A（增量累积）**：每轮新增一个 LoRA 适配器挂在 $\mathcal{L}_K^{(k)}$ 上，最终 Target_vK 同时挂 $K$ 组适配器（推理时合并）；
- **方式 B（重 base 化）**：每轮训练完直接把 LoRA merge 进 Target 权重，下一轮在合并后的 Target 上重新选层。

> **建议主线用方式 B**（更接近真实部署、消去 K 组适配器同时存在的工程复杂度）；方式 A 留作消融。

#### 2.4.3 收敛判定

满足任一条件即停止：

1. $|\overline{F}^{(k)} - \overline{F}^{(k-1)}| < \varepsilon$（flip rate 跨轮变化小于 2%）；
2. $\overline{F}^{(k)} < 0.1 \cdot \overline{F}^{(0)}$（已吸收 90% 信号）；
3. 通用域守护集 acc 退化超过 1.0%（触发回退到 $v_{k-1}$，并停止飞轮）；
4. $k = K_{\max} = 3$（硬上限）。

---

## 3. 飞轮的护栏与已知失败模式

### 3.1 失败模式 1：收敛到 Draft（伪进步）

**症状**：飞轮使 acc 提升，但 $\text{KL}(\text{Target}_{v_K}\,\|\,\text{Draft})$ 单调下降，且通用域 acc 显著退化。

**护栏**：
- 必做实验：每轮在 **MMLU 子集（500 题）** 上测 acc，绘制"飞轮轮次 vs 通用域 acc"曲线；
- 通用域退化 > 1.0% 时强制停飞轮；
- 报告"Target 与 Draft 的 logit cosine 相似度"随飞轮轮次的演化。

### 3.2 失败模式 2：收益递减到不可观测

**症状**：Round 0→1 flip rate 大降，Round 1→2 几乎不动；$K^\ast = 1$。

**应对（不必修复，写成结论即可）**：
- 老实把 $K^\ast = 1$ 报告出来——这本身是一条**可量化的经验定律**："在 Surgery 子集上，DSSD 引导的领域信号在 1 轮 LoRA 内即可被吸收"；
- 此时论文的方法贡献从"飞轮"降级回"FDLP 单轮 + 收敛判据"，但仍合格。

### 3.3 失败模式 3：Flip 热点跨轮漂移

**症状**：Round 0 与 Round 1 的 Top-K 热层 Jaccard 相似度 < 0.3，飞轮像在"打地鼠"。

**护栏（也是机制实验本身）**：
- 必做：**热点稳定性图**——画 Round k 与 Round k+1 的 Top-K 层 Jaccard 相似度；
- 高 Jaccard（>0.6）意味着飞轮在"反复加固同一片热区"；
- 低 Jaccard 意味着飞轮在"沿着轨迹拓宽吸收带"——这两种情况都可写成有意义的发现，关键是把现象刻画清楚。

### 3.4 失败模式 4：训练-推理动力学不匹配

**症状**：FDLP 选出的层在 SFT loss 上没有明显梯度优势，导致 LoRA 训练 underfitting。

**护栏**：
- 在 LoRA 训练阶段额外报告"每层 LoRA $A,B$ 矩阵的范数演化"，如果热层的 LoRA 范数显著大于其他层 → 验证 FDLP 选层在 SFT 动力学下仍然有效；
- 否则需在论文中讨论"推理敏感度 ≠ SFT 敏感度"的差距。

---

## 4. 可行性分析

### 4.1 工程可行性：**中**

**优势：**
1. 第一点的 DSSD pipeline、TelemetryLogger、StaticCache 全部可复用；
2. 飞轮的工程增量主要是：(a) 扩展日志字段记录 $A_t, B_t, F_t$；(b) 写一个 FDLP 打分脚本（只需在 Target 上做反向传播但不更新权重）；(c) 写一个 LLaMA-Factory 配置生成器，把 FDLP 选层结果转为 `lora_target` 字段；
3. 32B Target 的 LoRA 训练在 H200 单卡上勉强可行（bf16 32B ≈ 64 GB + 优化器状态 + 激活，约 110 GB，预留 30 GB margin）。

**难点：**
1. 单轮训练时间约 1–2 小时（取决于训练步数和 batch size），3 轮飞轮约 4–6 小时；
2. FDLP 反向传播显存峰值需要严格控制（建议 batch=1、micro-batch、按需 hook 抓取）；
3. 多轮 flip 日志的存储与去重需要工程化处理（每题约 100–500 个 flip 事件，3 轮约百 MB 级日志）。

### 4.2 科学可行性：**中**

- **乐观情景**：飞轮在 2 轮内收敛，最终 Target_v2 在 Surgery 上 acc 接近 0.690（与第一点 DSSD 在线引导持平），但 **不再需要 Draft + Base 联合解码**，纯 Target 推理 tps 恢复到 ~27；
- **中性情景**：飞轮在 1 轮内饱和，acc 提升 ~3 个百分点（0.650 → ~0.680），保留"FDLP 选层比全层 LoRA 训练更高效"的工程价值；
- **悲观情景**：飞轮过程中通用域退化 > 1.0% 触发回退，承认"DSSD 信号无法在不损害通用能力前提下被完全沉淀"——仍是有效阴性结论，需要提前与导师对齐预期。

### 4.3 与第一点的关系：**强叠加**

- 第一点：**零训练**的解码期融合，证明"领域信号的存在性"；
- 第二点：**最小训练**的解码-训练协同进化，证明"领域信号的可吸收性"；
- 联合叙事：DSSD 不是终点而是手段——它产生的引导事件本身就是一种 *自动数据标注*，把"模型边界改写"的关键 token 显式标记出来供 PEFT 使用。

---

## 5. 相关工作与差异点

> 本节按"邻居距离"由近到远列出，所有最危险的撞车候选必须在论文中正面对比。

### 5.1 PEFT 层位／rank 分配（最近邻，必须正面对比）

| 工作 | 与本文最大重叠 | 我们的差异 |
|------|---------------|-----------|
| **Flexora** (ACL 2025) | 自动 LoRA 层选择（HPO + 展开微分） | Flexora 用 *task-level* HPO，我们用 *event-conditioned* 监督 |
| **Act-LoRA** (MDPI Information 2026) | 激活幅值选层 | 我们用梯度，但加 flip 条件化；明确比较两者稳定性 |
| **IGU-LoRA** | Integrated Gradients 做层内 rank 分配 | IGU 用全数据集 IG，我们仅在 $F_t=1$ 子集上 |
| **GoRA** (2025) | 梯度驱动的自适应 rank + 初始化 | 同上，事件条件化是关键差异 |
| **AdaLoRA / SoRA / PiLoRA / LoSA / PE-DyRA** | 自适应 rank 分配 | 选 1–2 个作为 L8 强基线 |
| **LISA** (2024) | Layerwise Importance Sampling | 概念邻居 |
| **CAST** (2025) | head-level conflict-aware sparse tuning | 概念邻居 |

关键词：`adaptive rank allocation`, `LoRA layer selection`, `importance-based PEFT`.

### 5.2 Token 级加权 SFT（精神同构，必须 cite 并区分）

| 工作 | 它的事件定义 | 我们的差异 |
|------|-------------|-----------|
| **GIFT** (Diffusion LM) | 按 token entropy 加权 loss | 必须在 E3 中对比 entropy 选层 |
| **ProFit** (2025) | 按 token probability 选择高价值 token | 概念同构 |
| **ssToken** (2025) | 用历史模型 loss 差选 token | 概念同构（但他们也是迭代的——必须仔细区分） |

### 5.3 投机解码与 draft 微调（叙事承接面）

- **EDA (Efficient Draft Adaptation)**：迭代 Draft 不迭代 Target；
- **LK Losses**：直接优化 acceptance rate；
- **Variational SD (VSD)**：把 draft 训练成最大化 acceptance；
- **DEL (Dynamic Exit Layer, COLM 2025)**：用 acceptance rate 调整 exit layer，**与我们"用 SD 信号反推架构决策"思路最近**，但他们是 self-SD 的内部 exit，我们是 PEFT 放置；
- **Online Speculative Decoding** (Liu et al., 2024)：在线更新 Draft，但不形成飞轮闭环。

### 5.4 自蒸馏 / 迭代 SFT（飞轮模式的远邻）

- **Self-Distillation**：用模型自身输出作为目标；我们的"目标"是 Draft 提议且被验收通过的 token，是 *跨模型蒸馏*；
- **Iterative SFT** / **Bootstrapping**：迭代 fine-tune 是已有 pipeline，但**用 SD 解码事件驱动迭代**未见。

### 5.5 机制可解释性（工具箱性质，不主张创新）

- **Activation patching / Causal tracing**：作为 FDLP 选层的因果验收工具；
- **Activation scaling for prediction flip** (Findings of EMNLP 2024)：他们也用 "flip" 概念，但用于 steering 干预，不是 LoRA placement；必须在 related work 主动澄清差异。

---

## 6. 实验设计（建议直接做成论文表格）

### 6.1 评测基准与指标（与第一点一致）

- **域内主指标**：MedMCQA Surgery val acc（n=200，与第一点一致）；
- **通用域守护集**：MMLU 随机子集 500 题（每轮飞轮后必测，监控退化）；
- **系统指标**：Tokens/sec（纯 Target 推理 vs 第一点 DSSD vs 飞轮后的 Target）；
- **飞轮专属指标**：
  - 每轮 flip rate $\overline{F}^{(k)}$；
  - 跨轮热点 Jaccard 相似度；
  - $\text{KL}(\text{Target}_{v_k}\,\|\,\text{Draft})$；
  - 累计可训练参数量与累计训练 wall-clock。

### 6.2 Phase 0：单轮 FDLP 与事件必要性对照（无飞轮，作为 Round 0 的基础消融）

| ID | 设置 | 目的 |
|----|------|------|
| E0 | 统计 flip rate 随第一点超参（C9 的 α、τ）变化 | 选定飞轮使用的稳定 C9 配置 |
| E1 | **FDLP（flip-weighted）**：仅在 $F_t=1$ 上统计 $S_\ell$ | 主方法的层重要性图 |
| E2 | **All-token 对照**：所有步统计 $S_\ell$（不加 flip 权重） | 证明 flip 加权的必要性 |
| E3 | **Entropy 对照**：仅在高熵步（top-q%）统计 $S_\ell$ | 证明"挑不确定步"不足以替代 flip |
| E4 | **Disagreement 对照**：仅用 draft vs target 的 top-1 不一致事件 | 证明"任何分歧"不足以替代 flip |
| E5 | 少量 flip 样本上做 activation patching，得 $\Delta_\ell$ | 因果验证：FDLP 热层与 patching 峰值层一致性 |

### 6.3 Phase 1：飞轮主实验（核心）

| ID | 方法 | 描述 |
|----|------|------|
| F0 | 不做飞轮，纯 Target | acc=0.650，tps=27.3（第一点已测） |
| F1 | 第一点 DSSD（C9 最强配置）  | acc≈0.690，tps 大降（在线推理基线） |
| F2 | **DAF 飞轮（K=1）** | 单轮 FDLP + LoRA，验证 Round 0 的吸收幅度 |
| F3 | **DAF 飞轮（K=2）** | 二轮迭代，验证飞轮模式存在 |
| F4 | **DAF 飞轮（K=3 或自适应停止）** | 主推方法，自动收敛 |
| F5 | **DAF 飞轮 + 在飞轮后的 Target 上再开 DSSD** | 看"已吸收的 Target + DSSD 在线引导"是否还能再涨（叠加测试） |

**对每个 $F_k$ 报告**：acc、tps、可训练参数量、累计训练 wall-clock、通用域退化、$\text{KL}(\text{Target}_{v_k}\,\|\,\text{Draft})$、flip rate。

### 6.4 Phase 2：事件对照飞轮（防御性创新，关键！）

| ID | 飞轮事件类型 | 目的 |
|----|-------------|------|
| G1 | **Flip 飞轮**（=F4） | 主方法 |
| G2 | **Entropy 飞轮**：用高熵步代替 flip 事件驱动 FDLP + LoRA 选数据 | 证明只有 flip 飞轮能稳定收敛 |
| G3 | **Disagreement 飞轮**：用 draft–target 一般分歧驱动 | 同上 |
| G4 | **All-token 飞轮**：等价于普通迭代 SFT（无事件加权） | 飞轮模式的 naive baseline |

**通过标准**：G1 在收敛轮数、最终 acc、通用域保持率上均显著优于 G2/G3/G4，证明 "flip 事件作为飞轮驱动信号的不可替代性"。**这是论文最强的一组实验**，没有它整个飞轮叙事会被审稿人指为"包装迭代 SFT"。

### 6.5 Phase 3：固定预算下的强基线对照（必做）

| ID | 方法 | 说明 |
|----|------|------|
| L1 | 全层 LoRA（同 $R_{\text{total}}$） | 强对照 |
| L2 | **DAF 飞轮（=F4）** | 本文主方法 |
| L3 | 随机 K 层 LoRA（多种子） | 排除"稀疏就好" |
| L4 | 经验层（仅后 1/3 / 仅 attention） | 工程基线 |
| L8-IGU | IGU-LoRA 自适应 rank | 正面对照"梯度-based 重要性分配" |
| L8-GoRA | GoRA 自适应 rank | 第二个自适应基线 |
| L9-Flexora | Flexora HPO 选层 | 自动层选择对照 |
| L10-GIFT | GIFT 风格的 entropy-weighted SFT | 直接证明"flip 加权" vs "entropy 加权"差异 |

> 时间紧时至少补 L1 / L8-IGU / L10。

### 6.6 Phase 4（可选增强）：飞轮收敛律的经验拟合

- 定义"领域吸收率"：$\rho_k = 1 - \overline{F}^{(k)} / \overline{F}^{(0)}$；
- 在多个超参点／多个种子下拟合 $\rho_k$ vs $k$ 曲线（指数？幂律？）；
- 给出"达到 $\rho \geq 0.9$ 所需的 wall-clock 预算"作为新指标。

### 6.7 Phase 5（可选增强）：双向飞轮

- 同步用 flip 事件做 Draft 的 FDLP，每轮更新 Draft_v_k；
- 比较"单边飞轮（仅 Target）"vs"双向飞轮"在 acceptance rate 与 acc 上的差异；
- 工程价值：证明"用 flip + LoRA 即可低成本生产领域 Draft"，避免每个新领域都要做 FFT。

---

## 7. 预期结果与可写结论（分情景）

### 7.1 乐观情景（最理想结论）

- **飞轮存在性**：F4 显著优于 F2，证明"多轮迭代有意义"；
- **事件必要性**：G1 显著优于 G2/G3/G4，证明 flip 事件不可替代；
- **吸收等效性**：F4 acc 接近 F1（DSSD 在线引导），但推理 tps 恢复至 ~27（接近纯 Target）；
- **预算高效性**：L2 在固定预算下显著优于 L1/L3，与 L8 系列持平或更优；
- **通用域保持**：飞轮全程 MMLU 退化 < 1.0%。

**一句话结论（写进 abstract）**：
> Speculative decoding 内生的 flip 事件可被赋予双重角色——同时作为 PEFT placement 的稀疏监督锚点和飞轮收敛的可观测指标——形成"解码 → 选层 → 微调 → 再解码"的协同进化闭环；该闭环在 Surgery 子集上将在线引导的领域信号离线沉淀进 Target 权重，实现 acc 接近 DSSD 在线引导、tps 接近纯 Target 的双重收益。

### 7.2 中性情景（仍可成篇）

- 飞轮 1 轮内饱和（$K^\ast = 1$），acc 提升 ~3 个百分点；
- 收缩主张为："**FDLP 单轮 + flip-rate 收敛判据**作为一种 SD-aware PEFT 方法，比传统层选择方法在领域吸收效率上更优"；
- 飞轮叙事降级为"探索性发现：在该任务上 1 轮即收敛"。

### 7.3 悲观情景（提前预案）

- 通用域退化 > 1.0% 触发回退；
- 或 G2/G3 飞轮也能收敛，flip 事件不可替代性主张崩盘；
- 此时**老实把工作定位为"DSSD 信号作为 PEFT 监督的可行性研究 + 严格阴性发现"**，重点做好以下守势：
  - 把 G1 vs G2/G3/G4 的对比图画详细，作为"虽然没赢但有现象"的科学结论；
  - 强调 FDLP 单轮版本仍优于全层 / 随机层（L1/L3）；
  - 在 future work 讨论"什么样的领域 / 什么样的 Target–Draft 组合下飞轮可能成立"。

---

## 8. 写作检查清单（避免答辩被问倒）

- [ ] flip 定义是否与第一点 `TelemetryLogger` 字段逐字段一致？
- [ ] 每轮 flip rate / 通用域 acc / KL(Target‖Draft) 是否同时报告？
- [ ] 飞轮收敛判据是否单一且可机器检查（不是"看图说话"）？
- [ ] LoRA 总预算是否在 L1/L2/L3/L8 之间严格相等？
- [ ] 是否在飞轮训练后**关闭**第一点 DSSD 引导，单独评测 Target_vK？（避免飞轮收益与在线引导收益混淆）
- [ ] G2/G3/G4 飞轮对照是否完整？（如果缺则退守 FDLP 单轮版本）
- [ ] 是否声明"飞轮训练不是为了打榜，而是为了证明 DSSD 信号可被参数化吸收"？

---

## 9. 附录 A：撞车自检清单

> 检索时间窗：截至 2026 年 4 月。

**强警示（搜到一条即必须正面对比）：**
- `speculative decoding` + `LoRA` + `iterative` / `flywheel`
- `accept-reject` + `event-conditioned` + `fine-tuning` / `placement`
- `flip event` + `PEFT` / `adapter`
- `decode-loop` + `co-evolution` + `placement`

**中警示（必须 cite 并区分）：**
- `iterative SFT` / `self-distillation` + `speculative decoding`
- `online speculative decoding` + `target adaptation`
- `disagreement-driven` + `fine-tuning`

**低警示（工具箱性质）：**
- `activation patching` / `causal tracing`
- `gradient norm` + `layer importance`
- `entropy-weighted` SFT

---

## 10. 附录 B：与上一版第二点（FDLP 单方案）的关系

- 上一版方案 `毕业论文第二点_Flip驱动LoRA层位选择.md` **作为本版的子模块（2.3 节）保留**，不丢弃；
- 上一版若飞轮失败可作为"退守方案"——把贡献收缩为 FDLP 单轮 + 严格事件对照；
- **本版与上一版的核心区别**：
  - 上一版：FDLP 是核心方法，单轮使用；
  - **本版：FDLP 是飞轮的一个步骤；核心方法是飞轮闭环本身，加上 flip 事件的"双重角色"创新**；
- 写作时建议把上一版的实验矩阵（E1–E5、L1–L8、C1–C4）整体并入本版的 Phase 0 与 Phase 3，不要重复设计。

---

**文档版本：** v1（飞轮版第二点主叙事；FDLP 作为子模块保留；强对照实验 G2/G3/G4 是创新性主张能否成立的关键。）
