# 毕业论文第二点：Flip 驱动的 LoRA 层位选择（从领域引导解码反推 PEFT 放置策略）

**定位：** 与第一点「免训练、仅在输出/验收侧融合 Draft−Base 对比信号」形成递进：第二点从“免训练”进一步走向“精准微调”——在允许极小代价参数更新（LoRA）时，讨论 **如何利用推理期的动态信号（Token Flip）指导参数预算投放**，回答“同样的 LoRA 预算，挂在哪些层最省力/最有效”。关键变化是：我们不再依赖“开关验收导致 Target 激活变化”（事实上验收是系统外部决策，通常不回流改变当步前向），而是把你们框架内可观测的 **Token Flip（改口）事件**当作强语义监督信号，在 **Target 内部**用两类可复现实验量反推“最有杠杆的层”：
- **Flip-weighted 梯度/敏感度（用于 LoRA 放置）**：哪些层的参数（或可挂 LoRA 的模块）对提高 flip 目标 token 概率最敏感。
- **因果验证（用于机制解释与消融）**：用反事实干预验证“热层”不是相关性，而是在 flip 的因果链条/杠杆点上。

这避免了跨尺度模型（Draft/Base vs Target）的隐空间强行对齐，也避免了不严谨的“激活漂移口径”。

---

## 0. 审稿人视角：最可能“撞车”的点，以及如何自证创新

站在严格审稿人角度，本工作里**最容易被判定为“已有套路”的部分**是：

- 用梯度/敏感度（gradient norm、Fisher、loss increase 等）做层/模块重要性排序，再选 Top-K 层挂 LoRA；
- 用 activation patching / causal tracing 做机制验证；
- 在固定预算下做“全层 vs 稀疏层 vs 随机层 vs 经验层”的 LoRA 消融。

这些方法论在 PEFT/剪枝/重要性分配/机制可解释性文献中广泛存在，因此**不能把它们本身当作创新点**。

本第二点若要成立，必须把“创新锚点”收敛到你们系统独有的信号：

> **用 speculative decoding/accept–reject 体系内产生的 token flip 事件**（由第一点的 Draft–Base 对比引导触发）作为**强语义、稀疏、机制相关**的监督权重（flip-weighted / event-conditioned），反推出 PEFT（LoRA）的层位放置与容量分配。

并且必须用强对照证明：如果把 flip 换成“全 token 平均”“高熵步”“一般分歧步”，效果会显著变差；否则审稿人有理由认为你只是做了一个常规的“重要性分配”。

> **可证伪门槛（写作时必须承认）：** 若 flip-weighted 与 all-token/entropy/disagreement 的选层效果无显著差异，则本工作“创新性”应收缩为工程消融，而不应强行主张方法创新。

---

## 1. 论文故事：为什么重要、为什么难、创新是什么

### 1.1 为什么重要（Important）

1. **第一点已证明**免训练融合能涨点，但现实部署常仍会遇到边界：仅靠最后一层或验收侧修正，未必能覆盖所有错误类型；**PEFT（LoRA）**仍是工业界补齐领域能力的主流手段。  
2. **LoRA 的工程痛点**非常具体：同样总 rank/总可训练参数，**“加在哪些层”**往往靠经验或暴力网格，缺少与任务机制绑定的原则。  
3. 你们的解码框架里存在一个**强语义事件**：**token flip**——Target 在标准行为下会输出的 token，在领域引导/软验收后变为 Draft 提议的 token。该事件可被视为「领域信号真正改变了 Target 的局部决策边界」的可观测代理。

**故事一句话：**  
我们不是泛泛研究 LoRA，而是研究 **“当领域融合真实发生（flip）时，大模型内部哪一层最先/最强烈地响应”**，并据此回答 **PEFT 层位预算该投向哪里**。

### 1.2 为什么难（Hard）

1. **flip 是稀疏事件**：并非每一步都 flip，统计需要足够日志与稳定协议。  
2. **“激活变化大”≠“因果重要”**：可能是后续层被动放大；需要消融与对照排除伪相关。  
3. **与第一点耦合**：flip 定义依赖你们的验收/引导策略，必须在论文中 **固定策略版本** 与 **超参**，否则结论不可复现。  
4. **预算公平性**：比较「只加热层」与「全层 LoRA」时，必须 **固定总可训练参数量或总 rank 预算**，否则结论会被“参数更多所以更好”污染。

### 1.3 我们“到底哪里新”（创新边界 + 可证伪条件）

为了避免“换皮重要性分配”的嫌疑，本工作把创新点严格收敛为两层：

1. **问题定义（更像创新点的核心）：Flip-conditioned PEFT placement**  
   提出 **Flip-driven LoRA Placement（FDLP）**：把你们解码系统里由第一点（Draft–Base 对比引导）诱导的 **token flip 事件**当作“领域融合真实发生”的证据，用它来定义 *event-conditioned* 的层位选择问题：
   - 不是对所有 token/所有步平均；
   - 而是对“决策边界被领域信号改写”的稀疏事件加权（flip-weighted）。

2. **方法链路（强调严谨性而非发明工具）：flip-weighted 选层 + 因果验收**  
   - **Flip-weighted 梯度/敏感度打分**用于在固定预算下做稀疏层放置/容量分配（梯度重要性本身不新，但 flip 条件化是关键）；
   - **activation patching/causal tracing**用作“因果验收”，证明热层位于 flip 的因果链条上（patching 本身不新，但它防止我们只报告一个相关性分数）。

3. **系统叙事（与你们第一点绑定的递进）：从 SD 事件反推训练预算**  
   第一点在推理时产生结构化事件（flip）与日志；第二点把该事件作为监督锚点，回答工程上长期依赖经验的“LoRA 应该加在哪里”。这条“从 decode-loop 反推 PEFT 放置”的闭环，是本工作的系统差异点。

**可证伪条件（写进局限/讨论更可信）：**
- 若 FDLP 与下述替代事件/权重（all-token 平均、高熵步、一般分歧步）选层效果相近，则创新主张应降级为“工程消融与经验总结”。

> **与“随便做可解释性”区别：** 我们不靠挑神经元讲故事，而是把创新锚定在“flip 事件”这一可复现系统日志上，并用严格对照把它与常规重要性分配区分开。

---

## 2. 详细思路（方法草案）

### 2.1 定义：什么是 Token Flip（“改口”事件；需写进论文的精确定义）

在固定解码协议下，对第 $t$ 步：

- 记基座模型（Target）在不受领域引导影响下“原本倾向输出”的 token 为
  $$A \triangleq x^{\mathrm{target}}_t = \arg\max_x\, P_{\mathrm{target}}(x\mid p_t)$$
  其中 $p_t$ 是第 $t$ 步的 prefix。
- 记在投机解码 + 领域引导/软验收后，实际进入序列的 token 为
  $$B \triangleq x^{\mathrm{final}}_t$$

若 $B \neq A$，并且 $B$ 满足你们框架的“被领域侧改写/来自 Draft 提议（或等价判据）”，则记 **Token Flip（改口）事件** $F_t=1$。

> 实现上应与现有日志字段一致（accept/reject、chosen_token、draft_token 等），避免口径漂移。

### 2.2 层打分：用 flip 作为监督信号反推「最有杠杆的层」

#### 2.2.0 先把口径说清：为什么不能用“开关验收的激活漂移”选层

在你们第一点的主设定里（如 C1/C6），领域引导主要发生在 **验收概率/accept 规则**侧：
- Target 在给定 prefix 下的当步前向 $P_{\mathrm{target}}(\cdot\mid\text{prefix})$ 通常并不会因为“我们是否接受 Draft token”而改变；
- **accept 是系统外部决策**，往往不回流改变 Target 当步 hidden states。

因此，“同一步开/关引导得到 $h$ vs $\tilde h$”在理论上很容易变成不严谨或信号极弱。

本第二点改用一个更可解释、也更贴 LoRA 的反推问题：

> 已知在某个 prefix 下，我们希望 Target 把下一 token 推向 $B$（改口后的目标 token），那么 **改动 Target 哪些层最省、最有效**？

这可以通过 **flip-weighted 梯度/敏感度**得到可操作的层排序；再用 **因果验证（patching / counterfactual / echo）**做机制验证。

---

#### 2.2.1 方法一：Flip-weighted 梯度/敏感度选层（主方法，直接服务 LoRA 放置）

对每个 flip 事件的第 $t$ 步，固定 prefix 为 $p_t$。

**符号说明（本节统一口径，避免读者猜符号）：**
- $t$：时间步/解码步索引（第 $t$ 次生成下一 token）。
- $p_t$：第 $t$ 步的 prefix（上下文），即已生成/已给定的 token 序列；可写作 $p_t = x_{<t}$。
- $P_{\mathrm{target}}(\cdot\mid p_t)$：基座模型（Target）在给定 prefix $p_t$ 时的下一 token 条件分布。
- $P_{\theta}(\cdot\mid p_t)$：同一 Target 模型的条件分布，但显式写出参数 $\theta$（用于强调“对哪些参数求梯度/做 LoRA”）。通常 $P_{\theta}=P_{\mathrm{target}}$。
- $\theta$：Target 的参数集合（冻结或 LoRA 可更新的那部分都属于 $\theta$）。
- $A$：Target 在无领域引导下的“原本倾向输出”的 token，定义为 $A=\arg\max_x P_{\mathrm{target}}(x\mid p_t)$。
- $B$：在投机解码 + 领域引导/软验收后，实际进入序列的 token（改口后的 token）。
- $F_t\in\{0,1\}$：是否发生 Token Flip（改口）事件的指示变量；本节的期望/统计都在 $F_t=1$ 的子集上进行。
- $\mathcal{L}(p_t,B)$：把概率“推向 $B$”的交叉熵式目标，定义为 $-\log P_{\theta}(B\mid p_t)$。
- $\ell$：Transformer 的层索引（第 $\ell$ 层）。
- $m$：层内模块索引（例如 attention 的某个投影或 MLP 的某个投影）；$\mathcal{M}$ 是候选模块集合。
- $W_{\ell,m}$：第 $\ell$ 层第 $m$ 个候选模块的权重矩阵（即你打算挂 LoRA 的“宿主权重”）。
- $\nabla_{W_{\ell,m}}\,\mathcal{L}$：损失对该模块权重的梯度。
- $\lVert\cdot\rVert_F$：Frobenius 范数；$\lVert\cdot\rVert_0$：非零元素个数（此处可近似为参数量 numel）。
- $\epsilon$：数值稳定项，防止除零。

- 基线下一 token（仅用于定义 flip）：
  $$A = x^{\mathrm{target}}_t = \arg\max_x\; P_{\mathrm{target}}(x\mid p_t)$$
- 改口后的目标 token：
  $$B = x^{\mathrm{final}}_t$$
  实现上要求 $B$ 满足“被领域侧改写/来自 Draft（或等价判据）”。（若是带标注任务，也可在额外实验里把 $B$ 替换为“答案相关 token”，但主线口径保持为改口 token。）

定义一个与“把概率推向 $B$”一致的目标（无需真的训练，只做打分）：
$$
\mathcal{L}(p_t, B) = -\log P_{\theta}(B \mid p_t)
$$

**层/模块打分思想：** 哪些层的参数（或可挂 LoRA 的模块）对降低该损失最敏感，说明“在这些位置做小幅 PEFT 更新最有杠杆”。

一种可直接落地、且能做到预算公平的打分方式是“按模块归一化的梯度范数”：

- 设第 $\ell$ 层某个候选模块权重为 $W_{\ell,m}$（例如 attention 的 `q_proj/v_proj/o_proj`，或 MLP 的 `up_proj/down_proj`），则模块分数：
  $$
  s_{\ell,m} = \mathbb{E}_{t:F_t=1}\left[ \frac{\lVert \nabla_{W_{\ell,m}}\,\mathcal{L}(p_t, B) \rVert_F}{\sqrt{\lVert W_{\ell,m}\rVert_0}+\epsilon} \right]
  $$
  其中分母用参数量（或 $\sqrt{\text{numel}}$）做归一化，避免“大矩阵天然梯度更大”导致的偏置。

- 汇总成层分数（两种常见汇总口径，论文里固定其一即可）：
  $$
  S_\ell = \sum_{m\in\mathcal{M}} s_{\ell,m} \quad \text{或} \quad S_\ell = \max_{m\in\mathcal{M}} s_{\ell,m}
  $$

得到层排序 $\ell_1,\ell_2,\dots$，取 **Top-K** 作为 LoRA 候选层集合 $\mathcal{L}_K$。

> 解释性：$S_\ell$ 的语义非常直接——“为了把 flip 目标 token 的概率推上去，哪一层最需要被更新”。这比“看某层激活变化大”更贴近 LoRA 的学习机制。

可选增强（如果你想把它写得更像“预算分配策略”而不只是选层）：在固定总 rank 预算 $R_{\text{total}}$ 下做 rank 分配：
$$
 r_\ell \propto \mathrm{clip}(S_\ell),\quad \sum_{\ell\in\mathcal{L}} r_\ell = R_{\text{total}}
$$
形成“层位选择 + 层内容量分配”的完整闭环。

---

#### 2.2.2 因果验证路径一：Activation patching / causal tracing（把 Bad 变成 Good，验证“热层”在因果链条上）

梯度敏感度回答“LoRA 挂哪层最省力”，但论文需要进一步回答：这些“热层”是否只是相关性？这里引入反事实干预作为因果验证。

对同一 prefix $p_t$ 构造两条反事实轨迹：
- **Bad 轨迹**：让 Target 按自身行为继续（flip 步倾向输出 $A$）。
- **Good 轨迹**：在第 $t$ 步用 **teacher forcing** 强制下一 token 设为 $B$，得到与“改口后 token”一致的后续内部状态。

然后在某一层 $\ell$ 的 hook 点（推荐 residual stream / block 输出），把 Good 的表示 patch 到 Bad，再观察 $P(B)$ 的提升：
$$
\Delta_\ell = \mathbb{E}_{t:F_t=1}\Big[\log P(B\mid \text{patch at layer }\ell) - \log P(B\mid \text{no patch})\Big]
$$

若 $\Delta_\ell$ 在少数层出现明显峰值，说明这些层更接近“flip 的因果瓶颈/杠杆点”。

> 计算量控制：patching 只需在少量 flip 样本（例如 50–200 个）上做即可，用来支撑机制与消融，不必全量。

#### 2.2.3 因果验证路径二：Gradient-based Counterfactual（梯度下降伪造法；直接验证梯度分数的有效性）

该路径更“贴” 2.2.1 的梯度打分：既然我们用 $\nabla\mathcal{L}$ 评估层的杠杆性，就用它构造一个最小反事实干预，检验哪些层的干预最能把 $P(B)$ 推上去。

- **Bad 轨迹**：在 prefix $p_t$ 下做一次正常前向，缓存每层（或选定 hook 点）的隐藏状态 $h_{\text{bad},t}^{(\ell)}$。
- **Good 轨迹伪造**：对目标损失
  $$\mathcal{L}(p_t,B)=-\log P(B\mid p_t)$$
  计算隐藏态梯度 $\frac{\partial \mathcal{L}}{\partial h_t^{(\ell)}}$，并对该层隐藏态做微小扰动：
  $$h_{\text{good},t}^{(\ell)} = h_{\text{bad},t}^{(\ell)} - \eta\, \frac{\partial \mathcal{L}}{\partial h_t^{(\ell)}}$$
- **干预操作**：将 $h_{\text{good},t}^{(\ell)}$（或其残差增量）塞回第 $\ell$ 层对应的 hook 点，继续完成后续前向，观察 $P(B)$ 的提升幅度：
  $$\Gamma_\ell = \mathbb{E}_{t:F_t=1}\big[\log P(B\mid \text{inject at }\ell) - \log P(B\mid \text{no inject})\big]$$

结论口径：若某些层的 $\Gamma_\ell$ 显著高于其它层，说明“沿着该层的梯度方向做最小干预”就能更有效触发改口，因此这些层更可能是因果杠杆层；也为 2.2.1 的梯度热层提供直接验证。

#### 2.2.4 因果验证路径三：Teacher Forcing 跨时刻验证（Echo Analysis；观察“语义冲击”在 $t{+}1$ 被哪一层最先消化）

该路径利用 Transformer 的序列依赖：Flip 在第 $t$ 步发生后，其“语义冲击”会在第 $t{+}1$ 步的计算中体现为隐藏态差异。

- **对照组（未纠错）**：输入序列为 $[p_t, A]$，在 $t{+}1$ 步记录各层激活 $h_{t+1,\text{ctrl}}^{(\ell)}$。
- **实验组（已纠错）**：输入序列为 $[p_t, B]$（teacher forcing），在 $t{+}1$ 步记录各层激活 $h_{t+1,\text{exp}}^{(\ell)}$。
- **关键观测量**：每层的“回声强度”（示例定义）
  $$E_\ell = \mathbb{E}_{t:F_t=1}\big[\lVert h_{t+1,\text{exp}}^{(\ell)} - h_{t+1,\text{ctrl}}^{(\ell)}\rVert_2\big]$$

结论口径：在 $t{+}1$ 步反应最剧烈（$E_\ell$ 最大/出现峰值）的层，往往是最先感知并消化“领域修正信号”的层；它与 2.2.1 的梯度热层、2.2.2/2.2.3 的干预峰值层之间的一致性，可作为“机制闭环”的补强证据。

### 2.3 LoRA 放置（固定预算）

- **总预算约束：** 例如总 rank 之和固定：$\sum_{\ell \in \mathcal{L}} r_\ell = R_{\text{total}}$；或总可训练参数固定。  
- **FDLP：** 仅在 $\mathcal{L}_K$ 的注意力/MLP 投影上挂载 LoRA（具体模块与 LLaMA-Factory 配置对齐）。  
- **对照：** 全层均分同一 $R_{\text{total}}$；随机抽 $K$ 层；经验层（如仅中高层）等。

### 2.4 训练数据与目标（建议写清边界）

- **领域数据：** 与第一点同一 MedMCQA-Surgery 管线（或你们最终采纳的领域），格式与评测对齐。  
- **训练目标：** 标准 SFT/CE 即可（第二点贡献在 **层位选择**，不必同时发明新损失）。  
- **是否与引导联训：** 初版建议 **先离线选层，再 LoRA 微调**（两阶段），降低变量；若时间允许可做扩展：微调后仍用原引导解码评测。

---

## 3. 可行性分析

### 3.1 工程可行性：**中高**

- **优势：** 不需要 Draft hidden 注入 Target；也不需要在“同一步开关验收”上做不严谨的激活差分。主要工程增量是：
  1) 记录可复现的 flip 日志（$F_t$、$A$、$B$、prefix 对齐信息）；
  2) 在 Target 上做 **只读的 backward** 来统计梯度敏感度分数（2.2.1）；
  3) 生成对应的 LoRA 训练配置（仅 Top-K 层/模块）。
- **难点：**
  - 梯度统计需要能跑一次反向传播（但不更新权重），算力开销可控，且可以只对答案相关 token/少量 flip 样本统计；
  - activation patching（2.2.2）需要额外 forward，但只在小样本上做机制验证即可，不是主训练路径。

### 3.2 科学可行性：**中**

- **乐观：** Top-K 层 LoRA 在全层预算下达到 **相近 acc**，同时 **训练更快/显存更省**。  
- **中性：** acc 略低于全层，但 **显著优于随机层**，仍可主张「层位信息有价值」。  
- **风险：** Top-K 与随机差异不显著 → 需检查 flip 定义、样本量、归一化方式；或说明 **该任务下 LoRA 层位不敏感**（这也是有效科学结论，但需与导师对齐预期）。

### 3.3 与第一点关系：**不冲突**

- 第一点：**零训练** 的解码期融合。  
- 第二点：**小训练** 的层位归纳；评测仍可报告 **引导开/关** 两套曲线，展示互补或叠加。

---

## 4. 相关工作与差异点（写作时的对齐方式）

> 这里不手写不确定的 bib 条目，只给“你需要引用的方向 + 审稿人可能指出的撞车点 + 我们如何区分”。具体引用请用 Zotero/Google Scholar 补全。

### 4.1 PEFT / LoRA 的基础与默认放置实践（易被当作经验工程）
- **相关方向：** LoRA/QLoRA/PEFT 的基本方法与常见经验做法（只改 attention/只改 MLP、偏后层更有效、rank 网格等）。
- **审稿人会说：** “你只是换了个数据驱动的选层方式，本质仍是 placement ablation”。
- **我们如何区分：** 我们把 placement 的监督锚点明确绑定到 **speculative decoding 的 flip 事件**（event-conditioned），而不是对全量 token 平均或经验层位。

关键词：`LoRA`, `QLoRA`, `PEFT`, `adapter placement`, `module selection`.

### 4.2 自适应 rank/重要性分配（强相关、必须正面对照）
- **相关方向：** 自适应 rank 分配/剪枝/重要性分配（常见思路包括基于梯度或近似二阶信息的分配）。
- **审稿人会说：** “这类方法已经能在固定预算下做更优分配，你的 FDLP 是否只是弱化版？”
- **我们如何区分：**
  1) 我们的“重要性”不是从全训练过程端到端学出的，而是来自第一点系统产生的 **flip 事件（决策边界被改写的稀疏时刻）**；
  2) 我们强调 *flip-conditioned*：只对 $F_t=1$ 事件统计/加权；
  3) 我们承诺在实验中加入至少一个可复现的“自适应 rank/全层分配”基线（同预算），否则创新主张站不住。

关键词：`adaptive rank allocation`, `importance-based PEFT`, `AdaLoRA`（检索入口之一）.

### 4.3 层重要性/敏感度分析（梯度、Fisher、loss increase；本身不新）
- **相关方向：** 梯度范数、Fisher、删层/扰动导致的 loss increase 等。
- **审稿人会说：** “梯度选层不是新东西。”
- **我们如何区分：** 我们不把梯度打分本身当创新；创新锚点是 **用 flip 事件定义‘该对哪些 token/步计算梯度’**，并用 all-token/entropy/disagreement 对照证明 flip 的必要性。

关键词：`layer importance`, `gradient norm`, `Fisher information`, `loss increase`, `module sensitivity`.

### 4.4 机制可解释性：activation patching / causal tracing（工具箱，不作为创新点）
- **相关方向：** patching/tracing。
- **审稿人会说：** “patching 只是验证工具，不是方法贡献。”
- **我们如何使用：** 将其作为 **对 FDLP 选层的因果验收**与消融支撑（错配层、非 flip 统计），不把它当作创新点。

关键词：`activation patching`, `causal tracing`, `mechanistic interpretability`.

### 4.5 投机解码与验收（你们第一点的承接；差异点在‘事件’）
- **相关方向：** speculative decoding、draft verification、accept–reject。
- **我们如何区分：** 第二点不改验收公式，而是把 accept–reject 体系产生的 flip 事件当作“决策边界被领域信号改写”的可观测证据，用它反推 PEFT 放置——这是与常规 PEFT placement 工作的系统差异点。

关键词：`speculative decoding`, `draft verification`, `accept-reject`, `flip event`.

---

## 5. 需要做的实验（建议直接做成论文表格）

### 5.1 Phase 0：现象统计与“防撞车”信号验证（无训练更新）

| 编号 | 内容 | 输出 |
|------|------|------|
| E0 | 统计 flip 率随第一点超参（如 $\alpha/\lambda,\ \tau$）与策略（C1/C6 等）变化 | 曲线：flip rate vs 超参 |
| E1 | **FDLP（flip-weighted）**：只在 $F_t=1$ 上统计梯度敏感度得到 $S_\ell$（见 2.2.1） | 热层排序图（layer/module importance） |
| E2 | **All-token 对照**：在所有步/所有 token 上统计同样的 $S_\ell$（不加 flip 权重） | 与 E1 的排序差异 + 后续放置效果对照 |
| E3 | **Entropy 对照**：只在高熵步（$H_t$ 处于 top-q%）统计 $S_\ell$ | 证明“只是挑不确定步”不足以替代 flip |
| E4 | **Disagreement 对照**：仅用 draft vs target 的 top-1 不一致作为事件（不要求最终 flip）统计 $S_\ell$ | 证明“泛化分歧事件”不足以替代 flip |
| E5（可选） | 少量样本做 activation patching，得到 $\Delta_\ell$ 峰值层（见 2.2.2） | 机制验证图：$\Delta_\ell$ vs layer |

> 注：E2–E4 的目的不是“再造新方法”，而是给审稿人一个明确结论：**flip 作为监督锚点是否必要**。若 E2/E3/E4 与 E1 无差异，应主动收缩创新表述。

### 5.2 Phase 1：LoRA 对照（固定总预算；必须包含强基线）

| 编号 | 方法 | 说明 |
|------|------|------|
| L0 | 不训练 + 第一点最强解码 | 上限参照之一（推理侧融合） |
| L1 | 全层 LoRA（均分 rank） | 强对照（最常见的“堆参数到处学”做法） |
| L2 | **FDLP（flip-weighted）Top-K 层 LoRA** | 本文主方法 |
| L3 | 随机 K 层 LoRA（多种随机种子） | 排除「稀疏就好」 |
| L4 | 经验层（如仅后 1/3 层 / 仅中层 / 仅 attention 或仅 MLP） | 工程基线 |
| L5 | **All-token placement**：用 E2 的 all-token 统计选出来的 Top-K 层做 LoRA | 排除“梯度选层本身就够了” |
| L6 | **Entropy placement**：用 E3 的高熵步统计选层做 LoRA | 排除“只挑不确定步就行” |
| L7 | **Disagreement placement**：用 E4 的一般分歧事件统计选层做 LoRA | 排除“任意分歧事件都行” |
| L8（建议） | **自适应 rank/重要性分配基线**（同总预算） | 正面对照相关工作（例如某种自适应 rank/重要性分配实现） |

> 若只做“最小可答辩版本”，至少保证 L1/L2/L3 三组齐全，并在 Phase 2 做 C1（错配层）验证“热层”不是装饰性排序。

**必须报告：** `Accuracy`、`Tokens/sec`（推理）、**可训练参数量**、训练 wall-clock、（可选）域外轻量集上的 **退化 Δ**。

> 只要 L5/L6/L7 与 L2 表现相近，审稿人就会质疑“flip 的必要性”；只要 L8 明显更强，审稿人就会质疑“你只是弱化版自适应分配”。这两类风险必须用数据正面回答。

### 5.3 Phase 2：因果式与机制式消融（支撑「不是伪相关」）

| 编号 | 内容 |
|------|------|
| C1 | **错配层**：用相同的层分数排序，但把 LoRA 故意挂到错误层（或打乱层索引） |
| C2 | **非 flip 统计**：用非 flip 步统计得到的 Top-K 层来做 LoRA（对照“flip-weighted”的必要性） |
| C3 | K 敏感性：$K \in \{1,2,4,8,\dots\}$ 的帕累托前沿 |
| C4（可选） | **机制一致性**：比较 $S_\ell$（梯度热层）与 $\Delta_\ell$（patching 峰值层）的一致性（相关系数/峰值重合率） |

> 备注：若资源允许，可把 2.2.3 的 $\Gamma_\ell$ 与 2.2.4 的 $E_\ell$ 也纳入“机制一致性”对照，形成梯度打分 → 干预提升 → 跨时刻回声的三角闭环。

**通过标准（写作口径）：** L2 显著优于 L3；C1/C2 显著弱于 L2；若加入 C4，则给出“热层具有因果合理性”的额外证据。

---

## 6. 预期结果与可写结论（分情景）

### 6.1 乐观情景（最理想的论文结论）

- **必要性**：FDLP（L2）显著优于 L5/L6/L7（all-token/entropy/disagreement placement），证明“flip 事件”不是可替代的随便权重，而是关键监督锚点。  
- **有效性**：在 **固定 LoRA 总预算** 下，L2 的域内 acc 接近全层 LoRA（L1），同时训练时间/显存更优；随机层（L3）明显更差。  
- **严谨性**：错配层/非 flip 统计（C1/C2）显著弱于 L2，且（可选）patching 一致性（C4）支持热层的因果合理性。  
- **一句话结论（更审稿友好）：** speculative decoding 产生的 flip 事件，可作为 PEFT placement 的 *event-conditioned supervision*，从而在固定预算下更有效地决定 LoRA 应加在哪里。

### 6.2 中性情景（仍可成篇）

- L2 略低于 L1，但 **稳定优于 L3** 与 **部分经验层 L4**。  
- **结论改写为：** 在给定预算下，FDLP 提供一种 **数据驱动、与领域触发机制一致** 的层位初始化策略；全层仍略强，但 FDLP **性价比更优**。

### 6.3 悲观情景（提前预案）

- L2 与 L3 接近 → 优先排查：flip 定义是否过宽/过窄、flip 样本量是否不足、梯度分数的归一化口径（按参数量/按模块类型）、统计是否只覆盖了“答案相关 token”、以及训练随机性（种子/学习率/epoch）。  
- 若仍无差异：**不要硬吹层位神话**，可收缩贡献为「在该设定下 LoRA 层位不敏感」，并讨论 **任务形态（短答案/模板化输出）与 flip 稀疏度** 的限制——仍是有效结论，但需在开题/中期与导师对齐。

---

## 7. 总结论段（可直接改写进论文「第二点总结」）

本第二点从第一点框架中抽象出一个可观测、可复现的系统事件——**token flip**，并将其视为“领域信号真实改写了 Target 局部决策边界”的证据。由于验收通常是系统外部决策，不能用不严谨的“开关验收激活漂移”来选层，我们提出 **Flip-driven LoRA Placement**：在 **不依赖跨尺度隐藏态对齐** 的前提下，用 **flip-weighted 梯度/敏感度**在 Target 内部反推“最有杠杆的层/模块”，并在 **固定 PEFT 预算**下做稀疏层放置与（可选）rank 分配；同时用 **activation patching/causal tracing**对热层做机制验证与消融。

与常规“梯度选层/自适应分配”工作不同，我们把监督锚点绑定到 speculative decoding 的 **flip 事件（event-conditioned）**，并用 all-token/entropy/disagreement 等强对照检验“flip 是否必要”。只有当这些对照明显弱于 FDLP 时，本工作才主张其方法差异；否则我们将把贡献收缩为严格的工程消融与经验总结。

---

## 8. 附录：写作检查清单（避免答辩被问倒）

- [ ] flip 定义是否与代码日志 **逐字段一致**  
- [ ] LoRA 总预算是否与对照组 **严格相等**  
- [ ] Target 反事实前向定义是否 **单一且可实现**  
- [ ] 是否同时报告 **域内 acc** 与 **域外/通用退化**  
- [ ] 是否说明第二点 **允许 PEFT**，与第一点 **免训练** 的边界不混淆  

---

## 9. 附录：文献检索与“撞车”自检清单（建议）

> 目的：快速验证是否已有工作把“解码分歧/accept–reject 事件（flip 类事件）”直接用于 PEFT placement/rank 分配；若存在高度相似工作，应主动调整叙事或把贡献收缩到更窄的可证伪点。

建议检索组合（中英文均可）：
- `LoRA layer selection` / `adapter placement` / `module selection`
- `gradient norm` + `adapter` / `LoRA` + `placement`
- `adaptive rank allocation` + `LoRA`（例如以 `AdaLoRA` 作为入口扩展）
- `importance-based` + `PEFT` / `adapter` / `low-rank`
- `speculative decoding` + `finetuning` / `PEFT`
- `accept-reject` + `adapter` / `LoRA`
- `disagreement-driven` + `finetuning` / `adapter` / `PEFT`

若命中以下描述之一，需高度警惕“强撞车”：
- 明确把“解码时的分歧/flip/accept-reject 事件”用作 **placement 监督信号**；
- 在固定预算下提出与我们非常接近的“事件条件化（event-conditioned）重要性分配”。

---

**文档版本：** v3（加入审稿人视角的创新边界与可证伪门槛；补齐 all-token/entropy/disagreement 事件对照与 placement 基线，并要求正面对照自适应 rank/重要性分配相关工作。）
