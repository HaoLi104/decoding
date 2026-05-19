# Nano Banana 画图 Prompt 清单（开题报告 PPT 与文档共用）

> 共 10 张图。每张图均为学术风格的扁平化矢量插图（flat vector academic style），白色或浅灰背景，配色克制（建议 navy blue / teal / orange / dark gray 四色为主），便于直接嵌入 PPT 与 Word 文档。
>
> 使用方法：把每段 prompt 单独喂给 Gemini Nano Banana → 生成图片 → 命名为 `fig_01.png ~ fig_10.png` 放在工作目录的 `figures/` 文件夹下 → 重新运行生成脚本会自动嵌入。

---

## fig_01：总体技术路线架构图（封面 / 总览图）

```
A clean academic flowchart titled "Domain-Steered Speculative Decoding + Domain Absorption Flywheel".
Two horizontally arranged stacked panels:

(LEFT panel — "Innovation 1: DSSD, inference-time"):
Three boxes labeled "Target 32B (verifier)", "Draft 3B (proposer, fine-tuned)", "Base 3B (commonsense control)".
Arrows from Draft and Base into a fusion module labeled "ΔP = P_draft − P_base".
The fusion module feeds into "Soft-Guidance Acceptor" which outputs "accepted tokens".

(RIGHT panel — "Innovation 2: DAF, training-time"):
A circular flywheel with four nodes connected by clockwise arrows:
"1. Decode (collect flip events)" → "2. FDLP (place LoRA)" → "3. Train (LoRA SFT)" → "4. Re-decode (Round k+1)" → back to 1.
The center of the circle says "K iterations until flip rate converges".

A dashed arrow connects the LEFT panel's "accepted tokens" to the RIGHT panel's flywheel "1. Decode" node, labeled "flip events feed the flywheel".

Style: flat 2D vector, navy blue + teal + orange palette, white background, English labels, 16:9 ratio.
```

---

## fig_02：DSSD 三模型协同架构图（H200 单卡内聚）

```
A hardware-software architecture diagram titled "Single-GPU Tri-Model Co-Resident Architecture (H200 141GB)".
A large rounded rectangle representing one NVIDIA H200 GPU.
Inside it, three model boxes vertically stacked:
- "Target Qwen2.5-32B-Instruct (~64 GB, bfloat16)"
- "Draft Qwen2.5-3B-Instruct-Surgery (~6 GB, bfloat16)"
- "Base Qwen2.5-3B-Instruct (~6 GB, bfloat16)"

A horizontal "StaticCache (KV Buffer, pre-allocated)" bar spans below the three boxes, with arrows showing prefix-cache sharing among them.

On the right side, an info pill: "device_map=cuda:0 / no cross-card / Shadow Sync mode".

Style: clean technical diagram, dark gray + green + orange palette, white background, English labels.
```

---

## fig_03：C1–C9 策略演进树状图

```
A tree diagram titled "Evolution of Soft-Guidance Strategies (C1 → C9)".
Root node: "P_steered = P_target + α · (P_expert − P_base)".

Branches:
- Root → "C3: Probability-domain subsidy (P_t + αΔP) / P_d"
- Root → "C1: Ratio-domain addition P_t/P_d + αΔP" (highlighted as core)
- C1 → "C4: + Step-level Draft confidence gate S_t" (sparse activation)
- C1 → "C5: + Target entropy weight H_t/H_max" (uncertainty routing)
- {C4, C5} → "C6: Dual-signal AND gate (S_t × H_t)" (highlighted)
- C6 → "C8: Token-level gate ΔP(x) > τ" (highlighted as best)
- C8 → "C9: Binary gate + linear ΔP" (decoupled)

Each node labeled with its best Surgery acc:
- C3: 0.660 (+1.5pt)
- C1: 0.690 (+4.0pt)
- C4: 0.670 (+2.0pt)
- C5: 0.685 (+3.5pt)
- C6: 0.690 (+4.0pt)
- C8: 0.700 (+5.0pt)  ← star icon
- C9: 0.690 (+4.0pt, tps highest)

Style: hierarchical tree, navy + teal + orange, white background, English labels, 16:9 ratio.
```

---

## fig_04：C6 双信号联合门控架构图

```
A signal-flow architecture diagram titled "C6 Dual-Signal Gated Soft-Guidance".

Left side: two parallel signal extraction paths:
(Top) "Draft Confidence Signal": boxes "max P_draft" and "max P_base" feed into a subtractor producing "S_t = max(P_d) − max(P_b)" → through a threshold gate "𝟙(S_t > τ) · S_t" → output "G_draft".
(Bottom) "Target Entropy Signal": "P_target distribution" feeds into "H_t = −Σ P_t log P_t" → normalize → "H_t / H_max" → output "G_target".

Center: a multiplier (×) combining G_draft and G_target into "α_t = λ · G_draft · G_target".

Right: an "Acceptance formula" box showing "P'_accept = min(1, P_t/P_d + α_t · ΔP)".

Underneath: caption "Sparse activation only when Draft is confident AND Target is uncertain".

Style: signal-flow diagram, navy + orange palette, white background, English labels, 16:9 ratio.
```

---

## fig_05：DAF 飞轮闭环示意图

```
A circular flywheel diagram titled "Domain Absorption Flywheel (DAF)".
Four large nodes arranged on a circle with clockwise arrows:

1. (Top) "Decode with C9: collect flip events {F_t = 1, prefix p_t, draft token B_t}"
2. (Right) "FDLP: gradient sensitivity per layer S_ℓ; pick Top-K layers"
3. (Bottom) "Train: LoRA SFT on flip-conditioned data, merge into Target_v_{k+1}"
4. (Left) "Re-decode: measure new flip rate F̄^(k+1)"

Center: a circular logo with "Target_v_K, K ≤ 3, ε=2%".

Around the circle, three guard rails (small icons):
- "Guard 1: KL(Target ‖ Draft) bounded"
- "Guard 2: MMLU degradation ≤ 1%"
- "Guard 3: Hot-layer Jaccard > 0.3"

Style: circular flowchart, navy + teal + orange, white background, English labels, 16:9 ratio.
```

---

## fig_06：FDLP 选层算法流程图

```
A vertical algorithm flowchart titled "FDLP: Flip-driven LoRA Placement".

Step 1: "Collect flip set E = {(p_t, A_t, B_t) : F_t = 1}" (filled box)
Step 2: "For each flip, compute loss L = −log P_θ(B_t | p_t)" (rectangle)
Step 3: "Backward pass: compute ∇_{W_{ℓ,m}} L for each layer ℓ and module m" (rectangle)
Step 4: "Score s_{ℓ,m} = E[ ‖∇W_{ℓ,m}‖_F / √numel ]" (rectangle)
Step 5: "Aggregate to layer score S_ℓ = Σ_m s_{ℓ,m}" (rectangle)
Step 6: "Sort and pick Top-K layers L_K" (highlighted)
Step 7: "Allocate rank r_ℓ ∝ S_ℓ subject to Σ r_ℓ = R_total" (rectangle)
Step 8 (output): "LoRA placement plan: lora_target = L_K with ranks {r_ℓ}" (filled green)

Right side: a small heatmap thumbnail labeled "Example: layers 18, 24, 27, 31 are hot for Surgery".

Style: top-down algorithm flow, navy + green palette, white background, English labels, 16:9 ratio.
```

---

## fig_07：flip 事件双重角色图

```
A conceptual diagram titled "Dual Role of Flip Events".

Center: a single circular icon labeled "Flip Event F_t = 1 (Target's argmax overridden by Draft proposal)".

Two large arrows emanate from the center:

Left arrow (blue) → "Role A: Sparse Supervision Anchor for FDLP placement"
  - Sub-text: "Only F_t=1 steps contribute to gradient sensitivity score S_ℓ"

Right arrow (orange) → "Role B: Self-consistent Stopping Criterion for DAF flywheel"
  - Sub-text: "Flywheel stops when |F̄^(k) − F̄^(k-1)| < ε = 2%"

At the bottom: a horizontal arrow showing "Same event drives both training placement AND convergence detection — no acc-based early stopping needed".

Style: minimalist conceptual graphic, navy + orange duo-tone, white background, English labels, 16:9 ratio.
```

---

## fig_08：Pareto 前沿热图（acc vs tps 散点图）

```
A scatter plot titled "Pareto Frontier: Accuracy vs Throughput on MedMCQA Surgery (n=200)".

X-axis: "Tokens per second (tps)" range 0–30.
Y-axis: "Accuracy" range 0.55–0.72.

Plot the following labeled points:
- pure_target: (27.3, 0.650), gray cross, "baseline"
- standard_sd: (5.3, 0.650), gray dot, "standard SD, no domain"
- C1 α=0.10: (6.9, 0.690), blue dot
- C1 α=1.50: (17.7, 0.660), blue dot
- C5 λ=5: (8.2, 0.685), teal dot
- C6 λ=50: (11.5, 0.690), orange star
- C8 λ=20: (12.2, 0.700), red star (highlighted as best)
- C9 λ=100: (15.7, 0.690), orange dot
- C12 λ=2: (14.6, 0.685), purple dot
- C12 λ=20: (16.6, 0.665), purple dot
- hard_override: (17.8, 0.650), gray triangle

Draw a dashed line connecting the Pareto-optimal points (C1@0.1, C5@5, C8@20, C9@100, pure_target).

Annotation: "+5pt acc, 0.44× tps" near C8 point.

Style: scientific scatter plot with grid, white background, color-coded markers, legend in upper right.
```

---

## fig_09：熵分布对比图（三科目 Target 熵直方图）

```
A grouped histogram chart titled "Target Entropy Distribution: Correct vs Wrong Answers across Subjects".

Three subplots side by side (one per subject):
1. "Surgery (Target acc=59%)": two overlaid histograms — green (correct, mean H=0.387) and red (wrong, mean H=0.456). Annotation: "p=0.0008 ✓ significant".
2. "Pharmacology (Target acc=76%)": green H=0.321, red H=0.350. Annotation: "p=0.1218 — not significant (auto-degradation)".
3. "Anatomy (Target acc=74%)": green H=0.309, red H=0.409. Annotation: "p=0.0004 ✓ significant".

X-axis: "Shannon Entropy H_t (nats)" range 0–1.5.
Y-axis: "Frequency (count)".

Below the three plots, a single banner: "Entropy as a domain-knowledge gap detector: stronger gap → larger H_wrong − H_correct".

Style: scientific paper figure, green/red color pair, white background, clear axes labels, 16:9 ratio.
```

---

## fig_10：甘特图（剩余 12–18 个月研究进度）

```
A horizontal Gantt chart titled "Research Schedule: 2026.05 – 2027.06 (14 months)".

Timeline X-axis: months from May 2026 to June 2027.
Y-axis (rows, top to bottom):
1. "DSSD strategies finalize (C8/C9/C12 cleanup)" — May–Jun 2026 (2 mo, blue)
2. "Generalization tests (GSM8K, MMLU, Law-JECQA)" — Jun–Aug 2026 (3 mo, blue)
3. "DAF Phase 0: Flip event logging system" — Jul–Aug 2026 (2 mo, teal)
4. "DAF Phase 1: FDLP single-round placement" — Sep–Oct 2026 (2 mo, teal)
5. "DAF Phase 2: Flywheel K=2/K=3 iterations" — Oct–Dec 2026 (3 mo, teal)
6. "DAF Phase 3: Event-baseline ablation (entropy/disagreement)" — Jan–Feb 2027 (2 mo, teal)
7. "Engineering: vLLM integration / tree-based SD" — Mar–Apr 2027 (2 mo, orange)
8. "Thesis writing: chapters 1–3" — Jan–Mar 2027 (3 mo, gray)
9. "Thesis writing: chapters 4–6 + revision" — Apr–May 2027 (2 mo, gray)
10. "Defense preparation" — Jun 2027 (1 mo, red)

Mark milestones with diamond icons:
- M1 (end of Aug 2026): "DSSD paper submission"
- M2 (end of Dec 2026): "Flywheel mechanism verified"
- M3 (end of May 2027): "Thesis draft complete"

Style: clean Gantt chart with grouped color bars, white background, English labels, 16:9 ratio.
```
