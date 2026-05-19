PPT 与开题报告配图 Nano Banana 生成 Prompts

本文档收录 7 张关键配图的 Nano Banana / Gemini 2.5 Flash Image 生成提示词，每张图同时给出开题报告与 PPT 的放置建议。所有 prompt 都已包含明确的布局、配色、字体、文字内容（保持原样不翻译）与风格关键词，可直接复制粘贴到 Nano Banana 输入框使用。建议输出宽高比统一为 16:9，方便直接嵌入 PPT。

通用配色规范（贯穿 7 张图，保持视觉一致性）：
- Target / 通用大模型：蓝色 #3B82F6
- Draft / 领域专家小模型：橙色 #F97316
- Base / 常识对照：绿色 #10B981
- Target 熵 / 不确定性信号：紫色 #8B5CF6
- 强调 / 警示 / 翻转事件：红色 #EF4444
- 中性框 / 文字辅助：灰色 #6B7280

================================================================================

图 1　三模型协同框架总览图

放置位置：
- 开题报告：§3.2 (1) 三模型框架，紧跟"三模型必须同系列、同词表、同 tokenizer"那段
- PPT：第 3 章"研究内容一"第 1 页"模型架构设置"，作为该 slide 的主视觉

Prompt：

```
Generate a clean, modern academic infographic in landscape 16:9 ratio for a PhD thesis defense slide. The infographic illustrates a three-model collaboration framework for speculative decoding.

Layout: Three model boxes arranged around a central acceptance formula box.
- Top-left: A large blue rounded rectangle labeled "通用大模型 Target" with a smaller subtitle "Qwen2.5-32B-Instruct". Add a small icon of a brain or large neural network. Tag underneath: "综合推理 / 通用能力".
- Top-right: A medium orange rounded rectangle labeled "领域专家 Draft" with a smaller subtitle "Qwen2.5-3B-Surgery". Add a small medical cross icon. Tag underneath: "领域微调 / 高速".
- Bottom-right: A medium green rounded rectangle labeled "常识对照 Base" with a smaller subtitle "Qwen2.5-3B (未微调)". Tag underneath: "常识参考 / 与 Draft 同系列".

Center: A white rounded rectangle with the label "软引导验收公式（基础形式）" on top, and the formula inside written exactly as:
P_accept = min(1, (P_target(x) + α · ΔP(x)) / P_draft(x))

Directly below the main formula, a smaller gray italic line: "α 可被动态化为 α_base · P_draft(x) / 门控信号 等变体"

Three colored arrows flow from each model into the center formula box:
- Blue arrow from Target labeled "logit_T"
- Orange arrow from Draft labeled "logit_D"
- Green arrow from Base labeled "logit_B"

Below the formula box: A horizontal flow showing two boxes connected by an arrow: "Output Token Stream" → "KV Cache (StaticCache, 单卡 H200)".

Bottom-center annotation in small italic gray text: "三模型同 tokenizer / 共驻单张 H200 / bf16 精度".

Style: Flat vector infographic, pure white background, soft subtle drop shadows under each box, clean sans-serif font (Source Han Sans or similar), professional academic poster aesthetic similar to NeurIPS / ICML conference figures. Color palette: Target #3B82F6 blue, Draft #F97316 orange, Base #10B981 green, formula box white with #E5E7EB border. All Chinese, English and mathematical text must be rendered exactly as written, no translation. 16:9 aspect ratio.
```

================================================================================

图 2　标准 SD 在垂直领域失效与软引导修复对比图

放置位置：
- 开题报告：§1.2 末尾"标准投机解码的接受率仅为 0.212……"那段后；§4.2 开头"研究内容一目前已完成的工作包括……"前作为视觉总览
- PPT：第 1 章"研究背景"最后一页"投机解码面临的两个根本问题"作为佐证；第 4 章"目前进展"开头作为核心结果总览

Prompt：

```
Generate a clean academic chart figure in landscape 16:9 ratio for a PhD thesis defense slide. The figure shows two side-by-side bar charts demonstrating the failure of standard speculative decoding in vertical domains and the recovery achieved by the proposed soft-guidance methods.

Top center title (large, bold): "标准投机解码失效 vs 本文软引导修复"
Sub-title (smaller, gray): "MedMCQA Surgery 子集，n=200，贪婪解码"

Left chart:
- Chart title: "准确率 acc"
- Y-axis range: 0.60 to 0.72, with a horizontal red dashed line at y=0.650 labeled "Pure Target 基线 0.650"
- Four bars from left to right, each with the bar value displayed on top:
  Bar 1: "Pure Target", height 0.650, gray (#9CA3AF)
  Bar 2: "Standard SD", height 0.650, gray (#9CA3AF)
  Bar 3: "C9 (本文 概率域)", height 0.690, green (#10B981), with red "+4 pt" badge above
  Bar 4: "C12 (本文 logit 域)", height 0.690, green (#10B981), with red "+4 pt" badge above

Right chart:
- Chart title: "接受率 acc_rate"
- Y-axis range: 0.0 to 1.0, with a horizontal red dashed line at y=0.212 labeled "Standard SD 基线 0.212"
- Four bars from left to right, each with the bar value displayed on top:
  Bar 1: "Pure Target", height 1.000, gray with diagonal hatching (无投机基线)
  Bar 2: "Standard SD", height 0.212, gray (#9CA3AF), red "塌陷" label below the bar
  Bar 3: "C9", height 0.768, green (#10B981), red "+0.556" annotation above
  Bar 4: "C12", height 0.894, green (#10B981), red "+0.682 ★ SOTA" annotation above with a small gold star icon

Bottom annotation (small italic gray): "本文方法同时解决接受率塌陷（0.212 → 0.89）与性能上限突破（acc +4pt 超越 Pure Target）"

Style: Flat infographic, pure white background, clean modern sans-serif font, large readable axis labels and numbers, professional academic poster aesthetic. Use #9CA3AF gray for baselines, #10B981 green for proposed methods, #EF4444 red for annotations and threshold lines. All Chinese, English and numerical text must be rendered exactly as written. 16:9 aspect ratio.
```

================================================================================

图 3　研究三层递进总览图

放置位置：
- 开题报告：§3.1 总体研究目标，紧跟"三个研究目标覆盖在线借用、离线吸收、系统协同三个层次"那句
- PPT：第 2 章"研究目标"作为主视觉，强烈建议同时放在目录页作为研究路线图

Prompt：

```
Generate a clean modern academic infographic in landscape 16:9 ratio for a PhD thesis defense slide. The figure illustrates a three-layer progressive research framework.

Top center title (large bold): "研究三层递进体系：从在线借用到系统协同"

Layout: Three large horizontally aligned rounded rectangles of equal size, connected by thick right-pointing arrows. Each box has a different solid background color with white text inside.

Box 1 (left, blue background #3B82F6):
- Top label inside (small white text): "研究内容一"
- Title (large white text): "在线借用"
- Subtitle (medium white text): "基于对比概率差的软引导投机解码"
- Bottom small text: "ΔP 探针 + 双信号门控"

Box 2 (center, orange background #F97316):
- Top label inside: "研究内容二"
- Title: "离线吸收"
- Subtitle: "基于翻转事件的 Target 精准微调"
- Bottom small text: "FDLP 选层 + Top-K LoRA"

Box 3 (right, green background #10B981):
- Top label inside: "研究内容三"
- Title: "系统协同"
- Subtitle: "知识注入飞轮闭环系统 (DAF)"
- Bottom small text: "迭代收敛 → 单模型部署形态"

Below each box, large gray italic text shows the corresponding research question:
- Below Box 1: "能不能借用？"
- Below Box 2: "能不能吸收？"
- Below Box 3: "能否系统化迭代？"

Two large arrows "→" between boxes 1→2 and 2→3, with small annotation labels above the arrows: "在线信号沉淀" (between 1→2), "迭代闭环" (between 2→3).

Bottom-center annotation (italic small gray): "在线推理 → 离线训练 → 系统化协同演化"

Style: Flat modern infographic, pure white background, soft subtle drop shadows under each box, clean modern sans-serif font (Source Han Sans recommended), professional academic slide aesthetic similar to clean Keynote templates. All Chinese text must be rendered exactly as written. 16:9 aspect ratio.
```

================================================================================

图 6　双信号门控逻辑图

放置位置：
- 开题报告：§3.2 (4) 注入强度 α 的智能调节，作为 C4 / C5 / C6 公式群的总体可视化
- PPT：第 3 章"研究内容一"第 4 页"注入强度 α 的优化策略"，与 C4 / C5 / C6 公式同屏呈现

Prompt：

```
Generate a clean academic infographic in landscape 16:9 ratio for a PhD thesis defense slide. The figure illustrates the dual-signal gating mechanism for adaptive injection strength in soft-guided speculative decoding.

Top center title (large bold): "双信号门控：α_t 的智能调节"

Layout: Two gauge/meter dials on the left side stacked vertically, both connecting via wires to a logical AND gate symbol in the center, which then connects via a wire to an output formula box on the right.

Top-left gauge (orange theme #F97316):
- Label above: "Draft 自信度门控 (C4)"
- Dial face shows a needle pointing past a marked threshold τ
- Inside the dial: small formula text "S_t = P_draft − P_base"
- Below the dial: small green LED indicator with text "✓ S_t > τ 触发"

Bottom-left gauge (purple theme #8B5CF6):
- Label above: "Target 不确定度门控 (C5)"
- Dial face shows a needle pointing toward high entropy
- Inside the dial: small formula text "H_t / H_max"
- Subtitle: "Target 香农熵归一化"
- Below the dial: small green LED indicator with text "✓ Target 不确定 触发"

Center: A clean D-shaped logic AND gate symbol (dark gray #374151 outline) with two input wires from the two gauges and one output wire going right. Above the gate: "双信号 AND 门".

Right output: A large bright green rounded rectangle (#10B981) containing the unified formula written exactly as:
α_t = α_base · 𝕀(S_t > τ) · S_t · (H_t / H_max)
Below the formula in white text: "✓ Draft 有领域知识 + ✓ Target 真的需要 → 注入"

Bottom annotation (italic small gray, full-width): "保护通用位置流利度 / 仅在领域盲区强注入 / 双信号 AND 门优于任一单信号"

Style: Flat modern infographic, pure white background, clean modern sans-serif font, electronic-circuit-diagram inspired but minimalist with rounded corners. Color palette: Draft signal #F97316 orange, Target entropy #8B5CF6 purple, AND gate #374151 dark gray, output #10B981 green. All Chinese, English and mathematical text must be rendered exactly as written, the indicator function symbol 𝕀 must appear correctly. 16:9 aspect ratio.
```

================================================================================

图 7　Token 翻转事件示意图

放置位置：
- 开题报告：§3.3 (1) Token 翻转事件的形式化定义，紧跟定义两个条件那段
- PPT：第 3 章"研究内容二"开篇第 1 页，作为研究内容二最核心概念的视觉化锚点

Prompt：

```
Generate a clean academic infographic in landscape 16:9 ratio for a PhD thesis defense slide. The figure illustrates a "Token Flip Event" during three-model speculative decoding.

Top center title (large bold): "Token 翻转事件 (Token Flip Event)：领域知识缺失锚点"

Main illustration layout: A horizontal token-flow visualization in the middle of the canvas.

On the left, show a prefix sequence as a row of light-gray rounded token boxes containing this exact medical text (each token in its own box):
"The" | "patient" | "presents" | "with" | "acute" | "___"

After the prefix, a Y-shaped fork splits into two diverging arrows pointing right.

Upper branch (purple #A78BFA, semi-transparent to indicate "rejected"):
- A rounded token box containing the word "pain"
- Subtitle below the box (small italic): "A_t = Target argmax  (原本想输出)"
- The arrow from the fork to this token is drawn as a thin dashed purple arrow.

Lower branch (orange #F97316, solid bold to indicate "accepted"):
- A rounded token box containing the word "appendicitis"
- Subtitle below the box (small italic): "B_t = Draft 提议且被验收"
- The arrow from the fork to this token is drawn as a thick solid orange arrow.

At the fork point, draw a bright red lightning bolt icon ⚡ surrounded by a red glow, with a red label below: "F_t 翻转事件"

Below the lower (orange) branch, draw a small "→ 写入序列" arrow pointing further right to a continuation token box: "appendicitis"

Bottom annotation, full-width gray rounded box (#F3F4F6 background):
Line 1 (bold): "翻转事件定义：A_t (Target argmax) ≠ B_t (Draft 提议且被接受)"
Line 2: "→ 该位置即 Target 领域知识缺失的稀疏监督锚点"
Line 3: "→ 与 ΔP 较大的领域词高度重合，Target 熵显著高于非翻转位置"

Style: Flat modern infographic, pure white background, clean modern sans-serif font, slight medical-illustration touch (subtle rounded shapes). Color palette: prefix tokens #E5E7EB gray, A token #A78BFA purple semi-transparent, B token #F97316 orange solid, lightning #EF4444 red with glow, annotation box #F3F4F6 light gray. All Chinese and English text must be rendered exactly as written. 16:9 aspect ratio.
```

================================================================================

图 8　FDLP 选层算法流程图

放置位置：
- 开题报告：§3.3 (2) 基于梯度敏感度的 Top-K 层选取；§4.3 研究内容二完成情况说明的核心算法可视化
- PPT：第 3 章"研究内容二"中段"精准选层微调"slide

Prompt：

```
Generate a clean academic infographic in landscape 16:9 ratio for a PhD thesis defense slide. The figure illustrates the four-step FDLP (Flip-Driven Layer Placement) algorithm pipeline.

Top center title (large bold): "FDLP 选层算法：从翻转事件到 Top-K LoRA 挂载"

Layout: Four large stages arranged left-to-right with equal width, each stage as a rounded card with subtle drop shadow, connected by thick right-pointing arrows between stages.

Stage 1 — blue theme #3B82F6:
- Top icon: A jsonl document file icon
- Stage number badge "①" in top-left
- Title: "翻转事件采集"
- Middle small block: a code snippet style box showing fields:
  { prefix_ids, A, B, ΔP, H_t }
- Bottom annotation: "Round 0 解码 5000 题 → 翻转 jsonl"

Stage 2 — orange theme #F97316:
- Top icon: A neural network stack with a forward-arrow on top and a backward-arrow on bottom (representing forward + backward pass)
- Stage number badge "②" in top-left
- Title: "32B Target 反向传播"
- Middle: the formula written exactly as:
  L = − log P_target(B | prefix_ids)
- Bottom annotation: "事件级 batch=1 / gradient_checkpointing / 显存峰值 ~80 GiB"

Stage 3 — purple theme #8B5CF6:
- Top: A vertical bar chart with approximately 64 bars representing per-layer scores. The top 8 bars (sorted) are highlighted in bright red #EF4444, the rest in light gray. Sort order is from tallest red bars on the left to shorter gray bars on the right.
- Stage number badge "③" in top-left
- Title: "梯度范数打分 + Top-K 选层"
- Middle: the formula written exactly as:
  g_ℓ = ‖∇W_ℓ‖_F / sqrt(numel)
- Bottom annotation: "Top-K (K=8) 红色高亮 / 4 套对照子集并行供分"

Stage 4 — green theme #10B981:
- Top icon: A simplified 64-layer Transformer stack drawn as a tall vertical sequence of rectangles. Only 8 of the rectangles (corresponding to the Top-K layers from Stage 3) are highlighted in red and have a small "LoRA" adapter module attached to their right side.
- Stage number badge "④" in top-left
- Title: "Top-K 层挂载 LoRA 训练"
- Middle: small formula text:
  lora_target = Top-K 模块名
  lora_rank = ⌈r_total / K⌉ = 16
- Bottom annotation: "训练后合并回 32B → Target_{k+1}"

Style: Flat modern infographic, pure white background, soft drop shadows on each stage card, clean modern sans-serif font, professional academic poster aesthetic. All Chinese, English, mathematical text and formula symbols (especially ‖·‖_F, sqrt, ⌈·⌉) must be rendered exactly as written. 16:9 aspect ratio.
```

================================================================================

图 9　知识注入飞轮闭环示意图

放置位置：
- 开题报告：§3.4 研究内容三：基于投机解码框架的知识注入飞轮系统，作为该节主视觉
- PPT：第 3 章"研究内容三"主视觉，建议作为整张 PPT 中流程感最强的"压轴图"

Prompt：

```
Generate a clean academic infographic in landscape 16:9 ratio for a PhD thesis defense slide. The figure illustrates a closed-loop iterative knowledge-injection flywheel system for tri-model speculative decoding.

Top center title (large bold): "知识注入飞轮闭环系统"

Main layout: A large circular flywheel diagram occupying the center of the canvas. Four stage nodes are placed clockwise at the 12, 3, 6, and 9 o'clock positions. Thick clockwise circular arrows connect the nodes in sequence ① → ② → ③ → ④ → ①. The center of the circle has a stylized rotating flywheel icon with the Chinese label "知识注入飞轮" beneath it.

Node ① at 12 o'clock — blue theme #3B82F6:
- Title: "① 软引导投机解码（第 k 轮）"
- Subtitle: "三模型 + C9 策略 / α_base=50 / τ=0.05"

Node ② at 3 o'clock — orange theme #F97316:
- Title: "② 翻转事件采集"
- Subtitle: "记录前缀 / A 词 / B 词 / ΔP / Target 熵"

Node ③ at 6 o'clock — purple theme #8B5CF6:
- Title: "③ 梯度敏感度选层 + LoRA 微调"
- Subtitle: "Top-K=8 / rank=16 / 仅训练约 1% 参数"

Node ④ at 9 o'clock — green theme #10B981:
- Title: "④ 合并 → 新一代 Target"
- Subtitle: "替换原 Target，进入下一轮"

Around the outside of the flywheel, draw three gray dashed arrows pointing outward to three small "停止条件" boxes, each with a small red stop-sign icon:

Stop condition box (top-right corner of canvas):
"停止条件 A：翻转率跨轮下降幅度 < 0.02"
"翻转率不再显著下降"

Stop condition box (bottom-right corner of canvas):
"停止条件 B：累计翻转吸收率 ≥ 0.90"
"已吸收九成以上的翻转 token"

Stop condition box (bottom-left corner of canvas):
"停止条件 C：通用基准退化 > 1 个百分点"
"避免灾难性遗忘"

Bottom-center annotation (italic gray, full-width): "任一停止条件触发即停飞轮 → 输出最终一代 Target，单模型直接部署（无需 Draft / Base）"

Style: Flat modern infographic, pure white background, soft drop shadows under each node card, clean modern sans-serif font, professional academic slide aesthetic. The four nodes should clearly look like they form a closed loop with circular motion. Color palette: Node 1 blue #3B82F6, Node 2 orange #F97316, Node 3 purple #8B5CF6, Node 4 green #10B981, stop conditions gray #6B7280 boxes with red #EF4444 stop icons. All Chinese text must be rendered exactly as written, no translation, no English abbreviations except the standard mathematical symbols α, τ, K. 16:9 aspect ratio.
```

================================================================================

图 10　三模型协同执行时序图 & 融合验收引擎（Fancy 学术风格）

放置位置：
- 开题报告：§3.2 (1) 三模型框架 末尾，与图 1（静态架构）形成"静态 vs 动态"的配对；同时可在 §4.1 或 §4.2 作为方法运行流程总图
- PPT：第 3 章"研究内容一"第 2 页，紧跟图 1 之后作为"这一切是如何并行跑起来"的核心时序图；答辩时建议把这张图留 90 秒以上详细讲解

说明：这是图 1（静态框架）的动态对偶版本。图 1 告诉老师"有哪三个模型和验收公式"，图 10 告诉老师"三个模型在 H200 上如何并行、token 如何在三者之间流动、融合机制在哪一步触发"。

Prompt：

```
Generate a sophisticated, fancy, technical-blog-style infographic in landscape 16:9 ratio for a PhD thesis defense slide, in the visual aesthetic of OpenAI / DeepMind / Anthropic technical report figures — clean swim-lane timeline diagram with soft gradient cards, crisp arrows, and a dark-accent fusion panel at the bottom. This figure illustrates how the tri-model speculative decoding pipeline actually executes on a single GPU: which models run in parallel, how tokens flow between them, and where the soft-guidance fusion engine kicks in.

Top banner (slim horizontal bar spanning full width, dark slate gray #1F2937 background with white text):
Left side: "Single NVIDIA H200   |   bf16 precision   |   CUDA_VISIBLE_DEVICES=0"
Right side: "StaticCache 预分配 KV 缓冲   |   Prefix Cache 物理前缀共享   |   Copy-on-Write 分支"

Title (large bold, centered below banner): "三模型协同执行时序与融合验收引擎"
Subtitle (smaller gray, italic): "Tri-Model Speculative Decoding Pipeline — Execution Timeline View"

Main central area: Three horizontal swim-lane tracks stacked vertically, occupying about 55% of the canvas height. The horizontal axis is time, flowing strictly left to right, with faint vertical dashed time-step grid lines labeled at the top: "Prefill  |  Step k·γ  |  Step k·γ+1  |  Step k·γ+2  |  Step k·γ+3  |  Verify & Commit  |  Step (k+1)·γ".

Lane 1 (top, blue theme #3B82F6):
- Left-side lane label badge: "Target 32B · 验证官"
- Subtitle under label: "Qwen2.5-32B-Instruct"
- Prefill block: a long pale-blue rectangle saying "Prefill: 一次吃入 prompt，写入 StaticCache"
- Decode block: one single wide deep-blue rectangle spanning the 4 decode time-steps, labeled "一次并行前向（Tree-Mask）  →  logit_target[x_1..x_γ]", with a small icon of 4 parallel arrows inside to emphasize "single forward, γ outputs".

Lane 2 (middle, orange theme #F97316):
- Left-side lane label badge: "Draft 3B · 领域提案者"
- Subtitle under label: "Qwen2.5-3B-Surgery"
- Prefill block: pale-orange rectangle saying "Prefill 共享 prefix"
- Decode block: FOUR small orange rectangles in a chain, each labeled "x_1", "x_2", "x_3", "x_γ", connected by thin right-pointing arrows (autoregressive chain). Above this chain a bracket labeled "γ-step 自回归采样（串行）".

Lane 3 (bottom, green theme #10B981):
- Left-side lane label badge: "Base 3B · 常识对照"
- Subtitle under label: "Qwen2.5-3B-Instruct (未微调)"
- Prefill block: pale-green rectangle saying "Prefill 共享 prefix"
- Decode block: one single wide deep-green rectangle spanning the 4 decode time-steps, labeled "一次并行前向（Teacher-Forcing）  →  logit_base[x_1..x_γ]", with 4 parallel arrows icon inside.

Key parallelism callouts between the lanes, drawn as small curly braces and annotations:
- Between Target and Draft: a vertical dotted bracket with text "Target & Base 以 Draft 提案为条件一次性并行 / 节省 γ−1 次 Target 前向".
- A small clock icon next to this bracket with text "Wall-clock: 三模型近似串行 (Draft) + 并行 (Target ‖ Base)".

Data-flow arrows between lanes (thick colored arrows, subtle gradient):
- From each Draft token box (x_i), an orange vertical arrow pointing DOWN into the fusion engine, labeled "x_i, P_draft(x_i)".
- From the Target decode block, a blue downward arrow into the fusion engine, labeled "P_target(x_i), H_t".
- From the Base decode block, a green upward arrow into the fusion engine, labeled "P_base(x_i)".

Bottom fusion panel (dark slate card #111827 with glowing cyan border, occupying about 30% of canvas height, full width):
Panel title (top-left of the card, cyan #22D3EE): "Fusion & Acceptance Engine  ·  融合与验收引擎"

Inside the fusion panel, three sub-modules arranged horizontally left-to-right, each rendered as a softly glowing rounded tile with subtle gradient:

Sub-module A (left tile, green-tinted):
- Tile title: "① 领域探针"
- Formula (white monospaced font): "ΔP(x) = P_draft(x) − P_base(x)"
- Small caption: "对比概率差 / 领域偏好强度"

Sub-module B (center tile, purple-tinted):
- Tile title: "② 双信号门控 + 动态 α_t"
- Signal 1: "S_t = max P_draft − max P_base   （Draft 自信度）"
- Signal 2: "H_t / H_max   （Target 熵·不确定度）"
- Formula: "α_t = α_base · 𝕀(S_t > τ) · S_t · (H_t / H_max)"
- Small caption: "双信号 AND · 仅在领域盲区强注入"

Sub-module C (right tile, blue-tinted):
- Tile title: "③ 软引导验收概率"
- Formula (large, prominent): "P'_accept(x) = min( 1, ( P_target(x) + α_t · ΔP(x) ) / P_draft(x) )"
- Small caption: "标准投机解码 + 领域补贴项"

Right-end of fusion panel: A large decision diamond labeled "Verify", with two outputs:
- Upper branch: green ✓ labeled "Accept → commit x_i into output stream"
- Lower branch: red ✗ labeled "Reject → bonus token 采样 + 回滚"

Output stream strip (just above the fusion panel, right side): A horizontal row of 4 token slots labeled "x_1 ✓  x_2 ✓  x_3 ✓  x_γ ✗" with the first three in green outline and the last in red outline, followed by a small orange "bonus token" badge. To the right of this strip, "Output Stream →".

Feedback loop: A curved dashed arrow going from the "Reject → bonus token" output back up to the left side of the Draft lane, labeled "进入下一轮 γ-step". This closes the cycle and makes the figure feel like a living pipeline.

Top-right corner overlay (small callout card, white background with cyan border): A compact legend titled "关键设计":
- "① 三模型同 tokenizer，共享 KV 前缀"
- "② Target & Base 只需一次并行前向完成 γ-step 验收"
- "③ 双信号门控 → 只在领域盲区注入 ΔP"
- "④ 拒绝即 bonus token，无需重启序列"

Bottom footer (thin gray italic line, full width): "一次 γ-step 验证 = 1× Target forward + 1× Base forward + γ× Draft forward  →  接受率 ↑ · Target 调用次数 ↓ · 领域正确率 ↑"

Overall style: Sophisticated technical-report aesthetic, mostly white or very light gray #FAFAFA canvas with ONE dark fusion panel at the bottom to create contrast, soft drop shadows under all cards, subtle neon glows around the fusion engine to make it feel like the "brain" of the system, clean geometric sans-serif font (Inter or Source Han Sans CJK SC). Color palette kept consistent with the rest of the figure set: Target #3B82F6 blue, Draft #F97316 orange, Base #10B981 green, entropy #8B5CF6 purple, fusion engine dark #111827 with cyan #22D3EE accents, accept #10B981 green, reject #EF4444 red, banner #1F2937 dark slate. All Chinese, English, and mathematical text (especially 𝕀, ΔP, α_t, H_t/H_max, γ) must be rendered exactly as written without translation or substitution. No watermarks, no clipart, no decorative emojis beyond the ✓ ✗ marks. 16:9 aspect ratio, minimum equivalent font size 18pt for all labels.
```

================================================================================

通用使用建议

1. 中文文字渲染：Nano Banana 在中文文字渲染上偶有错字或漏字，建议每张图生成 3-5 张候选后人工挑选；如个别字渲染失败，可在 prompt 末尾追加 "All Chinese characters must be rendered with Source Han Sans / Noto Sans CJK SC, no missing strokes or substitution"。

2. 公式渲染：复杂的 LaTeX 符号（如 ‖·‖_F、⌈·⌉、ΔF̄）建议生成后用 PowerPoint / Keynote 自带的公式工具叠加在图上替换，比让 AI 直接渲染更稳。

3. 配色一致性：7 张图共用同一组配色（蓝橙绿紫红灰），如果生成出来颜色偏移，可在 prompt 中追加具体十六进制色号（已在每张 prompt 中给出）。

4. 字号控制：投影到大屏后图内文字字号要保证 18 pt 以上，如生成结果文字偏小，可在 prompt 末尾追加 "All text labels must be large and clearly readable from a distance, minimum font size equivalent to 18pt"。

5. 输出格式：建议导出为 PNG 格式（带透明背景）或 SVG（如 Nano Banana 支持），PPT 中插入后可二次缩放不失真。

6. 备份方案：如果某张图反复生成不理想，可改用 mermaid / draw.io 绘制矢量版本，告诉我我可以提供对应的 mermaid 语法。
