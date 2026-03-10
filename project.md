实验设置
实验前提：
- 通用大模型（target model）：Qwen14B
  - 路径：/data/ocean/decoding/model/Qwen/Qwen3-14B
- 领域专家小模型（draft model）：Intelligent-Internet/II-Medical-8B
  - 路径：/data/ocean/decoding/model/II-Medical-8B
- 领域专家小模型对应的基座模型（small_base model）：Qwen8B-base
  - 路径：/data/ocean/decoding/model/Qwen/Qwen3-8B-Base
实验目标：
在没有对大模型进行微调的情况下，让最终的模型架构，既能够拥有大模型的通用能力和强大的综合能力（比如CoT），同时具备小模型的领域知识（通过一些通用的领域benchmark来体现），同时拥有投机解码的加速。以轻微的推理速度损失，换取在垂直领域知识上的显著提升。

---
技术路线：基于对比特征蒸馏与双空间加权的有损投机解码
核心思路：
为了在零额外线上推理开销的前提下，让分类器具备判断“纯净医疗知识”的上帝视角，引入 small_base 模型参与离线数据挖掘与特征蒸馏。通过先实施特征空间蒸馏（方案B），再叠加概率差样本加权（方案A），实现对领域知识纠偏特征的精准捕捉。
分类器的输入与输出：
- 线上实际输入：仅包含 Target 和 Draft 两个模型的隐藏状态（Hidden States）拼接向量 $[H_{\text{target}}, H_{\text{draft}}]$，以保证极速推理。
- 离线教师输入：包含三个模型的隐藏状态拼接向量 $[H_{\text{target}}, H_{\text{draft}}, H_{\text{base}}]$。
- 输出：一个二分类结果（概率值），判断被小模型猜错的 Token 是否具备了领域知识且能修复 Target 的错误。
  - 重要 (True)：target model应该接受并放行这个 token。
  - 不重要 (False)：target model应该遵循原生逻辑拒绝这个 token。

---
分类器的训练流程：
第一步：使用自动挖掘与对比加权算法构建数据集
1. 基础筛选：先筛选“大模型做错，小模型做对”的 case。
2. 寻找分歧与强行续写：在这些 case 中，不断从前向后，截取出大模型（target model）的输出前缀（0～n-1）交给小模型，看看下个词的预测（n）是否与大模型的一致。若不一致，找到分歧点 Token。将前缀+分歧点强行交给大模型并继续输出。
3. 结果导向打标（Hard Label）：若大模型能够从此输出正确答案，判断这个词是重要的（Label=1/True）；反之则拒绝（Label=0/False）。注意：此处保留所有能修复答案的 Token，不硬性丢弃通用词，以保住最终正确率。
4. 获取特征：保存该节点上三个模型的隐藏状态 $[H_{\text{target}}, H_{\text{draft}}, H_{\text{base}}]$。
5. 计算领域知识增益（方案 A）：将相同前缀交给 small_base 模型，计算分歧 Token 的概率差值 $\Delta P = P_{\text{draft}} - P_{\text{base}}$。将 $\Delta P$ 映射为该样本的权重 (Sample Weight)。$\Delta P$ 越大，说明该 Token 越属于纯粹的领域增量知识。
第二步：从“上帝视角”到“零开销”的特征蒸馏训练 (Feature Distillation)
本阶段分为两步，逐步把 small_base 的信息压缩进轻量级分类器中：
1. Phase 1: 训练 Teacher 分类器（离线阶段）
  - 输入：$[H_{\text{target}}, H_{\text{draft}}, H_{\text{base}}]$。
  - 目标：拟合第一步得到的硬标签（True/False）。
  - 加权（引入方案 A）：在计算 Loss 时传入样本权重，迫使 Teacher 在高维特征空间中，重点关注那些 $\Delta P$ 极大的“核心医疗专业词汇”的特征模式。
  - 输出：Teacher 输出带有丰富暗知识的软标签 (Soft Labels)。
2. Phase 2: 训练 Student 分类器（线上实际使用）
  - 输入：$[H_{\text{target}}, H_{\text{draft}}]$（剔除 base，保证线上速度）。
  - 目标：使用 KL 散度（KL Divergence Loss）让 Student 去拟合 Teacher 输出的软标签。
  - 加权（引入方案 A）：在蒸馏 Loss 中再次引入样本权重，强迫参数有限的 Student 模型优先保证对“核心专业词汇”决策边界的完美模仿。
第三步：超参数调优与分类器选择
- 为了防止过拟合，通过网格搜索（Grid Search）在一个对数区间（例如 $10^0$ 到 $10^{-7}$ 之间）对正则化系数（参数 "C"）进行独立调优。
- 模型对比：在线上 Student 分类器的选择上，对比简单的逻辑线性回归（追求极致轻量化与兼容性）与单层 MLP（追求更好的特征融合能力）的 AUC 与 TPS（每秒生成 Token 数）表现。

---
第四步：消融实验设计 (Ablation Study)
为了严谨地验证上述设计的有效性，设计以下对照实验：
- Baseline (传统 AutoJudge)：不引入 small_base。仅使用 $[H_{\text{target}}, H_{\text{draft}}]$ 提取特征并直接拟合硬标签。
- Experiment 1 (仅实施方案 B - 特征空间蒸馏)：引入 small_base 训练 Teacher 并蒸馏给 Student，但不使用 $\Delta P$ 进行样本加权（所有样本权重相等）。验证高维特征空间三模型比对带来的增益。
- Experiment 2 (方案 B + 方案 A - 联合双空间对比蒸馏)：完整的最终方案。既使用三模型特征蒸馏，又引入概率差作为权重计算 Loss。验证输出空间（Logits）加权对领域知识敏感度的进一步提升。