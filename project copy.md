实验环境：
cd /data/ocean/decoding
conda activate kvner

实验前提：
一个通用大模型：target model：Qwen14B
路径：/data/ocean/decoding/model/Qwen/Qwen3-14B

一个领域专家小模型：draft model：Intelligent-Internet/II-Medical-8B
路径：/data/ocean/decoding/model/II-Medical-8B

领域专家小模型对应的基座模型：small_base model：Qwen8B-base
路径：/data/ocean/decoding/model/Qwen/Qwen3-8B-Base

实验目标：
在没有对大模型进行微调的情况下，让最终的模型架构，既能够拥有大模型的通用能力和强大的综合能力（比如CoT），同时具备小模型的领域知识（通过一些通用的领域benchmark来体现），同时拥有投机解码的加速。以轻微的推理速度损失，换取在领域知识上的提升。

技术路线1: 有损投机解码。训练一个分类器，当出现本应“拒绝”的时候，通过这个分类器，让target model 能接受这个token。


分类器的输入：三个模型（target、draft，small ）的隐藏状态（Hidden States）的拼接向量
分类器的输出：
输出是一个二分类结果（通常表现为概率值）：判断这个被小模型猜错的 Token 是否是具备了领域知识的，且如果target model接受了这个token，更容易生成最后的正确答案
重要 (True)：target model应该接受并放行这个token
不重要 (False)：target model应该拒绝这个token
分类器的训练：

第一步：使用自动挖掘训练数据算法
先筛选“大模型做错，小模型做对”的case。
然后在这些case中，不断从前向后，截取出大模型（target model）的输出前缀（0～n-1）交给小模型，看看下个词的预测（n）是否与大模型的一致，若不一致，就找到了一个分歧点。
将前缀+分歧点（0～n-1+n）交给大模型并继续输出，若大模型能够从此输出正确答案，那么判断这个词是重要的、应该被接受的。反之则认为这个token是应该被拒绝的。
这样就自动构建出了一个包含大量特征（拼接三个模型的hidden states）和标签（重要or不重要）的数据集。

第二步：训练逻辑回归 (Logistic Regression) 模型论文选择了一个极度简单的逻辑回归作为分类器。将第一步收集到的拼接隐藏状态作为输入，True/False 标签作为目标进行训练 。

第三步：超参数调优为了防止过拟合，作者通过网格搜索（Grid Search）在一个对数区间（例如 $10^0$ 到 $10^{-7}$ 之间）对 $L_2$ 正则化系数（参数 "C"）进行了独立的调优 。

分类器的选择：
可以做多个实验来选择，Auto Judge中使用的是简单的逻辑线形回归（auto judgeAUC=0.812）（优点是及其轻量化）
单层MLP（auto judge中AUC=0.799）