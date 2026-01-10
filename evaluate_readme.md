# DSD 评估流程指南

本文档说明如何使用 tmux 运行长时间评估任务，并将结果记录到日志文件中。

## 前置准备

### 1. 确认环境配置

确保以下配置正确：

- **模型路径**：检查 `config.py` 中的 `DRAFT_EXPERT` 是否指向微调后的模型
  ```python
  DRAFT_EXPERT: str = "/data/ocean/decoding/model/LLM-Research/Llama-3.2-1B-Instruct-medreason-ft"
  ```

- **评估任务**：当前默认评估三个任务：
  - MMLU - Professional Medicine
  - MMLU - Medical Genetics
  - MedMCQA

- **生成长度**：`main.py` 中 `gen_len = 2048`，保证 CoT 推理完整展现

### 2. 确认依赖环境

```bash
# 激活评估环境（通常是 kvner）
conda activate kvner

# 验证必要的库已安装
python -c "import torch, transformers, datasets; print('✓ 依赖检查通过')"
```

## 使用 tmux 运行评估

### 步骤 1：创建 tmux 会话

```bash
# 创建一个名为 evaluate 的 tmux 会话
tmux new -s evaluate

# 或者如果会话已存在，先删除再创建
tmux kill-session -t evaluate 2>/dev/null; tmux new -s evaluate
```

### 步骤 2：在 tmux 中准备环境

在 tmux 会话中执行：

```bash
# 进入项目目录
cd /data/ocean/decoding

# 激活 conda 环境
conda activate kvner

# 确认当前目录和模型路径
pwd
python -c "from config import ModelIDs; print('Expert model:', ModelIDs.DRAFT_EXPERT)"
```

### 步骤 3：运行评估并记录日志

在 tmux 会话中运行评估，同时将输出保存到日志文件：

```bash
# 运行评估，同时输出到终端和日志文件
python main.py 2>&1 | tee logs/evaluate_$(date +%Y%m%d_%H%M%S).log

# 或者使用更详细的日志文件名（包含任务信息）
python main.py 2>&1 | tee logs/evaluate_medreason_ft_$(date +%Y%m%d_%H%M%S).log
```

**说明**：
- `2>&1`：将标准错误重定向到标准输出
- `tee`：同时输出到终端和文件
- `$(date +%Y%m%d_%H%M%S)`：生成带时间戳的日志文件名

### 步骤 4：分离 tmux 会话（让任务在后台运行）

评估开始运行后，可以安全地分离会话：

```bash
# 按 Ctrl+B，然后按 D（Detach）
# 或者直接关闭终端窗口，任务会继续运行
```

**快捷键操作**：
1. 按 `Ctrl+B`（tmux 前缀键）
2. 松开，然后按 `D`（Detach）

### 步骤 5：重新连接 tmux 会话（查看进度）

```bash
# 列出所有 tmux 会话
tmux ls

# 重新连接到 evaluate 会话
tmux attach -t evaluate

# 或者简写
tmux a -t evaluate
```

### 步骤 6：查看日志文件

即使不在 tmux 会话中，也可以直接查看日志：

```bash
# 查看最新的日志文件
ls -lt logs/evaluate_*.log | head -n 1

# 查看日志的最后几行（实时监控）
tail -f logs/evaluate_*.log

# 查看完整日志
cat logs/evaluate_*.log | less

# 或者用编辑器打开
nano logs/evaluate_*.log
```

## 完整操作流程示例

### 第一次运行

```bash
# 1. 创建 tmux 会话
tmux new -s evaluate

# 2. 在 tmux 中执行
cd /data/ocean/decoding
conda activate kvner
mkdir -p logs  # 确保日志目录存在

# 3. 运行评估
python main.py 2>&1 | tee logs/evaluate_$(date +%Y%m%d_%H%M%S).log

# 4. 分离会话（Ctrl+B, D）
# 任务继续在后台运行
```

### 后续查看进度

```bash
# 重新连接查看进度
tmux attach -t evaluate

# 或者直接查看日志
tail -f logs/evaluate_*.log
```

### 评估完成后

```bash
# 1. 重新连接会话
tmux attach -t evaluate

# 2. 查看最终结果（会在终端显示）

# 3. 查看完整日志
cat logs/evaluate_*.log

# 4. 退出 tmux 会话（评估已完成）
exit
# 或者按 Ctrl+D
```

## 高级用法

### 只运行特定任务

```bash
# 只运行 MedMCQA
TASKS=medmcqa python main.py 2>&1 | tee logs/evaluate_medmcqa_$(date +%Y%m%d_%H%M%S).log

# 运行多个指定任务
TASKS=pro_med,med_gen python main.py 2>&1 | tee logs/evaluate_mmlu_$(date +%Y%m%d_%H%M%S).log
```

### 增加评估样本数量

编辑 `main.py`，修改 `prompt_limit`：

```python
prompt_limit = 50  # 从 20 增加到 50
```

### 监控 GPU 使用情况

在另一个终端或 tmux 窗口中：

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

### 查看评估进度（不进入 tmux）

```bash
# 查看日志的最后 50 行
tail -n 50 logs/evaluate_*.log

# 搜索特定关键词
grep "准确率" logs/evaluate_*.log
grep "Baseline" logs/evaluate_*.log
grep "Steered" logs/evaluate_*.log
```

## 常见问题

### Q: 如何确认评估正在运行？

```bash
# 方法1：查看 tmux 会话
tmux ls

# 方法2：查看进程
ps aux | grep "python main.py"

# 方法3：查看 GPU 使用
nvidia-smi
```

### Q: 评估被中断了怎么办？

```bash
# 重新连接 tmux 会话
tmux attach -t evaluate

# 如果会话不存在，重新创建并运行
tmux new -s evaluate
cd /data/ocean/decoding
conda activate kvner
python main.py 2>&1 | tee logs/evaluate_$(date +%Y%m%d_%H%M%S).log
```

### Q: 如何停止正在运行的评估？

```bash
# 方法1：在 tmux 会话中按 Ctrl+C

# 方法2：找到进程并终止
ps aux | grep "python main.py"
kill <PID>

# 方法3：终止整个 tmux 会话
tmux kill-session -t evaluate
```

### Q: 日志文件太大怎么办？

```bash
# 压缩旧日志
gzip logs/evaluate_*.log

# 或者只保留最新的几个
ls -t logs/evaluate_*.log | tail -n +10 | xargs rm
```

## 评估结果解读

评估完成后，日志文件会包含：

1. **每个任务的三种模式结果**：
   - Baseline：只使用 Target 模型（8B）
   - Expert-only：只使用微调后的专家模型（1B）
   - Steered：使用 DSD 架构（融合三个模型）

2. **最终总结**：所有任务的准确率对比

3. **详细预测结果**：每个样本的 GT（Ground Truth）和 Pred（Prediction）

**预期效果**：
- 微调后的专家模型在医疗任务上应比原始底座模型表现更好
- DSD 架构通过融合专家知识，可能比单独使用 Target 模型效果更好

## 注意事项

1. **评估时间**：每个任务约 20 个样本，三个任务，每个样本生成最多 2048 tokens，预计需要 1-3 小时（取决于 GPU 性能）

2. **显存占用**：确保有足够显存加载三个模型（Target 8B + Base 1B + Expert 1B）

3. **日志目录**：确保 `logs/` 目录存在且有写权限

4. **tmux 会话**：即使关闭终端，tmux 会话中的任务也会继续运行

5. **网络连接**：首次运行需要从 HuggingFace 下载数据集，确保网络畅通

