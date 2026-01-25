import argparse
import json
import re
from typing import Optional, List, Tuple

import torch
from tqdm import tqdm

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ------------------------------
# 鲁棒的答案提取逻辑
# ------------------------------
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_STRONG_PATTERNS = [
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"Final\s+answer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\bAnswer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\s*[.)]\s*", re.IGNORECASE),
]

_RE_LASTLINE_LETTER = re.compile(r"^\s*(?:option\s*)?([A-D])\s*$", re.IGNORECASE)
_RE_LASTLINE_LETTER_PUNCT = re.compile(r"^\s*(?:option\s*)?([A-D])\s*[.)]\s*$", re.IGNORECASE)

def load_model(model_path: str) -> Tuple[Optional[object], object, bool]:
    """加载模型 - 自动识别 GGUF 或 Transformers 格式"""
    if model_path.endswith('.gguf'):
        if not HAS_LLAMA_CPP:
            raise ImportError("请先安装 llama-cpp-python 以支持 GGUF 格式。")
        print(f"正在加载 GGUF 模型: {model_path}")
        # n_gpu_layers=-1 会将所有层卸载到 GPU (如 H200)
        model = Llama(
            model_path=model_path,
            n_ctx=8192,         # 增加上下文窗口以支持长推理
            n_gpu_layers=-1,    # 强制使用全显卡加速
            n_threads=16,       # CPU 线程数
            verbose=False,
        )
        return None, model, True
    else:
        if not HAS_TRANSFORMERS:
            raise ImportError("请安装 transformers 和 torch。")
        print(f"正在加载 Transformers 模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        return tokenizer, model, False

def create_prompt(question: str, options: dict) -> str:
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    return f"""You are a medical expert. Please analyze the following medical question step by step and provide your reasoning before giving the final answer.

Question: {question}

Options:
{options_text}

Please follow these steps:
1. Analyze the question and understand what is being asked
2. Consider each option carefully with medical knowledge
3. Provide your reasoning step by step
4. End your response with "The correct answer is: [LETTER]" where [LETTER] is STRICTLY one letter in [A,B,C,D].

Your response:"""

def generate_answers_gguf(
    model: Llama,
    prompts: List[str],
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.9,
):
    """针对 GGUF 格式的推理 (手动适配 Qwen/qwen3 模板)"""
    texts = []
    for prompt in prompts:
        # 手动包裹 Qwen 风格的 Chat 模板标签
        formatted_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response = model(
            formatted_input,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|im_start|>", "Question:"],
            echo=False,
        )
        texts.append(response['choices'][0]['text'])
    return texts

@torch.inference_mode()
def generate_answers_transformers(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.9,
):
    """针对 Transformers 格式的批量推理"""
    inputs_text = []
    for p in prompts:
        msg = [{"role": "user", "content": p}]
        inputs_text.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    batch = tokenizer(inputs_text, return_tensors="pt", padding=True).to(model.device)
    
    gen_ids = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_lens = batch["attention_mask"].sum(dim=1).tolist()
    texts = []
    for i, out_ids in enumerate(gen_ids):
        texts.append(tokenizer.decode(out_ids[prompt_lens[i]:], skip_special_tokens=True))
    return texts

def extract_final_answer(response: str, tail_chars: int = 1500) -> Optional[str]:
    if not response: return None
    text = _RE_THINK.sub("", response)
    m = _RE_ANSWER_BLOCK.search(text)
    scope = m.group(1) if m else text[-tail_chars:]

    for pat in _STRONG_PATTERNS:
        matches = list(pat.finditer(scope))
        if matches: return matches[-1].group(1).upper()

    lines = [ln.strip() for ln in scope.splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        mm = _RE_LASTLINE_LETTER.match(ln) or _RE_LASTLINE_LETTER_PUNCT.match(ln)
        if mm: return mm.group(1).upper()
    return None

def evaluate_model(model, tokenizer, dataset, is_gguf, **kwargs):
    correct, total = 0, 0
    results = []
    
    # 过滤无效数据
    valid_items = []
    for idx, item in enumerate(dataset):
        if item.get("question") and item.get("options") and item.get("answer_idx"):
            valid_items.append(item)
    
    if kwargs.get('max_samples', -1) > 0:
        valid_items = valid_items[:kwargs['max_samples']]

    batch_size = 1 if is_gguf else kwargs.get('batch_size', 1)

    for start in tqdm(range(0, len(valid_items), batch_size), desc="Evaluating"):
        chunk = valid_items[start : start + batch_size]
        prompts = [create_prompt(c["question"], c["options"]) for c in chunk]
        gts = [c["answer_idx"].upper().strip() for c in chunk]

        if is_gguf:
            resps = generate_answers_gguf(model, prompts, **{k:v for k,v in kwargs.items() if k in ['max_new_tokens','temperature','top_p']})
        else:
            resps = generate_answers_transformers(model, tokenizer, prompts, **{k:v for k,v in kwargs.items() if k in ['max_new_tokens','temperature','top_p']})

        for item, resp, gt in zip(chunk, resps, gts):
            pred = extract_final_answer(resp)
            is_correct = (pred == gt)
            if is_correct: correct += 1
            total += 1
            results.append({"question": item["question"], "gt": gt, "pred": pred, "correct": is_correct, "response": resp})

    acc = correct / total if total else 0
    return {"accuracy": acc, "correct": correct, "total": total, "results": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="GGUF文件路径或HF模型ID")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--output", default="./eval_results.json", type=str)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # 1. 修正后的解包逻辑
    tokenizer, model, is_gguf = load_model(args.model)

    # 2. 加载数据
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # 3. 执行评估
    eval_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        is_gguf=is_gguf,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # 4. 保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n评估完成！准确率: {eval_results['accuracy']:.2%}")
    print(f"结果已保存至: {args.output}")