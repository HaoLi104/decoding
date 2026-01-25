import argparse
import json
import re
import gc
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
# 增强型鲁棒答案提取逻辑
# ------------------------------
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_STRONG_PATTERNS = [
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"Final\s+answer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\bAnswer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"正确答案是\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
]

def extract_final_answer(response: str) -> Optional[str]:
    if not response: return None
    
    # 1. 优先在非思考块（正式回答部分）寻找强匹配
    formal_part = _RE_THINK.sub("", response)
    for pat in _STRONG_PATTERNS:
        matches = list(pat.finditer(formal_part))
        if matches: return matches[-1].group(1).upper()
        
    # 2. 寻找 <answer>X</answer> 标签
    m = _RE_ANSWER_BLOCK.search(response)
    if m:
        letters = re.findall(r"\b([A-D])\b", m.group(1))
        if letters: return letters[-1].upper()

    # 3. 兜底：如果模型被截断，去 <think> 思考块里找最后提到的选项
    # 很多推理模型会在思考结束时写 "So the answer should be B"
    think_matches = _RE_THINK.findall(response)
    if think_matches:
        think_text = " ".join(think_matches)
        think_pats = [
            re.compile(r"answer\s+(?:is|would\s+be|seems\s+to\s+be)\s*[:：]?\s*([A-D])\b", re.IGNORECASE),
            re.compile(r"choose\s+([A-D])\b", re.IGNORECASE),
            re.compile(r"选项\s*([A-D])\s*是正确的", re.IGNORECASE)
        ]
        for pat in think_pats:
            matches = list(pat.finditer(think_text))
            if matches: return matches[-1].group(1).upper()

    # 4. 极致兜底：找全文最后出现的那个 A/B/C/D
    all_letters = re.findall(r"\b([A-D])\b", response[-500:]) # 只看最后500字防止干扰
    if all_letters:
        return all_letters[-1].upper()

    return None

def load_model(model_path: str) -> Tuple[Optional[object], object, bool]:
    if model_path.endswith('.gguf'):
        if not HAS_LLAMA_CPP:
            raise ImportError("请安装 llama-cpp-python。")
        print(f"正在加载 GGUF 模型: {model_path}")
        model = Llama(
            model_path=model_path,
            n_ctx=16384,        # 既然有 H200，上下文给足，防止长推理溢出
            n_gpu_layers=-1,    # 全层卸载到 GPU
            n_threads=16,
            verbose=False,
        )
        return None, model, True
    else:
        # Transformers 逻辑保持不变
        print(f"正在加载 Transformers 模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
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

def generate_answers_gguf(model: Llama, prompts: List[str], **kwargs):
    texts = []
    for prompt in prompts:
        formatted_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        response = model(
            formatted_input,
            max_tokens=kwargs.get('max_new_tokens', 3072),
            temperature=kwargs.get('temperature', 0.0),
            top_p=kwargs.get('top_p', 0.9),
            stop=["<|im_end|>", "<|im_start|>", "Question:"],
            echo=False,
        )
        texts.append(response['choices'][0]['text'])
    return texts

@torch.inference_mode()
def generate_answers_transformers(model, tokenizer, prompts: List[str], **kwargs):
    inputs_text = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    batch = tokenizer(inputs_text, return_tensors="pt", padding=True).to(model.device)
    
    gen_ids = model.generate(
        **batch,
        max_new_tokens=kwargs.get('max_new_tokens', 3072),
        do_sample=(kwargs.get('temperature', 0.0) > 0),
        temperature=kwargs.get('temperature', 0.0) if kwargs.get('temperature', 0.0) > 0 else None,
        top_p=kwargs.get('top_p', 0.9) if kwargs.get('temperature', 0.0) > 0 else None,
    )
    prompt_lens = batch["attention_mask"].sum(dim=1).tolist()
    return [tokenizer.decode(g[prompt_lens[i]:], skip_special_tokens=True) for i, g in enumerate(gen_ids)]

def evaluate_model(model, tokenizer, dataset, is_gguf, **kwargs):
    correct, total = 0, 0
    results = []
    valid_items = [item for item in dataset if item.get("question") and item.get("options") and item.get("answer_idx")]
    
    if kwargs.get('max_samples', -1) > 0:
        valid_items = valid_items[:kwargs['max_samples']]

    batch_size = 1 if is_gguf else kwargs.get('batch_size', 1)

    for start in tqdm(range(0, len(valid_items), batch_size), desc="Evaluating"):
        chunk = valid_items[start : start + batch_size]
        prompts = [create_prompt(c["question"], c["options"]) for c in chunk]
        gts = [c["answer_idx"].upper().strip() for c in chunk]

        try:
            if is_gguf:
                resps = generate_answers_gguf(model, prompts, **kwargs)
            else:
                resps = generate_answers_transformers(model, tokenizer, prompts, **kwargs)
        except Exception as e:
            print(f"Error during generation: {e}")
            continue

        for item, resp, gt in zip(chunk, resps, gts):
            pred = extract_final_answer(resp)
            is_correct = (pred == gt)
            if is_correct: correct += 1
            total += 1
            # 对齐字段名，方便 jq 检查
            results.append({
                "id": total, 
                "ground_truth": gt, 
                "predicted_answer": pred, 
                "is_correct": is_correct, 
                "response": resp
            })

    return {"accuracy": correct / total if total else 0, "correct": correct, "total": total, "results": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--output", default="./eval_results.json", type=str)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3072) # 默认调高
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer, model, is_gguf = load_model(args.model)

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    eval_results = evaluate_model(
        model=model, tokenizer=tokenizer, dataset=dataset, is_gguf=is_gguf,
        max_samples=args.max_samples, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n评估完成！准确率: {eval_results['accuracy']:.2%}")
    
    # 显式清理，规避最后的 TypeError 报错
    if is_gguf:
        del model
        gc.collect()