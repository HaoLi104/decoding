import argparse
import json
import re
from typing import Optional
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ------------------------------
# Robust answer extraction
# ------------------------------
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_STRONG_PATTERNS = [
    # The correct answer is: B / The correct answer is：B / supports markdown **B**
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    # Final answer: B
    re.compile(r"Final\s+answer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    # Answer: B
    re.compile(r"\bAnswer\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\b", re.IGNORECASE),
    # Some models add punctuation right after the letter
    re.compile(r"The\s+correct\s+answer\s+is\s*[:：]?\s*(?:\*\*|\*)?\s*([A-D])\s*[.)]\s*", re.IGNORECASE),
]

# Fallback: only accept lines that are essentially just a letter
_RE_LASTLINE_LETTER = re.compile(r"^\s*(?:option\s*)?([A-D])\s*$", re.IGNORECASE)
_RE_LASTLINE_LETTER_PUNCT = re.compile(r"^\s*(?:option\s*)?([A-D])\s*[.)]\s*$", re.IGNORECASE)


def load_llm(model_path: str, max_model_len: int = 4096, gpu_memory_utilization: float = 0.90):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
    )
    return tokenizer, llm


def load_dataset(dataset_path: str):
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def create_prompt(question, options) -> str:
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    prompt = f"""You are a medical expert. Please analyze the following medical question step by step and provide your reasoning before giving the final answer.

Question: {question}

Options:
{options_text}

Please follow these steps:
1. Analyze the question and understand what is being asked
2. Consider each option carefully with medical knowledge
3. Provide your reasoning step by step
4. End your response with "The correct answer is: [LETTER]" where [LETTER] is STRICTLY one latter in [A,B,C,D].

Your response:"""
    return prompt


def apply_qwen_template(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text


def generate_answers_vllm(
    llm: LLM,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 4096,
):
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=max_new_tokens,
        top_p=0.9,
    )

    inputs = [apply_qwen_template(p, tokenizer) for p in prompts]

    outputs = llm.generate(inputs, sampling_params)

    texts = []
    for out in outputs:
        texts.append(out.outputs[0].text)
    return texts


def extract_final_answer(response: str, tail_chars: int = 1500) -> Optional[str]:
    """Extract final answer letter (A/B/C/D) from model response.

    Strategy:
    1) Strip <think>...</think> to avoid early conclusions inside thinking.
    2) Prefer parsing within <answer>...</answer> if present.
    3) Otherwise, search only the last `tail_chars` characters to reduce pollution
       from option analysis (A./B./C./D.).
    4) Use strong, explicit answer phrases; if multiple matches, take the last.
    5) Final fallback: look at the last few non-empty lines, accepting only lines
       that are basically a single letter.
    """
    if not response:
        return None

    text = _RE_THINK.sub("", response)

    m = _RE_ANSWER_BLOCK.search(text)
    scope = m.group(1) if m else text[-tail_chars:]

    for pat in _STRONG_PATTERNS:
        matches = list(pat.finditer(scope))
        if matches:
            return matches[-1].group(1).upper()

    # Fallback: inspect only the last few lines
    lines = [ln.strip() for ln in scope.splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        mm = _RE_LASTLINE_LETTER.match(ln) or _RE_LASTLINE_LETTER_PUNCT.match(ln)
        if mm:
            return mm.group(1).upper()

    return None


def evaluate_model(
    llm: LLM,
    tokenizer,
    dataset,
    max_samples: int | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 4096,
):
    correct = 0
    total = 0
    results = []

    if max_samples:
        dataset = dataset[:max_samples]

    items = []
    for idx, item in enumerate(dataset):
        question = item.get("question", "")
        options = item.get("options", {})
        gt = item.get("answer_idx", "").upper().strip()
        if not question or not options or not gt:
            continue
        items.append((idx, question, options, gt))

    for start in tqdm(range(0, len(items), batch_size), desc="Evaluating"):
        chunk = items[start : start + batch_size]
        prompts = [create_prompt(q, opt) for (_, q, opt, _) in chunk]

        try:
            responses = generate_answers_vllm(
                llm=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f"\nError in batch starting at {start}: {e}")
            import traceback
            traceback.print_exc()
            continue

        for (orig_idx, question, options, gt), resp in zip(chunk, responses):
            pred = extract_final_answer(resp)
            is_correct = (pred == gt)
            if is_correct:
                correct += 1
            total += 1

            results.append(
                {
                    "id": orig_idx,
                    "question": question,
                    "options": options,
                    "ground_truth": gt,
                    "predicted_answer": pred,
                    "response": resp,
                    "is_correct": is_correct,
                }
            )

    acc = correct / total if total else 0.0
    return {"accuracy": acc, "correct": correct, "total": total, "results": results}


def save_results(results, output_path: str, model_name: str = ""):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)
    print(f"\nFull results saved to {output_path}")

    summary_path = output_path.replace(".json", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("MEDICAL QA EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        if model_name:
            f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: MedQA\n")
        f.write(f"Total questions: {results['total']}\n")
        f.write(f"Correct answers: {results['correct']}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']:.2%})\n")
        f.write("=" * 50 + "\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/yang/PycharmProjects/ExpertDecoding/decoding-main/models/II-Medical-8B/models--Intelligent-Internet--II-Medical-8B/snapshots/545fa0238261e041fb1ef3f6ed644a5a8f8400e3", type=str)
    parser.add_argument("--dataset", default="./phrases_no_exclude_test.jsonl", type=str)
    parser.add_argument("--output", default="./evaluation_results.json", type=str)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    args = parser.parse_args()

    print("\nLoading tokenizer + vLLM engine...")
    tokenizer, llm = load_llm(
        model_path=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} samples from dataset")

    max_samples = args.max_samples if args.max_samples > 0 else None
    results = evaluate_model(
        llm=llm,
        tokenizer=tokenizer,
        dataset=dataset,
        max_samples=max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    save_results(results, args.output, model_name=args.model)
