import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path: str):
    """加载模型与分词器"""
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
    return tokenizer, model

def format_prompt(item: Dict[str, Any]) -> str:
    """构建优化的 Prompt，要求 CoT 过程嵌套在 JSON 中"""
    q = item.get("question", "")
    opts = item.get("options", {})
    gt = item.get("ground_truth", "")
    big_resp = item.get("big_model_response", "")
    big_pred = item.get("big_model_pred", "")
    opts_text = "\n".join([f"{k}. {v}" for k, v in opts.items()])
    
    prompt = (
        "You are a medical expert reviewer. Analyze the student's response to a medical MCQ step-by-step.\n"
        "1. Compare the student's logic with the correct answer.\n"
        "2. Identify the EXACT first phrase or sentence where the student's reasoning becomes incorrect.\n"
        "3. If the student's logic is entirely correct but the final choice is wrong, the error is at the final choice.\n\n"
        f"Question:\n{q}\n\nOptions:\n{opts_text}\n\nCorrect answer: {gt}\n\n"
        f"Student final choice (extracted): {big_pred}\n"
        f"Student response (full):\n{big_resp}\n\n"
        "Respond ONLY in this JSON format:\n"
        "{\n"
        "  \"thought\": \"brief step-by-step analysis\",\n"
        "  \"first_wrong_phrase\": \"the exact text snippet from the response\",\n"
        "  \"explanation\": \"why it is wrong\"\n"
        "}"
    )
    return prompt

def extract_first_phrase(text: str) -> Dict[str, str]:
    """增强型 JSON 提取，支持多种 key 名兼容"""
    # 尝试匹配最外层的 JSON 括号
    json_match = re.search(r"\{.*\}", text, flags=re.IGNORECASE | re.DOTALL)
    res = {"phrase": "", "explanation": "", "thought": ""}
    
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            # 兼容多种可能的 key 名
            res["phrase"] = str(obj.get("first_wrong_phrase") or obj.get("phrase") or "")
            res["explanation"] = str(obj.get("explanation") or "")
            res["thought"] = str(obj.get("thought") or "")
            return res
        except Exception:
            pass
            
    # 如果 JSON 解析完全失败，尝试正则提取短语
    m = re.search(r"first_wrong_phrase\"?\s*:\s*\"([^\"]+)\"", text, flags=re.IGNORECASE)
    if m:
        res["phrase"] = m.group(1)
    return res

def heuristic_first_error_pos(item: Dict[str, Any], phrase: str) -> int:
    """
    启发式定位：
    1. 模糊匹配错误短语
    2. 如果匹配不到，匹配错误的选项字母
    3. 全都匹配不到则回退到 1 或 0
    """
    resp = str(item.get("big_model_response", ""))
    phrase = phrase.strip()

    # 1. 尝试在原文中定位短语 (忽略大小写，允许简单的正则转义)
    if phrase and phrase.lower() not in ["null", "none", "n/a"]:
        # 尝试精确匹配
        idx = resp.find(phrase)
        if idx != -1:
            return idx + 1
        
        # 尝试忽略大小写的正则匹配（处理模型可能改变了大小写的情况）
        try:
            match = re.search(re.escape(phrase), resp, re.IGNORECASE)
            if match:
                return match.start() + 1
        except:
            pass

    # 2. 回退：如果预测和真值一致，说明没报错（可能是模型误判）
    gt = str(item.get("ground_truth", "")).strip().upper()
    pred = str(item.get("big_model_pred", "")).strip().upper()
    if gt and pred and gt == pred:
        return 0

    # 3. 回退：寻找学生最终选择字母在原文中出现的位置
    if pred in {"A", "B", "C", "D"} and resp:
        # 寻找诸如 "the answer is B" 或 "Choice: B" 这种模式
        m = re.search(rf"\b{pred}\b", resp)
        if m:
            return m.start() + 1

    # 4. 终极保底：如果确定有错但找不到位置，返回 1
    return 1

def judge_case(item: Dict[str, Any], tokenizer, model, max_new_tokens: int = 2048) -> Tuple[int, str]:
    """单例评判，增加 Token 长度以容纳 CoT"""
    prompt = format_prompt(item)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    gen_text = text[len(prompt):].strip()
    
    # 解析并定位
    parsed_res = extract_first_phrase(gen_text)
    pos = heuristic_first_error_pos(item, parsed_res["phrase"])
    
    # 返回位置和结构化的评论
    review_info = {
        "thought": parsed_res["thought"],
        "phrase": parsed_res["phrase"],
        "explanation": parsed_res["explanation"],
        "raw_gen": gen_text # 保留原始生成以便追溯
    }
    return pos, json.dumps(review_info, ensure_ascii=False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate first error position with CoT logic.")
    parser.add_argument("--cases", required=True, help="Path to cases JSON")
    parser.add_argument("--model", required=True, help="Reviewer model path")
    parser.add_argument("--output", default="logs/annotated_results.json", help="Output path")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    tokenizer, model = load_model(args.model)

    annotated = []
    print(f"Starting processing {len(cases)} cases...")
    
    for idx, item in enumerate(cases):
        try:
            pos, review_json = judge_case(item, tokenizer, model)
        except Exception as e:
            pos, review_json = -1, json.dumps({"error": str(e)})
        
        enriched = dict(item)
        enriched["review_first_error_pos"] = pos
        enriched["review_details"] = json.loads(review_json)
        annotated.append(enriched)
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(cases)} - Last Pos: {pos}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(annotated, ensure_ascii=False, indent=2))
    print(f"Success! Saved to {out_path}")

if __name__ == "__main__":
    main()