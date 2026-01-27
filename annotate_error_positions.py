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

def format_prompt_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """将 Prompt 转换为 Llama-3.1 要求的 Chat List 格式"""
    q = item.get("question", "")
    opts = item.get("options", {})
    gt = item.get("ground_truth", "")
    big_resp = item.get("big_model_response", "")
    big_pred = item.get("big_model_pred", "")
    opts_text = "\n".join([f"{k}. {v}" for k, v in opts.items()])
    
    system_prompt = (
        "You are a medical expert reviewer. Your task is to identify logical or medical errors "
        "in a student's response to a Multiple Choice Question. You must output a structured JSON."
    )
    
    user_prompt = (
        f"Med-MCQ Question:\n{q}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"Correct Answer: {gt}\n"
        f"Student Choice: {big_pred}\n\n"
        f"Student Reasoning Response:\n{big_resp}\n\n"
        "Instructions:\n"
        "1. Analyze the reasoning step-by-step.\n"
        "2. Identify the EXACT first phrase/sentence that contains an error.\n"
        "3. If only the final choice is wrong, identify the choice part.\n"
        "4. Output ONLY this JSON format:\n"
        "{\"thought\": \"analysis\", \"first_wrong_phrase\": \"text snippet\", \"explanation\": \"reasoning\"}"
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def extract_first_phrase(text: str) -> Dict[str, str]:
    """增强型 JSON 提取逻辑"""
    res = {"phrase": "", "explanation": "", "thought": ""}
    json_match = re.search(r"\{.*\}", text, flags=re.IGNORECASE | re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            res["phrase"] = str(obj.get("first_wrong_phrase") or obj.get("phrase", ""))
            res["explanation"] = str(obj.get("explanation") or "")
            res["thought"] = str(obj.get("thought") or "")
            return res
        except:
            pass
    m = re.search(r"first_wrong_phrase\"?\s*:\s*\"([^\"]+)\"", text, flags=re.IGNORECASE)
    if m:
        res["phrase"] = m.group(1)
    return res

def heuristic_first_error_pos(item: Dict[str, Any], phrase: str) -> int:
    """计算错误在原文中的字符偏移位置"""
    resp = str(item.get("big_model_response", ""))
    phrase = phrase.strip()

    if phrase and phrase.lower() not in ["null", "none", "n/a", ""]:
        idx = resp.find(phrase)
        if idx != -1:
            return idx + 1
        try:
            match = re.search(re.escape(phrase), resp, re.IGNORECASE)
            if match:
                return match.start() + 1
        except:
            pass

    gt = str(item.get("ground_truth", "")).strip().upper()
    pred = str(item.get("big_model_pred", "")).strip().upper()
    if gt and pred and gt != pred:
        m = re.search(rf"\b{pred}\b", resp)
        if m:
            return m.start() + 1
            
    return 1 if gt != pred else 0

def judge_case(item: Dict[str, Any], tokenizer, model, max_new_tokens: int = 512) -> Tuple[int, str]:
    """核心判分函数：显式传递 attention_mask 以获得可靠结果"""
    messages = format_prompt_messages(item)
    
    # 修改点：增加 return_dict=True 以同时获取 input_ids 和 attention_mask
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True 
    ).to(model.device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_length = input_ids.shape[1]
    
    if input_length > 8000:
        return -1, json.dumps({"error": "Input too long", "token_count": input_length})

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, # 显式传递掩码，消除警告并提高可靠性
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    gen_tokens = output_ids[0][input_length:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    
    if not gen_text:
        return -1, json.dumps({"error": "Model generated empty response", "raw_gen": ""})

    parsed_res = extract_first_phrase(gen_text)
    pos = heuristic_first_error_pos(item, parsed_res["phrase"])
    
    review_info = {
        "thought": parsed_res["thought"],
        "phrase": parsed_res["phrase"],
        "explanation": parsed_res["explanation"],
        "raw_gen": gen_text 
    }
    return pos, json.dumps(review_info, ensure_ascii=False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Case Error Annotator")
    parser.add_argument("--cases", required=True, help="Path to input JSON")
    parser.add_argument("--model", required=True, help="Path to Reviewer LLM")
    parser.add_argument("--output", default="logs/annotated_results.json", help="Output path")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    tokenizer, model = load_model(args.model)

    annotated = []
    print(f"Processing {len(cases)} cases...")
    
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
            print(f"[{idx + 1}/{len(cases)}] Current Pos: {pos}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(annotated, ensure_ascii=False, indent=2))
    print(f"Completed! Output saved to {out_path}")

if __name__ == "__main__":
    main()