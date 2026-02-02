import argparse
import json
import re
import sys
import traceback
import torch
from pathlib import Path
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force flush for debugging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

try:
    from data_loader import format_prompt
except ImportError:
    print("Error importing data_loader. Make sure data_loader.py is in the same directory.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# Regex patterns for extracting answers
ANSWER_PATTERNS = [
    re.compile(r"final answer\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"the answer is\s*([A-D])", re.IGNORECASE),
    re.compile(r"answer\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"option\s*([A-D])", re.IGNORECASE),
    re.compile(r"选项\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"答案\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"(?:final\s+answer|the\s+answer\s+is|conclusion|答案|结论)\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"(?:\[|【|\(|\（)\s*([A-D])\s*(?:\]|】|\)|\）)", re.IGNORECASE),
    re.compile(r"\b([A-D])\b\s*(?:is\s+the\s+correct\s+answer|是正确答案|是最终选择)", re.IGNORECASE),
]
LIST_FINAL_PATTERN = re.compile(r"\*\*([A-D])\.\s", re.IGNORECASE)

def extract_answer(text: str) -> str:
    """Extracts the final answer option from the model output."""
    if not text:
        return ""
    
    # Look at the end of the text first
    search_text = text[-400:] if len(text) > 400 else text

    for pattern in ANSWER_PATTERNS:
        m = pattern.search(search_text)
        if m:
            return m.group(1).upper()

    # If no pattern at the end, search specifically for bolded options as a fallback
    m_list = LIST_FINAL_PATTERN.findall(search_text)
    if m_list:
        return m_list[-1].upper()
    
    # Last resort: search the whole text
    for pattern in ANSWER_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).upper()

    return ""

def load_model_and_tokenizer(model_path: str):
    print(f"Loading model from {model_path}...")
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

def truncate_to_error_prefix(full_response: str, error_phrase: str) -> str:
    """
    Find the error phrase in the full response and truncate everything after it starts.
    Returns the prefix text (correct reasoning up to the error).
    """
    if not full_response or not error_phrase:
        return ""
    
    # Try exact match first
    idx = full_response.find(error_phrase)
    
    # If not found, distinct fallback could be implemented, but here we require a match
    if idx == -1:
        # Simple heuristic: ignore case or whitespace diffs could be added here
        return ""
        
    # We want the text BEFORE the error phrase
    return full_response[:idx]

def get_next_token_info(model, tokenizer, input_ids, top_k=5):
    with torch.inference_mode():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Get top-k
        probs, indices = torch.topk(torch.softmax(next_token_logits, dim=-1), k=top_k)
        
        results = []
        for i in range(top_k):
            tok_id = indices[0][i].item()
            prob = probs[0][i].item()
            tok_text = tokenizer.decode([tok_id])
            results.append({
                "token": tok_text,
                "prob": prob,
                "token_id": tok_id
            })
    return results

def generate_continuation(model, tokenizer, input_ids, max_new_tokens=1024):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False, # Use greedy for deterministic logic observation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # Only decode the *new* part
    input_len = input_ids.shape[1]
    generated_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return generated_text

def process_case(item, small_model, small_tok, big_model, big_tok, args):
    # 1. Get Context Info
    question = item.get("question", "")
    options = item.get("options", {})
    big_resp = item.get("big_model_response", "")
    
    # Retrieve the phrase identified by the reviewer
    # If using annotated file, it's inside 'review_details' -> 'phrase'
    # Or sometimes flat if processed. Let's look for review_details first.
    error_phrase = ""
    details = item.get("review_details", {})
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except:
            details = {}
    
    # Store raw review_details for output
    review_details_raw = details

    if isinstance(details, dict):
        error_phrase = details.get("phrase") or details.get("first_wrong_phrase", "")
    
    # If explicit position is available and trustworthy, we can use that too, 
    # but the user specifically asked to "find the phrase". 
    # Let's try to locate the phrase for exact logical cut-off.
    if not error_phrase:
        # print(f"Case {item.get('id')}: No error_phrase found in review_details. Skipping.")
        return None 

    prefix_text = truncate_to_error_prefix(big_resp, error_phrase)
    if not prefix_text:
        # Fallback: if finding phrase fails, maybe use review_first_error_pos if available
        err_pos = item.get("review_first_error_pos")
        if err_pos and isinstance(err_pos, int) and err_pos > 1:
             prefix_text = big_resp[:err_pos-1]
        else:
            print(f"Case {item.get('id')}: Could not locate error phrase '{error_phrase[:20]}...' in response. Skipping.")
            return None

    print(f"Case {item.get('id')}: Found prefix of length {len(prefix_text)} chars.")

    # 2. Construct Prompt inputs for both models
    # We need to recreate the prompt as if the model had just generated 'prefix_text'
    # Format: User Prompt + Assistant Start + prefix_text
    
    # Note: We must use the respective tokenizer's template
    # Small Model
    prompt_small = format_prompt(small_tok, question, options) 
    # format_prompt returns full string with <|im_start|>...
    # We apply the prefix. 
    # CAUTION: format_prompt typically adds generation prompt (e.g. <|im_start|>assistant\n)
    # So we just append the text.
    full_input_str_small = prompt_small + prefix_text
    input_ids_small = small_tok(full_input_str_small, return_tensors="pt").input_ids.to(small_model.device)
    
    # Big Model
    prompt_big = format_prompt(big_tok, question, options)
    full_input_str_big = prompt_big + prefix_text
    input_ids_big = big_tok(full_input_str_big, return_tensors="pt").input_ids.to(big_model.device)

    # 3. Inspect Logits (Micro)
    grad_small = get_next_token_info(small_model, small_tok, input_ids_small, top_k=5)
    grad_big = get_next_token_info(big_model, big_tok, input_ids_big, top_k=5)
    
    # 4. Generate Continuation (Macro)
    # See where they go from here. We use a larger max_new_tokens to hopefully reach the answer.
    cont_small = generate_continuation(small_model, small_tok, input_ids_small, max_new_tokens=1024)
    cont_big = generate_continuation(big_model, big_tok, input_ids_big, max_new_tokens=1024)

    # 5. Extract Answers and Check Correctness
    ground_truth = item.get("ground_truth", "").strip().upper()
    
    # We examine the continuation solely for the answer, 
    # assuming the model follows the "Final answer: X" format at the end.
    ans_small = extract_answer(cont_small)
    ans_big = extract_answer(cont_big)
    
    small_correct = (ans_small == ground_truth) if ans_small else False
    big_correct = (ans_big == ground_truth) if ans_big else False

    return {
        "id": item.get("id"),
        "ground_truth": ground_truth,
        "error_phrase_identified": error_phrase,
        "review_details": review_details_raw,
        "prefix_context": prefix_text,
        "small_model": {
            "next_token_top5": grad_small,
            "continuation": cont_small,
            "extracted_answer": ans_small,
            "is_correct_continuation": small_correct
        },
        "big_model": {
            "next_token_top5": grad_big,
            "continuation": cont_big,
            "extracted_answer": ans_big,
            "is_correct_continuation": big_correct
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated_file", required=True, help="Annotated JSON file")
    parser.add_argument("--small_model", default="/data/ocean/decoding/model/II-Medical-8B")
    parser.add_argument("--big_model", default="/data/ocean/decoding/model/Qwen/Qwen3-14B")
    parser.add_argument("--limit", type=int, default=10, help="Check first N cases")
    parser.add_argument("--output", default="logs/logic_divergence.json")
    args = parser.parse_args()

    # Load Data
    data = json.loads(Path(args.annotated_file).read_text())
    
    # Load Models
    small_tok, small_model = load_model_and_tokenizer(args.small_model)
    big_tok, big_model = load_model_and_tokenizer(args.big_model)
    
    results = []
    count = 0
    for item in data:
        if args.limit and count >= args.limit:
            break
            
        res = process_case(item, small_model, small_tok, big_model, big_tok, args)
        if res:
            results.append(res)
            count += 1
            print(f"Processed {count} cases.")
    
    # Save Full Results
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Done. Saved full results to {args.output}")

    # Generate and Save Filtered Results:
    # Criteria: Big Model Continuation is Wrong AND Small Model Continuation is Correct
    filtered_results = [
        r for r in results 
        if (not r['big_model']['is_correct_continuation']) and r['small_model']['is_correct_continuation']
    ]
    
    # We construct the filtered filename based on the output filename
    output_path = Path(args.output)
    filtered_filename = output_path.stem + "_filtered" + output_path.suffix
    filtered_path = output_path.parent / filtered_filename
    
    filtered_path.write_text(json.dumps(filtered_results, indent=2, ensure_ascii=False))
    print(f"Saved filtered results ({len(filtered_results)} cases) to {filtered_path}")

if __name__ == "__main__":
    print("Script started...", flush=True)
    try:
        main()
    except Exception as e:
        print("\n\nCRITICAL ERROR IN MAIN EXECUTION:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
