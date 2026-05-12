"""
Source-grounded inference for LoRA-adapted Gemma-3-12B models.

Uses the SAME prompt format and generation parameters as
run_sourcegrounded_inference.py to enable fair comparison with the
KorDef-LLM (Full SFT) and Gemma-3-12B (base) outputs.

Usage:
    python run_lora_inference.py \\
        --base /home/user/models/gemma3-12b \\
        --lora /home/user/Desktop/.../experiments/lora_comparison/model_LoRA-r8 \\
        --input /home/user/data/kordef_sourcegrounded_eval323_results/artifacts/eval_qa_329_aligned.jsonl \\
        --output outputs/outputs_lora_r8_sourcegrounded_eval323.jsonl \\
        --tag lora_r8
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def build_prompt(context: str, question: str) -> str:
    return (
        "아래 제공된 국방 행정규칙 문맥만 근거로 질문에 답하시오.\n"
        "문맥에 없는 내용은 추측하지 말고 '정보 부족'이라고 답하시오.\n"
        "답변은 한국어로 1~3문장으로 간결하게 작성하시오.\n\n"
        "[문맥]\n"
        f"{context}\n\n"
        "[질문]\n"
        f"{question}\n\n"
        "[답변]\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model path or HF id")
    parser.add_argument("--lora", required=True, help="LoRA adapter directory")
    parser.add_argument("--input", required=True, help="Eval set JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--tag", default=None, help="Short model tag for output (e.g. lora_r8)")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_tag = args.tag if args.tag else Path(args.lora).name

    print("[INFO] base:", args.base)
    print("[INFO] lora:", args.lora)
    print("[INFO] tag :", model_tag)
    print("[INFO] in  :", args.input)
    print("[INFO] out :", args.output)
    print("[INFO] cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] gpu :", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    print("[INFO] loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    print("[INFO] attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, args.lora)
    model.eval()
    print("[INFO] model ready.")

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for r in tqdm(rows):
            q = r["source_grounded_question"]
            context = r["context"]
            prompt = build_prompt(context, q)

            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = gen[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            row = {
                "eval_id": r["eval_id"],
                "question": q,
                "closed_book_question": r["closed_book_question"],
                "reference_answer": r["reference_answer"],
                "answer_evidence": r.get("answer_evidence", ""),
                "source_doc_id": r["source_doc_id"],
                "segment_id": r["segment_id"],
                "source_path": r["source_path"],
                "answer_type": r.get("answer_type"),
                "model": model_tag,
                "answer": answer,
            }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    print("[DONE]", args.output)


if __name__ == "__main__":
    main()
