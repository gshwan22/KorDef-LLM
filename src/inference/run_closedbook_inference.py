import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(question: str) -> str:
    return (
        "다음 국방 분야 질문에 대해 한국어로 간결하고 정확하게 답하시오.\n"
        "모르면 추측하지 말고 알 수 없다고 답하시오.\n\n"
        f"질문: {question}\n"
        "답변:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    print("[INFO] model:", args.model)
    print("[INFO] input:", args.input)
    print("[INFO] output:", args.output)
    print("[INFO] cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] gpu:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for r in tqdm(rows):
            q = r["closed_book_question"]
            prompt = build_prompt(q)

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
                "reference_answer": r["reference_answer"],
                "source_doc_id": r["source_doc_id"],
                "segment_id": r["segment_id"],
                "source_path": r["source_path"],
                "answer_type": r.get("answer_type"),
                "model": args.model,
                "answer": answer,
            }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            print(f"{r['eval_id']} | {answer[:160]}")

    print("[DONE]", args.output)


if __name__ == "__main__":
    main()
