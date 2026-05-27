"""
run_closedbook_inference.py
============================

Closed-book inference for Korean defense-domain question answering.

This script runs a Hugging Face causal language model on a JSONL evaluation
file in *closed-book* mode: the model is given only the question, with no
supporting context, and must answer from its parametric knowledge alone.
The companion script ``run_sourcegrounded_inference.py`` runs the same
evaluation in *source-grounded* (context-provided) mode.

Note on language
----------------
The prompt template and the model's outputs are intentionally in Korean
because this script is designed to evaluate Korean-language LLMs on
Korean defense administrative-rule questions. The Korean strings are
*not* code comments; they are functional data that must be preserved
verbatim for the evaluation to be reproducible. English explanations of
each Korean string are provided inline.

Usage
-----
    python run_closedbook_inference.py \
        --model graphuser/kordef-12b \
        --input  data/eval_qa_329_aligned.jsonl \
        --output outputs/kordef_closedbook.jsonl

Inputs
------
JSONL evaluation file with one item per line. Each item must contain at
least the following keys: ``eval_id``, ``closed_book_question``,
``reference_answer``, ``source_doc_id``, ``segment_id``, ``source_path``.
``answer_type`` is optional.

Outputs
-------
JSONL prediction file with one record per evaluation item. Each record
contains the original metadata, the model identifier, and the generated
``answer``.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(question: str) -> str:
    """Build the closed-book prompt in Korean for a single question.

    The Korean instructions translate as:
        Line 1: "Answer the following defense-domain question in Korean,
                 concisely and accurately."
        Line 2: "If you do not know, do not guess; reply that you cannot
                 determine the answer."
        Then:   "Question: {question}"
                "Answer:"
    """
    return (
        "다음 국방 분야 질문에 대해 한국어로 간결하고 정확하게 답하시오.\n"
        "모르면 추측하지 말고 알 수 없다고 답하시오.\n\n"
        f"질문: {question}\n"
        "답변:"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Closed-book inference for Korean defense QA."
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model id or local path.")
    parser.add_argument("--input", required=True,
                        help="Path to input JSONL evaluation file.")
    parser.add_argument("--output", required=True,
                        help="Path where predictions JSONL will be written.")
    parser.add_argument("--max-new-tokens", type=int, default=192,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Disable network access when loading model.")
    args = parser.parse_args()

    print("[INFO] model:", args.model)
    print("[INFO] input:", args.input)
    print("[INFO] output:", args.output)
    print("[INFO] cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] gpu:", torch.cuda.get_device_name(0))

    # --- load tokenizer and model -----------------------------------------
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

    # --- load evaluation set ----------------------------------------------
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # --- run inference, one item at a time, streaming to disk -------------
    with open(args.output, "w", encoding="utf-8") as out:
        for r in tqdm(rows):
            q = r["closed_book_question"]
            prompt = build_prompt(q)

            # Use the model's chat template if it provides one; otherwise
            # fall back to the raw prompt.
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
                    do_sample=False,           # greedy decoding for reproducibility
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens, not the prompt.
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

            # Per-item progress print, truncated for terminal readability.
            print(f"{r['eval_id']} | {answer[:160]}")

    print("[DONE]", args.output)


if __name__ == "__main__":
    main()
