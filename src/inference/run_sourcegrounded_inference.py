"""
run_sourcegrounded_inference.py
================================

Source-grounded inference for Korean defense-domain question answering.

This script runs a Hugging Face causal language model on a JSONL
evaluation file in *source-grounded* mode: each item supplies a context
passage drawn from a Korean defense administrative-rule document, and
the model must answer the question using only that passage. The
companion script ``run_closedbook_inference.py`` runs the same
evaluation without context.

Note on language
----------------
The prompt template and the model's outputs are intentionally in Korean
because this script is designed to evaluate Korean-language LLMs on
Korean defense administrative-rule passages. The Korean strings are
*not* code comments; they are functional data that must be preserved
verbatim for the evaluation to be reproducible. English explanations of
each Korean string are provided inline.

Usage
-----
    python run_sourcegrounded_inference.py \
        --model graphuser/kordef-12b \
        --input  data/eval_qa_329_aligned.jsonl \
        --output outputs/kordef_sourcegrounded.jsonl

Inputs
------
JSONL evaluation file with one item per line. Each item must contain at
least the following keys: ``eval_id``, ``source_grounded_question``,
``closed_book_question``, ``reference_answer``, ``answer_evidence``,
``context``, ``source_doc_id``, ``segment_id``, ``source_path``.
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


def build_prompt(context: str, question: str) -> str:
    """Build the source-grounded prompt in Korean.

    The Korean instructions translate as:
        Line 1: "Answer the question based only on the defense
                 administrative-rule context provided below."
        Line 2: "Do not guess about anything not contained in the
                 context; reply that there is 'insufficient information'."
        Line 3: "Write the answer in Korean, concisely in 1-3 sentences."
        Then:   "[Context] {context}"
                "[Question] {question}"
                "[Answer]"
    """
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
    parser = argparse.ArgumentParser(
        description="Source-grounded inference for Korean defense QA."
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

    # device_map={"": 0} pins the model entirely to GPU 0, which is the
    # typical configuration when the host has multiple GPUs and only one
    # is reserved for this job.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
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
            q = r["source_grounded_question"]
            context = r["context"]
            prompt = build_prompt(context, q)

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
                "closed_book_question": r["closed_book_question"],
                "reference_answer": r["reference_answer"],
                "answer_evidence": r.get("answer_evidence", ""),
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
