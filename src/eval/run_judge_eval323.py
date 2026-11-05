"""
run_judge_eval323.py
=====================

LLM-as-a-Judge evaluation of existing model outputs on the
source-grounded evaluation set (N=323), added in revision to address
reviewer requests for validation beyond lexical-overlap metrics.

Protocol (adapted from this project's earlier GPT-4o judge protocol):
  * Judge model: gpt-4o (set via --judge-model), temperature 0.
  * Blind: the judge never sees which system produced an answer.
  * Independent absolute scoring (no pairwise comparison), so there is
    no position bias to control for.
  * Four axes, 1-5 Likert each, tailored to source-grounded QA:
      - factual_consistency : no claims contradicting or absent from
                              the provided source context
      - correctness         : factual agreement with the reference answer
      - completeness        : covers the key content the question asks for
      - conciseness         : no unnecessary verbosity
  * Structured JSON output enforced via response_format.

The script is resumable: existing (model, eval_id) rows in the output
file are skipped, so it can be interrupted and re-run safely.

Run on: DGX Spark (or any machine with the prediction files)
  cd /home/kisti_platform/kordef_peerj_eval
  source venv/bin/activate
  pip install openai
  export OPENAI_API_KEY=sk-...
  python run_judge_eval323.py \
      --eval artifacts/eval_qa_329_aligned.jsonl \
      --outputs-dir outputs \
      --out judge/judge_scores.jsonl \
      --models gemma3_12b kordef exaone35 qwen25 llama31 axlight lora_r64

Cost control:
  --limit 10          score only the first 10 items per model (pilot)
  --models kordef     score one model at a time

After scoring, summarize with:
  python run_judge_eval323.py --summarize judge/judge_scores.jsonl
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

MODEL_FILES = {
    "gemma3_12b": "outputs_gemma3_12b_sourcegrounded_eval323.jsonl",
    "kordef":     "outputs_kordef_sourcegrounded_eval323.jsonl",
    "exaone35":   "outputs_exaone35_sourcegrounded_eval323.jsonl",
    "qwen25":     "outputs_qwen25_sourcegrounded_eval323.jsonl",
    "llama31":    "outputs_llama31_sourcegrounded_eval323.jsonl",
    "axlight":    "outputs_axlight_sourcegrounded_eval323.jsonl",
    "lora_r8":    "outputs_lora_r8_sourcegrounded_eval323.jsonl",
    "lora_r16":   "outputs_lora_r16_sourcegrounded_eval323.jsonl",
    "lora_r64":   "outputs_lora_r64_sourcegrounded_eval323.jsonl",
    "lora_r8_v2":      "outputs_lora_r8_v2_sourcegrounded_eval323.jsonl",
    "lora_r16_v2":     "outputs_lora_r16_v2_sourcegrounded_eval323.jsonl",
    "lora_r64_v2":     "outputs_lora_r64_v2_sourcegrounded_eval323.jsonl",
    "qlora_r64_v2":    "outputs_qlora_r64_v2_sourcegrounded_eval323.jsonl",
    "lora_r64_all_v2": "outputs_lora_r64_all_v2_sourcegrounded_eval323.jsonl",
}
BASE_MODEL = "gemma3_12b"
AXES = ["factual_consistency", "correctness", "completeness", "conciseness"]
ANSWER_KEYS = ("answer", "prediction", "output", "generated")

SYSTEM_PROMPT = (
    "You are an impartial evaluator of Korean question answering systems "
    "for defense administrative-rule documents. You will receive a source "
    "context (in Korean), a question (in Korean), a reference answer (in "
    "Korean), and a candidate answer (in Korean). Judge the candidate "
    "answer only; do not rewrite it. Score each axis as an integer from "
    "1 (very poor) to 5 (excellent):\n"
    "- factual_consistency: the candidate makes no claims that contradict "
    "the source context and does not assert facts absent from it. A "
    "justified refusal (stating the context lacks the information) when "
    "the context truly lacks it should score high here.\n"
    "- correctness: the candidate's factual content agrees with the "
    "reference answer.\n"
    "- completeness: the candidate covers the key content the question "
    "asks for, as reflected in the reference answer.\n"
    "- conciseness: the candidate avoids unnecessary verbosity and "
    "repetition.\n"
    "Respond ONLY with a JSON object of the form "
    '{"factual_consistency": n, "correctness": n, "completeness": n, '
    '"conciseness": n} with integer values 1-5 and nothing else.'
)

USER_TEMPLATE = (
    "[Source context]\n{context}\n\n"
    "[Question]\n{question}\n\n"
    "[Reference answer]\n{reference}\n\n"
    "[Candidate answer]\n{candidate}"
)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_answer(row):
    for k in ANSWER_KEYS:
        if k in row and row[k] is not None:
            return str(row[k])
    raise KeyError(f"No answer field in row keys={list(row.keys())[:10]}")


def summarize(scores_path):
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    rows = load_jsonl(scores_path)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No scores found.")
        return
    print(f"Judge summary ({scores_path})")
    print("=" * 72)
    base = df[df.model == BASE_MODEL].set_index("eval_id")
    for axis in AXES:
        print(f"\n[{axis}]")
        for name, sub in df.groupby("model"):
            vals = sub[axis].astype(float).values
            line = f"  {name:10s} n={len(vals):3d} mean={np.mean(vals):.3f}"
            if name != BASE_MODEL and not base.empty:
                merged = sub.set_index("eval_id")[[axis]].join(
                    base[[axis]], rsuffix="_b", how="inner").dropna()
                d = (merged[axis] - merged[f"{axis}_b"]).astype(float).values
                try:
                    p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
                except Exception:
                    p = math.nan
                line += f" | delta_vs_base={np.mean(d):+.3f} wilcoxon_p={p:.3g}"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", help="Aligned evaluation JSONL.")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--out", default="judge/judge_scores.jsonl")
    ap.add_argument("--models", nargs="*",
                    default=["gemma3_12b", "kordef", "exaone35", "qwen25",
                             "llama31", "axlight", "lora_r64"])
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, judge only the first N items per model.")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--summarize", metavar="SCORES_JSONL",
                    help="Only print a summary of an existing score file.")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return 0

    if not args.eval:
        ap.error("--eval is required unless --summarize is used.")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    from openai import OpenAI
    client = OpenAI()

    eval_rows = load_jsonl(args.eval)
    ref_by_id = {r["eval_id"]: r for r in eval_rows}
    print(f"[INFO] eval items: {len(ref_by_id)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for r in load_jsonl(out_path):
            done.add((r["model"], r["eval_id"]))
        print(f"[INFO] resuming: {len(done)} scores already present")

    out_f = open(out_path, "a", encoding="utf-8")
    total_calls = 0
    t0 = time.time()

    for name in args.models:
        fpath = Path(args.outputs_dir) / MODEL_FILES[name]
        if not fpath.exists():
            print(f"[WARN] missing predictions for {name}: {fpath} -- skipped")
            continue
        preds = [r for r in load_jsonl(fpath) if r.get("eval_id") in ref_by_id]
        if args.limit > 0:
            preds = preds[:args.limit]
        print(f"[INFO] {name}: {len(preds)} items to judge")

        for i, row in enumerate(preds):
            eid = row["eval_id"]
            if (name, eid) in done:
                continue
            ref = ref_by_id[eid]
            user_msg = USER_TEMPLATE.format(
                context=str(ref.get("context", "")).strip(),
                question=str(ref.get("source_grounded_question",
                                     ref.get("question", ""))).strip(),
                reference=str(ref["reference_answer"]).strip(),
                candidate=get_answer(row).strip() or "(empty answer)",
            )

            scores = None
            for attempt in range(args.max_retries):
                try:
                    resp = client.chat.completions.create(
                        model=args.judge_model,
                        temperature=0,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                    )
                    parsed = json.loads(resp.choices[0].message.content)
                    scores = {a: int(parsed[a]) for a in AXES}
                    for a in AXES:
                        if not 1 <= scores[a] <= 5:
                            raise ValueError(f"{a} out of range: {scores[a]}")
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"[RETRY] {name}/{eid} attempt {attempt+1}: "
                          f"{type(e).__name__}: {e} (sleep {wait}s)")
                    time.sleep(wait)
            if scores is None:
                print(f"[FAIL] {name}/{eid}: giving up after retries")
                continue

            rec = {"model": name, "eval_id": eid,
                   "judge_model": args.judge_model, **scores}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            done.add((name, eid))
            total_calls += 1
            if total_calls % 25 == 0:
                rate = total_calls / (time.time() - t0)
                print(f"[PROG] {total_calls} calls "
                      f"({rate:.2f}/s) -- latest {name} item {i+1}")

    out_f.close()
    print(f"[DONE] {total_calls} new judge calls -> {out_path}")
    print("Summarize with:\n"
          f"  python {Path(sys.argv[0]).name} --summarize {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
