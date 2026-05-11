#!/usr/bin/env python3
"""
Multi-model source-grounded scoring (extends score_sourcegrounded_eval323.py)

Uses IDENTICAL metric functions to the original script for paper consistency.
Adds support for arbitrary model output files via argparse.

Usage:
    python score_multi_model.py \\
        --eval artifacts/eval_qa_329_aligned.jsonl \\
        --output-dir outputs/metrics_multi \\
        --models \\
            gemma3_12b:outputs/outputs_gemma3_12b_sourcegrounded_eval323.jsonl \\
            kordef:outputs/outputs_kordef_sourcegrounded_eval323.jsonl \\
            exaone35:outputs/outputs_exaone35_sourcegrounded_eval323.jsonl \\
            qwen25:outputs/outputs_qwen25_sourcegrounded_eval323.jsonl \\
            llama31:outputs/outputs_llama31_sourcegrounded_eval323.jsonl \\
            axlight:outputs/outputs_axlight_sourcegrounded_eval323.jsonl \\
        --baseline gemma3_12b
"""
import argparse
import json
import re
import math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# === IDENTICAL TO score_sourcegrounded_eval323.py ===

REFUSAL_PATTERNS = [
    "알 수 없습니다",
    "정보 부족",
    "확인할 수 없습니다",
    "명시되어 있지 않습니다",
    "제공되지 않습니다",
    "문맥에 없습니다",
]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def korean_tokens(s):
    s = normalize_text(s)
    return re.findall(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]", s)


def token_f1(pred, ref):
    pt = korean_tokens(pred)
    rt = korean_tokens(ref)
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    pc = Counter(pt)
    rc = Counter(rt)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(rt)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a, b):
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(n):
        prev = 0
        for j in range(m):
            tmp = dp[j + 1]
            if a[i] == b[j]:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = tmp
    return dp[m]


def rouge_l_f1(pred, ref):
    pt = korean_tokens(pred)
    rt = korean_tokens(ref)
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    lcs = lcs_len(pt, rt)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pt)
    recall = lcs / len(rt)
    return 2 * precision * recall / (precision + recall)


def char_ngrams(s, n=3):
    s = re.sub(r"\s+", "", str(s))
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s)-n+1)}


def char_jaccard(pred, ref, n=3):
    a = char_ngrams(pred, n)
    b = char_ngrams(ref, n)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evidence_overlap(answer, evidence):
    at = set(korean_tokens(answer))
    et = set(korean_tokens(evidence))
    et = {x for x in et if len(x) >= 2}
    if not et:
        return 0.0
    return len(at & et) / len(et)


def is_refusal(answer):
    return any(p in answer for p in REFUSAL_PATTERNS)


def bootstrap_ci(values, n_boot=5000, seed=20260508):
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return (math.nan, math.nan)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        means.append(float(np.mean(sample)))
    return tuple(np.percentile(means, [2.5, 97.5]))


# === NEW: Multi-model wrapper ===

METRICS = ["token_f1", "rouge_l", "char3_jaccard", "evidence_token_recall", "answer_tokens", "refusal"]


def compute_per_item(model_name, ans, ref, evidence):
    return {
        "model": model_name,
        "token_f1": token_f1(ans, ref),
        "rouge_l": rouge_l_f1(ans, ref),
        "char3_jaccard": char_jaccard(ans, ref, 3),
        "evidence_token_recall": evidence_overlap(ans, evidence),
        "answer_chars": len(ans),
        "answer_tokens": len(korean_tokens(ans)),
        "refusal": int(is_refusal(ans)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="Reference eval set jsonl")
    ap.add_argument("--models", nargs="+", required=True,
                    help="List of MODEL_NAME:PATH_TO_OUTPUT_JSONL")
    ap.add_argument("--baseline", default="gemma3_12b",
                    help="Baseline model name for paired comparisons (default: gemma3_12b)")
    ap.add_argument("--output-dir", default="outputs/metrics_multi")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse model:path pairs
    model_paths = {}
    for spec in args.models:
        name, path = spec.split(":", 1)
        model_paths[name] = path

    print(f"Loaded models: {list(model_paths.keys())}")
    print(f"Baseline: {args.baseline}")

    # Load eval (use first model file's reference for evidence/ref answer)
    eval_rows = load_jsonl(args.eval)
    eval_by_id = {r["eval_id"]: r for r in eval_rows}
    print(f"Eval items: {len(eval_by_id)}")

    # Load all model predictions
    preds_by_model = {}
    for name, path in model_paths.items():
        try:
            rows = load_jsonl(path)
            preds_by_model[name] = {r["eval_id"]: r for r in rows}
            print(f"  {name}: {len(preds_by_model[name])} predictions")
        except FileNotFoundError:
            print(f"  WARNING: {path} not found, skipping {name}")

    # Common eval_ids across all models
    common_ids = set(eval_by_id)
    for name, p in preds_by_model.items():
        common_ids &= set(p)
    common_ids = sorted(common_ids)
    print(f"Common eval_ids: {len(common_ids)}")

    # Compute per-item metrics for each (model, eval_id)
    records = []
    for eid in common_ids:
        ref = eval_by_id[eid]
        ref_ans = ref.get("reference_answer", "")
        evidence = ref.get("answer_evidence", "")
        for model_name, preds in preds_by_model.items():
            ans = preds[eid].get("answer", "")
            row = compute_per_item(model_name, ans, ref_ans, evidence)
            row["eval_id"] = eid
            row["answer_type"] = ref.get("answer_type")
            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "per_model_metrics.csv", index=False)

    # === Per-model summary table (mean ± CI) ===
    summary_rows = []
    for model_name in preds_by_model:
        sub = df[df.model == model_name]
        row = {"model": model_name, "n": len(sub)}
        for m in METRICS:
            vals = sub[m].astype(float).values
            ci = bootstrap_ci(vals)
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_ci_lo"] = ci[0]
            row[f"{m}_ci_hi"] = ci[1]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "model_summary.csv", index=False)

    print("\n=== Per-model summary ===")
    print(summary_df[["model"] + [f"{m}_mean" for m in METRICS]].to_string(index=False))

    # === Pairwise comparison: each model vs baseline ===
    if args.baseline not in preds_by_model:
        print(f"\nWARNING: baseline '{args.baseline}' not found. Skipping pairwise.")
        return

    base_df = df[df.model == args.baseline].set_index("eval_id")

    pair_lines = ["", "=" * 80]
    pair_lines.append(f"Pairwise comparison vs baseline = {args.baseline}")
    pair_lines.append("=" * 80)

    for model_name in preds_by_model:
        if model_name == args.baseline:
            continue
        cmp_df = df[df.model == model_name].set_index("eval_id")
        common = base_df.index.intersection(cmp_df.index)
        pair_lines.append(f"\n[{model_name} vs {args.baseline}]  (n={len(common)})")
        for m in METRICS:
            deltas = (cmp_df.loc[common, m].astype(float) - base_df.loc[common, m].astype(float)).values
            mean_d = float(np.mean(deltas))
            ci = bootstrap_ci(deltas)
            try:
                if np.allclose(deltas, 0):
                    p = 1.0
                else:
                    p = wilcoxon(deltas).pvalue
            except Exception:
                p = math.nan

            wtl_str = ""
            if m != "refusal":
                wins = int(np.sum(deltas > 0))
                ties = int(np.sum(deltas == 0))
                losses = int(np.sum(deltas < 0))
                wtl_str = f"  W/T/L={wins}/{ties}/{losses}"
            pair_lines.append(
                f"  {m:25s}  Δ={mean_d:+.4f}  CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.4g}{wtl_str}"
            )

    summary = "\n".join(pair_lines)
    (out_dir / "pairwise_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    # === LaTeX-friendly table for paper ===
    latex_lines = ["", "=" * 80, "LaTeX table (cross-model comparison)", "=" * 80]
    header = "Model & Token-F1 & ROUGE-L & Char-3 Jac. & Evid. & Tokens & Refusal \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    for model_name in preds_by_model:
        sub = df[df.model == model_name]
        row_str = f"{model_name} & "
        row_str += f"{sub['token_f1'].mean():.4f} & "
        row_str += f"{sub['rouge_l'].mean():.4f} & "
        row_str += f"{sub['char3_jaccard'].mean():.4f} & "
        row_str += f"{sub['evidence_token_recall'].mean():.4f} & "
        row_str += f"{sub['answer_tokens'].mean():.2f} & "
        row_str += f"{sub['refusal'].mean():.4f} \\\\"
        latex_lines.append(row_str)

    latex_table = "\n".join(latex_lines)
    (out_dir / "latex_table.txt").write_text(latex_table, encoding="utf-8")
    print(latex_table)


if __name__ == "__main__":
    main()
