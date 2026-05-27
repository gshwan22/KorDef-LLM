"""
score_sourcegrounded_eval323.py
================================

Pairwise scoring for the Korean defense source-grounded QA evaluation
set (N=323). Compares the Gemma-3-12B baseline against KorDef-LLM
(fine-tuned from Gemma-3-12B on the Korean defense instruction corpus).

The script computes, per evaluation item and per model:
    * Token-level F1 against the reference answer
    * ROUGE-L F1 (token-level LCS) against the reference answer
    * Character 3-gram Jaccard similarity against the reference answer
    * Evidence-token recall: fraction of reference evidence tokens
      that appear in the model answer (proxy for source faithfulness)
    * Answer length (characters and tokens)
    * Refusal flag (1 if the answer matches a known Korean refusal
      pattern, 0 otherwise)

It then aggregates the per-item metrics into per-model means with 95%
bootstrap confidence intervals and reports paired Wilcoxon signed-rank
tests for the KorDef minus Gemma deltas.

Note on language
----------------
The refusal patterns are Korean phrases that the models emit when they
decline to answer. They are *not* code comments; they are functional
data and must remain in Korean for the matching to work. English
translations are provided inline.

Outputs
-------
Three files in ``outputs/metrics_eval323/``:
    1. per_model_metrics.csv   -- one row per (eval_id, model)
    2. paired_metrics.csv      -- one row per eval_id with both models'
                                  scores and deltas
    3. summary_metrics.txt     -- human-readable summary with bootstrap
                                  CIs, Wilcoxon p-values, and
                                  per-answer-type breakdown
Plus ``win_loss_examples.jsonl`` with the top-20 items where each
model wins by token-F1, for qualitative inspection.
"""

import json
import re
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# Hard-coded input/output paths. Edit these to point to your local
# prediction files produced by run_sourcegrounded_inference.py.
GEMMA = "outputs/outputs_gemma3_12b_sourcegrounded_eval323.jsonl"
KORDEF = "outputs/outputs_kordef_sourcegrounded_eval323.jsonl"
OUT_DIR = Path("outputs/metrics_eval323")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Korean refusal patterns. Translations:
#   "알 수 없습니다"          -> "I cannot determine."
#   "정보 부족"               -> "Insufficient information."
#   "확인할 수 없습니다"      -> "I cannot confirm."
#   "명시되어 있지 않습니다"  -> "It is not specified."
#   "제공되지 않습니다"       -> "Not provided."
#   "문맥에 없습니다"         -> "Not in the context."
REFUSAL_PATTERNS = [
    "알 수 없습니다",
    "정보 부족",
    "확인할 수 없습니다",
    "명시되어 있지 않습니다",
    "제공되지 않습니다",
    "문맥에 없습니다",
]


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(s):
    """Trim, collapse whitespace, and cast to str."""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def korean_tokens(s):
    """Tokenize a mixed Korean/English/numeric string.

    The regular expression intentionally separates Korean syllable
    blocks, ASCII alphabetic runs, numeric tokens, and any other
    non-whitespace character (which captures punctuation). The
    tokenizer is deliberately simple so that the metric is
    reproducible without external Korean morphological tools.
    """
    s = normalize_text(s)
    return re.findall(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]", s)


def token_f1(pred, ref):
    """Token-level F1 between predicted and reference answer strings."""
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
    """Token-level longest common subsequence length (rolling-array DP)."""
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
    """ROUGE-L F1 (LCS-based) between predicted and reference answers."""
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
    """Return the set of character n-grams in a whitespace-stripped string."""
    s = re.sub(r"\s+", "", str(s))
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s)-n+1)}


def char_jaccard(pred, ref, n=3):
    """Character n-gram Jaccard similarity, useful for agglutinative Korean."""
    a = char_ngrams(pred, n)
    b = char_ngrams(ref, n)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evidence_overlap(answer, evidence):
    """Fraction of reference-evidence tokens (length >= 2) recovered in
    the model answer. Used as a proxy for source faithfulness."""
    at = set(korean_tokens(answer))
    et = set(korean_tokens(evidence))
    et = {x for x in et if len(x) >= 2}
    if not et:
        return 0.0
    return len(at & et) / len(et)


def is_refusal(answer):
    """Return True if the answer contains any known Korean refusal phrase."""
    return any(p in answer for p in REFUSAL_PATTERNS)


def bootstrap_ci(values, n_boot=5000, seed=20260508):
    """Percentile bootstrap 95% confidence interval for the mean."""
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


# ---------------------------------------------------------------------
# Main: load both prediction files, align by eval_id, score per item,
# aggregate, and write outputs.
# ---------------------------------------------------------------------
gemma = load_jsonl(GEMMA)
kordef = load_jsonl(KORDEF)

g_by_id = {r["eval_id"]: r for r in gemma}
k_by_id = {r["eval_id"]: r for r in kordef}
ids = sorted(set(g_by_id) & set(k_by_id))

if len(ids) != len(gemma) or len(ids) != len(kordef):
    print("[WARN] ID mismatch")
    print("gemma:", len(gemma), "kordef:", len(kordef), "common:", len(ids))

records = []
for eid in ids:
    g = g_by_id[eid]
    k = k_by_id[eid]

    ref = g["reference_answer"]
    evidence = g.get("answer_evidence", "")
    q = g["question"]

    for model_name, row in [("gemma3_12b", g), ("kordef", k)]:
        ans = row.get("answer", "")
        records.append({
            "eval_id": eid,
            "model": model_name,
            "answer_type": row.get("answer_type"),
            "question": q,
            "reference_answer": ref,
            "answer": ans,
            "token_f1": token_f1(ans, ref),
            "rouge_l": rouge_l_f1(ans, ref),
            "char3_jaccard": char_jaccard(ans, ref, 3),
            "evidence_token_recall": evidence_overlap(ans, evidence),
            "answer_chars": len(ans),
            "answer_tokens": len(korean_tokens(ans)),
            "refusal": int(is_refusal(ans)),
        })

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "per_model_metrics.csv", index=False)

# Build a wide-format table with one row per eval_id and both models'
# scores side by side, plus the KorDef-minus-Gemma deltas.
wide_rows = []
for eid in ids:
    gr = df[(df.eval_id == eid) & (df.model == "gemma3_12b")].iloc[0]
    kr = df[(df.eval_id == eid) & (df.model == "kordef")].iloc[0]
    item = {
        "eval_id": eid,
        "answer_type": gr["answer_type"],
    }
    for metric in ["token_f1", "rouge_l", "char3_jaccard",
                   "evidence_token_recall", "answer_tokens", "refusal"]:
        item[f"gemma_{metric}"] = gr[metric]
        item[f"kordef_{metric}"] = kr[metric]
        item[f"delta_{metric}"] = kr[metric] - gr[metric]
    wide_rows.append(item)

wide = pd.DataFrame(wide_rows)
wide.to_csv(OUT_DIR / "paired_metrics.csv", index=False)

# Build a human-readable summary with bootstrap CIs and Wilcoxon tests.
summary_lines = []
summary_lines.append("KorDef source-grounded eval323 metrics")
summary_lines.append("=" * 70)
summary_lines.append(f"n_items: {len(ids)}")
summary_lines.append("")

for metric in ["token_f1", "rouge_l", "char3_jaccard",
               "evidence_token_recall", "answer_tokens", "refusal"]:
    summary_lines.append(f"[{metric}]")
    for model_name in ["gemma3_12b", "kordef"]:
        vals = df[df.model == model_name][metric].astype(float).values
        ci = bootstrap_ci(vals)
        summary_lines.append(
            f"  {model_name}: mean={np.mean(vals):.4f}, sd={np.std(vals):.4f}, "
            f"95CI=({ci[0]:.4f}, {ci[1]:.4f})"
        )

    deltas = wide[f"delta_{metric}"].astype(float).values
    ci = bootstrap_ci(deltas)
    try:
        if np.allclose(deltas, 0):
            p = 1.0
        else:
            p = wilcoxon(deltas).pvalue
    except Exception:
        p = math.nan

    summary_lines.append(
        f"  delta(kordef-gemma): mean={np.mean(deltas):.4f}, "
        f"95CI=({ci[0]:.4f}, {ci[1]:.4f}), wilcoxon_p={p:.6g}"
    )

    if metric != "refusal":
        wins = int(np.sum(deltas > 0))
        ties = int(np.sum(deltas == 0))
        losses = int(np.sum(deltas < 0))
        summary_lines.append(f"  wins/ties/losses for KorDef: {wins}/{ties}/{losses}")
    summary_lines.append("")

summary_lines.append("[By answer_type: token_f1 mean]")
for answer_type, sub in df.groupby("answer_type"):
    gmean = sub[sub.model == "gemma3_12b"]["token_f1"].mean()
    kmean = sub[sub.model == "kordef"]["token_f1"].mean()
    n = sub[sub.model == "gemma3_12b"].shape[0]
    summary_lines.append(
        f"  {answer_type}: n={n}, gemma={gmean:.4f}, "
        f"kordef={kmean:.4f}, delta={kmean-gmean:.4f}"
    )

summary = "\n".join(summary_lines)
(OUT_DIR / "summary_metrics.txt").write_text(summary, encoding="utf-8")
print(summary)

# Save the top-20 items where each model wins most by token-F1, for
# qualitative inspection by reviewers.
wide_sorted_k = wide.sort_values("delta_token_f1", ascending=False).head(20)
wide_sorted_g = wide.sort_values("delta_token_f1", ascending=True).head(20)

examples = []
for label, rows in [("kordef_wins", wide_sorted_k), ("gemma_wins", wide_sorted_g)]:
    for _, r in rows.iterrows():
        eid = r["eval_id"]
        g = g_by_id[eid]
        k = k_by_id[eid]
        examples.append({
            "group": label,
            "eval_id": eid,
            "answer_type": g.get("answer_type"),
            "question": g["question"],
            "reference_answer": g["reference_answer"],
            "gemma_answer": g.get("answer", ""),
            "kordef_answer": k.get("answer", ""),
            "delta_token_f1": float(r["delta_token_f1"]),
            "delta_rouge_l": float(r["delta_rouge_l"]),
            "delta_evidence_token_recall": float(r["delta_evidence_token_recall"]),
        })

with open(OUT_DIR / "win_loss_examples.jsonl", "w", encoding="utf-8") as out:
    for e in examples:
        out.write(json.dumps(e, ensure_ascii=False) + "\n")
