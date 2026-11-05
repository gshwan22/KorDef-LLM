"""
score_semantic_eval323.py
==========================

Semantic re-scoring of existing model outputs on the source-grounded
evaluation set (N=323), added in revision to address reviewer requests
for semantic (beyond-lexical) evaluation.

Two metrics per item:
  1. BERTScore F1 (bert-base-multilingual-cased, raw, no baseline
     rescaling) -- the same encoder used in the earlier evaluation
     protocol of this project, kept for consistency.
  2. Embedding cosine similarity between the generated answer and the
     reference answer, using BAAI/bge-m3 (CLS pooling, L2-normalized).

No new inference is run: the script joins each model's existing
predictions (by eval_id) against the aligned evaluation file and
re-scores the stored answers.

Run on: DGX Spark
  cd /home/kisti_platform/kordef_peerj_eval
  source venv/bin/activate
  pip install scipy pandas   # if missing
  python score_semantic_eval323.py \
      --eval artifacts/eval_qa_329_aligned.jsonl \
      --outputs-dir outputs \
      --out-dir metrics_semantic

Outputs (under --out-dir):
  semantic_per_item.csv   one row per (model, eval_id)
  semantic_summary.txt    per-model means, deltas vs. the Gemma base,
                          paired Wilcoxon p-values, bootstrap 95% CIs
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Model registry: short name -> prediction file name.
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
BASE_MODEL = "gemma3_12b"   # paired-comparison reference

ANSWER_KEYS = ("answer", "prediction", "output", "generated")


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


def bootstrap_ci(values, n_boot=10000, seed=20261028):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (math.nan, math.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------
# Embedding cosine with BGE-M3 via transformers (CLS pooling).
# ---------------------------------------------------------------------
class BgeM3Scorer:
    def __init__(self, device):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.model = AutoModel.from_pretrained(
            "BAAI/bge-m3", torch_dtype=torch.float16
        ).to(device).eval()
        self.device = device

    def embed(self, texts, batch_size=32, max_length=512):
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt"
                           ).to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state[:, 0]  # CLS
                h = torch.nn.functional.normalize(h, dim=-1)
            out.append(h.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def cosine(self, answers, references, batch_size=32):
        ea = self.embed(answers, batch_size)
        er = self.embed(references, batch_size)
        return (ea * er).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True,
                    help="Aligned evaluation JSONL (reference answers).")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--out-dir", default="metrics_semantic")
    ap.add_argument("--models", nargs="*", default=list(MODEL_FILES),
                    help="Subset of model names to score.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-bertscore", action="store_true",
                    help="Score with BGE-M3 cosine only.")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- reference data ------------------------------------------------
    eval_rows = load_jsonl(args.eval)
    ref_by_id = {r["eval_id"]: r for r in eval_rows}
    print(f"[INFO] eval items: {len(ref_by_id)}")

    # --- BERTScore smoke test ------------------------------------------
    use_bertscore = not args.skip_bertscore
    if use_bertscore:
        try:
            from bert_score import score as bert_score_fn
            # Encoder smoke test with two Korean sample sentences
            # ("This is a test sentence." / "This is a trial sentence.").
            _p, _r, _f = bert_score_fn(
                ["테스트 문장입니다."], ["시험 문장입니다."],
                model_type="bert-base-multilingual-cased",
                num_layers=9, rescale_with_baseline=False,
                device=device, verbose=False,
            )
            print(f"[INFO] BERTScore smoke test OK (F1={float(_f[0]):.4f})")
        except Exception as e:
            print(f"[WARN] BERTScore unavailable ({type(e).__name__}: {e}).")
            print("[WARN] Continuing with BGE-M3 cosine only. "
                  "Re-run with --skip-bertscore to silence this warning.")
            use_bertscore = False

    bge = BgeM3Scorer(device)

    # --- score every model ---------------------------------------------
    records = []
    for name in args.models:
        fpath = Path(args.outputs_dir) / MODEL_FILES[name]
        if not fpath.exists():
            print(f"[WARN] missing predictions for {name}: {fpath} -- skipped")
            continue
        preds = load_jsonl(fpath)
        pairs = []
        for row in preds:
            eid = row.get("eval_id")
            if eid in ref_by_id:
                pairs.append((eid, get_answer(row).strip(),
                              str(ref_by_id[eid]["reference_answer"]).strip()))
        print(f"[INFO] {name}: {len(pairs)} joined items")
        ids = [p[0] for p in pairs]
        answers = [p[1] if p[1] else " " for p in pairs]
        refs = [p[2] for p in pairs]

        cos = bge.cosine(answers, refs, args.batch_size)

        if use_bertscore:
            from bert_score import score as bert_score_fn
            _, _, f1 = bert_score_fn(
                answers, refs,
                model_type="bert-base-multilingual-cased",
                num_layers=9, rescale_with_baseline=False,
                device=device, batch_size=args.batch_size, verbose=False,
            )
            f1 = f1.numpy()
        else:
            f1 = np.full(len(answers), np.nan)

        for eid, c, b in zip(ids, cos, f1):
            records.append({"model": name, "eval_id": eid,
                            "bge_m3_cosine": float(c),
                            "bertscore_f1": float(b)})
        print(f"[DONE] {name}: cosine mean={np.mean(cos):.4f}"
              + (f", BERTScore mean={np.nanmean(f1):.4f}" if use_bertscore else ""))

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "semantic_per_item.csv", index=False)

    # --- summary with paired stats vs. base ----------------------------
    from scipy.stats import wilcoxon
    metrics = ["bge_m3_cosine"] + (["bertscore_f1"] if use_bertscore else [])
    lines = [f"Semantic re-scoring summary (N per model shown below)", "=" * 70]
    base = df[df.model == BASE_MODEL].set_index("eval_id")

    for metric in metrics:
        lines.append(f"\n[{metric}]")
        for name in args.models:
            sub = df[df.model == name]
            if sub.empty:
                continue
            vals = sub[metric].dropna().values
            lo, hi = bootstrap_ci(vals)
            line = (f"  {name:10s} n={len(vals):3d} "
                    f"mean={np.mean(vals):.4f} 95CI=({lo:.4f},{hi:.4f})")
            if name != BASE_MODEL and not base.empty:
                merged = sub.set_index("eval_id")[[metric]].join(
                    base[[metric]], rsuffix="_base", how="inner").dropna()
                deltas = (merged[metric] - merged[f"{metric}_base"]).values
                dlo, dhi = bootstrap_ci(deltas)
                try:
                    p = 1.0 if np.allclose(deltas, 0) else wilcoxon(deltas).pvalue
                except Exception:
                    p = math.nan
                line += (f" | delta_vs_base={np.mean(deltas):+.4f} "
                         f"95CI=({dlo:+.4f},{dhi:+.4f}) wilcoxon_p={p:.3g}")
            lines.append(line)

    summary = "\n".join(lines)
    (out_dir / "semantic_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\n[DONE] wrote {out_dir}/semantic_per_item.csv and semantic_summary.txt")


if __name__ == "__main__":
    sys.exit(main())
