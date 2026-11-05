#!/usr/bin/env python3
"""
prepare_train_corpus.py
========================

Converts the released full instruction corpus (kordef_full_corpus_v1.0
.jsonl, 568,830 instances) into the SFT training format used for LoRA
retraining: one {"text": ...} record per instance, with the Gemma chat
template applied and the same Korean defense-expert system framing
described in the paper's Methods.

Run on: GP24SV4
  python prepare_train_corpus.py
Output:
  /home/user/data/kordef_train_sft.jsonl   (+ a .stats.json sidecar)
"""
import json
from pathlib import Path

SRC = Path("/home/user/data/kordef_full_corpus_v1.0.jsonl")
DST = Path("/home/user/data/kordef_train_sft.jsonl")
MODEL_BASE = "/home/user/models/gemma3-12b"

# System prompt attached to every training instance (Korean, functional).
# English translation: "You are an expert in the Republic of Korea defense
# and military domain. Answer concisely in military-manual style."
SYSTEM_MSG = "당신은 대한민국 국방/군사 분야 전문가입니다. 군사 교범 스타일로 간결하게 답변하십시오."


def extract_qa(row):
    q = row.get("question") or row.get("instruction") or row.get("q")
    a = row.get("answer") or row.get("response") or row.get("output") or row.get("a")
    if not q or not a:
        return None
    q, a = str(q).strip(), str(a).strip()
    if not q or not a:
        return None
    return q, a


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    assert tok.chat_template, "tokenizer has no chat template"

    n_in = n_out = n_skip = 0
    src_counter = {}
    with open(SRC, encoding="utf-8") as fin, open(DST, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            qa = extract_qa(row)
            if qa is None:
                n_skip += 1
                continue
            q, a = qa
            # Gemma chat template has no separate system role; prepend the
            # expert framing to the user turn, matching inference usage.
            text = tok.apply_chat_template(
                [{"role": "user", "content": f"{SYSTEM_MSG}\n\n{q}"},
                 {"role": "assistant", "content": a}],
                tokenize=False,
            )
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_out += 1
            s = row.get("_source", row.get("metadata", {}).get("source", "unknown"))
            src_counter[s] = src_counter.get(s, 0) + 1
            if n_in % 100000 == 0:
                print(f"  {n_in} read / {n_out} written")

    stats = {"input_rows": n_in, "output_rows": n_out, "skipped": n_skip,
             "by_source": src_counter}
    Path(str(DST) + ".stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    assert n_out >= 500_000, f"SANITY FAIL: only {n_out} training rows"
    print(f"[DONE] {DST} ({n_out} rows)")


if __name__ == "__main__":
    main()
