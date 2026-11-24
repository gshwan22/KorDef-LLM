#!/usr/bin/env python3
"""Full-corpus train/eval overlap audit.
Checks all 568,825 serialized training texts against the 323 eval items:
exact question substring, exact answer substring, near-question
(>=80% of the question's character 5-grams contained in a training text)."""
import json, sys

EVAL = '/home/user/data/eval_qa_329_aligned.jsonl'
TRAIN = '/home/user/data/kordef_train_sft.jsonl'

items = [json.loads(l) for l in open(EVAL, encoding='utf-8') if l.strip()]
qk = next(k for k in ('source_grounded_question',) if k in items[0])
ak = next(k for k in ('reference_answer',) if k in items[0])
print(f"eval items: {len(items)}  (question key='{qk}', answer key='{ak}')", flush=True)

def grams(t, n=5):
    t = ''.join(t.split())
    return set(t[i:i+n] for i in range(max(len(t)-n+1, 1)))

qs = [''.join(it[qk].split()) for it in items]
answers = [''.join(it[ak].split()) for it in items]
qg = [grams(it[qk]) for it in items]
gram2items = {}
for i, g in enumerate(qg):
    for x in g:
        gram2items.setdefault(x, []).append(i)

exact_q = set(); exact_a = set(); near_q = set()
n_lines = 0
for line in open(TRAIN, encoding='utf-8'):
    n_lines += 1
    try:
        t = json.loads(line)['text']
    except Exception:
        continue
    tn = ''.join(t.split())
    for i, q in enumerate(qs):
        if q and q in tn: exact_q.add(i)
    for i, a in enumerate(answers):
        if a and a in tn: exact_a.add(i)
    hits = {}
    tg = set(tn[j:j+5] for j in range(max(len(tn)-4, 1)))
    for x in tg:
        for i in gram2items.get(x, ()):
            hits[i] = hits.get(i, 0) + 1
    for i, h in hits.items():
        if h >= 0.8 * len(qg[i]): near_q.add(i)
    if n_lines % 100000 == 0:
        print(f"  {n_lines:,} lines... exactQ={len(exact_q)} exactA={len(exact_a)} nearQ={len(near_q)}", flush=True)

print(f"\n=== FULL-CORPUS OVERLAP AUDIT ({n_lines:,} training texts vs {len(items)} eval items) ===")
print(f"exact question overlaps : {len(exact_q)}  {sorted(exact_q)[:10]}")
print(f"exact answer overlaps   : {len(exact_a)}  {sorted(exact_a)[:10]}")
print(f"near-question (>=80% 5-gram containment): {len(near_q)}  {sorted(near_q)[:10]}")
