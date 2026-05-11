#!/bin/bash
# Assumes execution from project root (or set KORDEF_ROOT)
KORDEF_ROOT="${KORDEF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$KORDEF_ROOT"
source venv/bin/activate

echo "=== EXAONE-3.5-7.8B KMMLU START $(date) ==="
lm_eval --model hf \
    --model_args "pretrained=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct,trust_remote_code=True,dtype=bfloat16" \
    --tasks kmmlu \
    --num_fewshot 5 \
    --output_path results/kmmlu/exaone35 \
    --batch_size 8 \
    2>&1 | tee logs/kmmlu_exaone35.log
echo "=== EXAONE DONE $(date) ==="

echo "=== A.X-4.0-Light KMMLU START $(date) ==="
lm_eval --model hf \
    --model_args "pretrained=skt/A.X-4.0-Light,trust_remote_code=True,dtype=bfloat16" \
    --tasks kmmlu \
    --num_fewshot 5 \
    --output_path results/kmmlu/axlight \
    --batch_size 8 \
    2>&1 | tee logs/kmmlu_axlight.log
echo "=== A.X DONE $(date) ==="

echo "=== ALL KMMLU $(date) ==="
