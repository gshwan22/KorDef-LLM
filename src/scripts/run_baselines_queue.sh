#!/bin/bash
# Assumes execution from project root (or set KORDEF_ROOT)
KORDEF_ROOT="${KORDEF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$KORDEF_ROOT"
source venv/bin/activate

set -e

echo "=== Qwen2.5-7B 시작 $(date) ==="
python scripts/run_sourcegrounded_inference.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --input artifacts/eval_qa_329_aligned.jsonl \
    --output outputs/outputs_qwen25_sourcegrounded_eval323.jsonl \
    > logs/qwen25_inference.log 2>&1
echo "=== Qwen2.5-7B 완료 $(date) ==="

echo "=== Llama-3.1-8B 시작 $(date) ==="
python scripts/run_sourcegrounded_inference.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input artifacts/eval_qa_329_aligned.jsonl \
    --output outputs/outputs_llama31_sourcegrounded_eval323.jsonl \
    > logs/llama31_inference.log 2>&1
echo "=== Llama-3.1-8B 완료 $(date) ==="

echo "=== A.X-4.0-Light 시작 $(date) ==="
python scripts/run_sourcegrounded_inference.py \
    --model skt/A.X-4.0-Light \
    --input artifacts/eval_qa_329_aligned.jsonl \
    --output outputs/outputs_axlight_sourcegrounded_eval323.jsonl \
    > logs/axlight_inference.log 2>&1
echo "=== A.X-4.0-Light 완료 $(date) ==="

echo "=== ALL DONE $(date) ==="
