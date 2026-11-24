#!/usr/bin/env python3
"""
train_fft_v2.py
===============

Full-parameter SFT of Gemma-3-12B on the frozen KorDef instruction corpus
(kordef_train_sft.jsonl, 568,825 instances), replacing the original run
whose training record could not be matched to any preserved data file.

Every hyperparameter mirrors the original run as recorded in
hyperparams_extracted.json (lr 5e-7, cosine schedule with 6% warmup,
3 epochs, per-device batch 1 with 8 gradient-accumulation steps,
max sequence length 4,096, no packing, bf16, gradient checkpointing,
AdamW fused, weight decay 0, gradient clipping 0.1, seed 42).
Only the input data file changes.

Three audit mechanisms are built in:
  1. Dataset guard: training aborts below MIN_ROWS instances, and the
     SHA256 of the training file is recorded.
  2. Pre-flight arithmetic: expected optimizer steps are computed and
     written to preflight.json BEFORE training starts, so the
     steps-versus-corpus consistency is verified up front.
  3. Post-run audit: the final trainer state is checked against the
     pre-flight expectation and written to train_audit.json.

Launch (see runbook): accelerate launch with the accompanying FSDP config.
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

MIN_ROWS = 500_000
EXPECTED_ROWS = 568_825


def sha256_of(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Base model path or HF id (Gemma-3-12B instruct, "
                         "same base as the adapter runs)")
    ap.add_argument("--data", required=True,
                    help="Frozen training corpus (kordef_train_sft.jsonl)")
    ap.add_argument("--output-dir", required=True,
                    help="Checkpoint output directory")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="Gradient accumulation steps. Keep "
                         "1 x grad_accum x WORLD_SIZE = 128 so the "
                         "effective batch matches the original run "
                         "(16 GPUs: 8; 2 GPUs: 64).")
    args = ap.parse_args()

    set_seed(args.seed)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    # ── 1. Dataset guard ─────────────────────────────────────────
    ds = load_dataset("json", data_files=args.data, split="train")
    n = len(ds)
    if n < MIN_ROWS:
        sys.exit(f"[GUARD] dataset has {n:,} rows (< {MIN_ROWS:,}); "
                 f"refusing to train. Wrong file?")
    if rank == 0:
        digest = sha256_of(args.data)
        print(f"[GUARD] dataset rows      : {n:,} "
              f"(expected {EXPECTED_ROWS:,})")
        print(f"[GUARD] dataset sha256    : {digest}")
        if n != EXPECTED_ROWS:
            print(f"[GUARD] WARNING: row count differs from the frozen "
                  f"corpus ({EXPECTED_ROWS:,}); proceeding because it "
                  f"passes MIN_ROWS, but verify the file.")

    # ── 2. Pre-flight arithmetic ─────────────────────────────────
    per_device_bs = 1
    grad_accum = args.grad_accum
    eff_batch = per_device_bs * grad_accum * world
    if eff_batch != 128:
        sys.exit(f"[GUARD] effective batch {eff_batch} != 128; "
                 f"adjust --grad-accum so 1 x accum x WORLD_SIZE = 128.")
    per_rank_batches = math.ceil(n / (per_device_bs * world))
    steps_per_epoch = math.ceil(per_rank_batches / grad_accum)
    expected_total = steps_per_epoch * args.epochs
    if rank == 0:
        preflight = {
            "dataset_rows": n,
            "dataset_sha256": digest,
            "world_size": world,
            "per_device_train_batch_size": per_device_bs,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": eff_batch,
            "expected_steps_per_epoch": steps_per_epoch,
            "num_train_epochs": args.epochs,
            "expected_total_steps": expected_total,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "preflight.json"), "w") as f:
            json.dump(preflight, f, indent=2)
        print(f"[PREFLIGHT] world={world}  eff_batch={eff_batch}  "
              f"steps/epoch={steps_per_epoch:,}  "
              f"expected total={expected_total:,}")
        print("[PREFLIGHT] verify these numbers against the corpus "
              "before the run is allowed to continue unattended.")

    # ── 3. Model and tokenizer ───────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Gemma-3 recommended
    )
    model.config.use_cache = False

    # ── 4. Training configuration (mirror of the original run) ───
    import dataclasses
    valid = {f.name for f in dataclasses.fields(SFTConfig)}
    desired = dict(
        output_dir=args.output_dir,
        run_name=os.path.basename(args.output_dir.rstrip("/")),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=math.ceil(0.06 * expected_total),
        weight_decay=0.0,
        max_grad_norm=0.1,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",
        dataset_kwargs={"add_special_tokens": False,
                        "append_concat_token": True},
        dataset_num_proc=32,
        group_by_length=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        seed=args.seed,
        report_to="none",
    )
    renames = {"max_seq_length": "max_length"}
    kwargs, dropped = {}, []
    for k, v in desired.items():
        if k in valid:
            kwargs[k] = v
        elif k in renames and renames[k] in valid:
            kwargs[renames[k]] = v
        else:
            dropped.append(k)
    if rank == 0:
        print(f"[CONFIG] warmup_steps = {kwargs.get('warmup_steps')}")
        if dropped:
            print(f"[CONFIG] DROPPED (unsupported in this trl): {dropped}")
    critical = {"learning_rate", "num_train_epochs",
                "per_device_train_batch_size",
                "gradient_accumulation_steps", "lr_scheduler_type",
                "warmup_steps", "max_grad_norm", "bf16",
                "gradient_checkpointing", "packing", "seed"}
    missing = critical - set(kwargs)
    if missing:
        sys.exit(f"[CONFIG] critical args unsupported: {missing}")
    cfg = SFTConfig(**kwargs)

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    result = trainer.train()
    trainer.save_model(args.output_dir)
    if rank == 0:
        tokenizer.save_pretrained(args.output_dir)

    # ── 5. Post-run audit ────────────────────────────────────────
    if rank == 0:
        state = trainer.state
        audit = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "expected_total_steps": expected_total,
            "step_match": abs(state.global_step - expected_total) <= args.epochs,
            "train_runtime_sec": result.metrics.get("train_runtime"),
            "final_loss": result.metrics.get("train_loss"),
            "dataset_rows": n,
            "dataset_sha256": digest,
            "effective_batch_size": eff_batch,
        }
        with open(os.path.join(args.output_dir, "train_audit.json"), "w") as f:
            json.dump(audit, f, indent=2)
        print(f"[AUDIT] global_step={state.global_step:,} "
              f"(expected {expected_total:,})  "
              f"match={audit['step_match']}")
        if not audit["step_match"]:
            print("[AUDIT] WARNING: step count deviates from pre-flight "
                  "expectation; investigate before using this checkpoint.")


if __name__ == "__main__":
    main()
