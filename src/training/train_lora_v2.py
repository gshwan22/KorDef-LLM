#!/usr/bin/env python3
"""
train_lora_v2.py
=================

Retrains ALL parameter-efficient variants on the real 568,830-instance
instruction corpus, replacing the earlier LoRA runs and adding the two
configurations requested in review.

Variants (shared protocol: lr=2e-4 cosine, 6% warmup, 1 epoch,
per_device_batch=1, grad_accum=16, bf16 compute, max_seq_length=1024,
packing, gradient checkpointing, seed 42):

  lora_r8        r=8,  alpha=16,  attention-only (q,k,v,o)
  lora_r16       r=16, alpha=32,  attention-only
  lora_r64       r=64, alpha=128, attention-only
  qlora_r64      r=64, alpha=128, attention-only, 4-bit NF4 base
  lora_r64_all   r=64, alpha=128, all linear (attention + MLP)

Data: /home/user/data/kordef_train_sft.jsonl (from prepare_train_corpus.py).
A hard guard aborts if the dataset has fewer than 500,000 rows, so the
earlier silent wrong-data failure cannot recur.

Run on: GP24SV4 (see the two-GPU schedule in the chat).
Outputs: experiments/lora_comparison_v2/model_<LABEL>/
"""
import argparse
import json
import os
import time
from pathlib import Path

PROJECT_DIR = Path("/home/user/Desktop/TestG/langchain-kr/18-FineTuning/JournalDefense25")
OUTPUT_DIR = PROJECT_DIR / "experiments" / "lora_comparison_v2"
MODEL_BASE = "/home/user/models/gemma3-12b"
DATA_FILE = Path("/home/user/data/kordef_train_sft.jsonl")
MIN_ROWS = 500_000  # guard against silent wrong-data training

os.environ.setdefault("TMPDIR", "/home/user/tmp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.makedirs("/home/user/tmp", exist_ok=True)

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL = ATTN + ["gate_proj", "up_proj", "down_proj"]

VARIANTS = {
    "lora_r8":      {"label": "LoRA-r8",      "r": 8,  "alpha": 16,  "targets": ATTN, "q4": False},
    "lora_r16":     {"label": "LoRA-r16",     "r": 16, "alpha": 32,  "targets": ATTN, "q4": False},
    "lora_r64":     {"label": "LoRA-r64",     "r": 64, "alpha": 128, "targets": ATTN, "q4": False},
    "qlora_r64":    {"label": "QLoRA-r64",    "r": 64, "alpha": 128, "targets": ATTN, "q4": True},
    "lora_r64_all": {"label": "LoRA-r64-all", "r": 64, "alpha": 128, "targets": ALL,  "q4": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    args = ap.parse_args()
    cfg = VARIANTS[args.variant]
    label = cfg["label"]
    model_out = OUTPUT_DIR / f"model_{label}"

    if (model_out / "adapter_config.json").exists():
        print(f"[SKIP] {label} already trained at {model_out}")
        return
    if model_out.exists():
        import shutil
        shutil.rmtree(model_out)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- data guard ----------------------------------------------------
    assert DATA_FILE.exists(), f"missing training data: {DATA_FILE}"
    n_rows = sum(1 for _ in open(DATA_FILE, encoding="utf-8"))
    assert n_rows >= MIN_ROWS, (
        f"SANITY FAIL: training file has {n_rows} rows (< {MIN_ROWS}). "
        "Refusing to train on wrong data.")
    print(f"[INFO] variant={args.variant} label={label}")
    print(f"[INFO] data: {DATA_FILE} ({n_rows} rows)")
    print(f"[INFO] targets={cfg['targets']} 4bit={cfg['q4']}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["q4"]:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_BASE, quantization_config=bnb,
            device_map={"": 0}, trust_remote_code=True)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_BASE, torch_dtype=torch.bfloat16,
            device_map={"": 0}, trust_remote_code=True)

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM,
                      r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.05,
                      target_modules=cfg["targets"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    dataset = load_dataset("json", data_files=str(DATA_FILE), split="train")
    assert len(dataset) >= MIN_ROWS

    sft_args = SFTConfig(
        output_dir=str(model_out),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        bf16=True,
        logging_steps=100,
        save_strategy="no",
        seed=42,
        max_seq_length=1024,
        packing=True,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = SFTTrainer(model=model, args=sft_args,
                         train_dataset=dataset, processing_class=tokenizer)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    trainer.save_model(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    meta = {"label": label, "variant": args.variant,
            "r": cfg["r"], "alpha": cfg["alpha"],
            "target_modules": cfg["targets"], "quantize_4bit": cfg["q4"],
            "data_file": str(DATA_FILE), "data_rows": n_rows,
            "packing": True, "max_seq_length": 1024,
            "trainable_params": trainable, "total_params": total,
            "train_time_sec": train_time, "seed": 42}
    (model_out / "training_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[DONE] {label} in {train_time/3600:.1f} h -> {model_out}")


if __name__ == "__main__":
    main()
