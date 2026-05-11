# KorDef-LLM

**An Open Pipeline for Domain-Adaptive Instruction Tuning of Korean Defense Large Language Models**

This repository accompanies the paper *"An Open Pipeline for Domain-Adaptive Instruction Tuning of Korean Defense Large Language Models"* (submitted to PeerJ Computer Science). It contains the inference, evaluation, and analysis code used to reproduce the experiments in the paper.

---

## Released Artifacts

| Component | Location | Notes |
|---|---|---|
| **Corpus and evaluation data** | [Zenodo `10.5281/zenodo.20083055`](https://doi.org/10.5281/zenodo.20083055) | Prompt-generated subset (235,367 QA pairs), source-document manifest (2,540 PDFs), cleaned text segments (34,472 segments), source-grounded evaluation set (N=323), train/eval overlap audit report |
| **Trained model weights** | [HuggingFace `jeong0313/koni-it-m-7875`](https://huggingface.co/jeong0313/koni-it-m-7875) | KorDef-LLM, 12B parameters, BF16, safetensors |
| **Code (this repo)** | [GitHub `gshwan22/KorDef-LLM`](https://github.com/gshwan22/KorDef-LLM) | Inference, scoring, and figure-generation code |

---

## Repository Structure

```
KorDef-LLM/
├── src/
│   ├── inference/
│   │   ├── run_sourcegrounded_inference.py   # Source-grounded QA inference (any HF model)
│   │   └── run_closedbook_inference.py        # Closed-book QA inference (any HF model)
│   ├── eval/
│   │   ├── score_sourcegrounded_eval323.py    # KorDef vs. Gemma paired comparison
│   │   └── score_multi_model.py               # Cross-model comparison (any set of models)
│   ├── figures/
│   │   ├── gen_eval323_figures.py             # Source-grounded evaluation figures
│   │   └── make_cross_model_figure.py         # Cross-model bootstrap CI figure
│   └── scripts/
│       ├── run_baselines_queue.sh             # Sequential baseline inference queue
│       └── run_kmmlu_extra.sh                 # KMMLU 5-shot evaluation queue
├── configs/
│   └── (training/inference configuration files)
├── requirements.txt                           # Pinned Python dependencies
├── LICENSE                                    # MIT
└── README.md                                  # This file
```

---

## Quick Start (Reproduce Source-Grounded Evaluation)

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the evaluation set

```bash
mkdir -p artifacts
# Download from Zenodo
wget https://zenodo.org/records/20083055/files/kordef_corpus_v1.0.tar.gz
tar -xzf kordef_corpus_v1.0.tar.gz
mv kordef_corpus_v1.0/eval_qa_329_final.jsonl artifacts/eval_qa_329_aligned.jsonl
```

### 3. Run source-grounded inference for KorDef-LLM

```bash
mkdir -p outputs logs
python src/inference/run_sourcegrounded_inference.py \
    --model jeong0313/koni-it-m-7875 \
    --input artifacts/eval_qa_329_aligned.jsonl \
    --output outputs/outputs_kordef_sourcegrounded_eval323.jsonl
```

### 4. Score and compare against the base model

```bash
# Also run Gemma-3-12B (the base model)
python src/inference/run_sourcegrounded_inference.py \
    --model google/gemma-3-12b-it \
    --input artifacts/eval_qa_329_aligned.jsonl \
    --output outputs/outputs_gemma3_12b_sourcegrounded_eval323.jsonl

# Paired comparison (Table 6 in the paper)
python src/eval/score_sourcegrounded_eval323.py
```

### 5. Reproduce cross-model comparison (Table 7, Figure 5)

```bash
# Run inference for additional baselines (sequential queue)
bash src/scripts/run_baselines_queue.sh

# Compute multi-model metrics
python src/eval/score_multi_model.py \
    --eval artifacts/eval_qa_329_aligned.jsonl \
    --baseline gemma3_12b \
    --output-dir outputs/metrics_multi \
    --models \
        gemma3_12b:outputs/outputs_gemma3_12b_sourcegrounded_eval323.jsonl \
        kordef:outputs/outputs_kordef_sourcegrounded_eval323.jsonl \
        exaone35:outputs/outputs_exaone35_sourcegrounded_eval323.jsonl \
        qwen25:outputs/outputs_qwen25_sourcegrounded_eval323.jsonl \
        llama31:outputs/outputs_llama31_sourcegrounded_eval323.jsonl \
        axlight:outputs/outputs_axlight_sourcegrounded_eval323.jsonl

# Cross-model figure with bootstrap CIs (Figure 5)
python src/figures/make_cross_model_figure.py
```

### 6. Reproduce KMMLU benchmark (Table 5)

```bash
# Requires lm-evaluation-harness >= 0.4.11
pip install lm-eval

# Edit run_kmmlu_extra.sh to add the models you want (or use as-is for EXAONE + A.X)
bash src/scripts/run_kmmlu_extra.sh
```

---

## Hardware Requirements

The experiments in the paper were run on the following hardware:

- **Source-grounded inference (per model, N=323)**: ~25–45 minutes on a single GPU with ≥16GB VRAM (BF16). Tested on NVIDIA GB10 (DGX Spark, 128GB unified memory).
- **KMMLU 5-shot evaluation (per model)**: ~1–2 hours on a single GPU.
- **Training of KorDef-LLM (12B)**: FSDP distributed training; please consult the model card on HuggingFace and the paper for details.

For models larger than ~24 GB in BF16, use `device_map={"": 0}` (single GPU) or appropriate distributed configuration. Note that `device_map="auto"` may unnecessarily offload weights to CPU and severely degrade inference speed on unified-memory systems.

---

## Citation

If you use this code or the released artifacts, please cite the paper:

```bibtex
@article{gwak2026kordef,
  title   = {An Open Pipeline for Domain-Adaptive Instruction Tuning of Korean Defense Large Language Models},
  author  = {Gwak, Sang-Hwan and Choi, Ji-Young and Jeong, Chang-Hoo and Lee, Gunwoo and Kim, Ina and Lee, Kyung-Ha},
  journal = {PeerJ Computer Science (submitted)},
  year    = {2026}
}
```

And the dataset:

```bibtex
@dataset{kordef_corpus_2026,
  title     = {KorDef-LLM: Korean Defense Domain Instruction Corpus and Source-Grounded Evaluation Set},
  author    = {Gwak, Sang-Hwan and others},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20083055}
}
```

---

## License

- **Code (this repository)**: MIT License (see `LICENSE`)
- **Released dataset (Zenodo)**: CC-BY-4.0
- **Released model (HuggingFace)**: Gemma Terms of Use (the model is fine-tuned from Gemma-3-12B; please consult the [Gemma Terms](https://ai.google.dev/gemma/terms))

---

## Intended Use and Limitations

KorDef-LLM is released for **research on Korean professional-domain language modeling and educational/reference-style question answering**. It is **not** intended for autonomous decision-making in military operations, procurement, maintenance, targeting, or safety-critical procedures without institutional review, retrieval grounding, and human expert oversight. See the paper's "Ethical and Safety Considerations" section and the HuggingFace model card for full out-of-scope use guidance.

---

## Contact

For questions about this code or the paper, please open an issue on GitHub or contact the corresponding author at `kyongha@kisti.re.kr`.

