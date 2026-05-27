# KorDef-LLM: Korean Defense Large Language Model

**Open pipeline for domain-adaptive instruction tuning of Korean defense large language models**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20083055.svg)](https://doi.org/10.5281/zenodo.20083055)
[![HuggingFace](https://img.shields.io/badge/HF-graphuser%2Fkordef--12b-yellow)](https://huggingface.co/graphuser/kordef-12b)

---

## 1. Description

KorDef-LLM is a Korean defense-domain large language model and accompanying **open, reproducible pipeline** for domain-adaptive instruction tuning of Korean LLMs in regulated professional domains. The project releases:

- A **~569K-pair Korean defense instruction corpus** constructed from publicly available, unclassified Korean Ministry of National Defense administrative-rule documents and educational materials,
- A **fine-tuned 12B-parameter model** (KorDef-LLM, based on Gemma-3-12B) and three LoRA-adapter variants (r=8, 16, 64),
- A **source-grounded evaluation framework** (N=323 items) measuring lexical overlap, evidence-token recall, answer style, and refusal behaviour,
- Six-model cross-comparison results against open Korean and multilingual baselines (EXAONE-3.5-7.8B, Qwen2.5-7B, Llama-3.1-8B, A.X-4.0-Light, plus the Gemma-3-12B base),
- A **network-analytic corpus diagnostic framework** for assessing domain coverage and cross-domain mixing of instruction corpora.

This repository accompanies the manuscript:

> Gwak, S.-H., Choi, J.-Y., Jeong, C.-H., Kim, I., Lee, K.-H., & Lee, G. (2026). *An open pipeline for domain-adaptive instruction tuning of Korean defense large language models.* PeerJ Computer Science (under review).

---

## 2. Dataset Information

### 2.1 Corpus

| Item | Value |
|---|---|
| Source documents | publicly available unclassified Korean Ministry of National Defense administrative-rule documents and educational materials |
| Final instruction pairs | ~569,000 |
| Mean instruction length | 28.4 tokens |
| Mean response length | 41.8 tokens |
| Language | Korean |
| Domain coverage | 12 defense sub-domains |

### 2.2 Evaluation set

| Item | Value |
|---|---|
| Source-grounded eval | 323 items, context-question-answer triples with evidence spans |
| Closed-book eval | 500 items, paired with source-grounded subset |
| Korean general-domain probe | KMMLU (35,030 questions, 5-shot) |

### 2.3 Availability

- **Archival (DOI-cited)**: https://doi.org/10.5281/zenodo.20083055
- **GitHub mirror**: https://github.com/gshwan22/KorDef-LLM

All released artifacts are publicly available, contain no classified or personally identifiable information, and are released under CC BY 4.0 (corpus) and MIT (code).

---

## 3. Code Information

### 3.1 Repository structure

```
KorDef-LLM/
├── README.md                          # This file
├── LICENSE                            # MIT for code, CC BY 4.0 for data
├── requirements.txt                   # Python dependencies
├── data/
│   ├── corpus_manifest.csv            # Source document manifest (Gebru-style datasheet)
│   ├── eval_qa_329_aligned.jsonl      # Source-grounded evaluation set (N=323)
│   └── README.md                      # Dataset-specific notes
├── training/
│   ├── train_full_sft.py              # Full supervised fine-tuning
│   ├── train_lora.py                  # LoRA fine-tuning (r=8, 16, 64)
│   └── configs/                       # Training hyperparameter YAMLs
├── inference/
│   ├── run_sourcegrounded_inference.py   # Source-grounded QA inference
│   ├── run_closedbook_inference.py       # Closed-book QA inference
│   └── run_lora_inference.py             # PEFT-based LoRA inference
├── evaluation/
│   ├── score_sourcegrounded_eval323.py   # Compute Tok-F1, ROUGE-L, Char-3 Jaccard,
│   │                                     # evidence recall, length, refusal
│   ├── score_multi_model.py              # Cross-model bootstrap CIs + Wilcoxon
│   └── score_kmmlu.py                    # KMMLU 5-shot wrapper
├── corpus_diagnostics/
│   ├── build_keyword_network.py          # Keyword co-occurrence network
│   ├── compute_domain_entropy.py         # Instance-level domain entropy
│   └── cross_domain_edges.py             # Cross-domain edge fraction
└── figures/
    └── generate_figures_pub_v6.py        # Publication figure generation
```

### 3.2 Trained model

- **HuggingFace**: https://huggingface.co/graphuser/kordef-12b
- Format: `safetensors`, fp16
- Base: `google/gemma-3-12b-it`
- License: Gemma-3 community license

---

## 4. Usage Instructions

### 4.1 Load the model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "graphuser/kordef-12b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Question: According to the following regulation, explain ..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 4.2 Reproduce source-grounded evaluation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the evaluation set from Zenodo
mkdir -p data
wget -O data/eval_qa_329_aligned.jsonl \
    "https://zenodo.org/records/20083055/files/eval_qa_329_aligned.jsonl"

# 3. Run inference for a target model
python inference/run_sourcegrounded_inference.py \
    --model_id graphuser/kordef-12b \
    --eval_file data/eval_qa_329_aligned.jsonl \
    --output_file outputs/kordef_predictions.jsonl

# 4. Compute metrics
python evaluation/score_sourcegrounded_eval323.py \
    --predictions outputs/kordef_predictions.jsonl \
    --reference data/eval_qa_329_aligned.jsonl \
    --output outputs/kordef_scores.json
```

### 4.3 Reproduce cross-model comparison (Table 7 / Figure 6)

```bash
python evaluation/score_multi_model.py \
    --models gemma-3-12b-it exaone-3.5-7.8b-instruct qwen2.5-7b-instruct \
             llama-3.1-8b-instruct a.x-4.0-light kordef-12b \
    --eval_file data/eval_qa_329_aligned.jsonl \
    --bootstrap_iters 1000 \
    --output_dir outputs/cross_model/
```

### 4.4 LoRA inference (r=8, 16, 64)

```bash
python inference/run_lora_inference.py \
    --base_model google/gemma-3-12b-it \
    --lora_adapter ./experiments/lora_comparison/model_LoRA-r64 \
    --eval_file data/eval_qa_329_aligned.jsonl \
    --output_file outputs/lora_r64_predictions.jsonl
```

---

## 5. Requirements

### 5.1 Hardware

| Use case | Minimum | Recommended |
|---|---|---|
| Inference (fp16) | 1 GPU with 24 GB VRAM | 1 A6000 / A100 (48-80 GB) |
| LoRA training | 1 GPU with 48 GB VRAM | 2 A6000 |
| Full SFT | 4 A100 (80 GB) | 8 H100 |

### 5.2 Software

- Python >= 3.9
- CUDA >= 12.1

Key dependencies (see `requirements.txt`):

```
torch>=2.8.0
transformers>=4.57.0
peft>=0.14.0
accelerate>=1.0.0
datasets>=2.20.0
bitsandbytes>=0.43.0   # optional, for 4-bit LoRA
trl>=0.10.0
evaluate
rouge-score
sacrebleu
scipy
numpy
pandas
matplotlib
networkx
```

### 5.3 Special model-specific notes

- **Gemma-3**: requires `token_type_ids` patch (see `inference/utils/gemma_patch.py`).
- **EXAONE-3.5**: requires `trust_remote_code=True`.

---

## 6. Methodology

### 6.1 Corpus construction

1. **Source collection**: Crawl publicly available unclassified Ministry of National Defense administrative-rule documents and educational materials.
2. **Preprocessing**: PDF parsing, deduplication, removal of front matter and tables of contents, paragraph-level segmentation.
3. **Instruction synthesis**: Two-stage source-grounded instruction generation using a teacher LLM, with quality filtering (length thresholds, repetition filters, language ID).
4. **Network-analytic diagnostics**: Keyword co-occurrence network construction, modularity-based community detection, cross-domain edge fraction, instance-level domain entropy.

### 6.2 Fine-tuning

1. **Full SFT**: Supervised fine-tuning of Gemma-3-12B on the full ~569K-pair corpus using paged AdamW, cosine LR schedule, packed sequences, max length 4096.
2. **LoRA variants**: rank r in {8, 16, 64}, alpha = 2r, dropout 0.05, target modules = all linear projections in attention and MLP blocks.

### 6.3 Evaluation

1. **Source-grounded QA** (N=323): Provide context + question, measure how faithfully the model reproduces evidence-span tokens from the reference. Metrics: Token-F1, ROUGE-L, Character 3-gram Jaccard, evidence-token recall, mean answer length, refusal rate.
2. **Statistical significance**: Paired Wilcoxon signed-rank tests vs Gemma-3-12B base, with Bonferroni correction over six metrics. 95% bootstrap CIs (1000 resamples) for cross-model comparison.
3. **Korean general-domain probe**: KMMLU 5-shot via the standardized `lm-evaluation-harness`.

---

## 7. Citation

If you use this code, dataset, or model, please cite:

```bibtex
@article{gwak2026kordef,
  title   = {An open pipeline for domain-adaptive instruction tuning of
             Korean defense large language models},
  author  = {Gwak, Sang-Hwan and Choi, Ji-Young and Jeong, Chang-Hoo
             and Kim, Ina and Lee, Kyung-Ha and Lee, Gunwoo},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Under review}
}

@dataset{gwak2026kordef_zenodo,
  title     = {KorDef-LLM: corpus, evaluation set, and trained model weights},
  author    = {Gwak, Sang-Hwan and Choi, Ji-Young and Jeong, Chang-Hoo
               and Kim, Ina and Lee, Kyung-Ha and Lee, Gunwoo},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.20083055},
  url       = {https://doi.org/10.5281/zenodo.20083055}
}
```

---

## 8. License & Contribution Guidelines

### 8.1 License

- **Code**: MIT License (see `LICENSE` file).
- **Corpus and evaluation set**: CC BY 4.0.
- **Model weights**: Gemma-3 community license.

### 8.2 Contributions

Contributions are welcome. Please:

1. Open an issue describing the proposed change before submitting a pull request.
2. For bug fixes, include a minimal reproducible example.
3. For new evaluation metrics or extended benchmarks, include unit tests and update this README.
4. Follow PEP 8 and run `ruff check` before submitting.

### 8.3 Disclaimer

KorDef-LLM is a research artifact. It is trained on publicly available unclassified administrative-rule documents, and is not intended for operational, classified, or mission-critical use. Outputs should be reviewed by qualified human experts before being acted upon.

---

## 9. Contact

For questions about the code or dataset:

- **Corresponding author**: Gunwoo Lee -- `gwlee@kisti.re.kr`
- **First author**: Sang-Hwan Gwak -- Korea Institute of Science and Technology Information (KISTI)

For issues with the model on HuggingFace, please use the HuggingFace discussion tab on the model page.

---

## 10. Acknowledgements

This work was supported by the Future Defense Bridge Technology Development Program through the National Research Foundation of Korea (NRF), funded by the Ministry of Science and ICT (MSIT) and the Defense Acquisition Program Administration (DAPA) of the Government of the Republic of Korea, under Grant No. RS-2024-00452972.
