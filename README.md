# KorDef-LLM

**An Open Pipeline for Domain-Adaptive Instruction Tuning of Korean Defense Large Language Models**

This repository contains the code accompanying the paper:

> Sang-Hwan Gwak, Ji-Young Choi, Chang-Hoo Jeong, Gunwoo Lee, Ina Kim, Kyung-Ha Lee.
> *An Open Pipeline for Domain-Adaptive Instruction Tuning of Korean Defense Large Language Models.*
> PeerJ Computer Science, 2026 (under review).

---

## Resources

| Artifact | Location |
|---|---|
| **Trained model weights** | [huggingface.co/jeong0313/koni-it-m-7875](https://huggingface.co/jeong0313/koni-it-m-7875) |
| **Corpus & evaluation data** | [Zenodo: 10.5281/zenodo.20083055](https://doi.org/10.5281/zenodo.20083055) |
| **Code (this repo)** | [github.com/gshwan22/KorDef-LLM](https://github.com/gshwan22/KorDef-LLM) |

---

## Repository Structure
---

## Quick Start

```bash
git clone https://github.com/gshwan22/KorDef-LLM.git
cd KorDef-LLM
pip install -r requirements.txt
```

Download data from Zenodo:
```bash
mkdir -p data && cd data
wget https://zenodo.org/records/20083055/files/kordef_corpus_v1.0.tar.gz
tar -xzf kordef_corpus_v1.0.tar.gz
cd ..
```

Reproduce paper figures:
```bash
python src/figures/gen_eval323_figures.py
```

---

## Reproducing the Paper

### Source-grounded evaluation (N=323) — main results
The metric values in Table 6 and Figures 5–7 are reproducible from outputs and references included in the Zenodo deposit.

### KMMLU benchmark (Table 5)
We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.5 with 5-shot prompting:
```bash
lm_eval --model hf \
    --model_args pretrained=jeong0313/koni-it-m-7875,trust_remote_code=True \
    --tasks kmmlu_direct \
    --num_fewshot 5
```

### Training
Full-parameter SFT on 16× NVIDIA H200 (FSDP, ~5h 23min). See `configs/train_config.yaml` for hyperparameters.

---

## Citation

```bibtex
@article{gwak2026kordef,
  title={An Open Pipeline for Domain-Adaptive Instruction Tuning of Korean Defense Large Language Models},
  author={Gwak, Sang-Hwan and Choi, Ji-Young and Jeong, Chang-Hoo and Lee, Gunwoo and Kim, Ina and Lee, Kyung-Ha},
  journal={PeerJ Computer Science},
  year={2026},
  note={under review}
}

@dataset{gwak2026kordef_data,
  title={KorDef-LLM: Korean Defense Domain Instruction Corpus and Source-Grounded Evaluation Set},
  author={Gwak, Sang-Hwan and others},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.20083055}
}
```

---

## License

Code: **MIT** (see `LICENSE`). Data: **CC-BY-4.0** (Zenodo). Model weights: **Gemma Terms of Use** (inherited from base model).

---

## Ethics and Intended Use

This research uses only public, unclassified Korean Ministry of National Defense administrative-rule documents. KorDef-LLM is **not intended for autonomous operational use** and should not be deployed for military decision-making, targeting, procurement, maintenance, or safety-critical procedures without retrieval grounding, institutional security review, and human expert oversight.

---

## Funding

This work was supported by the Future Defense Bridge Technology Development Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT (MSIT) and the Defense Acquisition Program Administration (DAPA) of the Korea government under Grant **RS-2024-00452972**.

---

## Contact

**Corresponding author:** Kyung-Ha Lee (kyongha@kisti.re.kr)
Large-Scale AI Research Center, KISTI, Daejeon, Republic of Korea
