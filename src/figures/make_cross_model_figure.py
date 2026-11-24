#!/usr/bin/env python3
"""Cross-model comparison figure: 6 models x 4 metrics, bootstrap CIs (n=323).
All rows recomputed from per-item scores (seed=42, 10,000 iterations)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('figures'); OUT.mkdir(exist_ok=True)

MODEL_RESULTS = {
    "Gemma-3-12B (base)": {"color": "#888888",
        "token_f1": (0.3983, 0.3803, 0.4158), "rouge_l": (0.3803, 0.3622, 0.3987),
        "char3": (0.2582, 0.2431, 0.2741), "evidence": (0.5337, 0.5055, 0.5609)},
    "EXAONE-3.5-7.8B": {"color": "#4477AA",
        "token_f1": (0.3774, 0.3555, 0.3995), "rouge_l": (0.3538, 0.3318, 0.3763),
        "char3": (0.2361, 0.2186, 0.2541), "evidence": (0.3957, 0.3705, 0.4215)},
    "Qwen2.5-7B": {"color": "#66CCEE",
        "token_f1": (0.4259, 0.3929, 0.4582), "rouge_l": (0.4088, 0.3765, 0.4415),
        "char3": (0.3122, 0.2837, 0.3414), "evidence": (0.3928, 0.3588, 0.4262)},
    "Llama-3.1-8B": {"color": "#228833",
        "token_f1": (0.5042, 0.4764, 0.5322), "rouge_l": (0.4885, 0.4604, 0.5178),
        "char3": (0.3621, 0.3344, 0.3906), "evidence": (0.5168, 0.4829, 0.5515)},
    "A.X-4.0-Light": {"color": "#CCBB44",
        "token_f1": (0.4981, 0.4724, 0.5241), "rouge_l": (0.4785, 0.4523, 0.5051),
        "char3": (0.3436, 0.3191, 0.3688), "evidence": (0.4702, 0.4414, 0.4990)},
    "KorDef-LLM (ours)": {"color": "#EE6677",
        "token_f1": (0.5162, 0.4948, 0.5373), "rouge_l": (0.4965, 0.4732, 0.5187),
        "char3": (0.3514, 0.3308, 0.3725), "evidence": (0.5264, 0.4989, 0.5525)},
}

ORDER = ["Gemma-3-12B (base)", "EXAONE-3.5-7.8B", "Qwen2.5-7B",
         "Llama-3.1-8B", "A.X-4.0-Light", "KorDef-LLM (ours)"]
METRICS = [("token_f1", "Token-F1"), ("rouge_l", "ROUGE-L"),
           ("char3", "Char-3 Jaccard"), ("evidence", "Evidence recall")]
PANEL = ['(a)', '(b)', '(c)', '(d)']


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.flatten()
    for ax_idx, (metric_key, metric_name) in enumerate(METRICS):
        ax = axes[ax_idx]
        y_positions = np.arange(len(ORDER))[::-1]
        for i, model_name in enumerate(ORDER):
            r = MODEL_RESULTS[model_name]
            mean, ci_lo, ci_hi = r[metric_key]
            ax.errorbar(mean, y_positions[i], xerr=[[mean - ci_lo], [ci_hi - mean]],
                        fmt='o', color=r["color"], ecolor=r["color"],
                        markersize=10, capsize=4, capthick=2,
                        markeredgecolor='black', markeredgewidth=0.7, elinewidth=2)
        ax.set_yticks(y_positions); ax.set_yticklabels(ORDER, fontsize=9)
        ax.set_xlabel(metric_name, fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.text(0.02, 0.98, PANEL[ax_idx], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        kordef_idx = ORDER.index("KorDef-LLM (ours)")
        ax.axhspan(y_positions[kordef_idx] - 0.4, y_positions[kordef_idx] + 0.4,
                   color='#EE6677', alpha=0.08, zorder=0)
    fig.savefig(OUT / "figure_cross_model.pdf", bbox_inches='tight', dpi=200)
    fig.savefig(OUT / "figure_cross_model.png", bbox_inches='tight', dpi=200)
    print("Saved: figures/figure_cross_model.pdf")


if __name__ == "__main__":
    main()
