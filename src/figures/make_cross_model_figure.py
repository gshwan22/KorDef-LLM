#!/usr/bin/env python3
"""
Generate cross-model comparison figure for paper.
6 models × 4 metrics with real bootstrap CIs (n=323).

Usage:
    python make_cross_model_figure.py
    -> figure_cross_model.pdf, figure_cross_model.png
"""
import numpy as np
import matplotlib.pyplot as plt

# Real bootstrap CIs (mean, ci_low, ci_high), seed=20260508, 5000 iterations
MODEL_RESULTS = {
    "Gemma-3-12B (base)": {
        "color": "#888888",
        "token_f1": (0.3983, 0.3804, 0.4155),
        "rouge_l":  (0.3803, 0.3621, 0.3980),
        "char3":    (0.2582, 0.2429, 0.2735),
        "evidence": (0.5337, 0.5057, 0.5599),
    },
    "EXAONE-3.5-7.8B": {
        "color": "#4477AA",
        "token_f1": (0.3774, 0.3555, 0.3998),
        "rouge_l":  (0.3538, 0.3320, 0.3762),
        "char3":    (0.2361, 0.2185, 0.2547),
        "evidence": (0.3957, 0.3705, 0.4210),
    },
    "Qwen2.5-7B": {
        "color": "#66CCEE",
        "token_f1": (0.4259, 0.3938, 0.4594),
        "rouge_l":  (0.4088, 0.3769, 0.4422),
        "char3":    (0.3122, 0.2842, 0.3421),
        "evidence": (0.3928, 0.3596, 0.4263),
    },
    "Llama-3.1-8B": {
        "color": "#228833",
        "token_f1": (0.5042, 0.4765, 0.5317),
        "rouge_l":  (0.4885, 0.4597, 0.5172),
        "char3":    (0.3621, 0.3343, 0.3896),
        "evidence": (0.5168, 0.4817, 0.5506),
    },
    "A.X-4.0-Light": {
        "color": "#CCBB44",
        "token_f1": (0.4981, 0.4716, 0.5240),
        "rouge_l":  (0.4785, 0.4515, 0.5049),
        "char3":    (0.3436, 0.3190, 0.3684),
        "evidence": (0.4702, 0.4413, 0.4986),
    },
    "KorDef-LLM (ours)": {
        "color": "#EE6677",
        "token_f1": (0.4284, 0.4089, 0.4478),
        "rouge_l":  (0.4023, 0.3821, 0.4222),
        "char3":    (0.2809, 0.2634, 0.2982),
        "evidence": (0.5489, 0.5202, 0.5771),
    },
}

ORDER = [
    "Gemma-3-12B (base)",
    "EXAONE-3.5-7.8B",
    "Qwen2.5-7B",
    "Llama-3.1-8B",
    "A.X-4.0-Light",
    "KorDef-LLM (ours)",
]

METRICS = [
    ("token_f1", "Token-F1"),
    ("rouge_l",  "ROUGE-L"),
    ("char3",    "Char-3 Jaccard"),
    ("evidence", "Evidence Recall (source faithfulness)"),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.flatten()

    for ax_idx, (metric_key, metric_name) in enumerate(METRICS):
        ax = axes[ax_idx]
        y_positions = np.arange(len(ORDER))[::-1]

        for i, model_name in enumerate(ORDER):
            r = MODEL_RESULTS[model_name]
            mean, ci_lo, ci_hi = r[metric_key]
            err = [[mean - ci_lo], [ci_hi - mean]]
            color = r["color"]
            ax.errorbar(
                mean, y_positions[i], xerr=err,
                fmt='o', color=color, ecolor=color,
                markersize=10, capsize=4, capthick=2,
                markeredgecolor='black', markeredgewidth=0.7,
                elinewidth=2,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(ORDER, fontsize=9)
        ax.set_xlabel(f"{metric_name} (higher is better)", fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # KorDef row highlight
        kordef_idx = ORDER.index("KorDef-LLM (ours)")
        ax.axhspan(
            y_positions[kordef_idx] - 0.4, y_positions[kordef_idx] + 0.4,
            color='#EE6677', alpha=0.08, zorder=0,
        )

    fig.suptitle("Cross-model source-grounded evaluation ($N{=}323$)",
                 fontsize=12, y=1.02)

    fig.savefig("figure_cross_model.pdf", bbox_inches='tight', dpi=200)
    fig.savefig("figure_cross_model.png", bbox_inches='tight', dpi=200)
    print("Saved: figure_cross_model.pdf, figure_cross_model.png")


if __name__ == "__main__":
    main()
