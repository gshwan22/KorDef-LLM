#!/usr/bin/env python3
"""Paper figures for source-grounded N=323 evaluation (retrained checkpoint).
Values fixed from the revised paper. Output: ./figures/ (PDF+PNG)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm

for name in ['Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumGothic', 'Malgun Gothic']:
    if any(name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = name
        break

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT = Path('figures'); OUT.mkdir(exist_ok=True)
C_GEMMA = '#5B9BD5'; C_KORDEF = '#E74C3C'; C_SIG = '#2171B5'; C_NS = '#999999'

# === FIXED metric values from revised paper (N=323, retrained checkpoint) ===
ANSWER_TYPES = {
    'criterion':       {'n': 96, 'gemma': 0.407, 'kordef': 0.538, 'delta': 0.131},
    'procedure':       {'n': 82, 'gemma': 0.424, 'kordef': 0.532, 'delta': 0.108},
    'definition':      {'n': 53, 'gemma': 0.402, 'kordef': 0.538, 'delta': 0.135},
    'responsibility':  {'n': 43, 'gemma': 0.351, 'kordef': 0.462, 'delta': 0.110},
    'eligibility':     {'n': 16, 'gemma': 0.428, 'kordef': 0.547, 'delta': 0.119},
    'purpose':         {'n': 16, 'gemma': 0.338, 'kordef': 0.406, 'delta': 0.068},
    'reporting':       {'n': 8,  'gemma': 0.347, 'kordef': 0.434, 'delta': 0.087},
    'document_status': {'n': 6,  'gemma': 0.375, 'kordef': 0.502, 'delta': 0.127},
    'exception':       {'n': 3,  'gemma': 0.362, 'kordef': 0.436, 'delta': 0.074},
}

# Paired deltas vs base: (delta, ci_lo, ci_hi, significant) — B=10,000 bootstrap
PAIRED = [
    ('Token-F1',        0.1180,  0.1024,  0.1335, True),
    ('ROUGE-L',         0.1162,  0.1001,  0.1326, True),
    ('Char-3 Jaccard',  0.0931,  0.0781,  0.1083, True),
    ('Evidence recall', -0.0073, -0.0298, 0.0147, False),
    ('Refusal rate',    -0.1703, -0.2105, -0.1300, True),
]
PAIRED_TOKENS = ('Answer tokens', -13.84, -15.02, -12.64, True)


def fig_answer_style():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    models = ['Gemma-3-12B', 'KorDef-LLM']
    colors = [C_GEMMA, C_KORDEF]

    means = [45.1486, 31.3096]
    bars = ax1.bar(models, means, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.annotate('', xy=(1.25, means[1]), xytext=(1.25, means[0]),
                 arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
    ax1.text(1.38, (means[0]+means[1])/2,
             r'$\Delta$ = $-$13.8' + '\n' + r'$p < 10^{-45}$',
             fontsize=8, ha='left', va='center', color='#333')
    ax1.set_ylabel('Mean answer tokens')
    ax1.set_title('(a) Answer length', fontweight='bold')
    ax1.set_ylim(0, 52)

    refusal = [0.1765, 0.0062]
    bars2 = ax2.bar(models, refusal, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars2, refusal):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Refusal rate')
    ax2.set_title('(b) Refusal rate', fontweight='bold')
    ax2.set_ylim(0, 0.25)
    ax2.text(0.5, 0.225, r'$\Delta$ = $-$0.170, $p < 10^{-13}$',
             fontsize=9, ha='center', color=C_SIG)

    plt.tight_layout()
    path = OUT / 'figure_sourcegrounded_answer_style_eval323.pdf'
    plt.savefig(path); plt.savefig(path.with_suffix('.png'), dpi=150); plt.close()
    print(f'  saved: {path.name}')


def fig_answer_type():
    items = sorted(ANSWER_TYPES.items(), key=lambda x: x[1]['n'], reverse=True)
    labels = [f"{k} (n={v['n']})" for k, v in items]
    g_vals = [v['gemma'] for _, v in items]
    k_vals = [v['kordef'] for _, v in items]
    d_vals = [v['delta'] for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(items))[::-1]
    for i in range(len(items)):
        ax.plot([g_vals[i], k_vals[i]], [y[i], y[i]], color='#ddd', linewidth=2, zorder=1)
        ax.scatter(g_vals[i], y[i], color=C_GEMMA, s=55, zorder=3, edgecolors='white', linewidth=0.5)
        ax.scatter(k_vals[i], y[i], color=C_KORDEF, s=55, zorder=3, edgecolors='white', linewidth=0.5)
        ax.text(0.60, y[i], f'+{d_vals[i]:.3f}', va='center', fontsize=8, color='#555', family='monospace')

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Token-F1'); ax.set_xlim(0.30, 0.66)
    ax.scatter([], [], color=C_GEMMA, s=55, label='Gemma-3-12B')
    ax.scatter([], [], color=C_KORDEF, s=55, label='KorDef-LLM')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_title('Token-F1 by answer type (N=323)', fontweight='bold')

    plt.tight_layout()
    path = OUT / 'figure_tokenf1_by_answer_type_eval323.pdf'
    plt.savefig(path); plt.savefig(path.with_suffix('.png'), dpi=150); plt.close()
    print(f'  saved: {path.name}')


def fig_forest():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6),
                                   gridspec_kw={'width_ratios': [3, 1]})
    names = [p[0] for p in PAIRED]
    y = np.arange(len(PAIRED))[::-1]
    for i, (name, d, lo, hi, sig) in enumerate(PAIRED):
        c = C_SIG if sig else C_NS
        ax1.errorbar(d, y[i], xerr=[[d - lo], [hi - d]], fmt='o', color=c,
                     ecolor=c, markersize=7, capsize=4, capthick=1.5, elinewidth=1.8)
    ax1.axvline(0, color='#333', linewidth=0.8, linestyle='--')
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel(r'Mean paired $\Delta$ (KorDef $-$ base)')
    ax1.set_title('(a) Score-scale metrics', fontweight='bold')

    name, d, lo, hi, sig = PAIRED_TOKENS
    ax2.errorbar(d, 0, xerr=[[d - lo], [hi - d]], fmt='o', color=C_SIG,
                 ecolor=C_SIG, markersize=7, capsize=4, capthick=1.5, elinewidth=1.8)
    ax2.axvline(0, color='#333', linewidth=0.8, linestyle='--')
    ax2.set_yticks([0]); ax2.set_yticklabels([name], fontsize=9)
    ax2.set_ylim(-1, 1); ax2.set_xlim(-18, 2)
    ax2.set_xlabel(r'Mean paired $\Delta$ (tokens)')
    ax2.set_title('(b) Answer length', fontweight='bold')

    plt.tight_layout()
    path = OUT / 'figure_sourcegrounded_eval323_forest.pdf'
    plt.savefig(path); plt.savefig(path.with_suffix('.png'), dpi=150); plt.close()
    print(f'  saved: {path.name}')


if __name__ == '__main__':
    print('Generating paper figures (N=323, retrained checkpoint)...')
    fig_answer_style(); fig_answer_type(); fig_forest()
    print(f'\nDone. Output in: {OUT.resolve()}')
