#!/usr/bin/env python3
"""
Generate paper figures for source-grounded N=323 evaluation.

Usage:
    python src/figures/gen_eval323_figures.py

Output: figures/ directory with PDF + PNG.

All metric values are FIXED from the paper. Run after evaluation outputs are
in place (or with the values hard-coded for paper-figure reproduction).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm

# Korean font (best-effort)
for name in ['Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumGothic', 'Malgun Gothic']:
    if any(name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = name
        break

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUT = Path('figures')
OUT.mkdir(exist_ok=True)

C_GEMMA = '#5B9BD5'
C_KORDEF = '#E74C3C'
C_SIG = '#2171B5'
C_NS = '#999999'

# === FIXED metric values from paper (N=323) ===
ANSWER_TYPES = {
    'criterion':       {'n': 96, 'gemma': 0.4070, 'kordef': 0.4361, 'delta': 0.0291},
    'procedure':       {'n': 82, 'gemma': 0.4239, 'kordef': 0.4565, 'delta': 0.0326},
    'definition':      {'n': 53, 'gemma': 0.4024, 'kordef': 0.4229, 'delta': 0.0205},
    'responsibility':  {'n': 43, 'gemma': 0.3514, 'kordef': 0.3733, 'delta': 0.0219},
    'eligibility':     {'n': 16, 'gemma': 0.4275, 'kordef': 0.4486, 'delta': 0.0211},
    'purpose':         {'n': 16, 'gemma': 0.3381, 'kordef': 0.3564, 'delta': 0.0183},
    'reporting':       {'n': 8,  'gemma': 0.3472, 'kordef': 0.4343, 'delta': 0.0870},
    'document_status': {'n': 6,  'gemma': 0.3750, 'kordef': 0.4383, 'delta': 0.0633},
    'exception':       {'n': 3,  'gemma': 0.3623, 'kordef': 0.5383, 'delta': 0.1760},
}


def fig_answer_style():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    models = ['Gemma-3-12B', 'KorDef-LLM']
    colors = [C_GEMMA, C_KORDEF]

    means = [45.1486, 41.2322]
    bars = ax1.bar(models, means, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.annotate('', xy=(1.25, means[1]), xytext=(1.25, means[0]),
                 arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
    ax1.text(1.38, (means[0]+means[1])/2,
             r'$\Delta$ = $-$3.9' + '\n' + r'$p < 10^{-11}$',
             fontsize=8, ha='left', va='center', color='#333')
    ax1.set_ylabel('Mean answer tokens')
    ax1.set_title('(a) Answer length', fontweight='bold')
    ax1.set_ylim(0, 52)

    refusal = [0.1765, 0.1486]
    bars2 = ax2.bar(models, refusal, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars2, refusal):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Refusal rate')
    ax2.set_title('(b) Refusal rate', fontweight='bold')
    ax2.set_ylim(0, 0.25)
    ax2.text(0.5, 0.225, r'$\Delta$ = $-$0.028 (n.s.)',
             fontsize=9, ha='center', color=C_NS, style='italic')

    plt.tight_layout()
    path = OUT / 'figure_sourcegrounded_answer_style_eval323.pdf'
    plt.savefig(path)
    plt.savefig(path.with_suffix('.png'), dpi=150)
    plt.close()
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
        ax.text(0.555, y[i], f'+{d_vals[i]:.4f}', va='center', fontsize=8, color='#555', family='monospace')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Token-F1')
    ax.set_xlim(0.30, 0.58)
    ax.scatter([], [], color=C_GEMMA, s=55, label='Gemma-3-12B')
    ax.scatter([], [], color=C_KORDEF, s=55, label='KorDef-LLM')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_title('Token-F1 by answer type (N=323)', fontweight='bold')

    plt.tight_layout()
    path = OUT / 'figure_tokenf1_by_answer_type_eval323.pdf'
    plt.savefig(path)
    plt.savefig(path.with_suffix('.png'), dpi=150)
    plt.close()
    print(f'  saved: {path.name}')


if __name__ == '__main__':
    print('Generating paper figures (N=323 source-grounded evaluation)...')
    fig_answer_style()
    fig_answer_type()
    print(f'\nDone. Output in: {OUT.resolve()}')
