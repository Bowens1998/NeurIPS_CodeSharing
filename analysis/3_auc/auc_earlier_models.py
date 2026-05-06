#!/usr/bin/env python3
"""
Step 3 — AUC Computation  (Earlier-Generation Models, Paper §3.3)
===================================================================
Integrates the OLR S-curve to produce a scalar AUC (Area Under Curve) per
model per task.  AUC = ∫₀¹ E[R_D | B_C] dB_C.

Reads from : utils.py  (shared extraction, OLR/AUC fitting)
Input data : data/earlier_models/auc_N100/{DNC,CTD,FIP}/{model}_100.json
Output     : figures/auc_earlier/
  AUC_Barchart_SortedBy{DNC,FIP,CTD}_EarlierModels.png
  AUC_Barchart_AllModels_EarlierModels.png
  AUC_Ranking_EarlierModels.png

Relation to other steps:
  ← Step 1 (reliability_earlier_models.py) confirmed B_C / R_D measurements
    are reliable for this earlier-generation model set.
  → Step 4 (risk_attitude_ranking.py) uses the AUC values produced here
    together with frontier AUC values to build cross-generation rankings.

Full dataset: https://huggingface.co/datasets/LLMsRiskAttitudeDataShare/DataShare_NeurIPS2026_LLMsRiskAttitude
"""

import sys, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# ── shared utilities ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    EARLIER_KEYS, EARLIER_LABELS, TASKS, TASK_COLORS,
    extract_earlier, fit_olr_auc,
)

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / 'data' / 'earlier_models' / 'auc_N100'
OUT  = ROOT / 'figures' / 'auc_earlier'
OUT.mkdir(parents=True, exist_ok=True)


# ── load earlier model data ───────────────────────────────────────────────────

def load_model_task(mkey, task):
    fpath = BASE / task / f'{mkey}_100.json'
    if not fpath.exists():
        print(f'  WARNING: {fpath.name} not found — download full dataset from HuggingFace')
        return np.array([]), np.array([], dtype=int)
    with open(fpath, encoding='utf-8') as f:
        d = json.load(f)
    bc_list, rd_list = [], []
    for t in d.get('trials', []):
        bc, rd = extract_earlier(t, task)
        if bc is not None:
            bc_list.append(bc)
            rd_list.append(rd)
    return np.array(bc_list), np.array(rd_list, dtype=int)


# ── compute AUC per model per task ────────────────────────────────────────────

print('=' * 60)
print('Step 3 — AUC / Beta Analysis  (Earlier-Generation Models, N=100)')
print('=' * 60)

results = {task: {} for task in TASKS}

for task in TASKS:
    print(f'\n--- {task} ---')
    for mkey in EARLIER_KEYS:
        label = EARLIER_LABELS[EARLIER_KEYS.index(mkey)]
        bc_arr, rd_arr = load_model_task(mkey, task)
        if len(bc_arr) == 0:
            results[task][mkey] = {'auc': None, 'beta': None, 'n': 0}
            continue
        auc, beta = fit_olr_auc(bc_arr, rd_arr)
        results[task][mkey] = {'auc': auc, 'beta': beta, 'n': len(bc_arr)}
        if auc:
            print(f'  {label:<20}: n={len(bc_arr):3d}  AUC={auc:.4f}  β={beta:+.4f}')
        else:
            print(f'  {label:<20}: fit failed')

print('\n--- AUC Summary ---')
header = f"{'Model':<22}" + ''.join(f'{t:>10}' for t in TASKS) + f"{'Mean':>10}"
print(header)
for mkey in EARLIER_KEYS:
    label = EARLIER_LABELS[EARLIER_KEYS.index(mkey)]
    vals  = [results[t][mkey]['auc'] for t in TASKS]
    valid = [v for v in vals if v is not None]
    mean_v = np.mean(valid) if valid else float('nan')
    row = (f'{label:<22}'
           + ''.join(f'{v:>10.4f}' if v else f'{"N/A":>10}' for v in vals)
           + f'{mean_v:>10.4f}')
    print(row)

rank_table = {}
for task in TASKS:
    pairs = [(m, results[task][m]['auc']) for m in EARLIER_KEYS
             if results[task][m]['auc'] is not None]
    pairs.sort(key=lambda x: x[1])
    rank_table[task] = {m: i + 1 for i, (m, _) in enumerate(pairs)}


# ── Figure: AUC grouped bar charts ───────────────────────────────────────────

def plot_auc_barchart(models_order, fname, sort_task=None):
    labels_order = [EARLIER_LABELS[EARLIER_KEYS.index(m)] for m in models_order]
    n, w = len(models_order), 0.26
    offsets = [-w, 0, w]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, model in enumerate(models_order):
        means = []
        for j, task in enumerate(TASKS):
            val = results[task][model]['auc']
            if val is None:
                continue
            alpha = 0.95 if task == sort_task else 0.75
            edge  = '#333333' if task == sort_task else 'white'
            lw    = 1.2 if task == sort_task else 0.5
            ax.bar(i + offsets[j], val, w, color=TASK_COLORS[task],
                   alpha=alpha, zorder=3, edgecolor=edge, linewidth=lw)
            means.append(val)
        if means:
            ax.hlines(np.mean(means), i - w * 1.5, i + w * 1.5,
                      colors='#1a1a1a', linewidths=1.4, linestyles='dashed', zorder=6)
    for x in np.arange(0.5, n - 0.5):
        ax.axvline(x, color='#cccccc', linewidth=0.6, zorder=1)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels_order, fontsize=10.5)
    ax.set_ylabel('AUC (Area Under OLR S-Curve)', fontsize=11)
    ax.set_xlim(-0.55, n - 0.45)
    sort_lbl = f' — sorted by {sort_task} AUC' if sort_task else ''
    ax.annotate(f'← More Risk-Averse{sort_lbl}', xy=(0.20, 1.01),
                xycoords='axes fraction', fontsize=10.5, fontweight='bold', ha='center')
    ax.annotate('More Risk-Taking →', xy=(0.76, 1.01),
                xycoords='axes fraction', fontsize=10.5, fontweight='bold', ha='center')
    patches = [mpatches.Patch(color=TASK_COLORS[t], alpha=0.85,
                               label=f'$B_C$→$R_D$ in {t}') for t in TASKS]
    mean_line = plt.Line2D([0], [0], color='#1a1a1a', linewidth=1.4,
                           linestyle='dashed', label='Mean AUC')
    ax.legend(handles=patches + [mean_line], loc='upper left',
              fontsize=9.5, framealpha=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


# ── Figure: Ranking heatmap + bump chart ─────────────────────────────────────

def plot_ranking_chart(fname):
    tasks_ext  = TASKS + ['Mean']
    rank_data  = {}
    for m in EARLIER_KEYS:
        per_task  = [rank_table[t].get(m, len(EARLIER_KEYS)) for t in TASKS]
        rank_data[m] = per_task + [np.mean(per_task)]

    model_order = sorted(EARLIER_KEYS, key=lambda m: rank_data[m][-1])
    labels_ord  = [EARLIER_LABELS[EARLIER_KEYS.index(m)] for m in model_order]
    cmap_rank   = LinearSegmentedColormap.from_list(
        'rank', ['#4A90D9', '#A8D5A2', '#F7DC6F', '#E59866', '#C0392B'], N=6)
    MCOLS       = ['#E07070', '#5B9BD5', '#70B870', '#F4A460', '#9370DB', '#20B2AA']
    col_map     = {m: MCOLS[i] for i, m in enumerate(model_order)}

    fig, (ax_heat, ax_bump) = plt.subplots(
        1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1.0, 1.2]})
    n_rows, n_cols = len(model_order), len(tasks_ext)
    ax_heat.set_xlim(0, n_cols); ax_heat.set_ylim(0, n_rows)
    ax_heat.set_aspect('equal')

    for row_i, model in enumerate(model_order):
        for col_j, task in enumerate(tasks_ext):
            rk      = rank_data[model][col_j]
            norm_rk = (rk - 1) / 5.0
            rect    = plt.Rectangle((col_j, n_rows - 1 - row_i), 1, 1,
                                     facecolor=cmap_rank(norm_rk),
                                     edgecolor='white', linewidth=1.5, zorder=2)
            ax_heat.add_patch(rect)
            val_str = f'{int(round(rk))}' if task != 'Mean' else f'{rk:.1f}'
            txt_col = 'white' if norm_rk > 0.6 or norm_rk < 0.2 else '#1a1a1a'
            ax_heat.text(col_j + 0.5, n_rows - 0.5 - row_i, val_str,
                         ha='center', va='center', fontsize=11,
                         fontweight='bold', color=txt_col, zorder=3)

    ax_heat.set_xticks([j + 0.5 for j in range(n_cols)])
    ax_heat.set_xticklabels(tasks_ext, fontsize=10.5, fontweight='bold')
    ax_heat.set_yticks([n_rows - 0.5 - i for i in range(n_rows)])
    ax_heat.set_yticklabels(labels_ord, fontsize=10)
    ax_heat.tick_params(length=0)
    ax_heat.set_title('Risk Attitude Rank (1 = Most Cautious)', fontsize=11, pad=8)

    task_x = np.arange(len(TASKS))
    ax_bump.set_xlim(-0.4, len(TASKS) - 0.6)
    ax_bump.set_ylim(6.6, 0.4)
    ax_bump.set_yticks(range(1, 7))
    ax_bump.set_yticklabels([f'#{i}' for i in range(1, 7)], fontsize=10)
    ax_bump.set_xticks(task_x); ax_bump.set_xticklabels(TASKS, fontsize=10.5, fontweight='bold')
    ax_bump.set_ylabel('Rank', fontsize=10.5)
    ax_bump.set_title('Ranking Across Tasks', fontsize=11, pad=8)
    ax_bump.grid(axis='y', alpha=0.25, zorder=0)
    ax_bump.spines['top'].set_visible(False); ax_bump.spines['right'].set_visible(False)

    for model in model_order:
        ranks = rank_data[model][:len(TASKS)]
        color = col_map[model]
        ax_bump.plot(task_x, ranks, '-o', color=color, linewidth=2.0,
                     markersize=9, zorder=4, solid_capstyle='round')
        ax_bump.text(len(TASKS) - 0.55, ranks[-1],
                     EARLIER_LABELS[EARLIER_KEYS.index(model)],
                     va='center', ha='left', fontsize=8.5,
                     color=color, fontweight='bold')

    ax_bump.set_xlim(-0.4, len(TASKS) + 0.5)
    plt.tight_layout(pad=2.0)
    plt.savefig(OUT / fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


print('\n--- Generating figures ---')
for sort_task in TASKS:
    order = sorted(EARLIER_KEYS, key=lambda m: results[sort_task][m]['auc'] or 0)
    plot_auc_barchart(order, f'AUC_Barchart_SortedBy{sort_task}_EarlierModels.png', sort_task)

mean_order = sorted(EARLIER_KEYS, key=lambda m: np.mean(
    [results[t][m]['auc'] for t in TASKS if results[t][m]['auc'] is not None]))
plot_auc_barchart(mean_order, 'AUC_Barchart_AllModels_EarlierModels.png')
plot_ranking_chart('AUC_Ranking_EarlierModels.png')
print('\nDone. Figures saved to:', OUT)
