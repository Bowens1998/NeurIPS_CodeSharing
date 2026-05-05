#!/usr/bin/env python3
"""
Step 4 — Cross-Task Ranking with Kendall's W  (Paper §3.4)
============================================================
Ranks all 12 models (6 frontier + 6 earlier-generation) by AUC within each
task, then tests cross-task consistency using Kendall's coefficient of
concordance (W).

Reads from : utils.py  (shared extraction, K-means, AUC fitting)
Input data : data/frontier_models/main_N100/  +  data/earlier_models/auc_N100/
Output     : figures/ranking/
  Ranking_Frontier_Heatmap_Bump.png
  Ranking_Earlier_Heatmap_Bump.png
  Ranking_Combined_Heatmap.png      — all 12 models
  KendallW_Summary.txt              — W, χ², df, p-value

Relation to other steps:
  ← Step 3 computed AUC per model per task for both model sets.
    This step aggregates those AUC values into cross-task rankings.

Full dataset: https://huggingface.co/datasets/[dataset-url]
"""

import sys, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2 as chi2_dist

# ── shared utilities ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    FRONTIER_KEYS, FRONTIER_LABELS,
    EARLIER_KEYS,  EARLIER_LABELS,
    TASKS, TASK_COLORS,
    load_frontier_task, discretize_frontier_task,
    extract_earlier, fit_olr_auc,
)

ROOT   = Path(__file__).parent.parent.parent
BASE_F = ROOT / 'data' / 'frontier_models' / 'main_N100'
BASE_E = ROOT / 'data' / 'earlier_models'  / 'auc_N100'
OUT    = ROOT / 'figures' / 'ranking'
OUT.mkdir(parents=True, exist_ok=True)


# ── load earlier model data ───────────────────────────────────────────────────

def load_earlier_task(task):
    result = {}
    for mkey in EARLIER_KEYS:
        fpath = BASE_E / task / f'{mkey}_100.json'
        if not fpath.exists():
            print(f'  WARNING: {fpath.name} not found — download full dataset from HuggingFace')
            result[mkey] = (np.array([]), np.array([]))
            continue
        with open(fpath, encoding='utf-8') as f:
            d = json.load(f)
        bc_list, rd_list = [], []
        for t in d.get('trials', []):
            bc, rd = extract_earlier(t, task)
            if bc is not None and rd is not None:
                bc_list.append(bc); rd_list.append(rd)
        result[mkey] = (np.array(bc_list), np.array(rd_list))
    return result


# ── compute AUC ───────────────────────────────────────────────────────────────

print('=' * 60)
print('Step 4 — Cross-Task Risk Attitude Ranking')
print('=' * 60)

auc_frontier = {task: {} for task in TASKS}
for task in TASKS:
    raw         = load_frontier_task(BASE_F, task, FRONTIER_KEYS)
    discretized = discretize_frontier_task(raw, task, FRONTIER_KEYS)
    for mkey in FRONTIER_KEYS:
        bc, rd = discretized[mkey]
        auc_frontier[task][mkey] = fit_olr_auc(bc, rd)[0] if len(bc) > 0 else None

auc_earlier = {task: {} for task in TASKS}
for task in TASKS:
    data = load_earlier_task(task)
    for mkey in EARLIER_KEYS:
        bc, rd = data[mkey]
        auc_earlier[task][mkey] = fit_olr_auc(bc, rd)[0] if len(bc) > 0 else None


def make_rank_table(keys, auc_dict):
    rtable = {}
    for task in TASKS:
        pairs = [(m, auc_dict[task][m]) for m in keys if auc_dict[task][m] is not None]
        pairs.sort(key=lambda x: x[1])
        rtable[task] = {m: i + 1 for i, (m, _) in enumerate(pairs)}
    return rtable


def print_auc_table(keys, labels, auc_dict, title):
    print(f'\n--- {title} ---')
    print(f"{'Model':<22}" + ''.join(f'{t:>10}' for t in TASKS) + f"{'Mean':>10}")
    for mkey, label in zip(keys, labels):
        vals  = [auc_dict[t][mkey] for t in TASKS]
        valid = [v for v in vals if v is not None]
        mean_v = np.mean(valid) if valid else float('nan')
        row = (f'{label:<22}'
               + ''.join(f'{v:>10.4f}' if v else f'{"N/A":>10}' for v in vals)
               + f'{mean_v:>10.4f}')
        print(row)


rank_frontier = make_rank_table(FRONTIER_KEYS, auc_frontier)
rank_earlier  = make_rank_table(EARLIER_KEYS,  auc_earlier)

print_auc_table(FRONTIER_KEYS, FRONTIER_LABELS, auc_frontier, 'Frontier Models — AUC')
print_auc_table(EARLIER_KEYS,  EARLIER_LABELS,  auc_earlier,  'Earlier Models — AUC')


# ── Kendall's W ───────────────────────────────────────────────────────────────

def kendalls_w(rank_matrix):
    """rank_matrix shape: (n_tasks, n_models)."""
    m, n  = rank_matrix.shape
    Ri    = rank_matrix.sum(axis=0)
    S     = np.sum((Ri - Ri.mean()) ** 2)
    W     = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2  = m * (n - 1) * W
    p_val = 1 - chi2_dist.cdf(chi2, n - 1)
    return W, chi2, n - 1, p_val


def compute_w(keys, rank_table, title):
    mat = np.array([[rank_table[t].get(m, len(keys)) for m in keys] for t in TASKS])
    W, chi2, df, p = kendalls_w(mat)
    print(f'\n{title}')
    print(f'  Kendall W = {W:.4f}  χ²({df}) = {chi2:.2f}  p = {p:.4f}')
    return W, chi2, df, p


w_f = compute_w(FRONTIER_KEYS, rank_frontier, 'Kendall W — Frontier Models')
w_e = compute_w(EARLIER_KEYS,  rank_earlier,  'Kendall W — Earlier Models')

summary = [
    'Cross-Task Risk Attitude Ranking — Kendall W Concordance',
    '=' * 55, '',
    f'Frontier Models (N=6): W={w_f[0]:.4f}  χ²({w_f[2]})={w_f[1]:.2f}  p={w_f[3]:.4f}',
    f'Earlier  Models (N=6): W={w_e[0]:.4f}  χ²({w_e[2]})={w_e[1]:.2f}  p={w_e[3]:.4f}',
    '', 'W=1 → perfect agreement across tasks; W=0 → no agreement.',
]
(OUT / 'KendallW_Summary.txt').write_text('\n'.join(summary), 'utf-8')
print('\nSaved: KendallW_Summary.txt')


# ── Plotting ──────────────────────────────────────────────────────────────────

CMAP_RANK = LinearSegmentedColormap.from_list(
    'rank', ['#4A90D9', '#A8D5A2', '#F7DC6F', '#E59866', '#C0392B'])


def heatmap_bump(keys, labels, rank_table, title, fname):
    tasks_ext   = TASKS + ['Mean']
    rank_data   = {m: [rank_table[t].get(m, len(keys)) for t in TASKS] for m in keys}
    for m in keys:
        rank_data[m].append(float(np.mean(rank_data[m])))
    model_order = sorted(keys, key=lambda m: rank_data[m][-1])
    labels_ord  = [labels[keys.index(m)] for m in model_order]
    n_models    = len(model_order)
    MCOLS       = ['#E07070', '#5B9BD5', '#70B870', '#F4A460', '#9370DB', '#20B2AA']
    col_map     = {m: MCOLS[i] for i, m in enumerate(model_order)}

    fig, (ax_heat, ax_bump) = plt.subplots(
        1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1.0, 1.2]})
    n_rows, n_cols = n_models, len(tasks_ext)
    ax_heat.set_xlim(0, n_cols); ax_heat.set_ylim(0, n_rows)
    ax_heat.set_aspect('equal')

    for row_i, model in enumerate(model_order):
        for col_j, task in enumerate(tasks_ext):
            rk      = rank_data[model][col_j]
            norm_rk = (rk - 1) / max(n_models - 1, 1)
            rect    = plt.Rectangle((col_j, n_rows - 1 - row_i), 1, 1,
                                     facecolor=CMAP_RANK(norm_rk),
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
    ax_heat.set_title(f'{title} — Rank (1=Most Cautious)', fontsize=11, pad=6)

    task_x = np.arange(len(TASKS))
    ax_bump.set_xlim(-0.4, len(TASKS) - 0.6)
    ax_bump.set_ylim(n_models + 0.6, 0.4)
    ax_bump.set_yticks(range(1, n_models + 1))
    ax_bump.set_yticklabels([f'#{i}' for i in range(1, n_models + 1)], fontsize=10)
    ax_bump.set_xticks(task_x); ax_bump.set_xticklabels(TASKS, fontsize=10.5, fontweight='bold')
    ax_bump.set_ylabel('Rank', fontsize=10.5)
    ax_bump.set_title('Ranking Across Tasks', fontsize=11, pad=6)
    ax_bump.grid(axis='y', alpha=0.25, zorder=0)
    ax_bump.spines['top'].set_visible(False); ax_bump.spines['right'].set_visible(False)

    for model in model_order:
        ranks = rank_data[model][:len(TASKS)]
        color = col_map[model]
        ax_bump.plot(task_x, ranks, '-o', color=color, linewidth=2.0,
                     markersize=9, zorder=4, solid_capstyle='round')
        ax_bump.text(len(TASKS) - 0.55, ranks[-1],
                     labels[keys.index(model)],
                     va='center', ha='left', fontsize=8.5,
                     color=color, fontweight='bold')

    ax_bump.set_xlim(-0.4, len(TASKS) + 0.5)
    plt.tight_layout(pad=2.0)
    plt.savefig(OUT / fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


heatmap_bump(FRONTIER_KEYS, FRONTIER_LABELS, rank_frontier,
             'Frontier Models', 'Ranking_Frontier_Heatmap_Bump.png')
heatmap_bump(EARLIER_KEYS, EARLIER_LABELS, rank_earlier,
             'Earlier-Generation Models', 'Ranking_Earlier_Heatmap_Bump.png')


# ── Combined 12-model heatmap ─────────────────────────────────────────────────

def combined_heatmap(fname):
    all_keys   = FRONTIER_KEYS   + EARLIER_KEYS
    all_labels = FRONTIER_LABELS + EARLIER_LABELS
    all_auc    = {task: {**auc_frontier[task], **auc_earlier[task]} for task in TASKS}
    rank_comb  = {}
    for task in TASKS:
        pairs = [(m, all_auc[task][m]) for m in all_keys if all_auc[task][m] is not None]
        pairs.sort(key=lambda x: x[1])
        rank_comb[task] = {m: i + 1 for i, (m, _) in enumerate(pairs)}

    rank_data   = {m: [rank_comb[t].get(m, len(all_keys)) for t in TASKS] for m in all_keys}
    for m in all_keys:
        rank_data[m].append(float(np.mean(rank_data[m])))
    model_order = sorted(all_keys, key=lambda m: rank_data[m][-1])
    gen_tag     = {m: 'F' if m in FRONTIER_KEYS else 'E' for m in all_keys}
    labels_ord  = [f'{all_labels[all_keys.index(m)]}  [{gen_tag[m]}]' for m in model_order]

    n_rows, n_cols = len(model_order), len(TASKS) + 1
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows); ax.set_aspect('equal')

    for row_i, model in enumerate(model_order):
        for col_j, task in enumerate(TASKS + ['Mean']):
            rk      = rank_data[model][col_j]
            norm_rk = (rk - 1) / max(n_rows - 1, 1)
            rect    = plt.Rectangle((col_j, n_rows - 1 - row_i), 1, 1,
                                     facecolor=CMAP_RANK(norm_rk),
                                     edgecolor='white', linewidth=1.2, zorder=2)
            ax.add_patch(rect)
            val_str = f'{int(round(rk))}' if task != 'Mean' else f'{rk:.1f}'
            txt_col = 'white' if norm_rk > 0.6 or norm_rk < 0.15 else '#1a1a1a'
            ax.text(col_j + 0.5, n_rows - 0.5 - row_i, val_str,
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color=txt_col, zorder=3)

    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(TASKS + ['Mean'], fontsize=11, fontweight='bold')
    ax.set_yticks([n_rows - 0.5 - i for i in range(n_rows)])
    ax.set_yticklabels(labels_ord, fontsize=9.5)
    ax.tick_params(length=0)
    ax.set_title('Combined Risk Attitude Ranking — All 12 Models\n'
                 '[F]=Frontier  [E]=Earlier-Generation  |  Rank 1 = Most Risk-Averse',
                 fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


combined_heatmap('Ranking_Combined_Heatmap.png')
print('\nDone. Figures saved to:', OUT)
