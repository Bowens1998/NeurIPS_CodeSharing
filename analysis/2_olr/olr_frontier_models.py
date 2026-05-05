#!/usr/bin/env python3
"""
Step 2 — OLR S-Curves  (Paper §3.2)
=====================================
For each frontier model × task, fits Ordered Logistic Regression: R_D ~ B_C.
Plots E[R_D] vs B_C (point estimate) on a shared 2×3 panel and per-task overlays.

Reads from : utils.py  (shared B_C/R_D extraction, K-means, OLR fitting)
Input data : data/frontier_models/main_N100/{DNC,CTD,FIP}/{task}_{model}_N100.json
Output     : figures/olr_frontier/
  OLR_Scurves_ByModel_FrontierModels.png  — 2×3 panel (one subplot per model)
  OLR_Scurves_ByTask_{DNC,FIP,CTD}.png   — all 6 models overlaid per task

Feeds into: Step 3 (auc_frontier_models.py) which integrates the same OLR curves
            to produce scalar AUC values.

Full dataset: https://huggingface.co/datasets/[dataset-url]
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

# ── shared utilities ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    FRONTIER_KEYS, FRONTIER_LABELS, TASKS, TASK_NAMES,
    CB_GRID, RANDOM_SEED,
    load_frontier_task, discretize_frontier_task, fit_olr_curve,
)

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / 'data' / 'frontier_models' / 'main_N100'
OUT  = ROOT / 'figures' / 'olr_frontier'
OUT.mkdir(parents=True, exist_ok=True)

TASK_COLOR   = {'DNC': '#c0271f', 'FIP': '#1a8c1a', 'CTD': '#1a6fab'}
MODEL_COLORS = ['#E07070', '#5B9BD5', '#70B870', '#F4A460', '#9370DB', '#20B2AA']


# ── load and discretize all tasks ────────────────────────────────────────────

print('=' * 60)
print('Step 2 — OLR S-Curves  (Frontier Models, N=100)')
print('=' * 60)

all_data = {}
for task in TASKS:
    raw = load_frontier_task(BASE, task, FRONTIER_KEYS)
    all_data[task] = discretize_frontier_task(raw, task, FRONTIER_KEYS)


# ── fit OLR per model per task ────────────────────────────────────────────────

fits  = {}   # fits[task][mkey] = e_rd array | None
betas = {}

for task in TASKS:
    print(f'\n--- {task} ---')
    fits[task]  = {}
    betas[task] = {}
    for mkey in FRONTIER_KEYS:
        bc_arr, rd_arr = all_data[task][mkey]
        label = FRONTIER_LABELS[FRONTIER_KEYS.index(mkey)]
        if len(bc_arr) == 0:
            fits[task][mkey] = betas[task][mkey] = None
            continue
        e_rd, beta = fit_olr_curve(bc_arr, rd_arr)
        fits[task][mkey]  = e_rd
        betas[task][mkey] = beta
        if beta is not None:
            print(f'  {label:<20}: n={len(bc_arr):3d}  β={beta:+.4f}')
        else:
            print(f'  {label:<20}: fit failed')

print('\n--- β Coefficient Summary ---')
header = f"{'Model':<22}" + ''.join(f'{t:>10}' for t in TASKS) + f"{'Mean':>10}"
print(header)
for mkey in FRONTIER_KEYS:
    label = FRONTIER_LABELS[FRONTIER_KEYS.index(mkey)]
    vals  = [betas[t][mkey] for t in TASKS]
    valid = [v for v in vals if v is not None]
    mean_b = np.mean(valid) if valid else float('nan')
    row = (f'{label:<22}'
           + ''.join(f'{v:>+10.4f}' if v is not None else f'{"N/A":>10}' for v in vals)
           + f'{mean_b:>+10.4f}')
    print(row)


# ── Figure 1: 2×3 panel — one subplot per model ─────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True, sharex=True)
fig.subplots_adjust(hspace=0.38, wspace=0.08)

for ax, mkey, label in zip(axes.flat, FRONTIER_KEYS, FRONTIER_LABELS):
    for task in TASKS:
        e_rd = fits[task][mkey]
        if e_rd is None:
            continue
        ax.plot(CB_GRID, e_rd, color=TASK_COLOR[task], linewidth=2.5, zorder=4)
    ax.set_title(label, fontsize=19, fontweight='bold')
    ax.set_xlim(-2, 102)
    ax.set_ylim(0.7, 5.3)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25)

for ax in axes[:, 0]:
    ax.set_ylabel(r'$R_D$', fontsize=18)
for ax in axes[1, :]:
    ax.set_xlabel(r'$B_C$', fontsize=18)

patches = [mpatches.Patch(color=TASK_COLOR[t], label=TASK_NAMES[t]) for t in TASKS]
fig.legend(handles=patches, loc='lower center', ncol=3,
           fontsize=16, framealpha=0.85, bbox_to_anchor=(0.5, -0.06))
plt.savefig(OUT / 'OLR_Scurves_ByModel_FrontierModels.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'\nSaved: OLR_Scurves_ByModel_FrontierModels.png')


# ── Figure 2: per-task — all 6 models overlaid ──────────────────────────────

for task in TASKS:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (mkey, label) in enumerate(zip(FRONTIER_KEYS, FRONTIER_LABELS)):
        e_rd = fits[task][mkey]
        if e_rd is None:
            continue
        ax.plot(CB_GRID, e_rd, color=MODEL_COLORS[i], linewidth=2.2, label=label)
    ax.set_xlabel(r'$B_C$ (Contextual Belief)', fontsize=13)
    ax.set_ylabel(r'$R_D$ (Expected Risk Decision)', fontsize=13)
    ax.set_title(f'{task} — {TASK_NAMES[task]}', fontsize=13, fontweight='bold')
    ax.set_xlim(-2, 102)
    ax.set_ylim(0.7, 5.3)
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / f'OLR_Scurves_ByTask_{task}_FrontierModels.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved: OLR_Scurves_ByTask_{task}_FrontierModels.png')

print('\nDone. Figures saved to:', OUT)
