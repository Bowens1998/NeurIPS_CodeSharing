#!/usr/bin/env python3
"""
Step 1 — Reliability  (Frontier Models, Paper §3.1)
====================================================
For each frontier model × task, loads the N=90 reliability trials
(30 per condition × 3 conditions) and verifies two criteria:
  P1 — Contextual Belief reliability: mean RSD ≤ 20%
  P2 — Risk Decision reliability: dominant R_D class ≥ 80% of trials

IC data format: {"experiment": "DNC", "trials": [{"condition": int, "contextual_belief": float, "risk_decision": int}]}
Identical format to earlier-model reliability data.

Reads from : utils.py  (shared constants)
Input data : data/frontier_models/reliability_N90/{DNC,CTD,FIP}/{model}_90.json
Output     : figures/reliability_frontier/
  Reliability_{task}_{model}_N90.png  — per-model 5-panel diagnostic
  Reliability_Summary_FrontierModels.png — cross-model P1/P2/IC summary table

Relation to other steps:
  → Step 2 (olr_frontier_models.py) and Step 3 (auc_frontier_models.py)
    assume B_C and R_D are reliable; this script provides that confirmation.
  Note: Grok 4 fails reliability in all tasks (high σ_r ≈ 2.89) and is
    excluded from the 5-model Kendall W analysis in Step 4.

Full dataset: https://huggingface.co/datasets/LLMsRiskAttitudeDataShare/DataShare_NeurIPS2026_LLMsRiskAttitude
"""

import json, math, sys, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ── shared utilities ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import FRONTIER_KEYS, FRONTIER_LABELS, TASKS

ROOT    = Path(__file__).parent.parent.parent
IC_DIR  = ROOT / 'data' / 'frontier_models' / 'reliability_N90'
FIG_DIR = ROOT / 'figures' / 'reliability_frontier'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = dict(zip(FRONTIER_KEYS, FRONTIER_LABELS))

CONDITION_COLORS = {1: '#3b82f6', 2: '#22c55e', 3: '#ef4444'}
RD_COLORS        = {1: '#1e3a8a', 2: '#3b82f6', 3: '#94a3b8', 4: '#f59e0b', 5: '#dc2626'}


def load_model_task(mkey, task):
    fpath = IC_DIR / task / f'{mkey}_90.json'
    if not fpath.exists():
        print(f'  WARNING: {fpath.name} not found — download full dataset from HuggingFace')
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int)
    with open(fpath, encoding='utf-8') as f:
        d = json.load(f)
    ctxs, rds, conds = [], [], []
    for t in d.get('trials', []):
        cond = t.get('condition')
        bc   = t.get('contextual_belief')
        rd   = t.get('risk_decision')
        if cond is not None and bc is not None and rd is not None:
            ctxs.append(float(bc))
            rds.append(int(rd))
            conds.append(int(cond))
    return np.array(ctxs), np.array(rds, dtype=int), np.array(conds, dtype=int)


def _normalized_entropy(values):
    cnt = Counter(values)
    n, K = len(values), len(cnt)
    if K <= 1:
        return 0.0
    H = -sum((c / n) * math.log(c / n) for c in cnt.values())
    return H / math.log(K)


def _rsd_ci95(values):
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if mean == 0:
        return 0.0, 0.0
    rsd = sd / mean * 100
    se = rsd * math.sqrt(1 / (2 * (n - 1)) + 1 / n)
    return max(0.0, rsd - 1.96 * se), rsd + 1.96 * se


def _prop_ci95(k, n):
    if n == 0:
        return 0.0, 0.0
    p, z = k / n, 1.96
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def compute_stats(ctx_arr, rd_arr, cond_arr):
    stats = {}
    for cond in sorted(set(cond_arr)):
        mask = cond_arr == cond
        ctxs = ctx_arr[mask]
        rds = rd_arr[mask]
        n = len(ctxs)
        if n == 0:
            continue
        ctx_mean = float(np.mean(ctxs))
        ctx_sd = float(np.std(ctxs))
        ci_half = 1.96 * ctx_sd / math.sqrt(n) if n > 1 else ctx_sd
        band5 = float(np.mean(np.abs(ctxs - ctx_mean) <= 5))
        cnt = Counter(rds.tolist())
        mode_rd, mode_n = cnt.most_common(1)[0]
        median_rd = float(np.median(rds))
        pm1_prop = float(np.mean(np.abs(rds - median_rd) <= 1))
        entropy = _normalized_entropy(rds.tolist())
        rsd = ctx_sd / ctx_mean * 100 if ctx_mean > 0 else 0.0
        rsd_lo, rsd_hi = _rsd_ci95(ctxs.tolist())
        dom_lo, dom_hi = _prop_ci95(mode_n, n)
        p1_pass = (ctx_sd / ctx_mean * 100 <= 20) if ctx_mean > 0 else False
        p2_pass = (mode_n / n) >= 0.80
        stats[cond] = dict(
            n=n, ctxs=ctxs.tolist(), rds=rds.tolist(),
            ctx_mean=ctx_mean, ctx_sd=ctx_sd,
            ctx_ci=(ctx_mean - ci_half, ctx_mean + ci_half),
            ctx_band5=band5, ctx_rsd=rsd,
            rsd_ci=(rsd_lo, rsd_hi),
            rd_mode=mode_rd, rd_mode_freq=mode_n / n,
            rd_median=median_rd, rd_pm1_prop=pm1_prop,
            rd_entropy=entropy,
            rd_dist={k: v / n for k, v in sorted(cnt.items())},
            dom_ci=(dom_lo, dom_hi),
            p1_pass=p1_pass, p2_pass=p2_pass,
        )
    return stats


def plot_ic(ctx_arr, rd_arr, cond_arr, model, task, out_path):
    stats = compute_stats(ctx_arr, rd_arr, cond_arr)
    conditions = sorted(stats.keys())
    y_pos = list(range(len(conditions)))
    all_rd = sorted(set(rd_arr.tolist()))
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 0.85], hspace=0.48, wspace=0.38)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, :])

    for cond in conditions:
        s = stats[cond]
        jit = rng.uniform(-0.18, 0.18, size=len(s['ctxs']))
        axA.scatter([cond + j for j in jit], s['ctxs'],
                    color=CONDITION_COLORS.get(cond, '#888'), s=50, alpha=0.72,
                    zorder=3, edgecolors='white', linewidths=0.4, label=f'C{cond}')
        q1, q3 = np.percentile(s['ctxs'], [25, 75])
        med = np.median(s['ctxs'])
        axA.plot([cond - 0.22, cond + 0.22], [med, med], color='#1e293b', linewidth=2.0, zorder=4)
        axA.add_patch(plt.Rectangle((cond - 0.22, q1), 0.44, q3 - q1,
                                    fill=False, edgecolor='#1e293b', linewidth=1.2, zorder=4))
    axA.set_xticks(conditions)
    axA.set_xticklabels([f'C{c}' for c in conditions], fontsize=11)
    axA.set_xlim(min(conditions) - 0.6, max(conditions) + 0.6)
    axA.set_ylim(-5, 105)
    axA.set_xlabel('Condition', fontsize=11)
    axA.set_ylabel('Contextual Belief ($B_C$)', fontsize=11)
    axA.set_title('A', fontsize=10, loc='left', fontweight='bold')
    axA.axhline(50, color='#aaa', linewidth=0.8, linestyle='--')
    axA.grid(True, axis='y', alpha=0.3)
    axA.legend(fontsize=9, loc='upper left', framealpha=0.8)

    for cond in conditions:
        s = stats[cond]
        jit = rng.uniform(-0.15, 0.15, size=len(s['rds']))
        axB.scatter(s['ctxs'], np.array(s['rds']) + jit,
                    color=CONDITION_COLORS.get(cond, '#888'), s=50, alpha=0.72,
                    zorder=3, edgecolors='white', linewidths=0.4, label=f'C{cond}')
    axB.set_xlim(-2, 102)
    axB.set_ylim(0.5, max(all_rd) + 0.5)
    axB.set_yticks(all_rd)
    axB.set_xlabel('Contextual Belief ($B_C$)', fontsize=11)
    axB.set_ylabel('Risk Decision ($R_D$)', fontsize=11)
    axB.set_title('B', fontsize=10, loc='left', fontweight='bold')
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=9, loc='best', framealpha=0.8)

    ctx_means = [stats[c]['ctx_mean'] for c in conditions]
    xerr_lo = [max(0.0, stats[c]['ctx_mean'] - stats[c]['ctx_ci'][0]) for c in conditions]
    xerr_hi = [max(0.0, stats[c]['ctx_ci'][1] - stats[c]['ctx_mean']) for c in conditions]
    colors = [CONDITION_COLORS.get(c, '#888') for c in conditions]
    axC.barh(y_pos, ctx_means, xerr=[xerr_lo, xerr_hi], color=colors, alpha=0.82, height=0.55,
             error_kw=dict(ecolor='#333', capsize=5, linewidth=1.5), zorder=3)
    for i, c in enumerate(conditions):
        s = stats[c]
        marker = '[OK]' if s['p1_pass'] else '[!!]'
        col = '#16a34a' if s['p1_pass'] else '#dc2626'
        axC.text(min(s['ctx_mean'] + xerr_hi[i] + 1.5, 90), i,
                 f'{marker}  SD={s["ctx_sd"]:.1f}  RSD={s["ctx_rsd"]:.0f}%',
                 va='center', fontsize=8, color=col)
    axC.set_yticks(y_pos)
    axC.set_yticklabels([f'C{c}  (n={stats[c]["n"]})' for c in conditions], fontsize=10)
    axC.set_xlim(0, 120)
    axC.set_xlabel('Contextual Belief ($B_C$)', fontsize=11)
    axC.set_title('C', fontsize=10, loc='left', fontweight='bold')
    axC.axvline(50, color='#aaa', linewidth=0.8, linestyle='--', zorder=1)
    axC.grid(True, axis='x', alpha=0.3)
    axC.text(0.01, 0.02, 'Pass: RSD ≤ 20%', transform=axC.transAxes, fontsize=8, color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8', edgecolor='#ccc'))

    for i, c in enumerate(conditions):
        s, left = stats[c], 0.0
        for cat in all_rd:
            prop = s['rd_dist'].get(cat, 0.0)
            if prop == 0.0:
                continue
            axD.barh(i, prop, left=left, color=RD_COLORS.get(cat, '#888'), alpha=0.85, height=0.55, zorder=3)
            if prop >= 0.08:
                axD.text(left + prop / 2, i, f'RD{cat}', ha='center', va='center',
                         fontsize=7.5, color='white', fontweight='bold')
            left += prop
        marker = '[OK]' if s['p2_pass'] else '[!!]'
        col = '#16a34a' if s['p2_pass'] else '#dc2626'
        axD.text(1.01, i,
                 f'{marker}  dom={s["rd_mode_freq"]:.0%}  ±1={s["rd_pm1_prop"]:.0%}  H={s["rd_entropy"]:.2f}',
                 va='center', fontsize=8, color=col, transform=axD.get_yaxis_transform())
    axD.set_yticks(y_pos)
    axD.set_yticklabels([f'C{c}' for c in conditions], fontsize=10)
    axD.set_xlim(0, 1)
    axD.set_xlabel('Proportion of trials', fontsize=11)
    axD.set_title('D', fontsize=10, loc='left', fontweight='bold')
    axD.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axD.grid(True, axis='x', alpha=0.3)
    patches = [Patch(color=RD_COLORS.get(c, '#888'), label=f'RD={c}') for c in all_rd]
    axD.legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.85)
    axD.text(0.01, 0.02, 'Pass: dom≥80%  |  H = normalised entropy',
             transform=axD.transAxes, fontsize=8, color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8', edgecolor='#ccc'))

    axE.axis('off')
    col_labels = ['Cond', 'n', '$B_C$ mean±SD', 'RSD% [95%CI]', 'P1',
                  'Mode $R_D$', 'Dom% [95%CI]', '±1 prop', 'H_norm', 'P2', 'Overall']
    rows = []
    for c in conditions:
        s = stats[c]
        rlo, rhi = s['rsd_ci']
        dlo, dhi = s['dom_ci']
        p1 = 'PASS' if s['p1_pass'] else 'FAIL'
        p2 = 'PASS' if s['p2_pass'] else 'FAIL'
        ov = 'PASS' if s['p1_pass'] and s['p2_pass'] else 'FAIL'
        rows.append([f'C{c}', str(s['n']), f"{s['ctx_mean']:.1f}±{s['ctx_sd']:.1f}",
                     f"{s['ctx_rsd']:.1f}% [{rlo:.1f},{rhi:.1f}]", p1,
                     f"RD={s['rd_mode']}", f"{s['rd_mode_freq']:.0%} [{dlo:.0%},{dhi:.0%}]",
                     f"{s['rd_pm1_prop']:.0%}", f"{s['rd_entropy']:.3f}", p2, ov])
    tbl = axE.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1e293b')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    pass_col = '#dcfce7'
    fail_col = '#fee2e2'
    neu = '#f8fafc'
    alt = '#f1f5f9'
    for ri, c in enumerate(conditions, start=1):
        s = stats[c]
        for ci in range(len(col_labels)):
            cell = tbl[ri, ci]
            if ci == 4:
                cell.set_facecolor(pass_col if s['p1_pass'] else fail_col)
                cell.set_text_props(fontweight='bold', color='#15803d' if s['p1_pass'] else '#b91c1c')
            elif ci == 9:
                cell.set_facecolor(pass_col if s['p2_pass'] else fail_col)
                cell.set_text_props(fontweight='bold', color='#15803d' if s['p2_pass'] else '#b91c1c')
            elif ci == 10:
                ok = s['p1_pass'] and s['p2_pass']
                cell.set_facecolor(pass_col if ok else fail_col)
                cell.set_text_props(fontweight='bold', color='#15803d' if ok else '#b91c1c')
            else:
                cell.set_facecolor(neu if ri % 2 == 1 else alt)
    axE.set_title('E', fontsize=10, loc='left', fontweight='bold', pad=12)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    p1n = sum(1 for c in conditions if stats[c]['p1_pass'])
    p2n = sum(1 for c in conditions if stats[c]['p2_pass'])
    ic_pass = p1n >= 2 and p2n >= 2
    return {c: stats[c] for c in conditions}, ic_pass


def plot_summary_table(all_results):
    fig, ax = plt.subplots(figsize=(14, 0.7 * len(FRONTIER_KEYS) + 2.5))
    ax.axis('off')
    col_labels = ['Model',
                  'DNC\nP1', 'FIP\nP1', 'CTD\nP1',
                  'DNC\nP2', 'FIP\nP2', 'CTD\nP2',
                  'DNC\nIC', 'FIP\nIC', 'CTD\nIC']
    rows = []
    for m in FRONTIER_KEYS:
        row = [MODEL_LABELS[m]]
        for phase in ['p1_pass', 'p2_pass', 'ic_pass']:
            for t in TASKS:
                val = all_results.get(t, {}).get(m, {}).get(phase)
                row.append('PASS' if val else 'FAIL')
        rows.append(row)
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1e293b')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    pass_col = '#dcfce7'
    fail_col = '#fee2e2'
    neu = '#f8fafc'
    alt = '#f1f5f9'
    for ri, m in enumerate(FRONTIER_KEYS, start=1):
        for ci in range(len(col_labels)):
            cell = tbl[ri, ci]
            if ci == 0:
                cell.set_facecolor(neu if ri % 2 == 1 else alt)
                cell.set_text_props(fontweight='bold')
            else:
                txt = rows[ri - 1][ci]
                ok = txt == 'PASS'
                cell.set_facecolor(pass_col if ok else fail_col)
                cell.set_text_props(fontweight='bold', color='#15803d' if ok else '#b91c1c')
    ax.text(0.5, -0.04,
            'P1 — Contextual Belief Consistency: mean RSD ≤ 20%  |  '
            'P2 — Risk Decision Consistency: dominant $R_D$ class ≥ 80%  |  '
            'IC: P1 ≥ 2/3 conditions AND P2 ≥ 2/3 conditions',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9.5, color='#444444',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f1f5f9', edgecolor='#cbd5e1'))
    plt.tight_layout()
    out = FIG_DIR / 'Reliability_Summary_FrontierModels.png'
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Summary table -> {out.name}')


if __name__ == '__main__':
    print('=' * 60)
    print('Reliability Analysis — Frontier Models (N=90)')
    print('=' * 60)

    all_results = {t: {} for t in TASKS}

    for task in TASKS:
        print(f'\n--- {task} ---')
        for model in FRONTIER_KEYS:
            try:
                ctx, rd, cond = load_model_task(model, task)
                if len(ctx) == 0:
                    all_results[task][model] = {'p1_pass': False, 'p2_pass': False, 'ic_pass': False}
                    continue
                out_path = FIG_DIR / f'Reliability_{task}_{model}_N90.png'
                stats, ic_pass = plot_ic(ctx, rd, cond, model, task, out_path)
                p1n = sum(1 for s in stats.values() if s['p1_pass'])
                p2n = sum(1 for s in stats.values() if s['p2_pass'])
                all_results[task][model] = {
                    'p1_pass': p1n >= 2, 'p2_pass': p2n >= 2, 'ic_pass': ic_pass
                }
                status = 'PASS' if ic_pass else 'FAIL'
                print(f'  {MODEL_LABELS[model]:20s}: IC={status}  P1={p1n}/3  P2={p2n}/3')
            except Exception as e:
                print(f'  {model}: ERROR — {e}')
                all_results[task][model] = {'p1_pass': False, 'p2_pass': False, 'ic_pass': False}

    print('\n--- Summary Table ---')
    plot_summary_table(all_results)
    print('\nDone. Figures saved to:', FIG_DIR)
