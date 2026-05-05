"""
analysis/utils.py
=================
Shared utilities for all analysis scripts in this repository.

Pipeline overview (see run_pipeline.py for full execution order):

  Step 1 — Reliability    : load IC data → compute P1/P2 per condition
  Step 2 — OLR S-curves   : load main_N100 → fit R_D ~ B_C → plot curves
  Step 3 — AUC            : load main_N100 → integrate OLR curve → AUC per model/task
  Step 4 — Ranking        : AUC-based ranks → Kendall W concordance

All extraction functions share the same B_C / R_D definitions:
  B_C (Contextual Belief)  — model's perceived risk level given situational context (0–100)
  R_D (Risk Decision)      — ordinal action chosen in response to that belief (1–5)

Full dataset:  https://huggingface.co/datasets/[dataset-url]
Sample format: data/{frontier,earlier}_models/  (one file per task in this repo)
"""

import warnings
import numpy as np
from sklearn.cluster import KMeans
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.integrate import trapezoid as sci_trap

warnings.filterwarnings('ignore')

EPS         = 1e-6
RANDOM_SEED = 42
CB_GRID     = np.linspace(0, 100, 201)   # B_C evaluation grid (percentage)

# ── Model registries ──────────────────────────────────────────────────────────

FRONTIER_KEYS   = ['DeepSeekV3.2', 'Gemini3Pro', 'GPT5.2', 'Grok4', 'Qwen3Max', 'ClaudeSonnet4.5']
FRONTIER_LABELS = ['DeepSeek V3.2', 'Gemini 3 Pro', 'GPT-5.2', 'Grok 4', 'Qwen3 Max', 'Sonnet 4.5']

EARLIER_KEYS    = ['DeepSeekV3', 'Gemini2.5Pro', 'GPT4.1', 'Grok3', 'Qwen3', 'Sonnet4']
EARLIER_LABELS  = ['DeepSeek V3', 'Gemini 2.5 Pro', 'GPT-4.1', 'Grok 3', 'Qwen 3', 'Sonnet 4']

TASKS       = ['DNC', 'FIP', 'CTD']
TASK_COLORS = {'DNC': '#E07070', 'FIP': '#70B870', 'CTD': '#5B9BD5'}
TASK_NAMES  = {
    'DNC': 'Drone Navigation Control',
    'FIP': 'Financial Investment Portfolio',
    'CTD': 'Clinical Triage Decision',
}


# ── B_C / R_D extraction — frontier models ────────────────────────────────────
# Frontier model JSON structure: {"trials": [{"steps": [...], "finalized": {...}}]}
# Steps contain per-move navigation data; B_C and R_D must be derived.

def extract_dnc(trial):
    """
    DNC — Drone Navigation Control
      B_C = last step's self-reported belief (0–100)
      SI  = log((VCR + ε) / (HPR + ε))  on drift steps only (drift_row ≠ 0)
            VCR = vertical correction rate (UP/DOWN actions)
            HPR = horizontal progression rate (RIGHT actions)
    Returns (B_C, SI) or (None, None).
    """
    steps = trial.get('steps', [])
    if not steps:
        return None, None
    bc          = float(steps[-1]['belief'])
    drift_steps = [s for s in steps if s.get('drift_row', 0) != 0] or steps
    vcr = sum(1 for s in drift_steps if s.get('action') in ('UP', 'DOWN'))
    hpr = sum(1 for s in drift_steps if s.get('action') == 'RIGHT')
    return bc, np.log((vcr + EPS) / (hpr + EPS))


def extract_fip(trial):
    """
    FIP — Financial Investment Portfolio
      B_C    = report.contextual.risk (0–100)
      h_prop = H / (L + M + H)  — high-risk asset allocation proportion
    Returns (B_C, h_prop) or (None, None).
    """
    report = trial.get('report', {})
    bc     = report.get('contextual', {}).get('risk')
    alloc  = report.get('alloc', {})
    l, m, h = (float(alloc.get(k, 0)) for k in ('L', 'M', 'H'))
    total  = l + m + h
    if total == 0 or bc is None:
        return None, None
    return float(bc), h / total


def extract_ctd(trial):
    """
    CTD — Clinical Triage Decision
      B_C = last BC_updates entry's ctx value (0–100)
      ESI = finalized.ESI  (1–5, already ordinal — used directly as R_D)
    Returns (B_C, ESI) or (None, None).
    """
    bc = None
    for s in reversed(trial.get('steps', [])):
        bu = s.get('BC_updates', [])
        if bu:
            bc = float(bu[-1]['ctx'])
            break
    esi = trial.get('finalized', {}).get('ESI')
    if bc is None or esi is None:
        return None, None
    return bc, int(esi)


# ── B_C / R_D extraction — earlier models ────────────────────────────────────
# Earlier model JSON structure: {"trials": [{"contextual_belief": X, "risk_decision": Y, ...}]}
# B_C and R_D are stored as top-level trial fields (pre-computed by the experiment interface).

def extract_earlier(trial, task):
    """
    Earlier-generation models: B_C and R_D stored directly at trial level.
    DNC: contextual_belief  | FIP: report.contextual.risk  | CTD: steps[-1].BC_updates[-1].ctx
    R_D: risk_decision field (already discretized 1–5).
    Returns (B_C, R_D) or (None, None).
    """
    if task == 'DNC':
        bc = trial.get('contextual_belief')
    elif task == 'FIP':
        bc = trial.get('report', {}).get('contextual', {}).get('risk')
    else:  # CTD
        steps = trial.get('steps', [])
        if not steps:
            return None, None
        bu = steps[-1].get('BC_updates', [])
        bc = float(bu[-1]['ctx']) if bu else None
    rd = trial.get('risk_decision')
    if bc is None or rd is None:
        return None, None
    return float(bc), int(rd)


# ── Global K-means discretization ─────────────────────────────────────────────

def discretize_kmeans(values, n_clusters=5, ascending=True):
    """
    Map continuous values to ordinal labels 1..n_clusters via K-means.
    Called ONCE per task across all models to ensure consistent cluster boundaries.

    ascending=True  (DNC): lowest SI cluster → R_D=1 (most aggressive/risk-taking)
    ascending=False (FIP): highest h_prop cluster → R_D=1 (most aggressive)

    CTD uses ESI directly (no discretization needed).
    """
    X       = np.array(values).reshape(-1, 1)
    km      = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels  = km.fit_predict(X)
    order   = np.argsort(km.cluster_centers_.flatten())
    if not ascending:
        order = order[::-1]
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[l] for l in labels])


# ── OLR fitting ───────────────────────────────────────────────────────────────

def fit_olr_auc(bc_arr, rd_arr):
    """
    Fit Ordered Logistic Regression: R_D ~ B_C (normalised to [0, 1]).
    Returns (auc, beta) or (None, None).

    AUC = ∫₀¹ E[R_D | B_C] dB_C  (trapezoid rule over 201-point grid)
    β   = OLR slope coefficient (risk sensitivity)
    """
    y = rd_arr.astype(int)
    x = (bc_arr / 100.0).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return None, None
    try:
        model  = OrderedModel(y, x, distr='logit')
        res    = model.fit(method='bfgs', disp=False)
        beta   = float(res.params[0])
        grid   = np.linspace(0, 1, 201)
        probs  = res.predict(exog=grid.reshape(-1, 1))
        cats   = model.labels.astype(float)
        auc    = float(sci_trap((probs * cats).sum(axis=1), grid))
        return auc, beta
    except Exception:
        return None, None


def fit_olr_curve(bc_arr, rd_arr, cb_grid=None):
    """
    Fit OLR and return E[R_D] curve over cb_grid (units: 0–100).
    Returns (e_rd_array, beta) or (None, None).
    Used by: 2_olr/olr_frontier_models.py
    """
    if cb_grid is None:
        cb_grid = CB_GRID
    y  = rd_arr.astype(int)
    x  = (bc_arr / 100.0).reshape(-1, 1)
    xg = (cb_grid / 100.0).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return None, None
    try:
        model = OrderedModel(y, x, distr='logit')
        res   = model.fit(method='bfgs', disp=False)
        cats  = model.labels.astype(float)
        beta  = float(res.params[0])
        e_rd  = np.clip((res.predict(exog=xg) * cats).sum(axis=1), 1, 5)
        return e_rd, beta
    except Exception:
        return None, None


# ── Frontier data loader (shared by steps 2, 3, 4) ───────────────────────────

def load_frontier_task(base_dir, task, model_keys):
    """
    Load all frontier model files for one task.
    Returns dict: model_key -> (bc_arr, raw_val_arr)
    where raw_val = SI (DNC), h_prop (FIP), or ESI (CTD).
    """
    import json
    raw = {}
    for mkey in model_keys:
        fpath = base_dir / task / f'{task}_{mkey}_N100.json'
        if not fpath.exists():
            print(f'  WARNING: {fpath.name} not found — download full dataset from HuggingFace')
            raw[mkey] = (np.array([]), np.array([]))
            continue
        with open(fpath, encoding='utf-8') as f:
            d = json.load(f)
        bc_list, val_list = [], []
        for t in d.get('trials', []):
            bc, val = (extract_dnc(t) if task == 'DNC' else
                       extract_fip(t) if task == 'FIP' else
                       extract_ctd(t))
            if bc is not None and val is not None:
                bc_list.append(bc)
                val_list.append(val)
        raw[mkey] = (np.array(bc_list), np.array(val_list))
    return raw


def discretize_frontier_task(raw, task, model_keys):
    """
    Apply global K-means across all models for DNC and FIP.
    CTD returns ESI directly (already ordinal 1–5).
    Returns dict: model_key -> (bc_arr, rd_arr).
    """
    if task == 'CTD':
        return {m: (bc, rd.astype(int)) for m, (bc, rd) in raw.items()}
    all_vals = np.concatenate([v for _, v in raw.values() if len(v) > 0])
    g_labels = discretize_kmeans(all_vals, ascending=(task == 'DNC'))
    result, offset = {}, 0
    for mkey in model_keys:
        bc, vals = raw[mkey]
        n = len(vals)
        result[mkey] = (bc, g_labels[offset:offset + n])
        offset += n
    return result
