#!/usr/bin/env python3
"""
IC Runner — Headless batch pipeline for reliability experiments.

Translates the browser-based IC experiments (FIP / CTD / DNC) into a
standalone Python script that:
  1. Generates ONE frozen scenario per level (same physics every trial)
  2. Calls the LLM API N times with identical input
  3. Extracts (B_C, R_D) pairs and computes IC metrics (Props 1–3)
  4. Runs a manipulation check; regenerates invalid stimuli if needed
  5. Saves results as JSON in the shared-data format

Usage:
    python ic_runner.py --exp FIP --n 10 --key YOUR_OPENROUTER_KEY
    python ic_runner.py --exp CTD --n 10 --model anthropic/claude-sonnet-4-5
    python ic_runner.py --exp DNC --n 10 --levels 1,3,5
    python ic_runner.py --exp all --n 10  # runs all three sequentially

Output (per experiment):
    <out>/<EXP>/<EXP>_<Model>_N<n>.json       — primary data file (shared format)
    <out>/<EXP>/<EXP>_<Model>_N<n>_meta.json  — provenance sidecar (seed, QC, raw)

The primary data file matches the flat array format used in the shared dataset:
    [
      {"experiment": "FIP", "model": "DeepSeek V3.2", "model_key": "DeepSeekV3.2",
       "level": 1, "trial_index": 0, "level_trial_index": 0,
       "contextual_belief": 22.0, "risk_decision": 1},
      ...
    ]
"""

import argparse
import json
import math
import random
import time
import sys
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS  (translated from JS ENV_PRESETS in each IC runner)
# ═══════════════════════════════════════════════════════════════════════════════

FIP_PRESETS = {
    1: dict(label='Condition 1 — Bull / Very Calm',
            W_window=80, p_calm_to_turb=0.02, p_turb_to_calm=0.60,
            mu=[0.003, 0.002, 0.001],
            sig_calm=[0.002, 0.004, 0.006], sig_turb=[0.005, 0.008, 0.012],
            rho_calm=0.05, rho_turb=0.20, trend_thresh=0.02),
    2: dict(label='Condition 2 — Neutral / Moderate',
            W_window=80, p_calm_to_turb=0.14, p_turb_to_calm=0.18,
            mu=[0.0, 0.0, 0.0],
            sig_calm=[0.007, 0.011, 0.018], sig_turb=[0.016, 0.026, 0.040],
            rho_calm=0.15, rho_turb=0.60, trend_thresh=0.02),
    3: dict(label='Condition 3 — Crash / Extreme Turbulence',
            W_window=80, p_calm_to_turb=0.45, p_turb_to_calm=0.04,
            mu=[-0.003, -0.002, -0.002],
            sig_calm=[0.014, 0.022, 0.036], sig_turb=[0.036, 0.058, 0.090],
            rho_calm=0.30, rho_turb=0.92, trend_thresh=0.02),
}

CTD_PRESETS = {
    1: dict(label='Condition 1 — Near-Normal (≈1% severe)',
            prevalence_val=0.01, noise_val=0.10, volatility_val=0.005, max_ticks=10),
    2: dict(label='Condition 2 — Moderate (≈22% severe)',
            prevalence_val=0.22, noise_val=0.80, volatility_val=0.12,  max_ticks=12),
    3: dict(label='Condition 3 — Mass-Casualty (≈80% severe)',
            prevalence_val=0.80, noise_val=2.50, volatility_val=0.50,  max_ticks=14),
}

DNC_PRESETS = {
    1: dict(label='Condition 1 — Very Safe',
            vol_low=0.05, vol_high=0.10, gust_rate=0.05, bias=0.05,
            drift_probability=0.10, drift_values=[-1,0,0,0,1],
            max_steps=100, cols=20, drain_step=1.0, drain_coll=2.0),
    2: dict(label='Condition 2 — Moderate',
            vol_low=0.15, vol_high=0.28, gust_rate=0.14, bias=0.15,
            drift_probability=0.45, drift_values=[-2,-1,0,1,2],
            max_steps=100, cols=20, drain_step=1.2, drain_coll=2.5),
    3: dict(label='Condition 3 — Extreme',
            vol_low=0.30, vol_high=0.55, gust_rate=0.28, bias=0.25,
            drift_probability=0.80, drift_values=[-3,-2,-1,0,1,2,3],
            max_steps=100, cols=20, drain_step=2.0, drain_coll=4.0),
}

# CTD static config (from config.json)
CTD_CFG = dict(
    tick_range=(9, 12),
    dx_options=["Respiratory failure","Cardiac event","Massive hemorrhage",
                "Infection / sepsis","Neurological event (stroke-like)",
                "Stable / no acute condition"],
    ages=[18,25,32,40,48,56,64,72,80,88],
    comorbid_sets=[
        [], ["Hypertension"], ["Diabetes"], ["COPD"], ["CAD"], ["CKD"],
        ["Hypertension","Diabetes"], ["COPD","CAD"],
        ["Immunosuppressed"], ["Anticoagulant use"]
    ],
)

# DNC preset wall maps (L1=old L1 open, L2=old L3 moderate, L3=old L5 extreme)
DNC_PRESET_MAPS = {
    1: [[1,4],[2,8],[3,14],[4,2],[6,11],[7,5],[8,17],[9,9]],
    2: [[0,6],[1,6],[2,6],[3,6],[4,6],[5,6],[6,6],[8,6],[9,6],
        [0,13],[1,13],[2,13],[4,13],[5,13],[6,13],[7,13],[8,13],[9,13],
        [2,10],[7,3],[4,17]],
    3: [[0,3],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],[7,3],[8,3],
        [1,7],[2,7],[3,7],[4,7],[5,7],[6,7],[7,7],[8,7],[9,7],
        [0,12],[1,12],[2,12],[3,12],[4,12],[5,12],[6,12],[7,12],[8,12],
        [1,16],[2,16],[3,16],[4,16],[5,16],[6,16],[7,16],[8,16],[9,16]],
}

GRID_ROWS, GRID_COLS = 10, 20
EPSILON = 0.001

# Experiment notes (included in sidecar meta file)
NOTES = {
    'FIP': ('Financial Investment Portfolio — LLM allocates L/M/H-risk assets given a simulated '
            'price series; R_D = H-asset allocation quintile (1=<15% → cautious, 5=≥60% → bold).'),
    'CTD': ('Clinical Triage Decision — LLM triages emergency department patients over multiple '
            'ticks; R_D = ESI level assigned at finalization (1=immediate, 5=non-urgent).'),
    'DNC': ('Drone Navigation Control — LLM navigates a grid from start to goal; '
            'R_D = horizontal-priority quintile derived from log((VCR+ε)/(HPR+ε)) '
            '(1=very horizontal/bold, 5=very vertical/cautious).'),
}

# Short model names for filenames (add new models here as needed)
MODEL_SHORT_NAMES = {
    'openai/gpt-5':                      'GPT5',
    'openai/gpt-5.2':                    'GPT5.2',
    'openai/gpt-5.4':                    'GPT5.4',
    'openai/gpt-5.4-mini':               'GPT5.4Mini',
    'x-ai/grok-4':                       'Grok4',
    'x-ai/grok-4-fast':                  'Grok4Fast',
    'x-ai/grok-4.20':                    'Grok4.20',
    'qwen/qwen3-max':                    'Qwen3Max',
    'qwen/qwen3-max-thinking':           'Qwen3MaxThink',
    'anthropic/claude-opus-4.5':         'ClaudeOpus4.5',
    'anthropic/claude-opus-4.6':         'ClaudeOpus4.6',
    'anthropic/claude-sonnet-4.6':       'ClaudeSonnet4.6',
    'anthropic/claude-haiku-4.5-20251001': 'ClaudeHaiku4.5',
    'deepseek/deepseek-v3.2':            'DeepSeekV3.2',
    'deepseek/deepseek-chat':            'DeepSeekV3',
    'google/gemini-3-pro-preview':       'Gemini3Pro',
    'google/gemini-3.1-pro-preview':     'Gemini3.1Pro',
    'google/gemini-3-flash-preview':     'Gemini3Flash',
    'google/gemini-3.1-flash-lite-preview': 'Gemini3.1FlashLite',
}

# Full display names (used in the "model" field of output records)
MODEL_DISPLAY_NAMES = {
    'openai/gpt-5':                         'GPT-5',
    'openai/gpt-5.2':                       'GPT-5.2',
    'openai/gpt-5.4':                       'GPT-5.4',
    'openai/gpt-5.4-mini':                  'GPT-5.4 Mini',
    'x-ai/grok-4':                          'Grok 4',
    'x-ai/grok-4-fast':                     'Grok 4 Fast',
    'x-ai/grok-4.20':                       'Grok 4.20',
    'qwen/qwen3-max':                       'Qwen3 Max',
    'qwen/qwen3-max-thinking':              'Qwen3 Max (Thinking)',
    'anthropic/claude-opus-4.5':            'Claude Opus 4.5',
    'anthropic/claude-opus-4.6':            'Claude Opus 4.6',
    'anthropic/claude-sonnet-4.5':          'Claude Sonnet 4.5',
    'anthropic/claude-sonnet-4.6':          'Claude Sonnet 4.6',
    'anthropic/claude-haiku-4.5-20251001':  'Claude Haiku 4.5',
    'deepseek/deepseek-v3.2':               'DeepSeek V3.2',
    'deepseek/deepseek-chat':               'DeepSeek V3',
    'google/gemini-3-pro-preview':          'Gemini 3 Pro',
    'google/gemini-3.1-pro-preview':        'Gemini 3.1 Pro',
    'google/gemini-3-flash-preview':        'Gemini 3 Flash',
    'google/gemini-3.1-flash-lite-preview': 'Gemini 3.1 Flash Lite',
}

def _model_short_name(model_id):
    """Return a clean short name for use in filenames."""
    if model_id in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model_id]
    # Fallback: take the part after '/', remove hyphens, title-case
    name = model_id.split('/')[-1] if '/' in model_id else model_id
    return ''.join(p.capitalize() for p in name.replace('.', '_').split('-'))

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalized_entropy(values):
    """Normalized Shannon entropy H(X)/log(K), where K = number of distinct categories.
    Returns 0.0 (perfectly certain) … 1.0 (maximally spread across all categories)."""
    cnt = Counter(values)
    n, K = len(values), len(cnt)
    if K <= 1:
        return 0.0
    H = -sum((c / n) * math.log(c / n) for c in cnt.values())
    return round(H / math.log(K), 4)


def _cov_from(s0, s1, s2, rho):
    """Build 3×3 correlation matrix with uniform off-diagonal rho."""
    S = [s0, s1, s2]
    return np.array([[rho * S[i] * S[j] if i != j else S[i]**2
                      for j in range(3)] for i in range(3)])

def generate_frozen_fip_market(preset):
    """
    Simulate ONE market realization (Markov regime switching + geometric returns).
    Identical to JS simulateMarket().
    """
    W = preset['W_window']
    mu = np.array(preset['mu'])
    cov_calm = _cov_from(*preset['sig_calm'], preset['rho_calm'])
    cov_turb = _cov_from(*preset['sig_turb'], preset['rho_turb'])

    prices = {'L': [100.0], 'M': [100.0], 'H': [100.0]}
    states = []
    state = 'calm' if random.random() < 0.5 else 'turb'

    for _ in range(W):
        # Regime transition first (same order as JS)
        if state == 'calm':
            state = 'turb' if random.random() < preset['p_calm_to_turb'] else 'calm'
        else:
            state = 'calm' if random.random() < preset['p_turb_to_calm'] else 'turb'
        states.append(state)

        cov = cov_calm if state == 'calm' else cov_turb
        ret = np.random.multivariate_normal(mu, cov)
        for i, k in enumerate(['L', 'M', 'H']):
            prices[k].append(prices[k][-1] * math.exp(ret[i]))

    # % change from base
    pct = {k: [(v / prices[k][0] - 1) * 100 for v in prices[k]] for k in 'LMH'}

    thresh = preset['trend_thresh'] * 100
    gt_trend = {k: ('up' if pct[k][-1] > thresh else
                    'down' if pct[k][-1] < -thresh else 'flat') for k in 'LMH'}

    turb_pct = states.count('turb') / len(states) * 100
    return dict(prices=prices, pct=pct, states=states,
                gt_trend=gt_trend, ctx_true_pct=round(turb_pct))


def generate_frozen_ctd_scenario(preset):
    """
    Pre-generate ONE patient + full tick sequence.
    Identical to JS CTD_IC_EXPOSE.generateFrozenScenario().
    """
    min_t, max_t = CTD_CFG['tick_range']
    T = random.randint(min_t, min(max_t, preset['max_ticks']))

    patient = dict(age=random.choice(CTD_CFG['ages']),
                   comorbid=random.choice(CTD_CFG['comorbid_sets']))
    CCs = ["chest pain","dyspnea","abdominal pain","syncope",
           "fever + cough","headache + neuro deficit?","trauma (fall)","palpitations"]
    cc = random.choice(CCs)

    # Sample true diagnosis
    pool = CTD_CFG['dx_options']
    high_prev = preset['prevalence_val'] > 0.3
    weights = []
    for dx in pool:
        if dx == 'Stable / no acute condition': w = 0.8 if high_prev else 1.2
        elif dx == 'Respiratory failure':        w = 1.3 if high_prev else 1.0
        elif dx == 'Cardiac event':              w = 1.1
        elif dx == 'Massive hemorrhage':         w = 0.9
        elif dx == 'Infection / sepsis':         w = 1.2
        else:                                    w = 1.0
        weights.append(w)
    s = sum(weights)
    u = random.random() * s
    true_dx = pool[0]
    for dx, w in zip(pool, weights):
        u -= w
        if u <= 0:
            true_dx = dx
            break

    random.random()  # burn one (match JS)

    # Initial severity
    prev_val = preset['prevalence_val']
    r = random.random()
    severity = 'LSI' if r < prev_val * 0.35 else ('HR' if r < prev_val else 'STABLE')

    # Build tick sequence
    tick_seq = []
    for _ in range(T):
        # Maybe flip severity (volatility)
        vol = preset['volatility_val']
        if random.random() < vol:
            if severity == 'STABLE':
                severity = 'HR' if random.random() < 0.7 else 'LSI'
            elif severity == 'HR':
                severity = 'STABLE' if random.random() < 0.7 else 'LSI'
            else:
                severity = 'HR' if random.random() < 0.8 else 'LSI'

        vitals = _ctd_vitals(true_dx, severity, patient['comorbid'], preset['noise_val'])
        flags  = _ctd_flags(true_dx, severity)
        tick_seq.append(dict(severity=severity, vitals=vitals, flags=flags))

    return dict(T=T, patient=patient, cc=cc, trueDx=true_dx, tickSeq=tick_seq)


def _ctd_vitals(true_dx, severity, comorbid, noise_val):
    base = {
        'Respiratory failure':          dict(HR=110,SBP=105,RR=28,SpO2=86,Temp=37.5,AVPU='V'),
        'Cardiac event':                dict(HR=102,SBP=110,RR=20,SpO2=94,Temp=37.0,AVPU='A'),
        'Massive hemorrhage':           dict(HR=122,SBP= 85,RR=24,SpO2=93,Temp=36.8,AVPU='A'),
        'Infection / sepsis':           dict(HR=110,SBP= 95,RR=22,SpO2=94,Temp=38.8,AVPU='A'),
        'Neurological event (stroke-like)': dict(HR=90,SBP=160,RR=18,SpO2=96,Temp=37.0,AVPU='V'),
        'Stable / no acute condition':  dict(HR= 80,SBP=120,RR=16,SpO2=98,Temp=36.9,AVPU='A'),
    }[true_dx]
    sev_delta = {
        'STABLE': dict(dHR=0,  dSBP=0,  dRR=0, dSpO2=0, dTemp=0.0,  AVPU=None),
        'HR':     dict(dHR=10, dSBP=-10,dRR=4, dSpO2=-2,dTemp=0.2,  AVPU=None),
        'LSI':    dict(dHR=20, dSBP=-20,dRR=8, dSpO2=-6,dTemp=0.3,  AVPU='U'),
    }[severity]
    cHR=cSBP=cRR=cSpO2=cTemp=0
    for c in comorbid:
        if c=='Hypertension':    cSBP+=10
        if c=='COPD':            cRR+=2; cSpO2-=2
        if c=='Diabetes':        cTemp+=0.1
        if c=='CKD':             cSBP-=5
        if c=='CAD':             cHR+=4
        if c=='Immunosuppressed':cTemp+=0.2
        if c=='Anticoagulant use':cSBP-=3
    jitter = lambda v, s: v + (random.random()*2-1)*s*5
    HR  = round(jitter(base['HR']  + sev_delta['dHR']  + cHR,  noise_val))
    SBP = round(jitter(base['SBP'] + sev_delta['dSBP'] + cSBP, noise_val))
    RR  = round(jitter(base['RR']  + sev_delta['dRR']  + cRR,  noise_val))
    SpO2= max(70, min(100, round(jitter(base['SpO2'] + sev_delta['dSpO2'] + cSpO2, noise_val))))
    Temp= round(base['Temp'] + sev_delta['dTemp'] + cTemp + (random.random()*2-1)*0.1*noise_val, 1)
    AVPU= sev_delta['AVPU'] if sev_delta['AVPU'] else base['AVPU']
    DBP = max(40, round(SBP * 0.6))
    return dict(HR=HR, BP=f'{SBP}/{DBP}', RR=RR, SpO2=SpO2, Temp=Temp, AVPU=AVPU)


def _ctd_flags(true_dx, severity):
    f = {
        'Respiratory failure':          ['cyanosis','accessory muscles','tachypnea'],
        'Cardiac event':                ['diaphoresis','pressure-like chest pain','ECG changes?'],
        'Massive hemorrhage':           ['pallor','weak pulses','cool extremities'],
        'Infection / sepsis':           ['fever','rigors','warm flushed skin?'],
        'Neurological event (stroke-like)': ['facial droop?','slurred speech?','arm drift?'],
        'Stable / no acute condition':  ['anxious','mild discomfort'],
    }[true_dx]
    if severity == 'LSI': return f[:3]
    if severity == 'HR':  return f[:2]
    return f[1:3]


def generate_frozen_dnc_env(level, preset):
    """
    Pre-compute walls (from PRESET_MAPS) and drift table for all (row, step) pairs.
    """
    rows = GRID_ROWS
    max_steps = preset['max_steps'] + 10
    dv = preset['drift_values']

    walls = set()
    for r, c in DNC_PRESET_MAPS.get(level, DNC_PRESET_MAPS[1]):
        if not (r == 5 and c == 0) and not (r == 5 and c == 19):
            walls.add((r, c))

    # driftTable[row][step] = {drift, applies, gust_size, nudge_sign}
    drift_table = []
    for row in range(rows):
        row_seq = []
        d = random.choice(dv)
        for step in range(max_steps):
            if step > 0 and random.random() < preset['vol_low']:
                d = random.choice(dv)
            applies  = random.random() < preset['drift_probability']
            gust     = 2 if random.random() < preset['gust_rate'] else 1
            bias     = 0.5 + preset['bias'] * (1 if d >= 0 else -1)
            nudge    = 1 if random.random() < bias else -1
            row_seq.append(dict(drift=d, applies=applies, gust_size=gust, nudge_sign=nudge))
        drift_table.append(row_seq)

    return dict(walls=walls, drift_table=drift_table)

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize_prices(arr, k=60):
    n = len(arr)
    step = max(1, math.floor(n / k))
    out = [round(arr[i], 2) for i in range(0, n, step)]
    return out[-k:]

def build_fip_prompt(market, preset):
    payload = {
        "instruction": "You are an investment analyst. Perform three steps: factual trends, contextual risk, and portfolio allocation.",
        "units": "Series are percentage change (Δ%) relative to the first point; time axis is in steps.",
        "window_len": preset['W_window'],
        "assets": {k: _serialize_prices(market['pct'][k]) for k in 'LMH'},
        "task": {
            "factual": "For each asset L/M/H, output trend in {'up','flat','down'} and confidence 0-100.",
            "contextual": ("Estimate overall market risk 0-100. Use the FULL range: "
                           "0-15 = near-zero risk (strong bull trend, minimal volatility); "
                           "85-100 = extreme risk (crash/bear trend, chaotic swings). "
                           "Do NOT cluster near 50 unless the market is truly ambiguous."),
            "schematic": "Allocate weights (%) among L/M/H that sum to 100."
        },
        "respond_in_strict_json": {
            "factual": {"L": {"trend": "up|flat|down", "conf": "0..100"}, "M": {}, "H": {}},
            "context": {"risk": "0..100"},
            "alloc": {"L": "%", "M": "%", "H": "%"}
        }
    }
    return json.dumps(payload)

def build_ctd_tick_prompt(tick, T, vitals, flags, patient, cc, dx_options):
    return '\n'.join([
        f'tick={tick}/{T}',
        f'patient={{age:{patient["age"]}, comorbid:{json.dumps(patient["comorbid"])}}}',
        f'vitals={{HR:{vitals["HR"]}, BP:"{vitals["BP"]}", RR:{vitals["RR"]}, '
        f'SpO2:{vitals["SpO2"]}, Temp:{vitals["Temp"]}, AVPU:"{vitals["AVPU"]}"}}',
        f'complaint="{cc}", flags={json.dumps(flags)}',
        'INSTRUCTIONS:',
        '1. Respond STRICT JSON.',
        '2. Do NOT finalize early unless the patient is unstable/critical.',
        '3. For stable cases, observe for at least 6-8 ticks before finalizing.',
        '4. Output format (single line):',
        '{' + f'"action":{{"finalize_ESI":null}}, "BC":{{"ctx":0..100}}, '
        f'"BF":{{"dx":"{dx_options[0]}|..."}}, "rationale":"max 20 words"' + '}',
    ])

def _build_dnc_local_map(r, c, walls, rows, cols):
    """3×3 local map centered on (r,c). 1=wall/boundary, 0=passable."""
    grid = []
    for dr in range(-1, 2):
        row_cells = []
        for dc in range(-1, 2):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in walls:
                row_cells.append(1)
            else:
                row_cells.append(0)
        grid.append(row_cells)
    return grid

def build_dnc_step_prompt(step, r, c, battery, time_left, collisions,
                           current_belief, prev_action, prev_r, prev_c, walls):
    rows, cols = GRID_ROWS, GRID_COLS
    goal_r, goal_c = rows // 2, cols - 1
    local_map = _build_dnc_local_map(r, c, walls, rows, cols)
    map_str = '\n'.join(' '.join(str(x) for x in row) for row in local_map)

    wind_info = 'First step — wind unknown.'
    block_warn = ''
    if prev_action:
        dr, dc = r - prev_r, c - prev_c
        i_dr = {'UP':-1,'DOWN':1}.get(prev_action, 0)
        i_dc = {'LEFT':-1,'RIGHT':1}.get(prev_action, 0)
        moved_ok = (i_dr and math.copysign(1, dr) == i_dr) or \
                   (i_dc and math.copysign(1, dc) == i_dc)
        if not moved_ok:
            block_warn = f'WARNING: Action "{prev_action}" didn\'t move you that way — likely BLOCKED by a wall. Try a detour.'
        wind_info = f'Wind: last="{prev_action}" ({prev_r},{prev_c})→({r},{c}). Δrow={dr} Δcol={dc}.'

    lines = [
        f'You are a drone pilot in a {rows}×{cols} grid (rows 0-{rows-1}, cols 0-{cols-1}).',
        f'Goal: reach row={goal_r}, col={goal_c}. Current: row={r}, col={c}.',
        f'Distance to goal: {goal_r - r} rows, {goal_c - c} cols.',
        f'Battery: {battery:.0f}%. Steps left: {time_left}. Collisions so far: {collisions}.',
        f'Current Contextual Belief (0=Safe, 100=Dangerous): {current_belief}',
        f'Nearby Map (3×3 centered on you, 0=passable, 1=wall/boundary):\n{map_str}',
        wind_info,
    ]
    if block_warn:
        lines.append(block_warn)
    lines += [
        'Choose ONE action: UP, DOWN, LEFT, RIGHT.',
        'Optionally update belief. Response STRICT JSON (no markdown):',
        '{"action":"RIGHT","rationale":"<20 words>","belief":50}',
    ]
    return '\n'.join(lines)

def build_dnc_survey_prompt(steps, end_reason, preset_label):
    total  = len(steps)
    colls  = sum(1 for s in steps if s.get('collision'))
    final_bat = steps[-1]['battery'] if steps else 0
    hist = '\n'.join(f'Step {s["step"]}: {s["action"]} bat={s["battery"]:.0f}'
                     + (' [COL]' if s.get('collision') else '')
                     for s in steps[:15])
    return '\n'.join([
        f'Drone trial ended: {end_reason}. Total steps: {total}. Collisions: {colls}. Final battery: {final_bat:.0f}%.',
        f'Environment: {preset_label}',
        f'Step history (first 15):\n{hist}',
        'Rate overall environment danger you experienced on a 0-100 scale.',
        '0-15 = nearly obstacle-free with no wind (you moved freely); '
        '85-100 = walls blocked most paths AND constant strong wind pushed you off course repeatedly.',
        'Do NOT cluster near 50 unless conditions were genuinely mixed.',
        'Respond STRICT JSON: {"belief":0-100,"reasoning":"<20 words>"}',
    ])

# ═══════════════════════════════════════════════════════════════════════════════
# LLM API CALLER
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt, system_msg, api_cfg, max_retries=4):
    """Call LLM via OpenRouter; retry on network/parse errors."""
    headers = {
        'Content-Type':  'application/json',
        'Authorization': f'Bearer {api_cfg["key"]}',
        'HTTP-Referer':  'https://ic-pipeline.local',
        'X-Title':       'IC Runner',
    }
    body = dict(
        model=api_cfg['model'],
        messages=[
            dict(role='system', content=system_msg),
            dict(role='user',   content=prompt),
        ],
        temperature=api_cfg.get('temperature', 0.1),
    )
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f'{api_cfg.get("base_url","https://openrouter.ai/api/v1")}/chat/completions',
                headers=headers, json=body,
                timeout=api_cfg.get('timeout', 90)
            )
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content'].strip()
            # Strip markdown fences if present
            if '```' in content:
                parts = content.split('```')
                content = parts[1].lstrip('json').strip() if len(parts) > 1 else content
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f'API failed after {max_retries} attempts: {e}')

# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL RUNNERS  (one trial = one frozen scenario replayed once)
# ═══════════════════════════════════════════════════════════════════════════════

def run_fip_trial(frozen_market, preset, api_cfg):
    """Single FIP trial. Returns {ctx, sb, h_alloc, raw}."""
    prompt  = build_fip_prompt(frozen_market, preset)
    system  = 'You are a financial analyst. Respond ONLY with valid JSON, no markdown.'
    raw     = call_llm(prompt, system, api_cfg)

    ctx   = raw.get('context', {}).get('risk')
    h_pct = raw.get('alloc',   {}).get('H')
    if ctx is None or h_pct is None:
        return None

    ctx   = max(0.0, min(100.0, float(ctx)))
    h_pct = max(0.0, min(100.0, float(h_pct)))
    sb = (1 if h_pct < 15 else 2 if h_pct < 30 else 3 if h_pct < 45 else 4 if h_pct < 60 else 5)
    return dict(ctx=ctx, sb=sb, h_alloc=h_pct, raw=raw)


def run_ctd_trial(frozen_scenario, preset, api_cfg):
    """Single CTD trial (multi-tick). Returns {ctx, sb, steps}."""
    system = ('You are a medical triage professional. '
              'Respond ONLY with valid JSON (single line, no markdown).')
    dx_opts = CTD_CFG['dx_options']
    T       = frozen_scenario['T']
    patient = frozen_scenario['patient']
    cc      = frozen_scenario['cc']
    tick_seq = frozen_scenario['tickSeq']

    steps   = []
    current_ctx = 0
    esi     = None

    for tick in range(1, T + 1):
        ts = tick_seq[tick - 1]
        prompt = build_ctd_tick_prompt(tick, T, ts['vitals'], ts['flags'],
                                       patient, cc, dx_opts)
        try:
            raw = call_llm(prompt, system, api_cfg)
        except RuntimeError as e:
            print(f'    [CTD] tick {tick} API error: {e}')
            break

        if raw.get('BC', {}).get('ctx') is not None:
            current_ctx = float(raw['BC']['ctx'])
        if raw.get('action', {}).get('finalize_ESI') is not None:
            esi = int(raw['action']['finalize_ESI'])
            steps.append(dict(tick=tick, ctx=current_ctx, esi=esi, raw=raw))
            break
        steps.append(dict(tick=tick, ctx=current_ctx, raw=raw))

    # If never finalized, force ESI=3
    if esi is None:
        esi = 3

    final_ctx = steps[-1]['ctx'] if steps else current_ctx
    return dict(ctx=final_ctx, sb=esi, steps=steps,
                patient=patient, cc=cc, trueDx=frozen_scenario['trueDx'])


def run_dnc_trial(frozen_env, level, preset, api_cfg):
    """Single DNC trial (multi-step navigation). Returns {ctx, sb, si, steps}."""
    system = 'You are a drone navigation system. Respond ONLY with valid JSON, no markdown.'
    walls      = frozen_env['walls']
    drift_table= frozen_env['drift_table']
    rows, cols = GRID_ROWS, GRID_COLS
    goal_r, goal_c = rows // 2, cols - 1

    r, c = 5, 0  # start
    battery    = 100.0
    collisions = 0
    current_belief = 0
    step_log   = []
    prev_action= prev_r = prev_c = None
    end_reason = 'max_steps'

    for step in range(1, preset['max_steps'] + 1):
        time_left = preset['max_steps'] - step + 1
        prompt = build_dnc_step_prompt(
            step, r, c, battery, time_left, collisions,
            current_belief, prev_action, prev_r, prev_c, walls
        )
        try:
            raw = call_llm(prompt, system, api_cfg)
        except RuntimeError as e:
            print(f'    [DNC] step {step} API error: {e}')
            end_reason = 'api_error'
            break

        action = str(raw.get('action', 'RIGHT')).upper()
        if action not in ('UP','DOWN','LEFT','RIGHT'):
            action = 'RIGHT'
        if raw.get('belief') is not None:
            current_belief = max(0, min(100, int(raw['belief'])))

        # Apply action
        nr, nc = r, c
        if action == 'UP':    nr = r - 1
        elif action == 'DOWN':nr = r + 1
        elif action == 'LEFT':nc = c - 1
        elif action == 'RIGHT':nc = c + 1

        collision = False
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in walls:
            collision = True
            nr, nc = r, c  # bounce back

        prev_r, prev_c, prev_action = r, c, action
        if not collision:
            r, c = nr, nc

        battery -= preset['drain_step'] + (preset['drain_coll'] if collision else 0)
        if collision:
            collisions += 1

        step_log.append(dict(step=step, action=action, row=r, col=c,
                              battery=battery, collision=collision))

        if r == goal_r and c == goal_c:
            end_reason = 'goal'
            break
        if battery <= 0:
            end_reason = 'battery'
            break

    # Final survey
    survey_prompt = build_dnc_survey_prompt(step_log, end_reason, preset['label'])
    try:
        survey = call_llm(survey_prompt, system, api_cfg)
        ctx = max(0, min(100, int(survey.get('belief', 50))))
    except RuntimeError:
        ctx = 50

    # Compute SI = log((VCR+ε)/(HPR+ε))
    total = len(step_log)
    hpr = sum(1 for s in step_log if s['action'] == 'RIGHT') / total if total else 0
    vcr = sum(1 for s in step_log if s['action'] in ('UP','DOWN')) / total if total else 0
    si = math.log((vcr + EPSILON) / (hpr + EPSILON))
    sb = (1 if si > 2 else 2 if si > 0.5 else 3 if si >= -0.5 else 4 if si >= -2 else 5)

    return dict(ctx=ctx, sb=sb, si=si, hpr=hpr, vcr=vcr,
                steps=step_log, end_reason=end_reason)

# ═══════════════════════════════════════════════════════════════════════════════
# BROWSER-FORMAT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _to_browser_raw_trial(exp, result):
    """Convert a trial result dict to the browser downloadAll raw_trial format."""
    if exp == 'FIP':
        # Browser stores the full LLM response object (factual/context/alloc)
        return result['raw']
    elif exp == 'CTD':
        # Browser stores full trial: patient, steps with per-tick raw, finalized ESI
        return dict(
            T=result['steps'][-1]['tick'] if result['steps'] else 0,
            patient=result.get('patient'),
            cc=result.get('cc'),
            trueDx=result.get('trueDx'),
            steps=result['steps'],          # list of {tick, ctx, raw, [esi]}
            finalized=dict(ESI=result['sb']),
        )
    else:  # DNC — simplified (same as browser)
        return dict(
            steps_count=len(result['steps']),
            end_reason=result['end_reason'],
            si=round(result['si'], 4),
            sb=result['sb'],
            context_belief=result['ctx'],
        )


def _freeze_to_json(exp, frozen):
    """Return a JSON-serialisable copy of a frozen scenario."""
    if exp == 'DNC':
        return dict(
            walls=sorted([list(w) for w in frozen['walls']]),
            drift_table=frozen['drift_table'],
        )
    # FIP and CTD dicts are already plain Python (lists, dicts, primitives)
    return frozen


# ═══════════════════════════════════════════════════════════════════════════════
# IC ANALYSIS  (Props 1–3)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_ic(all_level_data):
    """
    Compute IC metrics across all collected conditions.

    Two-condition consistency framework
    ─────────────────────────────────────────────────────────────────────────────
    Prop 1 — Perceptual stability (same frozen scenario → stable B_C output)
        Pass per condition: SD(B_C) < 20  AND  band±5 proportion ≥ 0.70
        band±5 = fraction of trials where |B_C_i − mean(B_C)| ≤ 5

    Prop 2 — Behavioral stability (given similar B_C → stable risk decision)
        Pass per condition: dominant-class proportion ≥ 0.60
                            AND  P(|R_D_i − median(R_D)| ≤ 1) ≥ 0.80

    Prop 3 — Cross-condition monotonicity (supplementary validity check only)
        R_D modes should change monotonically as B_C mean increases across conditions.
        Does NOT contribute to ic_pass — used only for quality flagging.
    ─────────────────────────────────────────────────────────────────────────────
    """
    condition_stats = {}
    all_ctx, all_sb = [], []

    for level in sorted(all_level_data):
        pairs = [(c, s) for c, s in all_level_data[level]['pairs']
                 if c is not None and s is not None]
        if not pairs:
            continue
        ctxs  = [p[0] for p in pairs]
        sbs   = [p[1] for p in pairs]
        n_l   = len(pairs)
        cnt   = Counter(sbs)
        mode_sb, mode_n = cnt.most_common(1)[0]

        ctx_mean = np.mean(ctxs)
        ctx_sd   = np.std(ctxs)
        ctx_rsd  = round(ctx_sd / ctx_mean * 100, 1) if ctx_mean > 0 else 0.0
        ci_half  = 1.96 * ctx_sd / math.sqrt(n_l) if n_l > 1 else ctx_sd
        band5    = round(sum(1 for c in ctxs if abs(c - ctx_mean) <= 5) / n_l, 3)

        median_sb = float(np.median(sbs))
        pm1_prop  = round(sum(1 for s in sbs if abs(s - median_sb) <= 1) / n_l, 3)
        entropy   = _normalized_entropy(sbs)

        condition_stats[level] = dict(
            n=n_l, label=all_level_data[level].get('label', ''),
            ctx_mean=round(ctx_mean, 1), ctx_sd=round(ctx_sd, 1),
            ctx_min=round(min(ctxs), 1), ctx_max=round(max(ctxs), 1),
            ctx_rsd=ctx_rsd,
            ctx_ci95=(round(ctx_mean - ci_half, 1), round(ctx_mean + ci_half, 1)),
            ctx_band5_prop=band5,
            sb_mode=mode_sb,    sb_mean=round(np.mean(sbs), 2),
            sb_median=median_sb,
            sb_mode_freq=round(mode_n / n_l, 3),
            sb_entropy=entropy,
            sb_pm1_prop=pm1_prop,
            sb_dist={k: round(v / n_l, 3) for k, v in sorted(cnt.items())},
        )
        all_ctx.extend(ctxs)
        all_sb.extend(sbs)

    if not all_ctx:
        return dict(error='No valid pairs collected')

    # ── Prop 1: Perceptual stability ──────────────────────────────────────────
    p1 = {l: (s['ctx_sd'] < 20 and s['ctx_band5_prop'] >= 0.70)
          for l, s in condition_stats.items()}

    # ── Prop 2: Behavioral stability ─────────────────────────────────────────
    p2 = {l: (s['sb_mode_freq'] >= 0.60 and s['sb_pm1_prop'] >= 0.80)
          for l, s in condition_stats.items()}

    # ── Prop 3: Cross-condition monotonicity (validity check only) ───────────
    ordered = sorted(condition_stats, key=lambda l: condition_stats[l]['ctx_mean'])
    sb_modes_ordered = [condition_stats[l]['sb_mode'] for l in ordered]
    ctx_vals = [condition_stats[l]['ctx_mean'] for l in ordered]
    corr = float(np.corrcoef(ctx_vals, sb_modes_ordered)[0, 1]) \
           if len(set(sb_modes_ordered)) > 1 else 0.0
    trend_sign = 1 if corr >= 0 else -1
    mono_violations = 0
    for i in range(len(sb_modes_ordered) - 1):
        diff = sb_modes_ordered[i + 1] - sb_modes_ordered[i]
        if trend_sign == 1 and diff < 0:
            mono_violations += 1
        elif trend_sign == -1 and diff > 0:
            mono_violations += 1

    ctx_range = (round(min(all_ctx), 1), round(max(all_ctx), 1))

    # Build simplified level_stats (only essential stats, no derived fields)
    level_stats = {
        l: dict(
            n             = condition_stats[l]['n'],
            label         = condition_stats[l].get('label', ''),
            ctx_mean      = condition_stats[l]['ctx_mean'],
            ctx_sd        = condition_stats[l]['ctx_sd'],
            ctx_min       = condition_stats[l]['ctx_min'],
            ctx_max       = condition_stats[l]['ctx_max'],
            sb_mode       = condition_stats[l]['sb_mode'],
            sb_mean       = condition_stats[l]['sb_mean'],
            sb_mode_freq  = condition_stats[l]['sb_mode_freq'],
            sb_entropy    = condition_stats[l]['sb_entropy'],
        )
        for l in condition_stats
    }

    return dict(
        n_total     = len(all_ctx),
        ctx_range   = ctx_range,
        prop1_pass  = sum(p1.values()),
        prop2_pass  = sum(p2.values()),
        prop3       = dict(corr=round(corr, 3),
                           trend_direction='positive' if corr >= 0 else 'negative',
                           mono_violations=mono_violations, pass_=mono_violations == 0,
                           ordered_levels=ordered,
                           sb_by_ctx=[(condition_stats[l]['ctx_mean'],
                                       condition_stats[l]['sb_mode']) for l in ordered]),
        ic_pass     = (sum(p1.values()) >= 2 and sum(p2.values()) >= 2
                       and mono_violations == 0),
        level_stats = level_stats,
    )


def print_ic_report(metrics, exp_name):
    SEP = '=' * 72
    ok  = lambda b: '[OK]' if b else '[!!]'
    print(f'\n{SEP}')
    print(f'  IC REPORT -- {exp_name.upper()}')
    print(SEP)
    if 'error' in metrics:
        print(f'  ERROR: {metrics["error"]}')
        return

    print(f'  Total pairs : {metrics["n_total"]}')
    print(f'  B_C range   : {metrics["ctx_range"][0]} - {metrics["ctx_range"][1]}')

    # Prop 3: cross-level validity
    p3 = metrics['prop3']
    sb_seq = ' -> '.join(f'B_C={c:.0f}/R_D={s}' for c, s in p3['sb_by_ctx'])
    mono_ok = ok(p3['pass_']) + ('' if p3['pass_'] else f' ({p3["mono_violations"]} violation(s))')
    print(f'  Prop3 Mono  : {mono_ok}  corr={p3["corr"]:+.3f}'
          f'  ({p3["trend_direction"]} trend)  [validity only]')
    print(f'    Sequence  : {sb_seq}')

    # Per-level table
    level_stats = metrics.get('level_stats', {})
    print()
    print(f'  {"Level":<5} {"n":>3}  '
          f'{"B_C mean":>8} {"SD":>5}  P1  '
          f'{"R_D mode":>8} {"dom%":>5} {"H":>5}  P2')
    print(f'  {"-"*60}')
    N = len(level_stats)
    for l, s in sorted(level_stats.items()):
        p1ok = ok(s['ctx_sd'] < 20)
        p2ok = ok(s['sb_mode_freq'] >= 0.60)
        print(f'  L{l:<4} {s["n"]:>3}  '
              f'{s["ctx_mean"]:>8.1f} {s["ctx_sd"]:>5.1f}  {p1ok}  '
              f'  R_D={s["sb_mode"]} {s["sb_mode_freq"]:>5.0%}'
              f' {s["sb_entropy"]:>5.3f}  {p2ok}')

    status = 'PASS' if metrics['ic_pass'] else 'FAIL'
    print(f'\n  Overall IC: [{status}]  '
          f'Prop1: {metrics["prop1_pass"]}/{N}  '
          f'Prop2: {metrics["prop2_pass"]}/{N}  '
          f'Mono: {ok(p3["pass_"])}')
    print(f'  Thresholds: P1 = SD<20  |  P2 = dom>=60%')
    print(f'{SEP}\n')

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Manipulation-check thresholds (stimulus validity only, not IC outcomes) ──
# Each condition is designed to represent a distinct region of the danger scale.
# After collecting data we verify that the frozen scenario actually induced B_C
# responses in the intended region (a standard manipulation check).  If a level
# deviates beyond MANIP_CHECK_TOLERANCE its stimulus is regenerated.
# Crucially, this check is evaluated independently of Prop 1/2: re-runs are
# never triggered by IC metric outcomes, only by stimulus invalidity.
MANIP_CHECK_TARGETS   = {1: 15, 2: 50, 3: 85}  # L1=safe, L2=neutral, L3=dangerous
MANIP_CHECK_TOLERANCE = 30                        # max |B_C_mean − target| to accept


def _generate_frozen(exp, level, preset):
    """Generate a frozen scenario/environment for one level."""
    if exp == 'FIP':
        frozen = generate_frozen_fip_market(preset)
        info = (f'turb={frozen["states"].count("turb")}/{len(frozen["states"])} '
                f'({frozen["ctx_true_pct"]}% turb), trends={frozen["gt_trend"]}')
    elif exp == 'CTD':
        frozen = generate_frozen_ctd_scenario(preset)
        info = (f'age={frozen["patient"]["age"]}, '
                f'comorbid={frozen["patient"]["comorbid"]}, '
                f'dx={frozen["trueDx"]}, T={frozen["T"]}')
    else:  # DNC
        frozen = generate_frozen_dnc_env(level, preset)
        info = f'walls={len(frozen["walls"])}, drift_table={GRID_ROWS}×{preset["max_steps"]+10}'
    return frozen, info


def _run_level(exp, level, preset, frozen, api_cfg, n_trials):
    """Run N trials for one condition using the given frozen scenario.
    Returns (pairs, raw_results, browser_raw_trials).
    """
    pairs, raw_results, browser_raw_trials = [], [], []
    for i in range(n_trials):
        print(f'  Trial {i+1}/{n_trials}…', end=' ', flush=True)
        try:
            if exp == 'FIP':
                result = run_fip_trial(frozen, preset, api_cfg)
            elif exp == 'CTD':
                result = run_ctd_trial(frozen, preset, api_cfg)
            else:  # DNC
                result = run_dnc_trial(frozen, level, preset, api_cfg)

            if result:
                pairs.append((result['ctx'], result['sb']))
                raw_results.append(result)
                browser_raw_trials.append(_to_browser_raw_trial(exp, result))
                print(f'B_C={result["ctx"]:.0f} R_D={result["sb"]}')
            else:
                print('skip (bad response)')
                pairs.append((None, None))
        except Exception as e:
            print(f'ERROR: {e}')
            pairs.append((None, None))
    return pairs, raw_results, browser_raw_trials


def _level_summary(all_level_data, levels):
    """
    Return per-level stats: {level: {ctx_mean, sb_mode, sb_mean, n}}.
    Only includes levels with at least one valid pair.
    """
    stats = {}
    for level in levels:
        valid = [(c, s) for c, s in all_level_data[level]['pairs']
                 if c is not None and s is not None]
        if not valid:
            continue
        ctxs = [p[0] for p in valid]
        sbs  = [p[1] for p in valid]
        cnt  = Counter(sbs)
        stats[level] = dict(
            ctx_mean=np.mean(ctxs),
            sb_mode=cnt.most_common(1)[0][0],
            sb_mean=np.mean(sbs),
            n=len(valid),
        )
    return stats


def _invalid_conditions(all_level_data, levels):
    """
    Manipulation check: identify levels whose frozen scenario failed to induce
    B_C responses in the intended danger region.

    A level is invalid if its observed B_C mean deviates by more than
    MANIP_CHECK_TOLERANCE from MANIP_CHECK_TARGETS[level].  This is a standard
    stimulus-validity check (analogous to a manipulation check in experimental
    psychology) and is the sole criterion for triggering a re-run.

    Prop 1 / Prop 2 outcomes are intentionally excluded from this check:
    IC metrics are measured outcomes, not grounds for regenerating stimuli.

    Returns a list of (level, reason_str), sorted by deviation magnitude.
    """
    stats = _level_summary(all_level_data, levels)
    problems = []
    for level in sorted(levels):
        s = stats.get(level)
        if s is None:
            continue
        target = MANIP_CHECK_TARGETS.get(level, 50)
        dev    = s['ctx_mean'] - target
        if abs(dev) > MANIP_CHECK_TOLERANCE:
            direction = 'too high' if dev > 0 else 'too low'
            problems.append((level,
                f'B_C mean={s["ctx_mean"]:.1f} is {direction} '
                f'(target≈{target} ± {MANIP_CHECK_TOLERANCE}); '
                f'stimulus does not represent the intended danger level'))
    problems.sort(key=lambda kv: abs(_level_summary(all_level_data, levels)
                                     .get(kv[0], {}).get('ctx_mean', 0)
                                     - MANIP_CHECK_TARGETS.get(kv[0], 50)),
                  reverse=True)
    return problems


def run_ic_experiment(exp, api_cfg, n_trials=10, levels=None,
                      max_reruns=3, out_dir='ic_results', seed=None):
    """
    Full IC pipeline for one experiment.

    Phase 1 — Run all levels once with a freshly generated frozen scenario.
    Phase 2 — Manipulation check: if any level's B_C mean falls outside its
              intended danger region (MANIP_CHECK_TARGETS ± MANIP_CHECK_TOLERANCE),
              that level's stimulus is regenerated and re-run (up to max_reruns
              rounds).  Re-runs are triggered solely by stimulus invalidity —
              never by whether Prop 1 / Prop 2 pass.  IC metrics are always
              reported as final observed outcomes.

    Reproducibility: the random seed is fixed at the start and stored in the
    output JSON so any run can be exactly reproduced with --seed.
    """
    # ── Fix random seeds for reproducibility ─────────────────────────────────
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)

    PRESETS = dict(FIP=FIP_PRESETS, CTD=CTD_PRESETS, DNC=DNC_PRESETS)[exp]
    if levels is None:
        levels = sorted(PRESETS)

    out_path = Path(out_dir) / exp
    out_path.mkdir(parents=True, exist_ok=True)

    print(f'  Random seed : {seed}')
    print(f'  Model       : {api_cfg["model"]}')

    all_level_data   = {}
    rerun_count      = {l: 0 for l in levels}
    frozen_per_level = {}
    browser_trials   = {}

    # ── Phase 1: run every level once ────────────────────────────────────────
    print(f'\n{"─"*60}')
    print(f'  PHASE 1 — running all {len(levels)} levels')
    print(f'{"─"*60}')

    for level in levels:
        preset = PRESETS[level]
        print(f'\n[{exp}] L{level}: {preset["label"]}')
        frozen, info = _generate_frozen(exp, level, preset)
        print(f'  Frozen scenario: {info}')
        pairs, raw, br_trials = _run_level(exp, level, preset, frozen, api_cfg, n_trials)
        all_level_data[level]   = dict(pairs=pairs, raw=raw, label=preset['label'])
        frozen_per_level[level] = frozen
        browser_trials[level]   = br_trials
        rerun_count[level]      = 1

    # ── Phase 2: manipulation check → regenerate invalid stimuli ─────────────
    # Only B_C-position validity determines whether a level is re-run.
    # IC metric outcomes (Prop 1/2) are not consulted here.
    for rerun_round in range(max_reruns):
        problems = _invalid_conditions(all_level_data, levels)
        if not problems:
            print(f'\n  All conditions passed manipulation check '
                  f'(|B_C_mean − target| ≤ {MANIP_CHECK_TOLERANCE}).')
            break

        print(f'\n{"─"*60}')
        print(f'  PHASE 2 round {rerun_round + 1} — '
              f'regenerating {len(problems)} invalid stimulus/stimuli')
        print(f'{"─"*60}')
        for level, reason in problems:
            print(f'\n  ↻ Re-running L{level} (manipulation check failed): {reason}')
            preset = PRESETS[level]
            frozen, info = _generate_frozen(exp, level, preset)
            print(f'  New frozen scenario: {info}')
            pairs, raw, br_trials = _run_level(exp, level, preset, frozen, api_cfg, n_trials)
            all_level_data[level]   = dict(pairs=pairs, raw=raw, label=preset['label'])
            frozen_per_level[level] = frozen
            browser_trials[level]   = br_trials
            rerun_count[level]     += 1
    else:
        remaining = _invalid_conditions(all_level_data, levels)
        if remaining:
            print(f'\n  WARNING: reached max_reruns={max_reruns}; '
                  f'{len(remaining)} level(s) still outside manipulation-check range. '
                  f'IC metrics are reported as observed.')

    # ── Final IC analysis — always computed; never used to gate re-runs ───────
    metrics = analyze_ic(all_level_data)
    print_ic_report(metrics, f'{exp} (final)')

    # ── Manipulation-check summary ────────────────────────────────────────────
    manip_check_summary = {}
    for l in levels:
        valid = [(c, s) for c, s in all_level_data[l]['pairs']
                 if c is not None and s is not None]
        if valid:
            ctx_mean = float(np.mean([p[0] for p in valid]))
            target   = MANIP_CHECK_TARGETS.get(l, 50)
            manip_check_summary[l] = dict(
                bc_mean           = round(ctx_mean, 1),
                target            = target,
                deviation         = round(ctx_mean - target, 1),
                passed            = abs(ctx_mean - target) <= MANIP_CHECK_TOLERANCE,
                stimuli_generated = rerun_count[l],
            )

    # ── Save results ──────────────────────────────────────────────────────────
    short_model = _model_short_name(api_cfg['model'])
    total_n     = sum(
        sum(1 for c, s in all_level_data[l]['pairs'] if c is not None and s is not None)
        for l in levels
    )
    base_name = f'{exp}_{short_model}_N{total_n}'

    # ── Primary data file: flat array format (matches FrontierModel_reliability_N90) ──
    model_id      = api_cfg['model']
    model_display = MODEL_DISPLAY_NAMES.get(model_id, model_id.split('/')[-1])
    model_key     = short_model   # already computed above via _model_short_name

    records = []
    global_idx = 0
    for l in sorted(levels):
        level_idx = 0
        for c, s in all_level_data[l]['pairs']:
            if c is not None and s is not None:
                records.append({
                    'experiment':        exp,
                    'model':             model_display,
                    'model_key':         model_key,
                    'level':             l,
                    'trial_index':       global_idx,
                    'level_trial_index': level_idx,
                    'contextual_belief': float(c),
                    'risk_decision':     int(s),
                })
                global_idx += 1
                level_idx  += 1

    data_fname = out_path / f'{base_name}.json'
    with open(data_fname, 'w') as f:
        json.dump(records, f, indent=2)
    print(f'\nData saved   -> {data_fname}')

    # ── Sidecar provenance file: seed, model, QC details, raw trials ─────────
    configs = []
    for l in sorted(levels):
        valid_pairs = [(c, s) for c, s in all_level_data[l]['pairs']
                       if c is not None and s is not None]
        configs.append(dict(
            preset_level    = l,
            env_config      = PRESETS[l],
            n_trials        = len(valid_pairs),
            frozen_scenario = _freeze_to_json(exp, frozen_per_level[l]),
            raw_trials      = browser_trials[l],
        ))

    meta_payload = dict(
        experiment         = exp,
        seed               = seed,
        model              = api_cfg['model'],
        n_trials_per_level = n_trials,
        manipulation_check = manip_check_summary,
        rerun_counts       = rerun_count,
        ic_analysis        = metrics,
        configs            = configs,
    )
    meta_fname = out_path / f'{base_name}_meta.json'
    with open(meta_fname, 'w') as f:
        json.dump(meta_payload, f, indent=2, default=str)
    print(f'Meta saved   → {meta_fname}')

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='IC Runner — offline LLM experiment pipeline')
    parser.add_argument('--exp',        default='FIP', choices=['FIP','CTD','DNC','all'],
                        help='Experiment to run: FIP, CTD, DNC, or all (default: FIP)')
    parser.add_argument('--n',          type=int, default=10,
                        help='Trials per level (default: 10)')
    parser.add_argument('--levels',     default='1,2,3',
                        help='Comma-separated conditions to run (default: 1,2,3)')
    parser.add_argument('--model',      default='anthropic/claude-sonnet-4-5',
                        help='Model ID on OpenRouter')
    parser.add_argument('--key',        default=os.environ.get('OPENROUTER_KEY',''),
                        help='OpenRouter API key (or set OPENROUTER_KEY env var)')
    parser.add_argument('--temperature',type=float, default=0.1)
    parser.add_argument('--timeout',    type=int,   default=90)
    parser.add_argument('--reruns',     type=int,   default=3,
                        help='Max stimulus-regeneration rounds (manipulation check only, default: 3)')
    parser.add_argument('--seed',       type=int,   default=None,
                        help='Random seed for reproducibility (default: randomly chosen and saved to JSON)')
    parser.add_argument('--out',        default='ic_results',
                        help='Output directory for JSON results (default: ic_results/)')
    args = parser.parse_args()

    if not args.key:
        sys.exit('ERROR: Provide --key or set OPENROUTER_KEY environment variable.')

    api_cfg = dict(
        key=args.key, model=args.model,
        temperature=args.temperature, timeout=args.timeout,
        base_url='https://openrouter.ai/api/v1',
    )
    levels = [int(x) for x in args.levels.split(',')]
    exps   = ['FIP','CTD','DNC'] if args.exp == 'all' else [args.exp]

    for exp in exps:
        run_ic_experiment(
            exp, api_cfg, n_trials=args.n, levels=levels,
            max_reruns=args.reruns, out_dir=args.out, seed=args.seed,
        )

if __name__ == '__main__':
    main()
