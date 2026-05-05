#!/usr/bin/env python3
"""
run_pipeline.py  —  NeurIPS 2026 Analysis Pipeline
====================================================
Entry point for the full analysis. Runs all steps in order.

Usage (from repository root):
    python analysis/run_pipeline.py

Data requirement:
    This repo contains one sample file per task (data format illustration).
    Full dataset (N=100/model/task × 12 models × 3 tasks) is available at:
        https://huggingface.co/datasets/[dataset-url]
    Download and place under data/ before running the full pipeline.

Pipeline:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  INPUT DATA                                                         │
    │  data/frontier_models/reliability_N90/{DNC,CTD,FIP}/         │
    │  data/earlier_models/reliability_N90/{DNC,CTD,FIP}/          │
    │  data/earlier_models/auc_N100/{DNC,CTD,FIP}/                       │
    │  data/frontier_models/main_N100/{DNC,CTD,FIP}/                     │
    └──────────────────┬──────────────────────────────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  STEP 1  Reliability / Intra-Consistency  (Paper §3.1)              │
    │  Script: 1_reliability/reliability_frontier_models.py               │
    │          1_reliability/reliability_earlier_models.py                │
    │  Logic:  IC data → P1 (RSD≤20%) + P2 (dom≥80%) per condition       │
    │  Output: figures/reliability_frontier/ + figures/reliability_earlier/│
    └──────────────────┬───────────────────────────────────────────────────┘
                       │  confirms B_C and R_D measurements are reliable
                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  STEP 2  OLR S-Curves  (Paper §3.2)                                 │
    │  Script: 2_olr/olr_frontier_models.py                               │
    │  Logic:  main_N100 → B_C/R_D extraction → OLR fit → E[R_D] curves  │
    │  Output: figures/olr_frontier/                                      │
    └──────────────────┬───────────────────────────────────────────────────┘
                       │  establishes B_C → R_D relationship per model
                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  STEP 3  AUC Computation  (Paper §3.3)                              │
    │  Script: 3_auc/auc_earlier_models.py  +  3_auc/auc_frontier_models  │
    │  Logic:  OLR curve → AUC = ∫ E[R_D] dB_C → scalar risk attitude    │
    │  Output: figures/auc_earlier/  +  figures/auc_frontier/             │
    └──────────────────┬───────────────────────────────────────────────────┘
                       │  AUC quantifies each model's risk attitude per task
                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  STEP 4  Cross-Task Ranking  (Paper §3.4)                           │
    │  Script: 4_ranking/risk_attitude_ranking.py                         │
    │  Logic:  AUC ranks → Kendall W concordance across DNC / FIP / CTD  │
    │  Output: figures/ranking/  +  KendallW_Summary.txt                  │
    └──────────────────────────────────────────────────────────────────────┘

All extraction functions are centralised in analysis/utils.py.
"""

import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SCRIPTS = [
    ROOT / 'analysis' / '1_reliability' / 'reliability_frontier_models.py',
    ROOT / 'analysis' / '1_reliability' / 'reliability_earlier_models.py',
    ROOT / 'analysis' / '2_olr'         / 'olr_frontier_models.py',
    ROOT / 'analysis' / '3_auc'         / 'auc_earlier_models.py',
    ROOT / 'analysis' / '3_auc'         / 'auc_frontier_models.py',
    ROOT / 'analysis' / '4_ranking'     / 'risk_attitude_ranking.py',
]

STEP_LABELS = [
    'Step 1 — Reliability (Frontier Models)',
    'Step 1 — Reliability (Earlier Models)',
    'Step 2 — OLR S-Curves',
    'Step 3 — AUC (Earlier Models)',
    'Step 3 — AUC (Frontier Models)',
    'Step 4 — Cross-Task Ranking',
]

print('=' * 68)
print('NeurIPS 2026  —  Full Analysis Pipeline')
print('=' * 68)

for label, script in zip(STEP_LABELS, SCRIPTS):
    print(f'\n>>> {label}')
    print(f'    {script.relative_to(ROOT)}')
    print('-' * 60)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f'  [ERROR] Script exited with code {result.returncode}')
        print('  Check that the full dataset is present in data/')
        print('  Full dataset: https://huggingface.co/datasets/[dataset-url]')

print('\n' + '=' * 68)
print('Pipeline complete.  Figures saved to:  figures/')
print('=' * 68)
