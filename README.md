# NeurIPS 2026 — Reproducibility Package

This repository contains all code and data needed to reproduce the main results
reported in the paper.

---

## Repository Structure

```
NeurIPS_CodeSharing/
├── data/
│   ├── frontier_models/
│   │   ├── main_N100/          — LLM response data (N=100 trials/model/task)
│   │   │   ├── DNC/            — 6 × DNC_{model}_N100.json
│   │   │   ├── CTD/            — 6 × CTD_{model}_N100.json
│   │   │   └── FIP/            — 6 × FIP_{model}_N100.json
│   │   └── reliability_N90/    — reliability data (N=90/model/task)
│   │       ├── DNC/            — 6 × {model}_90.json
│   │       ├── CTD/            — 6 × {model}_90.json
│   │       └── FIP/            — 6 × {model}_90.json
│   ├── earlier_models/
│   │   ├── auc_N100/           — 6 earlier-gen models × 3 tasks × N=100 trials
│   │   └── reliability_N90/    — reliability data for earlier-gen models
│   └── human_data/             — human participant data (CSV)
├── experiments/
│   ├── HumanExperimentConfig/  — browser-based interface for human participants
│   │   ├── index.html          — participant portal (task hub)
│   │   ├── Experiment_DNC/     — Drone Navigation Control
│   │   ├── Experiment_FIP/     — Financial Investment Portfolio
│   │   ├── Experiment_CTD/     — Clinical Triage Decision
│   │   └── deployment_guide.md — Google Sheets webhook setup
│   └── LLMsExperimentConfig/   — LLM API interface (OpenRouter)
│       ├── Experiment_DNC/
│       ├── Experiment_FIP/
│       └── Experiment_CTD/
├── analysis/
│   ├── 1_reliability/
│   │   ├── reliability_frontier_models.py
│   │   └── reliability_earlier_models.py
│   ├── 2_olr/
│   │   └── olr_frontier_models.py
│   ├── 3_auc/
│   │   ├── auc_earlier_models.py
│   │   └── auc_frontier_models.py
│   └── 4_ranking/
│       └── risk_attitude_ranking.py
├── figures/                    — all output figures (auto-created on run)
└── requirements.txt
```

---

## Setup

Python 3.10+ is required for the analysis scripts.

```bash
pip install -r requirements.txt
```

No other configuration is needed. All scripts resolve paths relative to the
repository root via `Path(__file__).parent.parent.parent`.

---

## Experiment Interfaces

The three tasks — DNC, FIP, CTD — each have two separate browser-based
implementations: one for human participants and one for LLM API calls.

### Human participants

```
experiments/HumanExperimentConfig/index.html        — participant portal
experiments/HumanExperimentConfig/Experiment_DNC/   — Drone Navigation Control
experiments/HumanExperimentConfig/Experiment_FIP/   — Financial Investment Portfolio
experiments/HumanExperimentConfig/Experiment_CTD/   — Clinical Triage Decision
```

Open `HumanExperimentConfig/index.html` in any modern browser (no server
required). Participants complete all three tasks; data is submitted automatically
to a Google Sheet via webhook. See `deployment_guide.md` for setup instructions.
Collected human data belongs in `data/human_data/` — see that directory's README.

### LLM API (OpenRouter)

```
experiments/LLMsExperimentConfig/Experiment_DNC/
experiments/LLMsExperimentConfig/Experiment_FIP/
experiments/LLMsExperimentConfig/Experiment_CTD/
```

Open the corresponding `index.html`, click **Config API**, and enter your
[OpenRouter](https://openrouter.ai) API key and model identifier. The interface
calls the LLM API and saves trial data in the same JSON format as the files in
`data/`. All LLM data is already included; re-running requires an active API key.

### Output format

Both interfaces produce JSON with the same structure:
```json
{
  "session": "DNC" | "FIP" | "CTD",
  "mode": "human" | "llm",
  "trials": [ { "steps": [...], "contextual_belief": ..., "risk_decision": ... } ]
}
```
This matches the format of all files in `data/frontier_models/main_N100/` and
`data/earlier_models/auc_N100/`.

---

## Reproducing Paper Results

### Quick start — run all steps at once

```bash
python analysis/run_pipeline.py
```

This runs all analysis steps in order and prints the data-flow diagram.

### Data requirement

The `data/` directory contains **one sample file per task** for format
illustration. The full dataset (N=100/model/task, 12 models, 3 tasks) is
available on HuggingFace:

> **[Full dataset →  https://huggingface.co/datasets/[dataset-url]]**

Download and unpack into `data/` before running the analysis scripts.

---

### Individual steps

All scripts share common extraction and fitting functions via `analysis/utils.py`.

#### Step 1 — Reliability / Intra-Consistency  (Paper §3.1)
```bash
python analysis/1_reliability/reliability_frontier_models.py
python analysis/1_reliability/reliability_earlier_models.py
```
Loads IC data → computes P1 (mean RSD ≤ 20%) and P2 (dominant R_D ≥ 80%) per
condition → confirms measurements are reliable before proceeding.
Output: `figures/reliability_frontier/`  +  `figures/reliability_earlier/`

#### Step 2 — OLR S-Curves  (Paper §3.2)
```bash
python analysis/2_olr/olr_frontier_models.py
```
Extracts B_C and R_D → fits OLR: R_D ~ B_C → plots E[R_D] curves.
Output: `figures/olr_frontier/`

#### Step 3 — AUC Computation  (Paper §3.3)
```bash
python analysis/3_auc/auc_earlier_models.py
python analysis/3_auc/auc_frontier_models.py
```
Integrates the OLR S-curve: AUC = ∫ E[R_D | B_C] dB_C → scalar risk attitude.
Output: `figures/auc_earlier/`  +  `figures/auc_frontier/`

#### Step 4 — Cross-Task Ranking  (Paper §3.4)
```bash
python analysis/4_ranking/risk_attitude_ranking.py
```
AUC-based ranks → Kendall's W concordance across DNC / FIP / CTD.
Output: `figures/ranking/`  +  `KendallW_Summary.txt`

---

## Key Constructs

| Symbol | Name | Description |
|--------|------|-------------|
| $B_C$ | Contextual Belief | Model's perceived danger level (0–100) given situational context |
| $R_D$ | Risk Decision | Categorical action chosen in response to that belief (1–5, 1=most aggressive) |

### $B_C$ extraction per task

| Task | Source field |
|------|-------------|
| DNC  | Last navigation step's `belief` value |
| FIP  | `report.contextual.risk` |
| CTD  | `steps[-1].BC_updates[-1].ctx` |

### $R_D$ discretization

| Task | Method |
|------|--------|
| DNC  | SI = log((VCR+ε)/(HPR+ε)) on drift steps only; global K-means(5) ascending |
| FIP  | H-allocation proportion; global K-means(5) descending |
| CTD  | `finalized.ESI` used directly (already 1–5) |

SI = Situation Index; VCR = vertical correction ratio; HPR = horizontal
progression ratio; ε = 1×10⁻⁶.

---

## Models Covered

**Frontier (6):** Grok 4, Claude Sonnet 4.5, DeepSeek V3.2, Gemini 3 Pro,
Qwen3 Max, GPT-5.2

**Earlier-generation (6):** Grok 3, Sonnet 4, DeepSeek V3, Gemini 2.5 Pro,
Qwen 3, GPT-4.1

**Tasks (3):** DNC · CTD · FIP

---

## Expected Runtimes

| Script | Typical time |
|--------|-------------|
| `reliability_frontier_models.py` | ~2 min |
| `reliability_earlier_models.py` | ~2 min |
| `olr_frontier_models.py` | ~2 min |
| `auc_earlier_models.py` | ~1 min |
| `auc_frontier_models.py` | ~1 min |
| `risk_attitude_ranking.py` | ~2 min |

Runtimes on a standard laptop CPU; parallelism not required.
