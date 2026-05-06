# NeurIPS 2026 — Reproducibility Package

This repository contains all **code** and **sample data** needed to verify the
analysis pipeline. The full dataset (N=100 trials/model/task, 12 models, 3 tasks)
is available on HuggingFace — see the [Data requirement](#data-requirement) section.

---

## Repository Structure

```
NeurIPS_CodeSharing/
├── data/                         ← sample files only; full dataset on HuggingFace
│   ├── frontier_models/
│   │   ├── main_N100/          — 1 sample file per task (format illustration)
│   │   │   ├── DNC/            — DNC_test_N100.json   (full: 6 × DNC_{model}_N100.json)
│   │   │   ├── CTD/            — CTD_test_N100.json   (full: 6 × CTD_{model}_N100.json)
│   │   │   └── FIP/            — FIP_test_N100.json   (full: 6 × FIP_{model}_N100.json)
│   │   └── reliability_N90/    — 1 sample file per task
│   │       ├── DNC/            — test_90.json   (full: 6 × {model}_90.json)
│   │       ├── CTD/            — test_90.json
│   │       └── FIP/            — test_90.json
│   ├── earlier_models/
│   │   ├── auc_N100/           — 1 sample per task  (full: 6 models × 3 tasks)
│   │   └── reliability_N90/    — 1 sample per task
│   └── human_data/
│       └── test.csv            — 15-subject sample (format illustration)
│                                 full dataset: human_features.csv on HuggingFace
├── experiments/
│   ├── HumanExperimentConfig/  — browser-based interface for human participants
│   │   ├── index.html          — participant portal (task hub)
│   │   ├── Experiment_DNC/     — Drone Navigation Control
│   │   ├── Experiment_FIP/     — Financial Investment Portfolio
│   │   ├── Experiment_CTD/     — Clinical Triage Decision
│   │   └── deployment_guide.md — Google Sheets webhook setup
│   └── LLMsExperimentConfig/   — LLM API interface (OpenRouter)
│       ├── ic_runner.py        — headless batch pipeline (alternative to browser)
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
│   ├── 4_ranking/
│   │   └── risk_attitude_ranking.py
│   └── 5_human_clustering/
│       └── human_risk_clustering.py
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

#### Python headless pipeline (`ic_runner.py`)

As an alternative to the browser interface, `ic_runner.py` provides a
command-line pipeline that runs all three experiments in batch and writes
output directly in the shared data format.

```bash
# Single experiment (30 trials total: 10 per condition)
python experiments/LLMsExperimentConfig/ic_runner.py \
    --exp FIP --n 10 --model anthropic/claude-sonnet-4-5 \
    --key YOUR_OPENROUTER_KEY --seed 42 \
    --out data/frontier_models/reliability_N90

# All three experiments sequentially
python experiments/LLMsExperimentConfig/ic_runner.py \
    --exp all --n 30 --key YOUR_OPENROUTER_KEY --seed 42 \
    --out data/frontier_models/reliability_N90
```

Each run writes two files per experiment:

| File | Contents |
|------|----------|
| `<EXP>_<Model>_N<n>.json` | Primary data — same format as all files in `data/` |
| `<EXP>_<Model>_N<n>_meta.json` | Provenance sidecar — seed, model, manipulation check, raw trials |

**Manipulation check:** conditions are re-run only if the frozen stimulus fails
to induce $B_C$ responses in its intended danger region (stimulus invalidity
check). IC metrics (Prop 1/2) never trigger re-runs — they are measured
outcomes reported as-is.

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

> **[Full dataset →  https://huggingface.co/datasets/LLMsRiskAttitudeDataShare/DataShare_NeurIPS2026_LLMsRiskAttitude]**

Download and unpack into `data/` before running the analysis scripts.

---

### Individual steps

All scripts share common extraction and fitting functions via `analysis/utils.py`.

#### Step 1 — Reliability  (Paper §3.1)
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

#### Step 5 — Human Participant Clustering  (Paper §4)
```bash
# Full dataset (download from HuggingFace first)
python analysis/5_human_clustering/human_risk_clustering.py

# Format illustration using the included 15-subject sample
python analysis/5_human_clustering/human_risk_clustering.py \
    --input data/human_data/test.csv
```
Reads per-subject pre-extracted behavioural features → K-means (k=3) →
assigns each participant to Cautious / Neutral / Aggressive cluster.

Input CSV columns:

| Column | Description |
|--------|-------------|
| `CTD_ESI_mean` | Mean ESI rating across CTD trials (1–5) |
| `FIP_alloc_continuous_mean` | Mean continuous allocation score across FIP trials (1–5) |
| `DNC_SI_scaled_mean` | Mean DNC Strategy Index scaled to [1, 5] across DNC trials |

The DNC column uses the logit-form SI = logit(VCR/(VCR+HPR+ε)) computed on
drift steps (nudge ≠ 0), mapped to [1, 5] via a two-tier percentile method
(floor trials → 5; non-floor quartile-binned to 1–4).

Output: `figures/human_clustering/`  +  `data/human_data/Human_Clustered_Data.csv`

---

## Key Constructs

| Symbol | Name | Description |
|--------|------|-------------|
| $B_C$ | Contextual Belief | Model's perceived danger level (0–100) given situational context |
| $R_D$ | Risk Decision | Categorical action chosen in response to that belief (1–5, 1=most cautious, 5=most aggressive) |

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
| `human_risk_clustering.py` | <1 min |

Runtimes on a standard laptop CPU; parallelism not required.
