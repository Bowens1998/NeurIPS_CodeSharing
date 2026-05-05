# Human Participant Data

This directory is intentionally empty in the repository.

## Running Your Own Human Experiments

The experiment interfaces in `experiments/HumanExperimentConfig/` support full
human participant data collection. Follow the steps below to collect and store
your own data here.

### Step 1 — Host the experiment portal

Deploy `experiments/HumanExperimentConfig/` to any static web host (e.g. GitHub
Pages). Participants open the portal URL in a browser and complete the three tasks:

| Directory | Task name |
|-----------|-----------|
| `Experiment_DNC/` | Drone Navigation Control (DNC) |
| `Experiment_FIP/` | Financial Investment Portfolio (FIP) |
| `Experiment_CTD/` | Clinical Triage Decision (CTD) |

### Step 2 — Connect to Google Sheets

Follow `experiments/HumanExperimentConfig/deployment_guide.md` to set up a
Google Apps Script webhook. After each participant completes all three tasks,
their data is automatically submitted to a Google Sheet you control.

### Step 3 — Export and place data here

Download the Google Sheet as CSV, or export the JSON column, and place the
resulting files in this directory. The data structure produced by the experiment
interfaces is identical to the LLM trial data in
`data/frontier_models/main_N100/` — the same analysis scripts can be applied
directly.

### Data format

Each submitted trial record follows the same structure as the LLM data:

```json
{
  "session": "DNC" | "FIP" | "CTD",
  "mode": "human",
  "trials": [
    { "steps": [...], "contextual_belief": 42.0, "risk_decision": 3 }
  ]
}
```

This means `B_C` (Contextual Belief) and `R_D` (Risk Decision) can be extracted
with the same `extract_*` functions in `analysis/utils.py` used for LLM data.
