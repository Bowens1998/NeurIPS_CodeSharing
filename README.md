# PNAS Data Share

Data repository accompanying the paper:
**"Risk Attitude and Risk Sensitivity in Large Language Models across Structured Decision Tasks"**

---

## Repository Structure

### `RiskAttitudeAnalysis_main_N100/`
Contains 18 JSON files (6 LLMs × 3 experiments in each folder named by tasks), each with 100 trials for the main risk attitude and risk sensitivity analysis.

File naming: `{experiment}_{model}_N100.json`

| Experiment | Description |
|------------|-------------|
| DNC | Drone Navigation Control |
| CTD | Clinical Triage Decision |
| FIP | Financial Investment Portfolio |

| Model | File key |
|-------|----------|
| Grok 4 | `Grok4` |
| Claude Sonnet 4.5 | `Sonnet4.5` |
| DeepSeek V3.2 | `DeepSeekV3.2` |
| Gemini 3 Pro | `Gemini3Pro` |
| Qwen3 Max | `Qwen3Max` |
| GPT 5.2 | `GPT5.2` |

Each file contains:
```json
{
  "experiment": "DNC",
  "model": "Grok 4",
  "n_trials": 100,
  "trials": [ { ... } ]
}
```

---

### `IntraConsistency_N90/`
Contains 4 JSON files from the Intra-Consistency (IC) analysis, each with 90 trials per model (30 per level × 3 levels).

| File | Contents |
|------|----------|
| `IC_DNC_merged.json` | 540 records (6 models × 90 trials), DNC experiment |
| `IC_CTD_merged.json` | 540 records, CTD experiment |
| `IC_FIP_merged.json` | 540 records, FIP experiment |
| `IC_ALL_merged.json` | 1620 records, all experiments combined |

Each record includes: `experiment`, `model`, `level`, `trial_index`, `contextual_belief`, and `risk_decision`.

---

## Contact
For questions about the data, please open an issue in this repository.
