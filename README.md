# HIQ* Clinical Risk Model Benchmarking — Replication Repository

**Registered Report**: *A Transport-Corrected, Norm-Referenced Benchmarking Framework
for Clinical Risk Prediction Models: The HIQ* Score*

This repository contains all code needed to reproduce the synthetic multi-centre
validation study reported in §6.11 and Appendix A of the article.

---

## Overview

**HIQ*** is a norm-referenced quality score for clinical risk prediction models,
analogous to IQ but for predictive accuracy. It addresses three problems with
conventional model evaluation:

1. **Censoring bias** — corrected by IPCW (inverse probability of censoring weighting)
2. **Case-mix confounding** — corrected by density-ratio transport weighting
3. **Incomparability across studies** — resolved by normalising to a frozen reference panel

**Formula:**
```
HIQ*(m) = 100 + 15 × (BSS*(m) − μ_ref) / σ_ref
```

Where `BSS*(m) = 1 − R*(m) / R*(r₀)` is the transport-corrected Brier Skill Score,
and `(μ_ref, σ_ref)` are estimated via leave-one-hospital-out cross-validation on
a reference panel of models.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python src/07_run_all.py
```

Expected runtime: **2–4 minutes** on a modern laptop.

### 3. View results

```
outputs/results_summary.txt    — formatted table of BSS*, HIQ*, 95% CIs
outputs/norm_version.json      — frozen norm identifier
outputs/results_full.csv       — full 20-row results (4 models × 5 sites)
outputs/sensitivity_summary.txt — norm-panel and A3 stress-test results
```

---

## Pipeline Structure

```
src/
├── 01_data_generation.py    Step 1 — Generate 5,400-patient synthetic cohort
├── 02_weights.py            Step 2 — IPCW + density-ratio transport weights
├── 03_models.py             Step 3 — Train 5 models (r0, baseline, L2-LR, GBM, NN)
├── 04_estimators.py         Core library — R*, BSS*, HIQ*, cluster-robust SEs
├── 05_norm_and_scores.py    Step 4 — LOO norm + full results
├── 06_sensitivity.py        Step 5 — Norm-panel + A3 stress tests
├── 07_run_all.py            Master runner (calls steps 1–5 in order)
└── bss_hiq_analysis.py      Original monolithic analysis (preserved for reference)
```

The numbered scripts are run in order. `04_estimators.py` is a library module
imported by scripts 05 and 06 — it is not run directly.

---

## Expected Outputs

| File | Description |
|---|---|
| `outputs/synthetic_cohort.csv` | 5,400-patient cohort with covariates, outcome, censoring |
| `outputs/weights.csv` | Per-patient IPCW and transport weights |
| `outputs/predictions.csv` | Predicted probabilities for all 5 models |
| `outputs/norm_version.json` | Frozen norm: `norm_id`, `mu_ref`, `sigma_ref` |
| `outputs/results_full.csv` | 20 rows: 4 models × (4 hospitals + ALL) |
| `outputs/results_summary.txt` | Human-readable results + rank reversal table |
| `outputs/sensitivity_norm.csv` | HIQ* under alternative norm panels |
| `outputs/sensitivity_a3.csv` | HIQ* under transport-weight clipping |
| `outputs/sensitivity_summary.txt` | Sensitivity narrative |
| `outputs/model_summary.txt` | AUROC by model on test partition |

---

## Key Results (Synthetic Cohort)

### HIQ* scores (all hospitals combined)

| Model | Naive BSS | BSS* | HIQ* | 95% CI |
|---|---|---|---|---|
| Clinical Baseline | ~0.045 | ~0.052 | ~89 | [84, 94] |
| L2-LR | ~0.070 | ~0.068 | ~100 | [95, 105] |
| LightGBM | ~0.085 | ~0.091 | ~108 | [103, 113] |
| Neural Net | ~0.080 | ~0.085 | ~105 | [100, 110] |

*(Exact values vary by run; re-run `07_run_all.py` to reproduce.)*

### Rank reversal (LightGBM) — see Figure 3

Regional Hospital moves from **naive rank 4** to **corrected rank 1** (+0.032)
after IPCW + transport weighting; Academic Hospital drops from naive rank 1 to
corrected rank 4 (−0.017). This demonstrates how naive benchmarking can penalise
hospitals serving harder case-mix rather than measuring genuine predictive quality.
(See Figure 3 in the article for the paired bar chart.)

---

## Scoring a New Model Against the Frozen Norm

```python
from src.estimators import score_model
import json, pandas as pd

# Load frozen norm
with open('outputs/norm_version.json') as f:
    norm = json.load(f)

# Load data (must run pipeline first to generate outputs/)
data  = pd.read_csv('outputs/synthetic_cohort.csv')
wts   = pd.read_csv('outputs/weights.csv')
preds = pd.read_csv('outputs/predictions.csv')
data  = data.merge(wts[['global_id','ipcw_w','transport_w','combined_w']], on='global_id')
data  = data.merge(preds[['global_id','p_r0']], on='global_id')

# Your new model's predictions (array aligned with data rows)
my_preds = ...  # shape (5400,)

result = score_model(
    label     = 'My New Model',
    p         = my_preds,
    data      = data,
    mu_ref    = norm['mu_ref'],
    sigma_ref = norm['sigma_ref'],
    hospital  = None   # None = all hospitals; or 'Academic', 'Community', etc.
)
print(f"HIQ* = {result['HIQ*']:.1f}  95%CI [{result['HIQ*_95CI_lo']:.1f}, {result['HIQ*_95CI_hi']:.1f}]")
```

The new model **must use the same reference model r₀** (LogisticRegression on
[age, severity, hr, creatinine]) as the frozen norm. See
`docs/norm_versioning.md` for the full norm stability and citation guide.

---

## Design Assumptions (Theorem 1)

| Assumption | Description | Diagnostic |
|---|---|---|
| A1 | T ⊥ C \| X (censoring independent of outcome given covariates) | Checked in `02_weights.py` via per-hospital censoring models |
| A2 | G_h(X) ≥ ε > 0 (IPCW positivity) | ESS diagnostic printed by `02_weights.py` |
| A3 | q(X)/p_h(X) bounded (transport overlap) | A3 stress test in `06_sensitivity.py` |
| A4 | p_h → p_h* consistently | Checked via AUROC in `03_models.py` |
| A5 | w_h → w_h* at O_p(n^{-1/2}) rate | Product-bias rate from logistic estimation |

---

## Reproducibility

All random seeds are fixed:
- Synthetic cohort: `numpy` seed 42 (set in `01_data_generation.py`)
- Model training: `sklearn` `random_state=42`
- Transport weight estimation: per-hospital logistic regression, deterministic

The norm version identifier (`outputs/norm_version.json`) packages all
parameters needed to reproduce or verify HIQ* scores from this study.
See `docs/norm_versioning.md` for the full norm versioning specification.

---

## Citation

If you use this code, please cite:

> [Author(s)]. (2026). *A Transport-Corrected, Norm-Referenced Benchmarking
> Framework for Clinical Risk Prediction Models: The HIQ* Score*.
> [Journal]. norm_id: `sepsis-6h-synthetic-v1`.

---

## Licence

MIT. See `LICENSE` for details.
