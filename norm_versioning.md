# HIQ* Norm Versioning — Extended Documentation

## Why Norm Versioning Matters

HIQ* is a **norm-referenced score**: it converts a raw Brier Skill Score (BSS*)
into an IQ-like scale where 100 = average reference-panel performance, 15 points =
one standard deviation. For this to be meaningful across studies and time,
*the norm must be frozen at the moment of publication and cited explicitly*.

Without a fixed norm, HIQ* scores from two studies are incommensurable — like
comparing IQ scores from different editions of the Wechsler test without knowing
which edition was used.

---

## Norm Version Identifier

A norm version is a small JSON object (`outputs/norm_version.json`):

```json
{
  "norm_id":           "sepsis-6h-synthetic-v1",
  "disease_task":      "sepsis-6h-synthetic",
  "horizon_hours":     6,
  "reference_model":   "LogisticRegression_4feat",
  "panel_models":      ["L2-LR", "LightGBM"],
  "loo_cv_date":       "2026-01-15",
  "mu_ref":            0.0715,
  "sigma_ref":         0.0714,
  "data_seed":         42,
  "n_panel_hospitals": 4,
  "note":              "HIQ* scores from other studies are comparable to this norm only if they cite this norm_id..."
}
```

### Field Definitions

| Field | Purpose |
|---|---|
| `norm_id` | Human-readable unique identifier. Version-bumped (`v2`, `v3`) when any field changes. |
| `disease_task` | The clinical prediction task (disease + time horizon). Different tasks → different norms. |
| `reference_model` | The frozen reference model r₀ that defines R*(r₀). Must match the actual model in `03_models.py`. |
| `panel_models` | The models whose BSS* distribution defines the norm scale. |
| `loo_cv_date` | When LOO-CV was run (for audit trail). |
| `mu_ref` | Mean BSS* across panel models × hospitals (the LOO-CV mean). |
| `sigma_ref` | Std BSS* across panel models × hospitals (the LOO-CV std, ddof=1). |
| `data_seed` | Random seed for synthetic cohort generation (for reproducibility). |

---

## Norm Estimation Procedure

The norm is estimated by **leave-one-hospital-out cross-validation (LOO-CV)**:

```
For each hospital h in {1, 2, 3, 4}:
  For each panel model m in {L2-LR, LightGBM}:
    Compute BSS*(m) restricted to patients from hospital h,
    using IPCW weights and transport weights specific to h
    → loo_values.append( BSS*(m, h) )

mu_ref    = mean(loo_values)
sigma_ref = std(loo_values, ddof=1)
```

**Why LOO-CV avoids circularity:**
- If we estimated the norm using the same hospitals where we then report HIQ* scores,
  the panel models would be optimised on the same distribution their scores are measured on.
- LOO-CV ensures that for each hospital's held-out score, the norm reflects the
  *other* hospitals' performance — removing the in-sample bias.

This is analogous to leave-one-out CV for calibration or shrinkage estimation.

---

## Scoring a New Model Against a Frozen Norm

A new model M can be scored against a previously published norm without
re-running LOO-CV:

```python
from src.estimators import score_model
import json, pandas as pd

# Load frozen norm
with open('outputs/norm_version.json') as f:
    norm = json.load(f)

# Load data, weights
data  = pd.read_csv('outputs/synthetic_cohort.csv')
# ... merge weights and predictions ...

# Score the new model
result = score_model(
    label     = 'My New Model',
    p         = my_predictions,
    data      = data,
    mu_ref    = norm['mu_ref'],
    sigma_ref = norm['sigma_ref'],
    hospital  = None   # None = all hospitals combined
)
print(f"HIQ* = {result['HIQ*']:.1f}  [{result['HIQ*_95CI_lo']:.1f}, {result['HIQ*_95CI_hi']:.1f}]")
```

**Important:** The new model must use the **same reference model r₀** as the
norm. The reference model defines R*(r₀), the BSS denominator. If r₀ changes,
BSS* changes for all models, and the norm is no longer valid.

---

## When to Create a New Norm Version

Create a new norm (increment `norm_id` to `v2`, etc.) when:

1. **The reference model changes** (different architecture, features, or training data).
2. **The panel composition changes** (different models, different number of hospitals).
3. **The disease task or outcome definition changes** (different ICD codes, different horizon).
4. **A major shift in target population** (new hospitals with substantially different case mix).
5. **A systematic recalibration** after audit shows drift in μ_ref or σ_ref > 10%.

Do **not** create a new norm version for:
- Adding new models to the leaderboard (they score against the existing norm).
- Routine annual updates of model coefficients (recalibration within the same architecture).
- Minor bug fixes that change BSS* by < 0.001.

---

## Norm Stability Diagnostics

The sensitivity analysis (`06_sensitivity.py`) reports HIQ* under alternative
norm specifications:

| Sensitivity check | Purpose |
|---|---|
| Leave-one-panel-model-out | How much does dropping one panel model shift the norm? |
| Alternative panel pairs | Is the norm robust to panel composition? |
| All-4-model panel | Upper bound on panel diversity. |
| A3 stress test (transport weight clipping) | How much does positivity violation inflate/deflate HIQ*? |

A norm is considered **stable** if HIQ* changes by < 5 points across all
leave-one-out variants. This corresponds to < 1/3 SD on the norm scale.

---

## Citation Format

When reporting HIQ* scores, cite the norm version as:

> "Models were scored using HIQ* (norm_id: `sepsis-6h-synthetic-v1`,
> μ_ref = 0.0715, σ_ref = 0.0714), estimated by leave-one-hospital-out CV
> on 4 hospitals using the L2-LR + LightGBM reference panel."

This allows independent groups to verify comparability of scores.
