# HIQ\* Clinical Risk — Multi-Centre Results

Synthetic ICU cohort · 5,400 patients · 4 hospitals · 4 models · sepsis-6h forecasting

**HIQ\*(m)** = 100 + 15 × (BSS\*(m) − μref) / σref · **BSS\*(m)** = 1 − R\*(m) / R\*(r₀) · μref = **0.0715**, σref = **0.0714**

Patients

5,400

4 hospital sites

Best model

113.2

LightGBM HIQ\*

Rank reversal

#4 → #1

Regional after correction

Neural Net

10.1

CI \[−3, +23\] — unstable

HIQ\* All Hospitals

Norm-referenced scores (100 = panel average, 15 pts = 1 SD). Error bars = cluster-robust 95% CI.

Naive BSS vs BSS\* — LightGBM

Transport correction reveals case-mix confounding across hospitals.

BSS\* by Hospital — All Models

Dots show corrected Brier Skill Score. Horizontal spread = real performance variation.

Forest Plot — HIQ\* with 95% CI

Cluster-robust sandwich SEs. Only LightGBM CI lies entirely above 100.

Rank Reversal — LightGBM

Hospital performance order after transport correction vs. naive BSS ranking.

Norm-Panel Sensitivity

HIQ\* under alternative reference panel compositions (LightGBM, all-site). Stability confirms the norm.

A3 Overlap Stress-Test

HIQ\* as transport-weight clipping tightens. Convergence to naive score (≈ 98) shows correction magnitude.

HIQ\* vs Naive BSS — Rating Comparison

5-dimension scorecard: HIQ\* vs naive BSS on estimator validity, statistical rigour, mix correction, censoring, and interpretability.