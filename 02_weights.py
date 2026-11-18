#!/usr/bin/env python3
"""
02_weights.py
─────────────
Compute IPCW weights and density-ratio transport weights.

Outputs
-------
outputs/weights.csv   – per-patient ipcw_w, transport_w, combined_w
outputs/weight_diagnostics.txt  – ESS, overlap checks

IPCW weights (Graf 1999)
────────────────────────
  G_h(X) = P(not censored | X, hospital h)
  Estimated per hospital via logistic regression on FEATURES.
  Censored patients receive weight 0 (they do not contribute to the
  Brier score estimator).
  Positivity floor: G_hat clipped to [0.10, 1.0] (Assumption A2).

Transport weights (Sugiyama et al. 2012)
─────────────────────────────────────────
  w_h(X) = q(X) / p_h(X)
  Estimated via logistic density-ratio:
      w_h(X) ≈ P(from Q | X) / P(from h | X)
  Target samples drawn from Q (N=2,000 per hospital).
  Weights normalised to mean 1 within each hospital.
  Positivity clip: [0.10, 10.0] (Assumption A3).

Effective sample size (ESS)
───────────────────────────
  ESS_h = (Σ w_h,i)² / Σ w_h,i²
  Hospitals with ESS < 60% of nominal n_h flagged as overlap warnings.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

SEED     = 42
rng      = np.random.default_rng(SEED)
FEATURES = ['age', 'severity', 'lactate', 'hr', 'creatinine']

TARGET_Q = {'severity_mean': 0.0, 'age_mean': 65.0,
            'severity_sd':   1.0, 'age_sd':   10.0}

IPCW_MIN    = 0.10   # positivity floor for censoring survival (A2)
IPCW_MAX    = 1.00
TRANS_MIN   = 0.10   # positivity floor for density ratio (A3)
TRANS_MAX   = 10.0
TARGET_N    = 2000   # reference samples drawn from Q per hospital


def ipcw_weights(data: pd.DataFrame) -> np.ndarray:
    """
    Per-patient IPCW weights.
    Censored patients → weight 0.
    Uncensored patients → 1 / G_hat(X).
    """
    weights = np.ones(len(data))
    for hosp in data['hospital'].unique():
        mask = (data['hospital'] == hosp).values
        X = data.loc[mask, FEATURES].values
        C = data.loc[mask, 'censored'].values
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        lr.fit(X, C)
        p_censor = lr.predict_proba(X)[:, 1]
        G_hat = np.clip(1 - p_censor, IPCW_MIN, IPCW_MAX)
        weights[mask] = 1.0 / G_hat
    # Zero out censored patients
    weights[data['censored'].values == 1] = 0.0
    return weights


def transport_weights(data: pd.DataFrame) -> np.ndarray:
    """
    Per-patient density-ratio transport weights toward target Q.
    """
    weights = np.ones(len(data))
    for hosp in data['hospital'].unique():
        mask = (data['hospital'] == hosp).values

        # Draw target samples
        tgt = pd.DataFrame({
            'age':       rng.normal(TARGET_Q['age_mean'],      TARGET_Q['age_sd'],      TARGET_N),
            'severity':  rng.normal(TARGET_Q['severity_mean'], TARGET_Q['severity_sd'], TARGET_N),
            'lactate':   rng.exponential(2.0, TARGET_N),
            'hr':        rng.normal(90, 15, TARGET_N).clip(40, 180),
            'creatinine':rng.exponential(1.2, TARGET_N),
            'is_target': 1,
        })

        hosp_df = data.loc[mask, FEATURES].copy()
        hosp_df['is_target'] = 0
        combined = pd.concat([hosp_df, tgt[FEATURES + ['is_target']]], ignore_index=True)

        X = combined[FEATURES].values
        y = combined['is_target'].values
        lr = LogisticRegression(C=0.5, max_iter=500, random_state=SEED)
        lr.fit(X, y)
        p_tgt = lr.predict_proba(X)[:, 1]

        # density ratio = p(target) / p(hospital)
        ratio = p_tgt / (1 - p_tgt + 1e-8)
        n_hosp = int(mask.sum())
        hosp_ratio = ratio[:n_hosp]
        hosp_ratio /= (hosp_ratio.mean() + 1e-8)   # normalise to mean 1
        hosp_ratio = np.clip(hosp_ratio, TRANS_MIN, TRANS_MAX)
        weights[mask] = hosp_ratio

    return weights


def effective_sample_size(weights: np.ndarray, mask: np.ndarray) -> float:
    w = weights[mask]
    w = w[w > 0]
    return w.sum()**2 / (w**2).sum() if len(w) > 0 else 0.0


def main():
    os.makedirs('outputs', exist_ok=True)
    data = pd.read_csv('outputs/synthetic_cohort.csv')

    ipcw_w     = ipcw_weights(data)
    transport_w = transport_weights(data)
    combined_w  = ipcw_w * transport_w

    data['ipcw_w']      = ipcw_w
    data['transport_w'] = transport_w
    data['combined_w']  = combined_w

    # Save
    data[['global_id', 'hospital', 'patient_id',
          'ipcw_w', 'transport_w', 'combined_w']].to_csv('outputs/weights.csv', index=False)

    # Diagnostics
    lines = ['Weight Diagnostics', '=' * 40]
    for hosp in data['hospital'].unique():
        mask = (data['hospital'] == hosp).values
        n    = mask.sum()
        ess_trans = effective_sample_size(transport_w, mask)
        uncens    = (data.loc[mask, 'censored'] == 0).sum()
        ess_ipcw  = effective_sample_size(ipcw_w, mask)
        flag_trans = ' *** LOW OVERLAP ***' if ess_trans < 0.6 * n else ''
        lines.append(
            f'{hosp:<12}  n={n:4d}  uncensored={uncens:4d}'
            f'  ESS_transport={ess_trans:5.0f} ({100*ess_trans/n:.0f}%){flag_trans}'
            f'  ESS_ipcw={ess_ipcw:5.0f}'
        )

    lines += [
        '',
        f'Overall: mean ipcw_w (uncensored) = '
        f'{ipcw_w[data["censored"]==0].mean():.3f}',
        f'         mean transport_w          = '
        f'{transport_w.mean():.3f}',
        f'         transport_w std           = '
        f'{transport_w.std():.3f}',
    ]
    with open('outputs/weight_diagnostics.txt', 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print('\nSaved: outputs/weights.csv, outputs/weight_diagnostics.txt')


if __name__ == '__main__':
    main()
