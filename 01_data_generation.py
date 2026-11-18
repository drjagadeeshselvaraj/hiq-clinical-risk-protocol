#!/usr/bin/env python3
"""
01_data_generation.py
─────────────────────
Generate the synthetic multi-centre ICU cohort used for replication.

Outputs
-------
outputs/synthetic_cohort.csv   – 5,400-row patient table with features,
                                  true probabilities, observed outcomes,
                                  censoring indicator
outputs/data_summary.txt       – cohort descriptive statistics

Design
------
Four hospital sites with heterogeneous patient mixes:

  Hospital    n      Severity shift  Age mean  Censoring rate
  ──────────  ─────  ──────────────  ────────  ──────────────
  Academic    1,800  +0.4            62        ~4.9 %
  Community   1,200  −0.3            68        ~12.1 %
  Teaching    1,500  +0.1            58        ~6.6 %
  Regional      900  −0.5            72        ~13.6 %

True outcome model (sepsis-6h log-odds):
  logit(p) = −2.5
             + 0.60·severity
             + 0.015·(age − 65)
             + 0.30·log(1 + lactate)
             + 0.008·(hr − 90)
             + 0.20·log(1 + creatinine)

Censoring is conditionally independent of outcome given features (Assumption A1).

Target population Q:
  severity ~ N(0, 1),  age ~ N(65, 10)
  (declared before any test-set access — mirrors pre-registration requirement)

References
----------
Graf et al. (1999) Statistics in Medicine 18:2529–2545   [IPCW]
Sugiyama et al. (2012) Cambridge University Press         [density-ratio transport]
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Hospital configuration ─────────────────────────────────────────────────────
HOSPITALS = {
    'Academic':  {'n': 1800, 'severity_shift': +0.4, 'age_mean': 62, 'censor_rate': 0.18},
    'Community': {'n': 1200, 'severity_shift': -0.3, 'age_mean': 68, 'censor_rate': 0.25},
    'Teaching':  {'n': 1500, 'severity_shift': +0.1, 'age_mean': 58, 'censor_rate': 0.20},
    'Regional':  {'n':  900, 'severity_shift': -0.5, 'age_mean': 72, 'censor_rate': 0.30},
}

# ── Target population Q ────────────────────────────────────────────────────────
TARGET_Q = {'severity_mean': 0.0, 'age_mean': 65.0,
            'severity_sd': 1.0,   'age_sd': 10.0}

# ── Feature names ──────────────────────────────────────────────────────────────
FEATURES     = ['age', 'severity', 'lactate', 'hr', 'creatinine']
REF_FEATURES = ['age', 'severity', 'hr', 'creatinine']


def generate_hospital(name: str, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate n patients for one hospital site."""
    n = cfg['n']

    age       = rng.normal(cfg['age_mean'], 12, n).clip(18, 95)
    severity  = rng.normal(cfg['severity_shift'], 1.0, n)
    lactate   = rng.exponential(2.0, n) * (1 + 0.3 * severity)
    hr        = rng.normal(90 + 5 * severity, 15, n).clip(40, 180)
    creatinine = rng.exponential(1.2, n) * (1 + 0.2 * (age - 65) / 10)

    # True log-odds — known data-generating process
    log_odds = (-2.5
                + 0.60 * severity
                + 0.015 * (age - 65)
                + 0.30 * np.log1p(lactate)
                + 0.008 * (hr - 90)
                + 0.20 * np.log1p(creatinine))

    true_prob = expit(log_odds)
    Y = rng.binomial(1, true_prob).astype(float)

    # Censoring — conditionally independent given features (A1)
    censor_logit = -1.5 - 0.3 * severity + 0.01 * (age - 65)
    censor_prob  = expit(censor_logit) * cfg['censor_rate'] * 2
    censor_prob  = np.clip(censor_prob, 0, cfg['censor_rate'] * 1.5)
    C = rng.binomial(1, censor_prob).astype(float)

    Y_obs = Y.copy()
    Y_obs[C == 1] = np.nan   # censored: outcome unobserved

    return pd.DataFrame({
        'hospital':    name,
        'patient_id':  np.arange(n),
        'age':         age,
        'severity':    severity,
        'lactate':     lactate,
        'hr':          hr,
        'creatinine':  creatinine,
        'true_prob':   true_prob,
        'Y':           Y,          # true (unobserved in real data)
        'Y_obs':       Y_obs,      # observed (NaN = censored)
        'censored':    C,
    })


def main():
    os.makedirs('outputs', exist_ok=True)

    frames = [generate_hospital(name, cfg, rng)
              for name, cfg in HOSPITALS.items()]
    data = pd.concat(frames, ignore_index=True)
    data['global_id'] = np.arange(len(data))

    data.to_csv('outputs/synthetic_cohort.csv', index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    summary_lines = [
        'Synthetic Multi-Centre ICU Cohort — Data Summary',
        '=' * 52,
        f'Total patients : {len(data):,}',
        '',
        data.groupby('hospital')[['censored', 'Y', 'severity', 'age']].agg(
            {'censored': 'mean', 'Y': 'mean',
             'severity': 'mean', 'age': 'mean'}
        ).round(3).to_string(),
        '',
        f'Target population Q: severity~N({TARGET_Q["severity_mean"]},{TARGET_Q["severity_sd"]})',
        f'                     age~N({TARGET_Q["age_mean"]},{TARGET_Q["age_sd"]})',
        '',
        f'Random seed: {SEED}',
    ]
    with open('outputs/data_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))

    print('\n'.join(summary_lines))
    print('\nSaved: outputs/synthetic_cohort.csv, outputs/data_summary.txt')


if __name__ == '__main__':
    main()
