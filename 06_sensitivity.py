#!/usr/bin/env python3
"""
06_sensitivity.py
─────────────────
Norm-panel sensitivity analysis (addresses Reviewer concern §1):
  How much do HIQ* scores change if the reference panel drops
  or replaces one model?

Also: Assumption A3 (positivity overlap) stress-test.
  Artificially extreme patient-mix shift → check ESS degradation
  and HIQ* bias from transport weight truncation.

Outputs
───────
outputs/sensitivity_norm.csv        – HIQ* under leave-one-panel-model-out
outputs/sensitivity_a3.csv          – HIQ* under progressively tighter overlap
outputs/sensitivity_summary.txt     – narrative summary
"""

import numpy as np
import pandas as pd
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from estimators import compute_norm_loo, score_model

PANEL_COLS  = ['p_l2lr', 'p_gbm']
MODEL_MAP   = {
    'p_baseline': 'Clinical Baseline',
    'p_l2lr':     'L2-LR',
    'p_gbm':      'LightGBM',
    'p_nn':       'Neural Net',
}
HOSPITALS = ['Academic', 'Community', 'Teaching', 'Regional']


def load_data():
    cohort = pd.read_csv('outputs/synthetic_cohort.csv')
    wts    = pd.read_csv('outputs/weights.csv')
    preds  = pd.read_csv('outputs/predictions.csv')
    return (cohort
            .merge(wts[['global_id', 'ipcw_w', 'transport_w', 'combined_w']],
                   on='global_id')
            .merge(preds[['global_id', 'p_r0',
                           'p_baseline', 'p_l2lr', 'p_gbm', 'p_nn']],
                   on='global_id'))


def norm_sensitivity(data):
    """
    Leave-one-model-out norm sensitivity.
    For each subset of panel_cols of size len-1, re-compute norm and HIQ*.
    """
    rows = []
    all_panel = ['p_baseline', 'p_l2lr', 'p_gbm', 'p_nn']

    # Full panel (all 4 models)
    for panel in [PANEL_COLS,                # original 2-model panel
                  ['p_l2lr'],                # just L2-LR
                  ['p_gbm'],                 # just GBM
                  all_panel,                 # all 4 as norm panel
                  ['p_baseline', 'p_nn'],    # alternative pair
                  ]:
        preds_sub = data[['global_id'] + panel + ['p_r0']].copy()
        mu, sigma  = compute_norm_loo(data, preds_sub, panel)
        for col, label in MODEL_MAP.items():
            row = score_model(label, data[col].values, data, mu, sigma, hospital=None)
            row['panel_desc'] = '+'.join(panel)
            row['mu_ref']     = round(mu, 4)
            row['sigma_ref']  = round(sigma, 4)
            rows.append(row)

    return pd.DataFrame(rows)


def a3_stress_test(data):
    """
    Assumption A3 stress-test: progressively clip transport weights harder.
    Lower TRANS_MAX → less extreme reweighting → HIQ* converges to naive BSS*.
    Shows sensitivity to overlap assumption.
    """
    with open('outputs/norm_version.json') as f:
        norm = json.load(f)
    mu_ref, sigma_ref = norm['mu_ref'], norm['sigma_ref']

    rows = []
    for clip_max in [10.0, 5.0, 3.0, 2.0, 1.5, 1.0]:
        data_clipped = data.copy()
        data_clipped['transport_w'] = data_clipped['transport_w'].clip(0.1, clip_max)
        # Renormalise per hospital to mean 1
        for hosp in data_clipped['hospital'].unique():
            mask = data_clipped['hospital'] == hosp
            w    = data_clipped.loc[mask, 'transport_w']
            data_clipped.loc[mask, 'transport_w'] = w / w.mean()
        data_clipped['combined_w'] = data_clipped['ipcw_w'] * data_clipped['transport_w']

        for col, label in [('p_gbm', 'LightGBM'), ('p_nn', 'Neural Net')]:
            row = score_model(
                label, data_clipped[col].values,
                data_clipped, mu_ref, sigma_ref, hospital=None)
            row['transport_clip_max'] = clip_max
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    os.makedirs('outputs', exist_ok=True)
    data = load_data()

    print('Running norm sensitivity analysis...')
    sens_norm = norm_sensitivity(data)
    sens_norm.to_csv('outputs/sensitivity_norm.csv', index=False)

    print('Running A3 (overlap) stress test...')
    sens_a3 = a3_stress_test(data)
    sens_a3.to_csv('outputs/sensitivity_a3.csv', index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    lines = ['Sensitivity Analysis Summary', '=' * 60, '']

    lines.append('1. Norm-panel sensitivity (all-hospital LightGBM HIQ*)')
    lines.append(f"  {'Panel':<35} {'mu_ref':>8} {'sigma_ref':>10} {'HIQ*':>8}")
    lines.append('  ' + '-' * 65)
    for _, row in sens_norm[sens_norm['Model'] == 'LightGBM'].iterrows():
        lines.append(
            f"  {row['panel_desc']:<35} "
            f"{row['mu_ref']:>8.4f} {row['sigma_ref']:>10.4f} {row['HIQ*']:>8.1f}"
        )

    lines += ['', '2. A3 overlap stress-test (all-hospital)']
    lines.append(f"  {'Model':<18} {'Clip max':>10} {'BSS*':>8} {'HIQ*':>8}")
    lines.append('  ' + '-' * 48)
    for _, row in sens_a3.iterrows():
        lines.append(
            f"  {row['Model']:<18} {row['transport_clip_max']:>10.1f} "
            f"{row['BSS*']:>8.4f} {row['HIQ*']:>8.1f}"
        )

    with open('outputs/sensitivity_summary.txt', 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print('\nSaved: outputs/sensitivity_norm.csv, outputs/sensitivity_a3.csv')


if __name__ == '__main__':
    main()
