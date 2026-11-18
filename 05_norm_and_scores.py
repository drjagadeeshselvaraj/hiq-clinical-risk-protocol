#!/usr/bin/env python3
"""
05_norm_and_scores.py
─────────────────────
Compute the reference norm via LOO-CV, then score all models at all
hospitals with BSS*, HIQ*, and cluster-robust 95% CIs.

Outputs
───────
outputs/norm_version.json      – frozen norm identifier (machine-readable)
outputs/results_full.csv       – 20-row results table (4 models × 5 sites)
outputs/results_summary.txt    – formatted summary for the article

Norm versioning (§4.7 of the article)
──────────────────────────────────────
The norm version identifier packages:
  - disease task  : 'sepsis-6h-synthetic'
  - reference model: 'LogisticRegression_4feat'
  - panel composition: ['L2-LR', 'LightGBM']
  - LOO-CV date
  - mu_ref, sigma_ref
  - seeds for data and weight generation

This JSON is published alongside any study reporting HIQ* scores so
that independent groups can score new models against the same norm.
"""

import numpy as np
import pandas as pd
import json, os
from datetime import date

# Import local estimator module
import sys
sys.path.insert(0, os.path.dirname(__file__))
from estimators import (compute_norm_loo, score_model,
                         bss_star, hiq_star, naive_bss, cluster_robust_se)

PANEL_COLS   = ['p_l2lr', 'p_gbm']   # reference panel models
MODEL_MAP    = {
    'p_baseline': 'Clinical Baseline',
    'p_l2lr':     'L2-LR',
    'p_gbm':      'LightGBM',
    'p_nn':       'Neural Net',
}
HOSPITALS    = ['Academic', 'Community', 'Teaching', 'Regional']


def main():
    os.makedirs('outputs', exist_ok=True)

    # ── Load data, weights, predictions ────────────────────────────────────────
    cohort = pd.read_csv('outputs/synthetic_cohort.csv')
    wts    = pd.read_csv('outputs/weights.csv')
    preds  = pd.read_csv('outputs/predictions.csv')

    data = cohort.merge(wts[['global_id', 'ipcw_w', 'transport_w', 'combined_w']],
                        on='global_id')\
                 .merge(preds[['global_id', 'p_r0',
                                'p_baseline', 'p_l2lr', 'p_gbm', 'p_nn']],
                        on='global_id')

    # ── Compute norm ────────────────────────────────────────────────────────────
    print('Computing reference norm via LOO-CV...')
    mu_ref, sigma_ref = compute_norm_loo(data, preds, PANEL_COLS)
    print(f'  mu_ref    = {mu_ref:.6f}')
    print(f'  sigma_ref = {sigma_ref:.6f}')

    # ── Freeze norm version identifier ─────────────────────────────────────────
    norm_version = {
        'norm_id':           f'sepsis-6h-synthetic-v1',
        'disease_task':      'sepsis-6h-synthetic',
        'horizon_hours':     6,
        'reference_model':   'LogisticRegression_4feat',
        'panel_models':      ['L2-LR', 'LightGBM'],
        'loo_cv_date':       str(date.today()),
        'mu_ref':            round(mu_ref, 6),
        'sigma_ref':         round(sigma_ref, 6),
        'data_seed':         42,
        'n_panel_hospitals': 4,
        'note':              (
            'HIQ* scores from other studies are comparable to this norm only if '
            'they cite this norm_id and use the same reference model and panel composition.'
        ),
    }
    with open('outputs/norm_version.json', 'w') as f:
        json.dump(norm_version, f, indent=2)
    print(f'  Frozen norm saved: outputs/norm_version.json')

    # ── Score all models × all hospitals ───────────────────────────────────────
    print('\nScoring models...')
    results = []

    # Add p_r0 to data for score_model's convenience
    data['p_r0'] = data['p_r0']  # already merged

    for col, label in MODEL_MAP.items():
        p = data[col].values
        for hosp in HOSPITALS:
            row = score_model(label, p, data, mu_ref, sigma_ref, hospital=hosp)
            results.append(row)
        # ALL hospitals combined
        row = score_model(label, p, data, mu_ref, sigma_ref, hospital=None)
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/results_full.csv', index=False)

    # ── Formatted summary ───────────────────────────────────────────────────────
    lines = [
        'HIQ* Results — All Models, All Hospitals',
        f'Norm: mu_ref={mu_ref:.4f}, sigma_ref={sigma_ref:.4f}',
        '=' * 95,
        f"{'Model':<20} {'Hospital':<12} {'Naive BSS':>10} {'BSS*':>8} "
        f"{'BSS* 95%CI':>20} {'HIQ*':>8} {'HIQ* 95%CI':>20}",
        '-' * 95,
    ]
    for _, row in results_df.iterrows():
        bci = f"[{row['BSS*_95CI_lo']:.3f}, {row['BSS*_95CI_hi']:.3f}]"
        hci = f"[{row['HIQ*_95CI_lo']:.1f}, {row['HIQ*_95CI_hi']:.1f}]"
        lines.append(
            f"{row['Model']:<20} {row['Hospital']:<12} "
            f"{row['Naive_BSS']:>10.4f} {row['BSS*']:>8.4f} "
            f"{bci:>20} {row['HIQ*']:>8.1f} {hci:>20}"
        )

    lines += [
        '',
        'Key finding — LightGBM per-hospital ranking reversal:',
        f"  {'Hospital':<12} {'Naive BSS':>10} {'BSS*':>8} {'HIQ*':>8} "
        f"{'Naive rank':>12} {'Corrected rank':>14}",
    ]
    gbm_rows = results_df[
        (results_df['Model'] == 'LightGBM') &
        (results_df['Hospital'] != 'ALL')
    ].copy()
    gbm_rows['naive_rank']  = gbm_rows['Naive_BSS'].rank(ascending=False).astype(int)
    gbm_rows['corr_rank']   = gbm_rows['BSS*'].rank(ascending=False).astype(int)
    for _, row in gbm_rows.sort_values('corr_rank').iterrows():
        delta = row['BSS*'] - row['Naive_BSS']
        arrow = '↑' if delta > 0 else '↓'
        lines.append(
            f"  {row['Hospital']:<12} {row['Naive_BSS']:>10.4f} "
            f"{row['BSS*']:>8.4f} {row['HIQ*']:>8.1f} "
            f"{'#'+str(row['naive_rank']):>12} {'#'+str(row['corr_rank']):>14}  "
            f"Δ={delta:+.3f} {arrow}"
        )

    with open('outputs/results_summary.txt', 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print('\nSaved: outputs/results_full.csv, outputs/results_summary.txt')


if __name__ == '__main__':
    main()
