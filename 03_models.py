#!/usr/bin/env python3
"""
03_models.py
────────────
Train the five prediction models on the development partition and
generate predictions on the full cohort.

Outputs
-------
outputs/predictions.csv   – per-patient predicted probabilities for all models
outputs/model_summary.txt – training partition sizes, AUROCs on test partition

Partitioning
────────────
Per hospital, 70% of non-censored patients → development set.
Remaining 30% non-censored + all censored → evaluation set.
(Censored patients contribute zero weight to the Brier estimator,
so they enter the evaluation partition but do not affect model selection.)

Models
------
  r0  : Frozen reference model  — LogisticRegression on [age, severity, hr, creatinine]
  m1  : Clinical Baseline       — LogisticRegression on [severity, age] only
  m2  : L2-LR                   — LogisticRegression (all features, C=0.1)
  m3  : LightGBM                — GradientBoostingClassifier (sklearn surrogate)
  m4  : Neural Net              — MLPClassifier (32→16, early stopping)

The reference model r0 is frozen before test-set access; its coefficients
are fixed and define R*(r0), the BSS denominator.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import os, json

SEED         = 42
FEATURES     = ['age', 'severity', 'lactate', 'hr', 'creatinine']
REF_FEATURES = ['age', 'severity', 'hr', 'creatinine']


def split_partitions(data: pd.DataFrame, dev_frac: float = 0.70):
    """Return (dev_idx, eval_idx) lists of integer positional indices."""
    dev_idx, eval_idx = [], []
    for hosp in data['hospital'].unique():
        mask = (data['hospital'] == hosp) & (data['censored'] == 0)
        idx  = data.index[mask].tolist()
        n_dev = int(dev_frac * len(idx))
        rng_local = np.random.default_rng(hash(hosp) % 2**31)
        rng_local.shuffle(idx)
        dev_idx.extend(idx[:n_dev])
        eval_idx.extend(idx[n_dev:])
    # All censored patients go to eval (zero weight, no info leakage)
    censored_idx = data.index[data['censored'] == 1].tolist()
    eval_idx.extend(censored_idx)
    return sorted(set(dev_idx)), sorted(set(eval_idx))


def build_models(data: pd.DataFrame):
    """Train all five models; return dict of fitted estimators."""
    dev_idx, _ = split_partitions(data)
    dev   = data.loc[dev_idx]
    X_dev = dev[FEATURES].values
    y_dev = dev['Y'].values

    models = {}

    # r0 — frozen reference (must match §6.7 of the article)
    models['r0'] = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    models['r0'].fit(dev[REF_FEATURES].values, y_dev)

    # m1 — clinical baseline
    models['m1_baseline'] = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    models['m1_baseline'].fit(dev[['severity', 'age']].values, y_dev)

    # m2 — L2-LR
    models['m2_l2lr'] = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(C=0.1, max_iter=500, random_state=SEED)),
    ])
    models['m2_l2lr'].fit(X_dev, y_dev)

    # m3 — GBM (sklearn surrogate for LightGBM)
    models['m3_gbm'] = GradientBoostingClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.05, random_state=SEED)
    models['m3_gbm'].fit(X_dev, y_dev)

    # m4 — Neural Net
    models['m4_nn'] = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp',    MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=300,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15)),
    ])
    models['m4_nn'].fit(X_dev, y_dev)

    return models


def predict_all(data: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Return dataframe of predicted probabilities for all models."""
    preds = data[['global_id', 'hospital', 'patient_id']].copy()
    preds['p_r0']      = models['r0'].predict_proba(data[REF_FEATURES].values)[:, 1]
    preds['p_baseline'] = models['m1_baseline'].predict_proba(data[['severity', 'age']].values)[:, 1]
    preds['p_l2lr']    = models['m2_l2lr'].predict_proba(data[FEATURES].values)[:, 1]
    preds['p_gbm']     = models['m3_gbm'].predict_proba(data[FEATURES].values)[:, 1]
    preds['p_nn']      = models['m4_nn'].predict_proba(data[FEATURES].values)[:, 1]
    return preds


def model_summary(data: pd.DataFrame, preds: pd.DataFrame, models: dict) -> list:
    """AUROC on test partition, by hospital."""
    _, eval_idx = split_partitions(data)
    eval_data  = data.loc[eval_idx]
    eval_preds = preds.loc[eval_idx]
    test_obs   = eval_data[eval_data['censored'] == 0]
    test_preds = eval_preds.loc[test_obs.index]
    y_true     = test_obs['Y'].values

    lines = ['Model AUROCs on Test Partition (non-censored only)', '=' * 52]
    for col, label in [
        ('p_r0',      'Reference (r0)  '),
        ('p_baseline','Clinical Baseline'),
        ('p_l2lr',    'L2-LR           '),
        ('p_gbm',     'LightGBM        '),
        ('p_nn',      'Neural Net      '),
    ]:
        auc = roc_auc_score(y_true, test_preds[col].values)
        lines.append(f'  {label}  AUROC = {auc:.4f}')

    dev_idx, _ = split_partitions(data)
    lines += [
        '',
        f'Development set size : {len(dev_idx):,}',
        f'Evaluation set size  : {len(eval_idx):,}  '
        f'(of which {(data.loc[eval_idx,"censored"]==0).sum():,} non-censored)',
    ]
    return lines


def main():
    os.makedirs('outputs', exist_ok=True)
    data  = pd.read_csv('outputs/synthetic_cohort.csv')

    print('Training models...')
    models = build_models(data)

    print('Generating predictions...')
    preds  = predict_all(data, models)
    preds.to_csv('outputs/predictions.csv', index=False)

    summary = model_summary(data, preds, models)
    with open('outputs/model_summary.txt', 'w') as f:
        f.write('\n'.join(summary))

    print('\n'.join(summary))
    print('\nSaved: outputs/predictions.csv, outputs/model_summary.txt')


if __name__ == '__main__':
    main()
