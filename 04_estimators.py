#!/usr/bin/env python3
"""
04_estimators.py
────────────────
Core statistical estimators for the HIQ* pipeline.

This module is importable by other scripts and by the notebook.
No I/O side-effects when imported.

Estimators
──────────
  r_star(p, Y_obs, ipcw_w, transport_w)
      → target-population Brier risk R*(m)
      Equation (1) of the article:
        R*(m) = Σ_h α_h · (1/n_h) · Σ_i w_h(X_i) · (1/G_hat_i) · (p_i − Y_i)²

  bss_star(p, Y_obs, ipcw_w, transport_w, p_r0)
      → corrected Brier Skill Score BSS*(m) = 1 − R*(m) / R*(r0)

  naive_bss(p, Y_obs, p_r0)
      → naive BSS with no IPCW and no transport correction

  hiq_star(bss_val, mu_ref, sigma_ref)
      → norm-referenced score HIQ* = 100 + 15·(BSS* − μ_ref) / σ_ref

  compute_norm_loo(data, preds, panel_models)
      → (mu_ref, sigma_ref) via LOO-CV on the reference panel

  cluster_robust_se(p, Y_obs, ipcw_w, transport_w, p_r0,
                    hospital_ids, sigma_ref)
      → (se_bss, se_hiq) sandwich variance, Propositions 1–2

  Theorem 1 conditions (checked in diagnostics, not enforced here):
      A1: T ⊥ C | X  (censoring independence)
      A2: G_h(X) ≥ ε > 0  (positivity — IPCW floor)
      A3: q(X)/p_h(X) bounded  (overlap — transport clip)
      A4: p_h → p_h* consistently
      A5: w_h → w_h* consistently  (product-bias rate O_p(n^{-1}))
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


# ── Low-level Brier estimators ─────────────────────────────────────────────────

def r_star(p: np.ndarray,
           Y_obs: np.ndarray,
           ipcw_w: np.ndarray,
           transport_w: np.ndarray) -> float:
    """
    Target-population Brier risk R*(m).

    Parameters
    ----------
    p           : predicted probabilities (n,)
    Y_obs       : observed outcomes; NaN for censored patients (n,)
    ipcw_w      : IPCW weights; 0 for censored patients (n,)
    transport_w : density-ratio transport weights (n,)

    Returns
    -------
    float — R*(m), or np.nan if no non-censored observations
    """
    combined = ipcw_w * transport_w
    mask     = combined > 0
    if mask.sum() == 0:
        return np.nan
    Y = np.where(np.isnan(Y_obs), 0.0, Y_obs)   # censored → 0 (zero weight)
    return float(np.sum(combined[mask] * (p[mask] - Y[mask])**2) / mask.sum())


def bss_star(p: np.ndarray,
             Y_obs: np.ndarray,
             ipcw_w: np.ndarray,
             transport_w: np.ndarray,
             p_r0: np.ndarray) -> float:
    """
    Corrected Brier Skill Score BSS*(m) = 1 − R*(m) / R*(r0).
    Returns np.nan if R*(r0) = 0.
    """
    r_m   = r_star(p,    Y_obs, ipcw_w, transport_w)
    r_ref = r_star(p_r0, Y_obs, ipcw_w, transport_w)
    if r_ref == 0 or np.isnan(r_ref):
        return np.nan
    return float(1 - r_m / r_ref)


def naive_bss(p: np.ndarray,
              Y_obs: np.ndarray,
              p_r0: np.ndarray) -> float:
    """
    Naive BSS — no IPCW, no transport.
    Uses only non-censored (non-NaN) observations.
    """
    mask = ~np.isnan(Y_obs)
    if mask.sum() == 0:
        return np.nan
    bs_m = np.mean((p[mask]    - Y_obs[mask])**2)
    bs_r = np.mean((p_r0[mask] - Y_obs[mask])**2)
    return float(1 - bs_m / bs_r) if bs_r > 0 else np.nan


# ── Norm computation ───────────────────────────────────────────────────────────

def compute_norm_loo(data: pd.DataFrame,
                     preds: pd.DataFrame,
                     panel_cols: List[str]) -> Tuple[float, float]:
    """
    Estimate (mu_ref, sigma_ref) via leave-one-hospital-out CV.

    For each hospital h held out, the BSS* of each panel model is computed
    using the weights specific to hospital h. The resulting distribution of
    held-out BSS* values is the reference panel distribution.

    Parameters
    ----------
    data       : cohort dataframe (must contain ipcw_w, transport_w, Y_obs)
    preds      : predictions dataframe (must contain p_r0 and each col in panel_cols)
    panel_cols : list of prediction column names for the reference panel
                 (e.g. ['p_l2lr', 'p_gbm'])

    Returns
    -------
    (mu_ref, sigma_ref) : float, float
    """
    merged = data.merge(preds[['global_id'] + panel_cols + ['p_r0']],
                        on='global_id', how='left')

    loo_values = []
    for h in merged['hospital'].unique():
        mask     = merged['hospital'] == h
        d        = merged[mask]
        Y_obs    = d['Y_obs'].fillna(0).values
        ipcw_w   = d['ipcw_w'].values
        trans_w  = d['transport_w'].values
        p_r0_h   = d['p_r0'].values

        for col in panel_cols:
            p = d[col].values
            val = bss_star(p, Y_obs, ipcw_w, trans_w, p_r0_h)
            if not np.isnan(val):
                loo_values.append(val)

    loo_arr = np.array(loo_values)
    return float(loo_arr.mean()), float(loo_arr.std(ddof=1))


# ── HIQ* transformation ────────────────────────────────────────────────────────

def hiq_star(bss_val: float,
             mu_ref: float,
             sigma_ref: float) -> float:
    """
    HIQ*(m) = 100 + 15 × (BSS*(m) − μ_ref) / σ_ref

    Monotone affine transformation of BSS*.
    Preserves model rankings (Proposition 0 / monotonicity property).
    """
    if sigma_ref == 0 or np.isnan(bss_val):
        return np.nan
    return float(100 + 15 * (bss_val - mu_ref) / sigma_ref)


# ── Cluster-robust standard error (Propositions 1–2) ─────────────────────────

def cluster_robust_se(p: np.ndarray,
                      Y_obs: np.ndarray,
                      ipcw_w: np.ndarray,
                      transport_w: np.ndarray,
                      p_r0: np.ndarray,
                      hospital_ids: np.ndarray,
                      sigma_ref: float) -> Tuple[float, float]:
    """
    Cluster-robust sandwich standard error for BSS* and HIQ*.

    Influence function for R*(m):
        ψ_i = w_i · (p_i − Y_i)² − R*(m)

    Sandwich variance (hospital-level clusters):
        V̂ = (1/n²) · Σ_h (Σ_{i∈h} ψ_i)²

    SE(BSS*)  = √V̂ / R*(r0)              [delta method]
    SE(HIQ*)  = (15/σ_ref) · (1/R*(r0)) · √V̂

    Parameters
    ----------
    hospital_ids : array of hospital labels aligned with p / Y_obs / weights

    Returns
    -------
    (se_bss, se_hiq) : float, float  (NaN if degenerate)
    """
    combined = ipcw_w * transport_w
    mask     = combined > 0
    n        = int(mask.sum())
    if n == 0:
        return np.nan, np.nan

    Y    = np.where(np.isnan(Y_obs), 0.0, Y_obs)
    p_m  = p[mask]
    p_r  = p_r0[mask]
    Y_m  = Y[mask]
    w_m  = combined[mask]
    h_m  = hospital_ids[mask]

    r_m  = np.sum(w_m * (p_m - Y_m)**2) / n
    r_r  = np.sum(w_m * (p_r - Y_m)**2) / n
    if r_r == 0:
        return np.nan, np.nan

    # Influence function
    psi = w_m * (p_m - Y_m)**2 - r_m

    # Sum influence function within each hospital cluster
    V = 0.0
    for h in np.unique(h_m):
        cluster_sum = psi[h_m == h].sum()
        V += cluster_sum**2
    V = V / (n**2)

    se_bss = float(np.sqrt(V) / r_r)
    se_hiq = float((15 / sigma_ref) * (1 / r_r) * np.sqrt(V))
    return se_bss, se_hiq


# ── Convenience: score one model across hospitals ─────────────────────────────

def score_model(label: str,
                p: np.ndarray,
                data: pd.DataFrame,
                mu_ref: float,
                sigma_ref: float,
                hospital: Optional[str] = None) -> dict:
    """
    Compute naive BSS, BSS*, HIQ*, cluster-robust SEs for one model.

    Parameters
    ----------
    label    : model name for the output dict
    p        : predicted probabilities (aligned with data)
    data     : dataframe with Y_obs, ipcw_w, transport_w, p_r0, hospital
    mu_ref   : frozen norm parameter
    sigma_ref: frozen norm parameter
    hospital : if None, scores on the full dataset; else filters to that hospital

    Returns
    -------
    dict with keys: Hospital, Model, Naive_BSS, BSS*, BSS*_SE,
                    BSS*_95CI_lo, BSS*_95CI_hi,
                    HIQ*, HIQ*_SE, HIQ*_95CI_lo, HIQ*_95CI_hi
    """
    if hospital is not None:
        mask = data['hospital'] == hospital
        d    = data[mask]
        p_   = p[mask.values]
    else:
        d    = data
        p_   = p

    Y_obs    = d['Y_obs'].fillna(0).values
    ipcw_w   = d['ipcw_w'].values
    trans_w  = d['transport_w'].values
    p_r0_arr = d['p_r0'].values
    hosp_ids = d['hospital'].values

    nb     = naive_bss(p_,   d['Y_obs'].values, p_r0_arr)
    bss    = bss_star(p_,    Y_obs, ipcw_w, trans_w, p_r0_arr)
    hiq    = hiq_star(bss,   mu_ref, sigma_ref)
    se_bss, se_hiq = cluster_robust_se(
        p_, Y_obs, ipcw_w, trans_w, p_r0_arr, hosp_ids, sigma_ref)

    return {
        'Hospital':       hospital or 'ALL',
        'Model':          label,
        'Naive_BSS':      nb,
        'BSS*':           bss,
        'BSS*_SE':        se_bss,
        'BSS*_95CI_lo':   bss - 1.96 * se_bss if se_bss and not np.isnan(se_bss) else np.nan,
        'BSS*_95CI_hi':   bss + 1.96 * se_bss if se_bss and not np.isnan(se_bss) else np.nan,
        'HIQ*':           hiq,
        'HIQ*_SE':        se_hiq,
        'HIQ*_95CI_lo':   hiq - 1.96 * se_hiq if se_hiq and not np.isnan(se_hiq) else np.nan,
        'HIQ*_95CI_hi':   hiq + 1.96 * se_hiq if se_hiq and not np.isnan(se_hiq) else np.nan,
    }
