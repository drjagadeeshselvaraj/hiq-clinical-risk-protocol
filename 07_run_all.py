#!/usr/bin/env python3
"""
07_run_all.py
─────────────
Master runner — executes the full HIQ* replication pipeline in order.

Usage
─────
  python src/07_run_all.py

Steps
─────
  Step 1   01_data_generation.py   Generate synthetic cohort
  Step 2   02_weights.py           IPCW + transport weights
  Step 3   03_models.py            Train models, generate predictions
  Step 4   05_norm_and_scores.py   LOO norm, BSS*, HIQ*, cluster-robust SEs
  Step 5   06_sensitivity.py       Norm-panel + A3 stress-tests

Expected runtime: ~2–4 minutes on a modern laptop.
All outputs land in outputs/.
"""

import subprocess, sys, time, os

STEPS = [
    ('Step 1 — Data generation',         'src/01_data_generation.py'),
    ('Step 2 — IPCW + transport weights', 'src/02_weights.py'),
    ('Step 3 — Model training',           'src/03_models.py'),
    ('Step 4 — Norm + HIQ* scores',       'src/05_norm_and_scores.py'),
    ('Step 5 — Sensitivity analyses',     'src/06_sensitivity.py'),
]


def run_step(label: str, script: str):
    print(f'\n{"─"*60}')
    print(f'{label}')
    print(f'{"─"*60}')
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # let output stream to console
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'\n*** FAILED: {script} (exit {result.returncode}) ***')
        sys.exit(result.returncode)
    print(f'\n[Done in {elapsed:.1f}s]')


def main():
    # Always run from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    print(f'Working directory: {os.getcwd()}')
    os.makedirs('outputs', exist_ok=True)

    t_total = time.time()
    for label, script in STEPS:
        run_step(label, script)

    print(f'\n{"="*60}')
    print(f'All steps complete in {time.time()-t_total:.1f}s')
    print(f'Results in: {os.path.abspath("outputs/")}')
    print('='*60)

    # Print final results
    summary_path = 'outputs/results_summary.txt'
    if os.path.exists(summary_path):
        print('\n' + open(summary_path).read())


if __name__ == '__main__':
    main()
