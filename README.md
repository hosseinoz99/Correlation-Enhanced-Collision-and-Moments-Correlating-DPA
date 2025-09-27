# Assignment 9 — Correlation-Enhanced Collision & Moments-Correlating DPA (AES)

This package contains a **ready-to-run, vectorized solver** that implements both required attacks and
produces **all deliverables** (6 plots + key.txt + description.txt + code).

> **No provided data are included.** Run locally with your `Traces00000.dat`..`Traces00009.dat`,
> `plaintexts.dat`, and `ciphertexts.dat`. The solver streams files and works even if only a subset
> of trace files is present (useful for quick tests).

## How to run
```bash
pip install numpy matplotlib

# Example (uses your /mnt/data paths by default)
python run_assignment9.py

# Or directly:
python assignment9_solver.py   --tracedir /path/to/traces_dir   --plaintexts /path/to/plaintexts.dat   --ciphertexts /path/to/ciphertexts.dat   --outdir assignment9_outputs   --byte_i 0 --byte_j 5   --roi 800 --steps 20 --lastname OmidiZadeh
```
**Tip:** For byte-difference 5 vs 10, invoke with `--byte_i 5 --byte_j 10`.

## What the solver does (high level)
- **ROI selection (variance-based):** Keep top-K time samples across all traces for speed (default K=800).
- **CECA:** For each Δk ∈ [0..255], use the subset of traces with `p_i ⊕ p_j == Δk`; compute the
  strongest inter-time correlation across ROI columns; record the maximum. Also provide progressive
  (#traces) curves and a correlation-vs-offset plot for the winning Δk.
- **MC-DPA:** For each Δk, define model `m = 1[p_i ⊕ p_j == Δk]` (zero-mean), and correlate `m`
  with cross-moments `X[:,t*] * X[:,u]` (centered) over ROI columns; take the maximum. Provide the
  same deliverables as for CECA.

## Submission 
- Include: `key.txt`, `plot_k_corr_*.png` (2 files), `plot_nrtraces_corr_*.png` (2 files),
  `plot_points_corr_*.png` (2 files), `description.txt`, and the **code**.
- Do **not** include the data files; avoid vector graphics; keep file sizes small.
- Name the archive: **`9.zip`**.
