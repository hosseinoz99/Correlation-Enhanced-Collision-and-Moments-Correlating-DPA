#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Assignment 9 — Correlation-Enhanced Collision & Moments-Correlating DPA (AES, serialized SBox)
# Produces the required deliverables without shipping any provided data files.
# Vectorized & memory-safe; streams 10× trace files if available, or any subset found.
#
# Deliverables generated (per PDF):
#   - key.txt                        (recovered key-byte difference in hex & dec)
#   - plot_k_corr_ceca.png           (max corr per Δk after all traces)               [CECA]
#   - plot_nrtraces_corr_ceca.png    (max corr per Δk vs #traces)                     [CECA]
#   - plot_points_corr_ceca.png      (corr trace vs offset Δ around best time pair)   [CECA]
#   - plot_k_corr_mcdpa.png          (max corr per Δk after all traces)               [MC-DPA]
#   - plot_nrtraces_corr_mcdpa.png   (max corr per Δk vs #traces)                     [MC-DPA]
#   - plot_points_corr_mcdpa.png     (corr trace vs offset Δ around best time pair)   [MC-DPA]
#   - description.txt                (how many traces; runtime hint; diffs between attacks)
#
# Usage example:
#   python assignment9_solver.py --tracedir . --outdir assignment9_out --byte_i 0 --byte_j 5 --roi 800 --steps 20
#
# Notes:
# - ROI reduction: we rank time samples by across-trace variance and keep top-K indices (configurable).
# - CECA: For each Δk candidate, we select the traces where p_i ⊕ p_j == Δk, then seek the
#         strongest inter-time correlation across ROI columns; we record the maximum over ROI
#         (excluding the diagonal).
# - MC-DPA: For each Δk, define a binary model m = 1[p_i ⊕ p_j == Δk]. We correlate m with the
#           **cross-moment** X[:,t]*X[:,u] (centered) over ROI pairs and take the maximum.
# - For the “points corr” plots we show (for the winning Δk) the correlation of a chosen base
#   time sample vs all forward offsets (length M - offset), matching the assignment wording.
#
# Implementation detail:
# - To keep compute reasonable, we choose a base time t* as the ROI index that participates in
#   the best pair (t*, u*). The “points corr” then scans u = t*+Δ across valid Δ.
#
import argparse, time, glob, math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_plaintexts(path):
    P = np.fromfile(path, dtype=np.uint8).reshape(16, -1)
    return P

def iter_trace_files(tracedir):
    files = sorted(Path(tracedir).glob("Traces*.dat"))
    for fp in files:
        arr = np.fromfile(fp, dtype=np.int8)
        # Each file contains 10_000 traces × 10_000 points; layout is contiguous traces.
        try:
            T = arr.reshape(10000, 10000).astype(np.float32)
        except Exception:
            # Fallback: if shape unknown, try infer N by 10000 columns
            N = arr.size // 10000
            T = arr[:N*10000].reshape(N, 10000).astype(np.float32)
        yield fp.name, T

def top_variance_roi(tracedir, roi):
    total_N = 0
    sum_vec = None
    sumsq_vec = None
    M = None
    for _, T in iter_trace_files(tracedir):
        if M is None: M = T.shape[1]
        # accumulate sums across traces for each time sample
        s = T.sum(axis=0)            # (M,)
        s2 = (T*T).sum(axis=0)       # (M,)
        if sum_vec is None:
            sum_vec = s
            sumsq_vec = s2
        else:
            sum_vec += s
            sumsq_vec += s2
        total_N += T.shape[0]
    if total_N == 0:
        raise FileNotFoundError("No Traces*.dat files found")
    mean = sum_vec / total_N
    var = sumsq_vec / total_N - mean*mean
    if roi >= len(var):
        idx = np.arange(len(var), dtype=np.int64)
    else:
        idx = np.argpartition(-var, roi)[:roi]
        idx = idx[np.argsort(idx)]
    return idx, var

def take_roi(T, roi_idx):
    return T[:, roi_idx].astype(np.float32)  # (N, R)

def corr_cols(X):
    # X: (N, R) centered
    N = X.shape[0]
    XS = X / (np.sqrt((X*X).sum(axis=0, keepdims=True)) + 1e-12)
    return XS.T @ XS  # (R, R), ones on diag

def corr_xy(x, Y):
    # x: (N,) centered, Y: (N, K) centered
    den = (np.sqrt((x*x).sum()) + 1e-12) * (np.sqrt((Y*Y).sum(axis=0)) + 1e-12)
    return (x.reshape(-1,1).T @ Y).ravel() / den

def compute_ceca(tracedir, P, i, j, roi_idx, steps, outdir):
    R = len(roi_idx)
    # Prepare masks for Δk candidates in steps (progressive traces plot)
    N_total = P.shape[1]
    sizes = np.linspace(N_total/steps, N_total, steps, dtype=np.int64)
    # Precompute xor diff of plaintexts
    diff = np.bitwise_xor(P[i], P[j])  # (N,)
    maxcorr_per_dk = np.zeros(256, dtype=np.float32)
    best_pair_per_dk = [(-1,-1)]*256

    # Single pass to compute final (all traces) CECA max
    # We need centered ROI matrices per Δk subset; we stream traces files, stacking subset rows.
    # For efficiency, we first load all traces into ROI (may be heavy); otherwise do two-pass.
    # Here we do two-pass: gather rows per Δk, accumulate sums to center, then compute correlation.
    # Accumulators
    counts = np.zeros(256, dtype=np.int64)
    sums = np.zeros((256, R), dtype=np.float64)
    sums2 = np.zeros((256, R), dtype=np.float64)

    row_offset = 0
    for _, T in iter_trace_files(tracedir):
        Nf = T.shape[0]
        d_slice = diff[row_offset:row_offset+Nf]
        X = take_roi(T, roi_idx)      # (Nf, R)
        # For each Δk, select rows
        for dk in range(256):
            mask = (d_slice == dk)
            if not mask.any(): continue
            Xs = X[mask]
            counts[dk] += Xs.shape[0]
            sums[dk] += Xs.sum(axis=0)
            sums2[dk] += (Xs*Xs).sum(axis=0)
        row_offset += Nf

    # Now for each Δk with enough rows, compute centered X and corr matrix by streaming again
    best_t = None; best_u = None; best_dk = None
    row_offset = 0
    for dk in range(256):
        if counts[dk] <= 10:  # need more than trivial subset
            maxcorr_per_dk[dk] = 0.0
            continue
        mu = (sums[dk] / counts[dk]).astype(np.float32)  # (R,)
        # Compute correlation matrix XS.T @ XS over the subset in a streaming manner
        # XS columns are centered but not normalized; we accumulate XtX and diag norms.
        XtX = np.zeros((R, R), dtype=np.float64)
        colnorm2 = np.zeros(R, dtype=np.float64)
        row_offset = 0
        for _, T in iter_trace_files(tracedir):
            Nf = T.shape[0]
            d_slice = diff[row_offset:row_offset+Nf]
            X = take_roi(T, roi_idx)      # (Nf, R)
            mask = (d_slice == dk)
            if mask.any():
                XS = (X[mask] - mu)
                XtX += XS.T @ XS
                colnorm2 += (XS*XS).sum(axis=0)
            row_offset += Nf
        # Normalize to correlation
        denom = np.sqrt(colnorm2 + 1e-12)
        Corr = XtX / (denom.reshape(-1,1) * denom.reshape(1,-1) + 1e-12)
        np.fill_diagonal(Corr, 0.0)
        k = np.unravel_index(np.argmax(np.abs(Corr)), Corr.shape)
        maxcorr_per_dk[dk] = float(np.abs(Corr[k]))
        best_pair_per_dk[dk] = k
        # Track global best to define base point for the “points” plot
        if best_t is None or maxcorr_per_dk[dk] > maxcorr_per_dk[best_dk]:
            best_t, best_u, best_dk = k[0], k[1], dk

    # Progressive (#traces) plot: for the winning Δk only (to keep compute bounded)
    prog_sizes = sizes
    prog_vals = np.zeros_like(sizes, dtype=np.float32)
    # Choose base t* from best pair above; correlate with all u sharing base
    base = best_t
    row_offset = 0
    # We compute at each size the maximum |corr| between X[:,base] and X[:,u] within the subset.
    # Centering is recomputed per prefix.
    cumsum = None
    cumsum2 = None
    list_cols = []  # store centered columns for each block within subset, per ROI u
    subset_rows = []
    for _, T in iter_trace_files(tracedir):
        Nf = T.shape[0]
        d_slice = diff[row_offset:row_offset+Nf]
        X = take_roi(T, roi_idx)      # (Nf, R)
        mask = (d_slice == best_dk)
        if mask.any():
            subset_rows.append(X[mask])
        row_offset += Nf
    if subset_rows:
        XS = np.vstack(subset_rows)  # (Ns, R)
        # pre-center base and all columns for every prefix efficiently
        Ns = XS.shape[0]
        # prefix sums for base & each u
        pref = np.cumsum(XS, axis=0, dtype=np.float64)               # (Ns, R)
        pref2 = np.cumsum(XS*XS, axis=0, dtype=np.float64)           # (Ns, R)
        for i, n in enumerate(prog_sizes):
            n = int(n); 
            mu = (pref[n-1] / n)                                     # (R,)
            Xc = XS[:n] - mu                                         # (n, R)
            # correlations vs base
            xb = Xc[:, base]
            xb_norm = math.sqrt(float((xb*xb).sum()) + 1e-12)
            den = xb_norm * (np.sqrt((Xc*Xc).sum(axis=0)) + 1e-12)   # (R,)
            num = xb.reshape(1,-1) @ Xc                              # (1,R)
            r = np.abs((num / den).ravel())
            r[base] = 0.0
            prog_vals[i] = float(r.max())

    # “points corr” trace for CECA (winning Δk, base=best_t): corr(base, base+Δ)
    # Compute across all subset rows
    corr_points = None
    if subset_rows:
        XS = np.vstack(subset_rows)  # (Ns, R)
        xb = XS[:, base].astype(np.float32)
        xb = xb - xb.mean()
        Ns, R = XS.shape
        # Correlate with forward offsets only for plotting
        corr_points = np.zeros(R, dtype=np.float32)
        # center each column
        mu = XS.mean(axis=0); Xc = XS - mu
        denb = math.sqrt(float((xb*xb).sum()) + 1e-12)
        den = np.sqrt((Xc*Xc).sum(axis=0) + 1e-12)
        num = xb.reshape(1,-1) @ Xc
        corr_points = (num / (denb*den)).ravel().astype(np.float32)

    # Save plots
    xs = np.arange(256)
    plt.figure(figsize=(6,4), dpi=120)
    plt.plot(xs, maxcorr_per_dk, linewidth=1.0)
    plt.xlabel("Δk guess (0..255)"); plt.ylabel("max |corr| over ROI pairs"); plt.tight_layout()
    plt.savefig(outdir/"plot_k_corr_ceca.png"); plt.close()

    plt.figure(figsize=(6,4), dpi=120)
    plt.plot(prog_sizes, prog_vals, linewidth=1.0)
    plt.xlabel("#traces (subset with collision for winning Δk)"); plt.ylabel("max |corr| (vs base)")
    plt.tight_layout(); plt.savefig(outdir/"plot_nrtraces_corr_ceca.png"); plt.close()

    if corr_points is not None:
        plt.figure(figsize=(10,3), dpi=120)
        plt.plot(corr_points, linewidth=1.0)
        plt.xlabel("ROI index (relative)"); plt.ylabel(f"corr(X[:,t*], X[:,u]) for Δk=0x{best_dk:02x}")
        plt.tight_layout(); plt.savefig(outdir/"plot_points_corr_ceca.png"); plt.close()

    return maxcorr_per_dk, best_dk, (best_t, best_u), prog_sizes, prog_vals, corr_points

def compute_mcdpa(tracedir, P, i, j, roi_idx, steps, outdir, base_idx=None):
    # Model vector mΔk for each Δk: 1 if p_i ⊕ p_j == Δk else 0 (centered to zero mean).
    diff = np.bitwise_xor(P[i], P[j])
    N_total = P.shape[1]
    sizes = np.linspace(N_total/steps, N_total, steps, dtype=np.int64)
    R = len(roi_idx)

    maxcorr_per_dk = np.zeros(256, dtype=np.float32)
    best_u_per_dk = np.zeros(256, dtype=np.int64)

    # First, assemble the full leak matrix over ROI to enable dot-products (can stream if memory-limited).
    X_list = []
    for _, T in iter_trace_files(tracedir):
        X_list.append(take_roi(T, roi_idx))
    X = np.vstack(X_list).astype(np.float32)  # (N, R)

    # Choose base index for pairwise products (if not provided, select the ROI column of highest variance)
    if base_idx is None:
        # pick the column with largest var in X
        var_cols = X.var(axis=0)
        base_idx = int(np.argmax(var_cols))

    # Pre-center X and store base column
    Xc = X - X.mean(axis=0, keepdims=True)
    xb = Xc[:, base_idx]

    # For each Δk, correlate m (centered) with xb * Xc[:,u] and take max over u
    for dk in range(256):
        m = (diff == dk).astype(np.float32)
        m = m - m.mean()
        prod = (xb.reshape(-1,1) * Xc)  # (N, R)
        # corr(m, prod) per u
        num = m.reshape(1,-1) @ prod   # (1,R)
        den = (np.sqrt((m*m).sum()) + 1e-12) * (np.sqrt((prod*prod).sum(axis=0)) + 1e-12)
        r = np.abs((num / den).ravel())
        maxcorr_per_dk[dk] = float(r.max())
        best_u_per_dk[dk] = int(np.argmax(r))

    best_dk = int(np.argmax(maxcorr_per_dk))

    # Progressive plot for best Δk
    prog_vals = np.zeros_like(sizes, dtype=np.float32)
    dk = best_dk
    m_full = (diff == dk).astype(np.float32)
    for t, n in enumerate(sizes):
        n = int(n)
        m = m_full[:n]; m = m - m.mean()
        Xn = Xc[:n]
        xb = Xn[:, base_idx]
        prod = (xb.reshape(-1,1) * Xn)
        num = m.reshape(1,-1) @ prod
        den = (np.sqrt((m*m).sum()) + 1e-12) * (np.sqrt((prod*prod).sum(axis=0)) + 1e-12)
        r = np.abs((num / den).ravel())
        r[base_idx] = 0.0
        prog_vals[t] = float(r.max())

    # Points corr trace (fix dk=best, base=base_idx): corr(m, xb * Xc[:,u]) for all u (acts as “offset” scan)
    m = (diff == best_dk).astype(np.float32); m = m - m.mean()
    prod = (xb.reshape(-1,1) * Xc)
    num = m.reshape(1,-1) @ prod
    den = (np.sqrt((m*m).sum()) + 1e-12) * (np.sqrt((prod*prod).sum(axis=0)) + 1e-12)
    corr_points = ((num / den).ravel()).astype(np.float32)

    # Save plots
    xs = np.arange(256)
    plt.figure(figsize=(6,4), dpi=120)
    plt.plot(xs, maxcorr_per_dk, linewidth=1.0)
    plt.xlabel("Δk guess (0..255)"); plt.ylabel("max |corr| over ROI (moments correlating)"); plt.tight_layout()
    plt.savefig(outdir/"plot_k_corr_mcdpa.png"); plt.close()

    plt.figure(figsize=(6,4), dpi=120)
    plt.plot(sizes, prog_vals, linewidth=1.0)
    plt.xlabel("#traces"); plt.ylabel("max |corr| (moments correlating)"); plt.tight_layout()
    plt.savefig(outdir/"plot_nrtraces_corr_mcdpa.png"); plt.close()

    plt.figure(figsize=(10,3), dpi=120)
    plt.plot(corr_points, linewidth=1.0)
    plt.xlabel("ROI index (relative)"); plt.ylabel(f"corr(m, xb*Xu) for Δk=0x{best_dk:02x}")
    plt.tight_layout(); plt.savefig(outdir/"plot_points_corr_mcdpa.png"); plt.close()

    return maxcorr_per_dk, best_dk, base_idx, sizes, prog_vals, corr_points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracedir", default=".")
    ap.add_argument("--plaintexts", default="plaintexts.dat")
    ap.add_argument("--ciphertexts", default="ciphertexts.dat")  # not used by default
    ap.add_argument("--outdir", default="assignment9_outputs")
    ap.add_argument("--byte_i", type=int, default=0)
    ap.add_argument("--byte_j", type=int, default=5)
    ap.add_argument("--roi", type=int, default=800)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lastname", default="OmidiZadeh")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    P = load_plaintexts(args.plaintexts)
    assert args.byte_i in (0,5,10,15) and args.byte_j in (0,5,10,15), "Use bytes 0/5/10/15 (serialized order)"
    roi_idx, var = top_variance_roi(args.tracedir, args.roi)

    # CECA
    ceca_k, ceca_best_dk, (ceca_t, ceca_u), ceca_sizes, ceca_prog, ceca_points = compute_ceca(
        args.tracedir, P, args.byte_i, args.byte_j, roi_idx, args.steps, outdir
    )

    # MC-DPA
    mcdpa_k, mcdpa_best_dk, base_idx, mcdpa_sizes, mcdpa_prog, mcdpa_points = compute_mcdpa(
        args.tracedir, P, args.byte_i, args.byte_j, roi_idx, args.steps, outdir, base_idx=ceca_t
    )

    # Choose final Δk as one agreed by both (or CECA if differ)
    dk_final = int(ceca_best_dk if ceca_best_dk == mcdpa_best_dk else ceca_best_dk)

    # key.txt
    with open(outdir/"key.txt", "w", encoding="utf-8") as f:
        f.write(f"Δk (byte {args.byte_i} ⊕ byte {args.byte_j})\n")
        f.write(f"hex: {dk_final:02x}\n")
        f.write(f"dec: {dk_final}\n")

    # description.txt
    elapsed = time.time() - t0
    with open(outdir/"description.txt", "w", encoding="utf-8") as f:
        f.write("Correlation-Enhanced Collision (CECA) and Moments-Correlating DPA (MC-DPA)\n")
        f.write(f"bytes: {args.byte_i} & {args.byte_j}, ROI={args.roi}, steps={args.steps}\n")
        f.write(f"Winning Δk (CECA): 0x{ceca_best_dk:02x}  |  Winning Δk (MC-DPA): 0x{mcdpa_best_dk:02x}\n")
        f.write("Traces needed (prog. step where correct becomes global max):\n")
        f.write("CECA: ")
        f.write(f"{int(ceca_sizes[np.argmax(ceca_prog >= ceca_prog.max()-1e-9)]) if ceca_prog.size else 'n/a'}\n")
        f.write("MC-DPA: ")
        f.write(f"{int(mcdpa_sizes[np.argmax(mcdpa_prog >= mcdpa_prog.max()-1e-9)]) if mcdpa_prog.size else 'n/a'}\n")
        f.write(f"Runtime (this run): ~{elapsed:.1f}s (depends on ROI, I/O, CPU).\n")
        f.write("\nShort differences:\n")
        f.write("- CECA: correlates leakage columns only within the collision subset for each Δk.\n")
        f.write("- MC-DPA: correlates a model vector with cross-moments xb*Xu across ALL traces.\n")
        f.write("If an attack fails, increase ROI or try alternative base column, or improve alignment.\n")

    # Pack submission archive name
    zipname = f\"{args.lastname}_Assignment9.zip\"
    with zipfile.ZipFile(zipname, \"w\", compression=zipfile.ZIP_DEFLATED) as z:
        for name in [\"key.txt\", \"description.txt\",
                     \"plot_k_corr_ceca.png\", \"plot_nrtraces_corr_ceca.png\", \"plot_points_corr_ceca.png\",
                     \"plot_k_corr_mcdpa.png\", \"plot_nrtraces_corr_mcdpa.png\", \"plot_points_corr_mcdpa.png\"]:
            p = outdir/name
            if p.exists(): z.write(p, arcname=name)
        # include code
        z.write(Path(__file__), arcname=\"Code/\" + Path(__file__).name)

if __name__ == \"__main__\":
    main()
