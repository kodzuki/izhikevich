


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# -------------------- Mapping --------------------
def guess_variable_mapping(data: np.ndarray) -> Dict[int, str]:
    """
    Guess mapping of columns {0..5} -> {'tau','distance','velocity','AX','FM','FR'}.
    Heuristics (can be overridden by user):
      - tau  ~ 0.0001..0.02 s (0.1–20 ms)
      - distance ~ 0..0.02 m (brain size ~1.6 cm)
      - velocity ~ 0.5..15 m/s (expected ~5 m/s)
    Returns dict mapping column_index -> name.
    """
    m = {}
    ncol = data.shape[1]
    cols = range(ncol)

    # Compute ranges
    ranges = {i: (np.nanmin(data[:, i]), np.nanmax(data[:, i])) for i in cols}

    # Identify distance
    cand_dist = [i for i, (mn, mx) in ranges.items() if 0 <= mn and mx <= 0.05]
    # Prefer column with max <= 0.02
    cand_dist.sort(key=lambda i: (abs(ranges[i][1] - 0.016), ranges[i][1]))
    if cand_dist:
        m[cand_dist[0]] = 'distance'

    # Identify velocity (around ~5 m/s typical)
    cand_vel = [i for i, (mn, mx) in ranges.items() if mx >= 0.5 and mn >= 0]
    if cand_vel:
        # choose by median closeness to ~5
        vel_scores = []
        for i in cand_vel:
            med = float(np.nanmedian(data[:, i]))
            vel_scores.append((abs(med-5.0), i))
        vel_idx = sorted(vel_scores)[0][1]
        if vel_idx not in m:
            m[vel_idx] = 'velocity'

    # Identify tau (seconds, small ~ms)
    cand_tau = [i for i, (mn, mx) in ranges.items() if 1e-5 <= mx <= 0.1]
    if cand_tau:
        # prefer with median ~0.0005 (0.5ms)
        tau_scores = []
        for i in cand_tau:
            med = float(np.nanmedian(data[:, i]))
            tau_scores.append((abs(med-5e-4), i))
        tau_idx = sorted(tau_scores)[0][1]
        if tau_idx not in m:
            m[tau_idx] = 'tau'

    # Fill remaining as microstructural
    names_left = ['AX', 'FM', 'FR']
    for i in cols:
        if i not in m:
            if names_left:
                m[i] = names_left.pop(0)
            else:
                m[i] = f"var_{i}"
    return m

# -------------------- Computations --------------------
def compute_tau(distance_m: np.ndarray, velocity_mps: np.ndarray) -> np.ndarray:
    """τ = distance / velocity (seconds). Safe division with NaN handling."""
    v = np.where(velocity_mps<=0, np.nan, velocity_mps.astype(float))
    d = distance_m.astype(float)
    tau = d / v
    return tau

def clean_delays_ms(tau_s: np.ndarray, method: str = 'iqr', min_ms: float = 0.05, max_ms: float = 50.0) -> np.ndarray:
    """
    Clean and bound τ distribution in milliseconds.
    - Bounds by [min_ms, max_ms] (biologically plausible range)
    - Remove outliers via IQR or z-score.
    """
    ms = tau_s * 1e3
    ms = ms[np.isfinite(ms)]
    ms = ms[(ms >= min_ms) & (ms <= max_ms)]
    if len(ms) == 0:
        return ms

    if method == 'iqr':
        q1, q3 = np.percentile(ms, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        ms = ms[(ms >= lo) & (ms <= hi)]
    elif method == 'zscore':
        mu, sd = ms.mean(), ms.std() if ms.std() > 0 else 1.0
        z = (ms - mu) / sd
        ms = ms[np.abs(z) <= 3.0]
    return ms

# -------------------- Selection --------------------
def select_delay_sets(
    tau_ms: np.ndarray,
    distance_m: Optional[np.ndarray] = None,
    velocity_mps: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Build a small library of delay sets:
      - 'all_clean': cleaned τ
      - 'short_range': D < 5 mm (if distance provided)
      - 'long_range':  D >= 5 mm (if distance provided)
      - 'slow_paths':  V in lowest quartile (if velocity provided)
      - 'fast_paths':  V in highest quartile (if velocity provided)
      - 'q1','q2','q3','q4': τ quartiles
    """
    out = {'all_clean': tau_ms.copy()}

    if distance_m is not None and len(distance_m)==len(tau_ms):
        mm = distance_m * 1e3
        mask_short = mm < 5.0
        mask_long  = mm >= 5.0
        out['short_range'] = tau_ms[mask_short]
        out['long_range']  = tau_ms[mask_long]

    if velocity_mps is not None and len(velocity_mps)==len(tau_ms):
        v = velocity_mps
        q1, q3 = np.nanpercentile(v, [25, 75])
        out['slow_paths'] = tau_ms[v <= q1]
        out['fast_paths'] = tau_ms[v >= q3]

    # quartiles of tau
    q = np.nanpercentile(tau_ms, [25, 50, 75])
    out['q1'] = tau_ms[tau_ms <= q[0]]
    out['q2'] = tau_ms[(tau_ms > q[0]) & (tau_ms <= q[1])]
    out['q3'] = tau_ms[(tau_ms > q[1]) & (tau_ms <= q[2])]
    out['q4'] = tau_ms[tau_ms > q[2]]
    return out

# -------------------- Export --------------------
def export_delay_sets(sets: Dict[str, np.ndarray], outdir: str, prefix: str = "delays"):
    os.makedirs(outdir, exist_ok=True)
    meta = {}
    for name, arr in sets.items():
        safe = arr.astype(float)
        path_npy = os.path.join(outdir, f"{prefix}_{name}.npy")
        path_csv = os.path.join(outdir, f"{prefix}_{name}.csv")
        np.save(path_npy, safe)
        pd.Series(safe).to_csv(path_csv, index=False, header=['tau_ms'])
        meta[name] = dict(n=int(len(safe)), file_npy=path_npy, file_csv=path_csv)
    # write a small manifest
    manifest = os.path.join(outdir, f"{prefix}_manifest.json")
    with open(manifest, "w") as f:
        import json
        json.dump(meta, f, indent=2)
    return meta

# -------------------- Quick Plot --------------------
def plot_delay_histograms(sets: Dict[str, np.ndarray], bins: int = 50):
    """Quick histograms for visual inspection (matplotlib only, no custom colors)."""
    import math
    names = list(sets.keys())
    k = len(names)
    cols = 2
    rows = math.ceil(k/cols)
    for idx, name in enumerate(names, 1):
        plt.figure()
        plt.hist(sets[name], bins=bins)
        plt.xlabel("τ (ms)")
        plt.ylabel("Count")
        plt.title(name)
        plt.tight_layout()
