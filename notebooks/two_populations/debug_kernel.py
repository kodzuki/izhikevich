"""
debug_kernel.py
===============
Versión Python (sin numba) del kernel de propagación, con logging completo
de distribuciones intermedias. Úsalo sobre UN archivo K=0 y UN archivo K>0
para comparar y diagnosticar el blow-up de normalización.

Uso:
    from debug_kernel import run_debug_comparison
    run_debug_comparison(fpath_k0, fpath_kpos, Ne=1000, warmup=500.0, window_ms=4.0)
"""

import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
from scipy import sparse
from pathlib import Path
from collections import defaultdict


# =============================================================================
# KERNEL PYTHON INSTRUMENTADO
# =============================================================================

def _kernel_debug(t_sorted, i_sorted, indptr, indices,
                   rev_indptr, rev_indices,
                   window, Ne, min_spikes,
                   label="", normalize=True):
    """
    Mirror no-numba del kernel V7 con logging de distribuciones.
    
    Parámetros
    ----------
    normalize : bool
        True  → weights[i] = 1/parent_hits  (versión nueva, problemática)
        False → weights[i] = 1.0            (versión baseline, sin normalizar)
    """
    n_events = len(t_sorted)
    weights = np.ones(n_events, dtype=np.float64)
    
    # -------------------------------------------------------------------------
    # BACKWARD PASS: calcular parent_hits y asignar weights
    # -------------------------------------------------------------------------
    parent_hits_all = []   # para todo spike con parents > 0
    parent_hits_zero = 0   # spikes con ningún parent en ventana
    weights_assigned = []  # weights finales tras backward pass

    left_ptr = 0
    for i in range(n_events):
        t_curr = t_sorted[i]
        node_curr = i_sorted[i]
        t_min = t_curr - window

        while left_ptr < i and t_sorted[left_ptr] < t_min:
            left_ptr += 1

        n_parents_total = rev_indptr[node_curr + 1] - rev_indptr[node_curr]
        if n_parents_total == 0:
            weights_assigned.append(weights[i])  # 1.0
            continue

        parent_hits = 0
        start_p = rev_indptr[node_curr]
        end_p = rev_indptr[node_curr + 1]
        window_len = i - left_ptr

        if window_len > 0:
            if window_len < n_parents_total:
                for k in range(left_ptr, i):
                    pre_cand = i_sorted[k]
                    l, r = start_p, end_p - 1
                    while l <= r:
                        mid = (l + r) >> 1
                        val = rev_indices[mid]
                        if val < pre_cand:
                            l = mid + 1
                        elif val > pre_cand:
                            r = mid - 1
                        else:
                            parent_hits += 1
                            break
            else:
                for k in range(left_ptr, i):
                    pre_cand = i_sorted[k]
                    for z in range(start_p, end_p):
                        if rev_indices[z] == pre_cand:
                            parent_hits += 1
                            break

        if parent_hits > 0:
            parent_hits_all.append(parent_hits)
            if normalize:
                weights[i] = 1.0 / parent_hits
            # else: weights[i] queda 1.0
        else:
            parent_hits_zero += 1

        weights_assigned.append(weights[i])

    # -------------------------------------------------------------------------
    # LOG BACKWARD PASS
    # -------------------------------------------------------------------------
    parent_hits_all = np.array(parent_hits_all) if parent_hits_all else np.array([0])
    weights_assigned = np.array(weights_assigned)

    print(f"\n{'─'*60}")
    print(f"  [{label}] BACKWARD PASS")
    print(f"{'─'*60}")
    print(f"  Total spikes:          {n_events}")
    print(f"  Spikes con parents>0:  {len(parent_hits_all)} ({100*len(parent_hits_all)/n_events:.1f}%)")
    print(f"  Spikes con parents=0:  {parent_hits_zero} ({100*parent_hits_zero/n_events:.1f}%)")
    if len(parent_hits_all) > 0 and parent_hits_all[0] > 0:
        print(f"  parent_hits stats:")
        print(f"    mean={parent_hits_all.mean():.2f}, median={np.median(parent_hits_all):.1f}")
        print(f"    p95={np.percentile(parent_hits_all, 95):.1f}, max={parent_hits_all.max()}")
    print(f"  Weight stats (post-backward):")
    print(f"    mean={weights_assigned.mean():.4f}, median={np.median(weights_assigned):.4f}")
    print(f"    n(weight<0.5): {np.sum(weights_assigned < 0.5)} ({100*np.mean(weights_assigned < 0.5):.1f}%)")
    print(f"    n(weight=1.0): {np.sum(weights_assigned == 1.0)} ({100*np.mean(weights_assigned == 1.0):.1f}%)")

    # -------------------------------------------------------------------------
    # FORWARD PASS: calcular p_val por spike pre
    # -------------------------------------------------------------------------
    neuron_counts = defaultdict(int)
    for nid in i_sorted:
        if nid < Ne:
            neuron_counts[nid] += 1

    p_vals = []
    sigma_vals = []
    weighted_sums = []
    n_children_list = []

    right_idx = 0
    for i in range(n_events):
        pre_id = i_sorted[i]

        if pre_id >= Ne or neuron_counts[pre_id] < min_spikes:
            continue

        t_curr = t_sorted[i]
        t_max = t_curr + window

        start_ptr = indptr[pre_id]
        end_ptr = indptr[pre_id + 1]
        n_neighbors = end_ptr - start_ptr
        if n_neighbors == 0:
            continue

        if right_idx < i + 1:
            right_idx = i + 1
        while right_idx < n_events and t_sorted[right_idx] <= t_max:
            right_idx += 1

        weighted_sum = 0.0
        raw_children = 0
        for k in range(i + 1, right_idx):
            post_id = i_sorted[k]
            l, r = start_ptr, end_ptr - 1
            is_child = False
            while l <= r:
                mid = (l + r) >> 1
                val = indices[mid]
                if val < post_id:
                    l = mid + 1
                elif val > post_id:
                    r = mid - 1
                else:
                    is_child = True
                    break
            if is_child:
                weighted_sum += weights[k]
                raw_children += 1

        p_val = weighted_sum / n_neighbors
        p_vals.append(p_val)
        sigma_vals.append(weighted_sum)
        weighted_sums.append(weighted_sum)
        n_children_list.append(raw_children)

    # -------------------------------------------------------------------------
    # LOG FORWARD PASS
    # -------------------------------------------------------------------------
    p_vals = np.array(p_vals)
    sigma_vals = np.array(sigma_vals)
    n_children_arr = np.array(n_children_list)

    print(f"\n  [{label}] FORWARD PASS")
    print(f"{'─'*60}")
    print(f"  Pre-spikes analizados: {len(p_vals)}")
    if len(p_vals) > 0:
        print(f"  P_transmission:")
        print(f"    mean={p_vals.mean():.5f}, std={p_vals.std():.5f}")
        print(f"    p50={np.percentile(p_vals, 50):.5f}, p95={np.percentile(p_vals, 95):.5f}")
        print(f"    n(p_val>0): {np.sum(p_vals>0)} ({100*np.mean(p_vals>0):.1f}%)")
        print(f"  Sigma (weighted_sum):")
        print(f"    mean={sigma_vals.mean():.3f}, std={sigma_vals.std():.3f}")
        print(f"    max={sigma_vals.max():.3f}")
        print(f"  Raw children por pre-spike:")
        print(f"    mean={n_children_arr.mean():.3f}, max={n_children_arr.max()}")
        
        # DIAGNÓSTICO CLAVE: diferencia raw vs weighted
        p_raw_equiv = n_children_arr / (np.array([
            indptr[int(i_sorted[i]) + 1] - indptr[int(i_sorted[i])]
            for i, pid in enumerate(i_sorted)
            if i_sorted[i] < Ne and neuron_counts[i_sorted[i]] >= min_spikes
               and (indptr[int(i_sorted[i]) + 1] - indptr[int(i_sorted[i])]) > 0
        ]) if len(n_children_arr) > 0 else np.ones(len(n_children_arr)))
        # Simplifiquemos: ratio weighted_sum / raw_children cuando hay hijos
        mask_c = n_children_arr > 0
        if mask_c.any():
            ratio_w_raw = weighted_sums_arr = np.array(weighted_sums)[mask_c] / n_children_arr[mask_c]
            print(f"  Ratio weighted/raw (cuando hay hijos):")
            print(f"    mean={ratio_w_raw.mean():.4f}, min={ratio_w_raw.min():.4f}")
            print(f"    → 1.0 = sin efecto, <1.0 = normalización activa")

    return {
        'p_mean': p_vals.mean() if len(p_vals) > 0 else 0,
        'sigma_mean': sigma_vals.mean() if len(sigma_vals) > 0 else 0,
        'parent_hits_all': parent_hits_all,
        'weights_assigned': weights_assigned,
        'p_vals': p_vals,
        'sigma_vals': sigma_vals,
        'n_children': n_children_arr,
    }


# =============================================================================
# FUNCIÓN DE CARGA Y PREPARACIÓN
# =============================================================================

def _load_and_prep(fpath, Ne, warmup):
    """Carga un pkl.gz y devuelve los arrays necesarios."""
    with gzip.open(fpath, 'rb') as f:
        data = pickle.load(f)

    st = data['spike_times']
    si = data['spike_indices']
    syn = data.get('synapses')
    T = data.get('T_total', 4000)

    # Filtrar warmup
    mask = st >= warmup
    st_f, si_f = st[mask], si[mask]

    # Ordenar por tiempo
    sort_idx = np.argsort(st_f)
    t_sorted = st_f[sort_idx].astype(np.float64)
    i_sorted = si_f[sort_idx].astype(np.int32)

    # Extraer sinapsis — robusto contra múltiples formatos de serialización
    if syn is None:
        # Sin sinapsis guardadas: reconstruir topología aleatoria E→E como proxy
        # (solo para poder buildear la CSR; w=0 → is_baseline=True)
        print("  ⚠️  syn=None: archivo sin sinapsis guardadas → skip")
        return None

    elif isinstance(syn, dict):
        # Formato dict {'i': ..., 'j': ..., 'w': ...}
        s_arr = np.array(syn['i'], dtype=np.int32)
        t_arr = np.array(syn['j'], dtype=np.int32)
        w_arr = np.array(syn.get('w', np.ones(len(s_arr))), dtype=np.float32)

    elif hasattr(syn, 'i') and hasattr(syn, 'j'):
        # Formato objeto Brian2 serializado
        s_arr = np.array(syn.i, dtype=np.int32)
        t_arr = np.array(syn.j, dtype=np.int32)
        w_arr = np.array(syn.w, dtype=np.float32) if hasattr(syn, 'w') else np.ones(len(s_arr))

    elif isinstance(syn, (list, tuple)) and len(syn) >= 2:
        # Formato tuple/list (s_arr, t_arr) o (s_arr, t_arr, w_arr)
        s_arr = np.array(syn[0], dtype=np.int32)
        t_arr = np.array(syn[1], dtype=np.int32)
        w_arr = np.array(syn[2], dtype=np.float32) if len(syn) > 2 else np.ones(len(s_arr))

    elif isinstance(syn, np.ndarray):
        # Formato array estructurado con campos, o matriz (N,2)/(N,3)
        if syn.dtype.names and 'i' in syn.dtype.names:
            s_arr = syn['i'].astype(np.int32)
            t_arr = syn['j'].astype(np.int32)
            w_arr = syn['w'].astype(np.float32) if 'w' in syn.dtype.names else np.ones(len(s_arr))
        elif syn.ndim == 2 and syn.shape[1] >= 2:
            s_arr = syn[:, 0].astype(np.int32)
            t_arr = syn[:, 1].astype(np.int32)
            w_arr = syn[:, 2].astype(np.float32) if syn.shape[1] > 2 else np.ones(len(s_arr))
        else:
            print(f"  ⚠️  syn es ndarray con shape={syn.shape}, dtype={syn.dtype} — no reconocido → skip")
            return None

    else:
        # Log del tipo real para ayudar a diagnosticar
        print(f"  ⚠️  Formato de sinapsis no reconocido: type={type(syn)}")
        print(f"       attrs disponibles: {[a for a in dir(syn) if not a.startswith('_')][:15]}")
        if hasattr(syn, '__dict__'):
            print(f"       __dict__ keys: {list(syn.__dict__.keys())[:10]}")
        print("       → skip")
        return None

    is_baseline = np.all(np.abs(w_arr) < 1e-10)
    print(f"  is_baseline: {is_baseline} | w_min={w_arr.min():.2e}, w_max={w_arr.max():.2e}")
    print(f"  N spikes (post-warmup): {len(t_sorted)}")
    print(f"  N synapses: {len(s_arr)}")

    # Filtrar solo E como pre
    mask_e = s_arr < Ne
    s_e = s_arr[mask_e]
    t_e = t_arr[mask_e]

    N_max = max(Ne, t_e.max() + 1) if len(t_e) > 0 else Ne
    coo = sparse.coo_matrix(
        (np.ones(len(s_e), dtype=bool), (s_e, t_e)),
        shape=(N_max, N_max)
    )
    csr = coo.tocsr(); csr.sort_indices()
    csc = coo.tocsc(); csc.sort_indices()

    return {
        't_sorted': t_sorted,
        'i_sorted': i_sorted,
        'csr': csr,
        'csc': csc,
        'T': T,
        'is_baseline': is_baseline,
    }


# =============================================================================
# FUNCIÓN PRINCIPAL DE COMPARACIÓN
# =============================================================================

def run_debug_comparison(fpath_k0, fpath_kpos, Ne=1000, warmup=500.0, window_ms=4.0,
                          min_spikes=1):
    """
    Corre el kernel debug en modo normalizado y no-normalizado para K=0 y K>0.
    Imprime logs detallados y genera una figura comparativa.
    
    Uso
    ---
    run_debug_comparison(
        fpath_k0   = "results/sweep/raw_data_k0.00_r9.7_t2_xxxx.pkl.gz",
        fpath_kpos = "results/sweep/raw_data_k10.00_r15.0_t1_xxxx.pkl.gz",
        Ne=1000, warmup=500.0, window_ms=4.0
    )
    """
    window = float(window_ms)

    print("=" * 70)
    print("CARGANDO K=0")
    print("=" * 70)
    d0 = _load_and_prep(fpath_k0, Ne, warmup)

    print("\n" + "=" * 70)
    print("CARGANDO K>0")
    print("=" * 70)
    dk = _load_and_prep(fpath_kpos, Ne, warmup)

    results = {}
    for label, d, norm in [
        ("K=0  | NO norm",  d0, False),
        ("K=0  | CON norm", d0, True),
        ("K>0  | NO norm",  dk, False),
        ("K>0  | CON norm", dk, True),
    ]:
        print(f"\n{'='*70}")
        print(f"  CASO: {label}")
        print(f"{'='*70}")
        r = _kernel_debug(
            d['t_sorted'], d['i_sorted'],
            d['csr'].indptr, d['csr'].indices,
            d['csc'].indptr, d['csc'].indices,
            window, Ne, min_spikes,
            label=label, normalize=norm
        )
        results[label] = r

    # -------------------------------------------------------------------------
    # FIGURA DIAGNÓSTICA
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Debug: Comparación K=0 vs K>0 (con/sin normalización)", fontsize=13)

    colors = {
        "K=0  | NO norm":  ("pink",    "solid"),
        "K=0  | CON norm": ("red",     "dashed"),
        "K>0  | NO norm":  ("skyblue", "solid"),
        "K>0  | CON norm": ("blue",    "dashed"),
    }

    # Plot 1: distribución parent_hits
    ax = axes[0, 0]
    ax.set_title("Distribución parent_hits")
    for key, res in results.items():
        ph = res['parent_hits_all']
        if ph.max() > 0:
            c, ls = colors[key]
            max_ph = min(int(ph.max()), 30)
            ax.hist(ph, bins=range(1, max_ph + 2), alpha=0.5,
                    color=c, linestyle=ls, label=key, density=True)
    ax.set_xlabel("parent_hits"); ax.legend(fontsize=7)

    # Plot 2: distribución de weights (post-backward)
    ax = axes[0, 1]
    ax.set_title("Distribución weights (backward pass)")
    for key, res in results.items():
        c, ls = colors[key]
        w = res['weights_assigned']
        ax.hist(w, bins=50, alpha=0.5, color=c, label=key, density=True)
    ax.set_xlabel("weight"); ax.legend(fontsize=7)

    # Plot 3: P_transmission por caso (bar)
    ax = axes[0, 2]
    ax.set_title("P_transmission por caso")
    labels_plot = list(results.keys())
    p_means = [results[k]['p_mean'] for k in labels_plot]
    bar_colors = [colors[k][0] for k in labels_plot]
    bars = ax.bar(range(len(labels_plot)), p_means, color=bar_colors)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=20, ha='right', fontsize=7)
    ax.set_ylabel("P_transmission")
    for b, v in zip(bars, p_means):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}", ha='center', fontsize=8)

    # Plot 4: distribución P_vals
    ax = axes[1, 0]
    ax.set_title("Distribución P_vals (por pre-spike)")
    for key, res in results.items():
        c, ls = colors[key]
        pv = res['p_vals']
        if len(pv) > 0:
            ax.hist(pv, bins=50, alpha=0.5, color=c, label=key, density=True)
    ax.set_xlabel("p_val"); ax.legend(fontsize=7)

    # Plot 5: distribución sigma (weighted_sum)
    ax = axes[1, 1]
    ax.set_title("Distribución sigma (weighted_sum)")
    for key, res in results.items():
        c, ls = colors[key]
        sv = res['sigma_vals']
        if len(sv) > 0:
            ax.hist(sv, bins=50, alpha=0.5, color=c, label=key, density=True)
    ax.set_xlabel("weighted_sum"); ax.legend(fontsize=7)

    # Plot 6: raw_children por pre-spike
    ax = axes[1, 2]
    ax.set_title("Raw children por pre-spike")
    for key, res in results.items():
        c, ls = colors[key]
        nc = res['n_children']
        if len(nc) > 0:
            max_c = min(int(nc.max()), 20)
            ax.hist(nc, bins=range(0, max_c + 2), alpha=0.5, color=c, label=key, density=True)
    ax.set_xlabel("n_children_raw"); ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = "/mnt/user-data/outputs/debug_normalization.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n📊 Figura guardada en: {out_path}")
    plt.show()

    # -------------------------------------------------------------------------
    # TABLA RESUMEN FINAL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLA RESUMEN")
    print("=" * 70)
    print(f"{'Caso':<22} {'P_mean':>10} {'sigma_mean':>12} {'w_mean':>10} {'ph_mean':>10}")
    print("-" * 70)
    for key, res in results.items():
        ph = res['parent_hits_all']
        ph_m = ph.mean() if len(ph) > 0 and ph.max() > 0 else 0.0
        wm = res['weights_assigned'].mean()
        print(f"  {key:<20} {res['p_mean']:>10.5f} {res['sigma_mean']:>12.4f} {wm:>10.4f} {ph_m:>10.2f}")
    print("=" * 70)

    return results


# =============================================================================
# FUNCIÓN DE DIAGNÓSTICO RÁPIDO (sin figura, solo logs)
# =============================================================================

def quick_diagnose(fpath, Ne=1000, warmup=500.0, window_ms=4.0, min_spikes=1, label="TEST"):
    """
    Diagnóstico rápido sobre un único archivo.
    Corre el kernel en ambos modos (norm/no-norm) y compara.
    Devuelve None si el archivo no tiene sinapsis válidas.
    """
    print(f"\n{'='*70}")
    print(f"QUICK DIAGNOSE: {Path(fpath).name}")
    print(f"{'='*70}")
    d = _load_and_prep(fpath, Ne, warmup)

    if d is None:
        print("  → Archivo skipeado (sin sinapsis válidas)\n")
        return None

    r = None
    for norm in [False, True]:
        tag = f"{label} | {'CON' if norm else 'NO'} norm"
        r = _kernel_debug(
            d['t_sorted'], d['i_sorted'],
            d['csr'].indptr, d['csr'].indices,
            d['csc'].indptr, d['csc'].indices,
            float(window_ms), Ne, min_spikes,
            label=tag, normalize=norm
        )

    return r
