# # """
# # spike_propagation_reanalysis.py
# # ================================
# # Re-análisis de propagación de spikes sobre raw data del barrido 2D existente.

# # Métricas calculadas:
# #     1. sigma_global      : <A(t+1)/A(t)>  — estimador Beggs & Plenz
# #     2. sigma_micro_EE    : tracking vecinos E→E por spike
# #     3. sigma_micro_Eall  : tracking vecinos E→(E+I) por spike
# #     4. p_prop_mean       : probabilidad media de activar un vecino concreto
# #     5. sigma_corrected   : sigma_global con substracción de baseline k=0

# # Corrección baseline:
# #     El caso k=0 (sin acoplamiento) actúa como referencia de actividad Poisson pura.
# #     ΔA(t) = A_k>0(t) - <A_k=0(t)>  → elimina contribución del input externo
# #     sigma_corrected se calcula sobre ΔA, capturando solo la dinámica de red.

# # Elección de bin temporal:
# #     dt_sim = 0.1 ms, tau_syn = 1.5 ms, window_mono = 4 ms
# #     → bin_ms = 1.0 ms  (10x dt_sim, <τ_syn×2, sensato para 1 generación sináptica)
# #     → Se valida con barrido [0.5, 1, 2, 4] ms y se verifica estabilidad de sigma

# # Referencias:
# #     - Beggs & Plenz (2003) J. Neurosci. 23:11167  — σ global
# #     - Wilting & Priesemann (2018) Nat. Commun. 9:2325  — MR estimator, baseline
# #     - Zierenberg et al. (2020) Phys. Rev. E 102:040301 — corrección input no estacionario
# # """

# # import numpy as np
# # import pandas as pd
# # import pickle
# # import gzip
# # from pathlib import Path
# # from collections import defaultdict
# # from tqdm import tqdm
# # import warnings
# # warnings.filterwarnings('ignore')

# # # ─────────────────────────────────────────────────────────────────────────────
# # # CONFIGURACIÓN (ajustar a tu sweep)
# # # ─────────────────────────────────────────────────────────────────────────────

# # SWEEP_DIR = Path('results/spike_propagation_2d/sweep_2d_XXXXXXXX_XXXXXX')  # ← EDITAR

# # REANALYSIS_CONFIG = {
# #     # Red
# #     'Ne': 800,
# #     'Ni': 200,
# #     # Simulación
# #     'dt_sim_ms': 0.1,
# #     'T_ms': 4000,
# #     'warmup_ms': 500,
# #     # Análisis
# #     'dt_bin_ms': 1.0,         # bin de análisis (10x dt_sim)
# #     'window_ms': 4.0,         # ventana monosináptica
# #     'min_spikes_ancestor': 3, # mínimo de spikes para incluir neurona como ancestro
# #     # Barrido de bins para validación (ms)
# #     'bin_sweep': [0.5, 1.0, 2.0, 4.0],
# # }


# # # ─────────────────────────────────────────────────────────────────────────────
# # # 1. CARGA DE DATOS
# # # ─────────────────────────────────────────────────────────────────────────────

# # def load_sweep_index(sweep_dir: Path) -> dict:
# #     """Carga el índice del sweep (results.pkl)."""
# #     results_path = sweep_dir / 'results.pkl'
# #     with open(results_path, 'rb') as f:
# #         data = pickle.load(f)
# #     return data


# # def load_raw_sim(filepath: str) -> dict:
# #     """Carga raw data de una simulación individual (.pkl.gz)."""
# #     with gzip.open(filepath, 'rb') as f:
# #         return pickle.load(f)


# # # ─────────────────────────────────────────────────────────────────────────────
# # # 2. CONSTRUCCIÓN DE ESTRUCTURAS INTERNAS
# # # ─────────────────────────────────────────────────────────────────────────────

# # def build_neighbors(syn_i, syn_j, syn_w, Ne, Ni,
# #                     mode='EE', min_weight=0.0):
# #     """
# #     Construye diccionario de vecinos desde arrays de sinapsis.

# #     Parameters
# #     ----------
# #     syn_i, syn_j, syn_w : arrays de pre, post e peso
# #     Ne : número de neuronas excitatorias (índices 0..Ne-1)
# #     Ni : número de neuronas inhibitorias (índices Ne..Ne+Ni-1)
# #     mode : 'EE'    → ancestros E, descendientes E
# #            'Eall'  → ancestros E, descendientes E+I
# #            'all'   → ancestros E+I, descendientes E+I
# #     min_weight : umbral mínimo de peso

# #     Returns
# #     -------
# #     neighbors : dict {pre_idx: array(post_idx)}
# #     """
# #     syn_i = np.asarray(syn_i, dtype=np.int32)
# #     syn_j = np.asarray(syn_j, dtype=np.int32)
# #     syn_w = np.asarray(syn_w, dtype=np.float32)

# #     exc_mask_pre  = syn_i < Ne
# #     exc_mask_post = syn_j < Ne
# #     weight_mask   = syn_w >= min_weight

# #     # Seleccionar ancestros (pre)
# #     if mode in ('EE', 'Eall'):
# #         anc_mask = exc_mask_pre
# #     else:  # 'all'
# #         anc_mask = np.ones(len(syn_i), dtype=bool)

# #     # Seleccionar descendientes (post)
# #     if mode == 'EE':
# #         desc_mask = exc_mask_post
# #     else:  # 'Eall' o 'all'
# #         desc_mask = np.ones(len(syn_j), dtype=bool)

# #     final_mask = anc_mask & desc_mask & weight_mask

# #     neighbors = defaultdict(list)
# #     for pre, post in zip(syn_i[final_mask], syn_j[final_mask]):
# #         neighbors[int(pre)].append(int(post))

# #     return {k: np.array(v, dtype=np.int32) for k, v in neighbors.items()}


# # def build_spike_dict(spike_times: np.ndarray, spike_indices: np.ndarray,
# #                      warmup_ms: float) -> dict:
# #     """
# #     Organiza spikes por neurona, filtrando el periodo de warmup.

# #     Returns
# #     -------
# #     spike_dict : {neuron_id: sorted_spike_times_array_ms}
# #     """
# #     mask = spike_times >= warmup_ms
# #     t = spike_times[mask].astype(np.float32)
# #     idx = spike_indices[mask].astype(np.int32)

# #     spike_dict = defaultdict(list)
# #     for time, nid in zip(t, idx):
# #         spike_dict[int(nid)].append(float(time))

# #     return {k: np.sort(v, kind='mergesort') for k, v in spike_dict.items()}


# # def build_population_activity(spike_dict: dict, dt_bin_ms: float,
# #                                T_ms: float, warmup_ms: float,
# #                                neuron_mask=None) -> np.ndarray:
# #     """
# #     Construye la serie temporal de actividad poblacional A(t).

# #     Parameters
# #     ----------
# #     spike_dict  : {nid: spike_times_ms}
# #     dt_bin_ms   : resolución del bin de análisis
# #     T_ms        : duración total de la simulación
# #     warmup_ms   : periodo excluido (ya filtrado en spike_dict, pero necesitamos T_analysis)
# #     neuron_mask : set de nids a incluir (None = todos)

# #     Returns
# #     -------
# #     A : array (n_bins,) con conteo de spikes por bin
# #     """
# #     T_analysis = T_ms - warmup_ms
# #     n_bins = int(T_analysis / dt_bin_ms)
# #     A = np.zeros(n_bins, dtype=np.float32)

# #     for nid, times in spike_dict.items():
# #         if neuron_mask is not None and nid not in neuron_mask:
# #             continue
# #         # Los spikes ya están filtrados por warmup; desplazamos al origen
# #         shifted = times - warmup_ms
# #         bins_idx = (shifted / dt_bin_ms).astype(np.int32)
# #         valid = (bins_idx >= 0) & (bins_idx < n_bins)
# #         np.add.at(A, bins_idx[valid], 1)

# #     return A


# # # ─────────────────────────────────────────────────────────────────────────────
# # # 3. ESTIMADORES DE SIGMA
# # # ─────────────────────────────────────────────────────────────────────────────

# # def sigma_global(A: np.ndarray, min_active: int = 1) -> float:
# #     """
# #     Estimador global Beggs-Plenz: <A(t+1)/A(t)>.

# #     Solo usa bins donde A(t) >= min_active para evitar divisiones por 0.
# #     """
# #     mask = A[:-1] >= min_active
# #     if not mask.any():
# #         return np.nan
# #     return float(np.mean(A[1:][mask] / A[:-1][mask]))


# # def sigma_corrected_k0(A_coupled: np.ndarray, A_baseline_mean: float,
# #                         min_active: float = 0.5) -> float:
# #     """
# #     Sigma global corregido por baseline k=0.

# #     ΔA(t) = A_coupled(t) - A_baseline_mean
# #     sigma_corr = <ΔA(t+1) / ΔA(t)>  donde ΔA(t) > 0

# #     Parameters
# #     ----------
# #     A_coupled      : actividad poblacional del caso k>0
# #     A_baseline_mean : media de A para k=0 al mismo rate_hz (el mismo número de trials)
# #     """
# #     delta_A = A_coupled - A_baseline_mean
# #     delta_A = np.clip(delta_A, 0, None)  # no puede ser negativo

# #     mask = delta_A[:-1] >= min_active
# #     if not mask.any():
# #         return np.nan
# #     return float(np.mean(delta_A[1:][mask] / delta_A[:-1][mask]))


# # def sigma_micro(spike_dict: dict, neighbors: dict,
# #                 window_ms: float, dt_bin_ms: float,
# #                 T_ms: float, warmup_ms: float,
# #                 min_spikes: int = 3):
# #     """
# #     Branching ratio microscópico con tracking de vecinos de 1er orden.

# #     Para cada spike de neurona ancestro i en bin t:
# #         σ_i(t) = # vecinos j ∈ N(i) que disparan en (t, t + window_ms]

# #     σ_micro = mean sobre todos los spikes ancestros

# #     Returns
# #     -------
# #     dict con:
# #         sigma        : float, branching ratio global
# #         p_prop_mean  : float, P(vecino específico se activa | spike ancestro)
# #         p_prop_dist  : array, distribución de p_prop por neurona
# #         per_neuron   : dict {nid: {'sigma': float, 'p_prop': float, 'n_spikes': int}}
# #     """
# #     T_analysis = T_ms - warmup_ms
# #     n_bins = int(T_analysis / dt_bin_ms)
# #     window_bins = max(1, int(window_ms / dt_bin_ms))

# #     # Construir raster: bin → set de nids activos
# #     # Usamos array de sets (más rápido para lookup que dict of lists)
# #     raster = [None] * n_bins
# #     for nid, times in spike_dict.items():
# #         shifted = times - warmup_ms
# #         bins_idx = (shifted / dt_bin_ms).astype(np.int32)
# #         for b in bins_idx:
# #             if 0 <= b < n_bins:
# #                 if raster[b] is None:
# #                     raster[b] = set()
# #                 raster[b].add(nid)

# #     # Reemplazar None por empty set para simplificar lookup
# #     for i in range(n_bins):
# #         if raster[i] is None:
# #             raster[i] = set()

# #     # Análisis de propagación
# #     total_descendants = 0
# #     total_spikes_analyzed = 0
# #     per_neuron = {}

# #     for pre_id, post_ids in neighbors.items():
# #         if pre_id not in spike_dict:
# #             continue

# #         pre_spikes_shifted = spike_dict[pre_id] - warmup_ms
# #         pre_bins = (pre_spikes_shifted / dt_bin_ms).astype(np.int32)
# #         valid_bins = pre_bins[(pre_bins >= 0) & (pre_bins < n_bins - window_bins)]

# #         if len(valid_bins) < min_spikes:
# #             continue

# #         post_set = set(post_ids.tolist())  # para lookup O(1)
# #         n_neighbors = len(post_ids)

# #         neuron_desc_sum = 0
# #         neuron_spikes = 0

# #         for t_bin in valid_bins:
# #             # Contar vecinos activos en ventana (t_bin, t_bin + window_bins]
# #             desc = 0
# #             for dt in range(1, window_bins + 1):
# #                 t_look = t_bin + dt
# #                 if t_look < n_bins:
# #                     desc += len(raster[t_look] & post_set)

# #             total_descendants += desc
# #             total_spikes_analyzed += 1
# #             neuron_desc_sum += desc
# #             neuron_spikes += 1

# #         # Estadísticas por neurona
# #         mean_desc = neuron_desc_sum / neuron_spikes
# #         p_prop = mean_desc / n_neighbors  # prob de activar un vecino específico

# #         per_neuron[pre_id] = {
# #             'sigma': mean_desc,
# #             'p_prop': p_prop,
# #             'n_spikes': neuron_spikes,
# #             'n_neighbors': n_neighbors
# #         }

# #     if total_spikes_analyzed == 0:
# #         return {
# #             'sigma': np.nan, 'p_prop_mean': np.nan,
# #             'p_prop_dist': np.array([]), 'per_neuron': {},
# #             'n_spikes_analyzed': 0
# #         }

# #     sigma_val = total_descendants / total_spikes_analyzed
# #     p_props = np.array([v['p_prop'] for v in per_neuron.values()])

# #     return {
# #         'sigma': float(sigma_val),
# #         'p_prop_mean': float(np.mean(p_props)) if len(p_props) > 0 else np.nan,
# #         'p_prop_dist': p_props,
# #         'per_neuron': per_neuron,
# #         'n_spikes_analyzed': total_spikes_analyzed
# #     }


# # def sigma_vs_binsize(spike_dict: dict, T_ms: float, warmup_ms: float,
# #                      bin_sizes_ms: list, Ne: int = None) -> dict:
# #     """
# #     Barre tamaños de bin y devuelve sigma_global para cada uno.
# #     Útil para elegir el bin óptimo (buscar plateau de σ).

# #     Parameters
# #     ----------
# #     Ne : si se especifica, solo cuenta neuronas excitatorias en A(t)
# #     """
# #     neuron_mask = set(range(Ne)) if Ne is not None else None
# #     results = {}

# #     for bin_ms in bin_sizes_ms:
# #         A = build_population_activity(spike_dict, bin_ms, T_ms, warmup_ms, neuron_mask)
# #         sig = sigma_global(A)
# #         mean_A = float(np.mean(A))
# #         results[bin_ms] = {
# #             'sigma': sig,
# #             'mean_activity': mean_A,
# #             'mean_rate_hz': mean_A / bin_ms * 1000  # Hz approx
# #         }

# #     return results


# # # ─────────────────────────────────────────────────────────────────────────────
# # # 4. ANÁLISIS COMPLETO DE UNA SIMULACIÓN
# # # ─────────────────────────────────────────────────────────────────────────────

# # def analyze_single_sim(raw_data: dict, cfg: dict,
# #                        A_baseline_mean: float = None) -> dict:
# #     """
# #     Análisis completo de una simulación raw.

# #     Parameters
# #     ----------
# #     raw_data       : {'spike_times', 'spike_indices', 'synapses': {i, j, w}}
# #     cfg            : REANALYSIS_CONFIG
# #     A_baseline_mean: media de A(t) del caso k=0 para el mismo rate_hz
# #                      (None si no hay baseline disponible todavía)

# #     Returns
# #     -------
# #     dict con todas las métricas
# #     """
# #     Ne        = cfg['Ne']
# #     Ni        = cfg['Ni']
# #     warmup    = cfg['warmup_ms']
# #     T_ms      = cfg['T_ms']
# #     dt_bin    = cfg['dt_bin_ms']
# #     window    = cfg['window_ms']
# #     min_spk   = cfg['min_spikes_ancestor']

# #     # ── Extraer arrays ──────────────────────────────────────────────────────
# #     spike_times   = np.asarray(raw_data['spike_times'],   dtype=np.float32)
# #     spike_indices = np.asarray(raw_data['spike_indices'], dtype=np.int32)
# #     syn_i = np.asarray(raw_data['synapses']['i'], dtype=np.int32)
# #     syn_j = np.asarray(raw_data['synapses']['j'], dtype=np.int32)
# #     syn_w = np.asarray(raw_data['synapses']['w'], dtype=np.float32)

# #     # ── Organizar spikes ────────────────────────────────────────────────────
# #     spike_dict = build_spike_dict(spike_times, spike_indices, warmup)

# #     # ── Actividad poblacional (solo excitatorias para A(t)) ─────────────────
# #     exc_set = set(range(Ne))
# #     A = build_population_activity(spike_dict, dt_bin, T_ms, warmup, exc_set)

# #     # ── Firing rate ─────────────────────────────────────────────────────────
# #     T_analysis = T_ms - warmup
# #     n_spikes_exc = sum(
# #         len(t) for nid, t in spike_dict.items() if nid < Ne
# #     )
# #     n_spikes_inh = sum(
# #         len(t) for nid, t in spike_dict.items() if nid >= Ne
# #     )
# #     firing_rate_exc = (n_spikes_exc / Ne / T_analysis) * 1000.0
# #     firing_rate_inh = (n_spikes_inh / Ni / T_analysis) * 1000.0 if Ni > 0 else 0.0
# #     firing_rate_all = ((n_spikes_exc + n_spikes_inh) / (Ne + Ni) / T_analysis) * 1000.0

# #     # ── Sigma global ────────────────────────────────────────────────────────
# #     sig_global = sigma_global(A)

# #     # ── Sigma corregido (si hay baseline) ───────────────────────────────────
# #     if A_baseline_mean is not None:
# #         sig_corr = sigma_corrected_k0(A, A_baseline_mean)
# #     else:
# #         sig_corr = np.nan

# #     # ── Conectividades E→E y E→All ──────────────────────────────────────────
# #     neighbors_EE   = build_neighbors(syn_i, syn_j, syn_w, Ne, Ni, mode='EE')
# #     neighbors_Eall = build_neighbors(syn_i, syn_j, syn_w, Ne, Ni, mode='Eall')

# #     # ── Sigma microscópico ──────────────────────────────────────────────────
# #     micro_EE   = sigma_micro(spike_dict, neighbors_EE,   window, dt_bin, T_ms, warmup, min_spk)
# #     micro_Eall = sigma_micro(spike_dict, neighbors_Eall, window, dt_bin, T_ms, warmup, min_spk)

# #     return {
# #         # Tasas de disparo
# #         'firing_rate_exc': firing_rate_exc,
# #         'firing_rate_inh': firing_rate_inh,
# #         'firing_rate_all': firing_rate_all,
# #         'n_spikes_exc': n_spikes_exc,
# #         'n_spikes_inh': n_spikes_inh,

# #         # Sigma global
# #         'sigma_global': sig_global,
# #         'sigma_global_corr': sig_corr,
# #         'A_mean': float(np.mean(A)),  # guardamos para baseline

# #         # Sigma microscópico E→E
# #         'sigma_micro_EE': micro_EE['sigma'],
# #         'p_prop_EE': micro_EE['p_prop_mean'],
# #         'n_spikes_analyzed_EE': micro_EE['n_spikes_analyzed'],

# #         # Sigma microscópico E→(E+I)
# #         'sigma_micro_Eall': micro_Eall['sigma'],
# #         'p_prop_Eall': micro_Eall['p_prop_mean'],
# #         'n_spikes_analyzed_Eall': micro_Eall['n_spikes_analyzed'],
# #     }


# # # ─────────────────────────────────────────────────────────────────────────────
# # # 5. PIPELINE PRINCIPAL: RE-ANÁLISIS DEL SWEEP COMPLETO
# # # ─────────────────────────────────────────────────────────────────────────────

# # def reanalyze_sweep(sweep_dir: Path, cfg: dict,
# #                     run_bin_sweep: bool = False) -> pd.DataFrame:
# #     """
# #     Re-analiza todos los raw data del sweep 2D.

# #     Estrategia en dos pasadas:
# #         Pasada 1 → procesar k=0 y construir tabla de baselines por rate_hz
# #         Pasada 2 → procesar k>0 usando baseline correspondiente por rate_hz

# #     Parameters
# #     ----------
# #     sweep_dir      : directorio del sweep (contiene results.pkl)
# #     cfg            : REANALYSIS_CONFIG
# #     run_bin_sweep  : si True, ejecuta barrido de bins en muestra representativa

# #     Returns
# #     -------
# #     df : DataFrame con métricas re-analizadas
# #     """
# #     # ── Cargar índice ────────────────────────────────────────────────────────
# #     print(f"\n{'='*60}")
# #     print(f"  Re-análisis de propagación de spikes")
# #     print(f"  Sweep: {sweep_dir}")
# #     print(f"{'='*60}\n")

# #     index_data = load_sweep_index(sweep_dir)
# #     file_paths = index_data['file_paths']   # {(k, rate, trial): filepath}
# #     K_values   = index_data['K_values']
# #     rate_values = index_data['rate_hz_values']
# #     n_trials   = index_data['n_trials']

# #     print(f"  K values   : {len(K_values)} ({min(K_values):.1f} – {max(K_values):.1f})")
# #     print(f"  rate_hz    : {len(rate_values)} ({min(rate_values):.1f} – {max(rate_values):.1f})")
# #     print(f"  Trials     : {n_trials}")
# #     print(f"  Total sims : {len(file_paths)}\n")

# #     # ── Pasada 1: k=0 → construir baselines ─────────────────────────────────
# #     print("── Pasada 1: calculando baselines k=0 ──")
# #     # baseline_table[rate] = {'A_means': list_of_trial_means}
# #     baseline_table = defaultdict(lambda: {'A_means': [], 'firing_rates': []})

# #     k0_keys = [(k, r, t) for (k, r, t) in file_paths if k == 0.0]

# #     for (k, rate, trial) in tqdm(k0_keys, desc='k=0 baseline'):
# #         fp = file_paths[(k, rate, trial)]
# #         raw = load_raw_sim(fp)

# #         spike_dict = build_spike_dict(
# #             np.asarray(raw['spike_times']),
# #             np.asarray(raw['spike_indices']),
# #             cfg['warmup_ms']
# #         )
# #         exc_set = set(range(cfg['Ne']))
# #         A = build_population_activity(
# #             spike_dict, cfg['dt_bin_ms'], cfg['T_ms'], cfg['warmup_ms'], exc_set
# #         )
# #         baseline_table[rate]['A_means'].append(float(np.mean(A)))

# #         # FR baseline
# #         T_analysis = cfg['T_ms'] - cfg['warmup_ms']
# #         n_spk_exc = sum(len(t) for nid, t in spike_dict.items() if nid < cfg['Ne'])
# #         fr_exc = (n_spk_exc / cfg['Ne'] / T_analysis) * 1000.0
# #         baseline_table[rate]['firing_rates'].append(fr_exc)

# #     # Media de baselines por rate_hz (promediada sobre trials)
# #     baseline_A_mean = {
# #         rate: np.mean(vals['A_means'])
# #         for rate, vals in baseline_table.items()
# #     }
# #     baseline_FR_mean = {
# #         rate: np.mean(vals['firing_rates'])
# #         for rate, vals in baseline_table.items()
# #     }
# #     print(f"  Baselines calculados para {len(baseline_A_mean)} valores de rate_hz\n")

# #     # ── Barrido de bins (opcional, sobre muestra) ────────────────────────────
# #     if run_bin_sweep:
# #         print("── Barrido de bin temporal ──")
# #         # Tomar una simulación representativa de k moderado, rate medio
# #         k_sample  = K_values[len(K_values) // 2]
# #         r_sample  = rate_values[len(rate_values) // 2]
# #         t_sample  = 0
# #         key_sample = min(file_paths.keys(),
# #                          key=lambda x: abs(x[0]-k_sample) + abs(x[1]-r_sample))

# #         raw_s = load_raw_sim(file_paths[key_sample])
# #         sd_s  = build_spike_dict(
# #             np.asarray(raw_s['spike_times']),
# #             np.asarray(raw_s['spike_indices']),
# #             cfg['warmup_ms']
# #         )
# #         bin_results = sigma_vs_binsize(
# #             sd_s, cfg['T_ms'], cfg['warmup_ms'],
# #             cfg['bin_sweep'], Ne=cfg['Ne']
# #         )
# #         print(f"  Muestra: k={key_sample[0]:.1f}, rate={key_sample[1]:.1f} Hz")
# #         print(f"  {'bin_ms':>8} {'sigma':>8} {'mean_A':>8} {'FR_exc':>10}")
# #         for bms, bres in sorted(bin_results.items()):
# #             print(f"  {bms:>8.1f} {bres['sigma']:>8.4f} {bres['mean_activity']:>8.2f} "
# #                   f"{bres['mean_rate_hz']:>10.2f}")
# #         print()

# #     # ── Pasada 2: todos los k → análisis completo ───────────────────────────
# #     print("── Pasada 2: análisis completo ──")
# #     all_results = []

# #     for (k, rate, trial), fp in tqdm(file_paths.items(), desc='Simulaciones'):
# #         try:
# #             raw = load_raw_sim(fp)
# #         except Exception as e:
# #             print(f"  [ERROR] k={k}, rate={rate}, trial={trial}: {e}")
# #             continue

# #         # Baseline para este rate_hz
# #         baseline_a = baseline_A_mean.get(rate, None)

# #         # Análisis completo
# #         metrics = analyze_single_sim(raw, cfg, baseline_a)

# #         all_results.append({
# #             'k': k,
# #             'rate_hz': rate,
# #             'trial': trial,
# #             **metrics
# #         })

# #     df = pd.DataFrame(all_results)
# #     df = df.sort_values(['k', 'rate_hz', 'trial']).reset_index(drop=True)

# #     print(f"\n  Simulaciones procesadas: {len(df)}")
# #     print(f"  Columnas: {list(df.columns)}\n")
# #     return df


# # # ─────────────────────────────────────────────────────────────────────────────
# # # 6. AGREGACIÓN Y CORRECCIÓN DELTA P
# # # ─────────────────────────────────────────────────────────────────────────────

# # def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
# #     """
# #     Agrega resultados sobre trials: media ± std para cada (k, rate_hz).
# #     También calcula ΔP = P_coupled - P_baseline  (misma rate_hz, k=0)
# #     """
# #     metrics = [
# #         'firing_rate_exc', 'firing_rate_inh', 'firing_rate_all',
# #         'sigma_global', 'sigma_global_corr',
# #         'sigma_micro_EE', 'sigma_micro_Eall',
# #         'p_prop_EE', 'p_prop_Eall',
# #         'A_mean'
# #     ]

# #     agg_dict = {m: ['mean', 'std'] for m in metrics if m in df.columns}
# #     df_agg = df.groupby(['k', 'rate_hz']).agg(agg_dict)
# #     df_agg.columns = ['_'.join(c) for c in df_agg.columns]
# #     df_agg = df_agg.reset_index()

# #     # ── ΔP: diferencia con baseline k=0 ─────────────────────────────────────
# #     k0_agg = df_agg[df_agg['k'] == 0.0].set_index('rate_hz')

# #     for metric in ['p_prop_EE', 'p_prop_Eall', 'sigma_micro_EE', 'sigma_micro_Eall']:
# #         col = f'{metric}_mean'
# #         delta_col = f'delta_{metric}'
# #         if col in df_agg.columns:
# #             df_agg[delta_col] = df_agg.apply(
# #                 lambda row: row[col] - k0_agg.loc[row['rate_hz'], col]
# #                 if row['rate_hz'] in k0_agg.index else np.nan,
# #                 axis=1
# #             )

# #     return df_agg


# # # ─────────────────────────────────────────────────────────────────────────────
# # # 7. GUARDADO
# # # ─────────────────────────────────────────────────────────────────────────────

# # def save_results(df_raw: pd.DataFrame, df_agg: pd.DataFrame,
# #                  sweep_dir: Path, cfg: dict):
# #     """Guarda resultados del re-análisis."""
# #     out = {
# #         'df_raw': df_raw,
# #         'df_aggregated': df_agg,
# #         'config': cfg
# #     }
# #     outpath = sweep_dir / 'reanalysis_propagation.pkl'
# #     with open(outpath, 'wb') as f:
# #         pickle.dump(out, f, protocol=4)

# #     # También CSV para inspección rápida
# #     df_agg.to_csv(sweep_dir / 'reanalysis_aggregated.csv', index=False)
# #     print(f"  Resultados guardados en:\n    {outpath}")
# #     print(f"  CSV: {sweep_dir / 'reanalysis_aggregated.csv'}\n")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # 8. VISUALIZACIÓN BÁSICA
# # # ─────────────────────────────────────────────────────────────────────────────

# # def plot_sigma_comparison(df_agg: pd.DataFrame, sweep_dir: Path = None):
# #     """
# #     Compara sigma_global, sigma_micro_EE y sigma_corrected vs K
# #     para distintos valores de rate_hz.
# #     """
# #     import matplotlib.pyplot as plt

# #     rate_sample = sorted(df_agg['rate_hz'].unique())
# #     rate_sample = rate_sample[::max(1, len(rate_sample)//5)]  # 5 curvas max

# #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# #     metrics_labels = [
# #         ('sigma_global_mean',    'σ global (Beggs-Plenz)'),
# #         ('sigma_global_corr_mean', 'σ global corregido (k=0)'),
# #         ('sigma_micro_EE_mean',  'σ micro E→E'),
# #     ]

# #     K_vals = sorted(df_agg['k'].unique())
# #     colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rate_sample)))

# #     for ax, (col, label) in zip(axes, metrics_labels):
# #         if col not in df_agg.columns:
# #             ax.text(0.5, 0.5, f'{col}\nno disponible',
# #                     ha='center', va='center', transform=ax.transAxes)
# #             continue
# #         for rate, color in zip(rate_sample, colors):
# #             sub = df_agg[df_agg['rate_hz'] == rate].sort_values('k')
# #             ax.plot(sub['k'], sub[col], 'o-', color=color,
# #                     label=f'{rate:.0f} Hz', markersize=5, linewidth=1.5)

# #         ax.axhline(1.0, color='red', ls='--', lw=1.2, label='σ=1 (crítico)')
# #         ax.set_xlabel('K (acoplamiento)', fontsize=12)
# #         ax.set_ylabel(label, fontsize=11)
# #         ax.set_title(label, fontsize=12)
# #         ax.legend(fontsize=8, loc='upper left')
# #         ax.grid(True, alpha=0.3)

# #     plt.suptitle('Comparación de estimadores de branching ratio', fontsize=14, y=1.01)
# #     plt.tight_layout()

# #     if sweep_dir:
# #         outpath = sweep_dir / 'reanalysis_sigma_comparison.png'
# #         plt.savefig(outpath, dpi=200, bbox_inches='tight')
# #         print(f"  Figura: {outpath}")
# #     plt.show()


# # def plot_delta_p(df_agg: pd.DataFrame, sweep_dir: Path = None):
# #     """Heatmap de ΔP_EE = P_coupled - P_baseline(k=0)."""
# #     import matplotlib.pyplot as plt
# #     from matplotlib import colors as mcolors

# #     if 'delta_p_prop_EE' not in df_agg.columns:
# #         print("  ΔP no disponible — ejecuta aggregate_results primero")
# #         return

# #     K_vals   = sorted(df_agg['k'].unique())
# #     rate_vals = sorted(df_agg['rate_hz'].unique())

# #     # Pivot
# #     pivot = df_agg.pivot(index='k', columns='rate_hz', values='delta_p_prop_EE')

# #     fig, ax = plt.subplots(figsize=(10, 7))
# #     vmax = np.nanpercentile(np.abs(pivot.values), 95)
# #     im = ax.imshow(pivot.values, aspect='auto', origin='lower',
# #                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
# #     ax.set_xticks(range(len(rate_vals)))
# #     ax.set_xticklabels([f'{r:.0f}' for r in rate_vals], rotation=45)
# #     ax.set_yticks(range(len(K_vals)))
# #     ax.set_yticklabels([f'{k:.1f}' for k in K_vals])
# #     ax.set_xlabel('rate_hz (Hz)', fontsize=12)
# #     ax.set_ylabel('K (acoplamiento)', fontsize=12)
# #     ax.set_title('ΔP_EE = P(k>0) − P_baseline(k=0)\n(Contribución neta de la red)', fontsize=12)
# #     plt.colorbar(im, ax=ax, label='ΔP')
# #     plt.tight_layout()

# #     if sweep_dir:
# #         outpath = sweep_dir / 'reanalysis_delta_p_heatmap.png'
# #         plt.savefig(outpath, dpi=200, bbox_inches='tight')
# #         print(f"  Figura: {outpath}")
# #     plt.show()


# """
# spike_propagation_reanalysis.py  — v2 (final)
# ===============================================
# Re-análisis de propagación de spikes sobre raw data del barrido 2D existente.

# Cambios respecto a v1:
#     - sigma_micro: resolución continua con searchsorted (sin binning)
#     - sigma_micro: causal_weighting opcional (V7-style, 1/n_parents)
#     - sigma_micro: baseline_correction='poisson' opcional  → P_net = (P_obs-P_base)/(1-P_base)
#     - sigma_corrected_k0: fórmula correcta según branching process con drive externo
#                           → (A(t+1) - h) / A(t)  [solo numerador corregido]

# Métricas calculadas:
#     1. sigma_global        : <A(t+1)/A(t)>             — Beggs & Plenz
#     2. sigma_global_corr   : <(A(t+1)-h) / A(t)>      — Wilting & Priesemann
#     3. sigma_micro_EE      : tracking vecinos E→E, tiempo continuo
#     4. sigma_micro_Eall    : tracking vecinos E→(E+I), tiempo continuo
#     5. p_prop_*            : P(vecino específico activado | spike ancestro)

# Modos de sigma_micro:
#     causal_weighting=False, baseline_correction=None    → conteo bruto (default)
#     causal_weighting=True,  baseline_correction=None    → ponderado por crédito causal
#     causal_weighting=False, baseline_correction='poisson' → corrección Poisson por spike
#     causal_weighting=True,  baseline_correction='poisson' → combinado

# Referencias:
#     - Beggs & Plenz (2003) J. Neurosci. 23:11167
#     - Wilting & Priesemann (2018) Nat. Commun. 9:2325
#     - Zierenberg et al. (2020) Phys. Rev. E 102:040301
# """

# import numpy as np
# import pandas as pd
# import pickle
# import gzip
# import multiprocessing as mp
# from pathlib import Path
# from collections import defaultdict
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. CARGA DE DATOS
# # ─────────────────────────────────────────────────────────────────────────────

# def load_sweep_index(sweep_dir: Path) -> dict:
#     with open(sweep_dir / 'results.pkl', 'rb') as f:
#         return pickle.load(f)


# def load_raw_sim(filepath: str) -> dict:
#     with gzip.open(filepath, 'rb') as f:
#         return pickle.load(f)


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. CONSTRUCCIÓN DE ESTRUCTURAS INTERNAS
# # ─────────────────────────────────────────────────────────────────────────────

# def build_neighbors(syn_i, syn_j, syn_w, Ne, Ni,
#                     mode='EE', min_weight=0.0) -> dict:
#     """
#     Construye diccionario de vecinos desde arrays de sinapsis.

#     mode : 'EE'   → ancestros E, descendientes E
#            'Eall' → ancestros E, descendientes E+I
#            'all'  → ancestros E+I, descendientes E+I
#     """
#     syn_i = np.asarray(syn_i, dtype=np.int32)
#     syn_j = np.asarray(syn_j, dtype=np.int32)
#     syn_w = np.asarray(syn_w, dtype=np.float32)

#     anc_mask  = (syn_i < Ne) if mode in ('EE', 'Eall') else np.ones(len(syn_i), bool)
#     desc_mask = (syn_j < Ne) if mode == 'EE' else np.ones(len(syn_j), bool)
#     final     = anc_mask & desc_mask & (syn_w >= min_weight)

#     nb = defaultdict(list)
#     for pre, post in zip(syn_i[final], syn_j[final]):
#         nb[int(pre)].append(int(post))

#     return {k: np.array(v, dtype=np.int32) for k, v in nb.items()}


# def build_spike_dict(spike_times: np.ndarray, spike_indices: np.ndarray,
#                      warmup_ms: float) -> dict:
#     """
#     Organiza spikes por neurona filtrando warmup.
#     Devuelve {neuron_id: sorted_spike_times_ms}.
#     """
#     mask = spike_times >= warmup_ms
#     t   = spike_times[mask].astype(np.float64)   # float64 para searchsorted preciso
#     idx = spike_indices[mask].astype(np.int32)

#     sd = defaultdict(list)
#     for time, nid in zip(t, idx):
#         sd[int(nid)].append(time)

#     return {k: np.sort(v) for k, v in sd.items()}


# def build_population_activity(spike_dict: dict, dt_bin_ms: float,
#                                T_ms: float, warmup_ms: float,
#                                neuron_mask=None) -> np.ndarray:
#     """Serie temporal de actividad poblacional A(t) en bins de dt_bin_ms."""
#     T_analysis = T_ms - warmup_ms
#     n_bins = int(T_analysis / dt_bin_ms)
#     A = np.zeros(n_bins, dtype=np.float64)

#     for nid, times in spike_dict.items():
#         if neuron_mask is not None and nid not in neuron_mask:
#             continue
#         shifted  = times - warmup_ms
#         bins_idx = (shifted / dt_bin_ms).astype(np.int32)
#         valid    = (bins_idx >= 0) & (bins_idx < n_bins)
#         np.add.at(A, bins_idx[valid], 1)

#     return A


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. ESTIMADORES DE SIGMA
# # ─────────────────────────────────────────────────────────────────────────────

# def sigma_global(A: np.ndarray, min_active: int = 1) -> float:
#     """
#     Estimador global Beggs-Plenz: <A(t+1)/A(t)>.
#     Excluye bins con A(t) < min_active para evitar 0/0.
#     """
#     mask = A[:-1] >= min_active
#     if not mask.any():
#         return np.nan
#     return float(np.mean(A[1:][mask] / A[:-1][mask]))


# def sigma_corrected_k0(A_coupled: np.ndarray, A_baseline_mean: float,
#                         min_active: int = 1) -> float:
#     """
#     Sigma global corregido — fórmula correcta del branching process con drive:

#         E[A(t+1)] = m·A(t) + h   →   m̂ = <(A(t+1) - h) / A(t)>

#     Solo se resta el ruido en el numerador (Wilting & Priesemann 2018).
#     El denominador A(t) permanece completo porque todos los spikes en t,
#     independientemente de su origen, generan despolarización real en t+1.
#     """
#     mask = A_coupled[:-1] >= min_active
#     if not mask.any():
#         return np.nan
#     num = A_coupled[1:][mask] - A_baseline_mean
#     den = A_coupled[:-1][mask]
#     return float(np.mean(num / den))


# def sigma_micro(spike_dict: dict, neighbors: dict,
#                 window_ms: float, T_ms: float, warmup_ms: float,
#                 min_spikes: int = 3,
#                 causal_weighting: bool = False,
#                 baseline_correction: str = None,
#                 p_base: float = 0.0) -> dict:
#     """
#     Branching ratio microscópico en tiempo continuo (searchsorted).

#     Para cada spike de neurona ancestro i en t_spike:
#         Para cada vecino post j ∈ N(i):
#             Busca el primer spike de j en (t_spike, t_spike + window_ms]
#             Si existe → cuenta como activación (ponderada si causal_weighting)

#     Parameters
#     ----------
#     spike_dict          : {nid: sorted_spike_times_ms}  (ya filtrado por warmup)
#     neighbors           : {pre_id: array(post_ids)}
#     window_ms           : ventana de propagación monosináptica (ms)
#     T_ms, warmup_ms     : para calcular n_spikes válidos
#     min_spikes          : mínimo de spikes para incluir neurona ancestro
#     causal_weighting    : si True, pondera por 1/n_padres que dispararon (V7-style)
#     baseline_correction : None | 'poisson'
#                           'poisson' → P_net = max(0, (P_obs - p_base)/(1 - p_base))
#     p_base              : probabilidad Poisson base (requerida si baseline_correction='poisson')
#                           p_base = 1 - exp(-FR_baseline_hz * window_s)

#     Returns
#     -------
#     dict con sigma, p_prop_mean, p_prop_dist, per_neuron, n_spikes_analyzed
#     """
#     # Grafo inverso para causal_weighting: post → {pre conectados}
#     if causal_weighting:
#         rev_neighbors = defaultdict(list)
#         for pre_id_, post_ids_ in neighbors.items():
#             for post_id_ in post_ids_:
#                 rev_neighbors[int(post_id_)].append(int(pre_id_))
#         rev_neighbors = {k: np.array(v, dtype=np.int32) for k, v in rev_neighbors.items()}

#     total_descendants    = 0.0
#     total_spikes_analyzed = 0
#     per_neuron = {}

#     for pre_id, post_ids in neighbors.items():
#         if pre_id not in spike_dict:
#             continue

#         pre_spikes = spike_dict[pre_id]   # sorted float64, ya sin warmup
#         if len(pre_spikes) < min_spikes:
#             continue

#         n_neighbors    = len(post_ids)
#         neuron_desc_sum = 0.0
#         neuron_spikes  = 0

#         for t_spike in pre_spikes:
#             t_max     = t_spike + window_ms
#             desc_count = 0.0

#             for post_id in post_ids:
#                 if post_id not in spike_dict:
#                     continue
#                 post_spikes = spike_dict[post_id]

#                 # Primer spike de post estrictamente después de t_spike
#                 idx = np.searchsorted(post_spikes, t_spike, side='right')
#                 if idx >= len(post_spikes) or post_spikes[idx] > t_max:
#                     continue           # sin activación en la ventana

#                 t_post = post_spikes[idx]

#                 if causal_weighting:
#                     # Cuántos padres conectados de post_id dispararon
#                     # en la misma ventana hacia atrás (t_post - window_ms, t_post]
#                     parents = rev_neighbors.get(post_id, set())
#                     n_parents_fired = 0
#                     for parent_id in parents:
#                         if parent_id not in spike_dict:
#                             continue
#                         p_spikes = spike_dict[parent_id]
#                         i0 = np.searchsorted(p_spikes, t_post - window_ms, side='right')
#                         i1 = np.searchsorted(p_spikes, t_post,             side='right')
#                         if i1 > i0:
#                             n_parents_fired += 1
#                     weight = 1.0 / n_parents_fired if n_parents_fired > 0 else 1.0
#                     desc_count += weight
#                 else:
#                     desc_count += 1.0

#             # Corrección Poisson por spike (nivel microscópico)
#             if baseline_correction == 'poisson' and p_base < 1.0:
#                 P_obs  = desc_count / n_neighbors
#                 P_net  = max(0.0, (P_obs - p_base) / (1.0 - p_base))
#                 desc_count = P_net * n_neighbors

#             total_descendants     += desc_count
#             total_spikes_analyzed += 1
#             neuron_desc_sum       += desc_count
#             neuron_spikes         += 1

#         if neuron_spikes == 0:
#             continue

#         mean_desc = neuron_desc_sum / neuron_spikes
#         p_prop    = mean_desc / n_neighbors

#         per_neuron[pre_id] = {
#             'sigma':       mean_desc,
#             'p_prop':      p_prop,
#             'n_spikes':    neuron_spikes,
#             'n_neighbors': n_neighbors,
#         }

#     if total_spikes_analyzed == 0:
#         return {
#             'sigma': np.nan, 'p_prop_mean': np.nan,
#             'p_prop_dist': np.array([]), 'per_neuron': {},
#             'n_spikes_analyzed': 0,
#         }

#     sigma_val = total_descendants / total_spikes_analyzed
#     p_props   = np.array([v['p_prop'] for v in per_neuron.values()])

#     return {
#         'sigma':            float(sigma_val),
#         'p_prop_mean':      float(np.mean(p_props)) if len(p_props) > 0 else np.nan,
#         'p_prop_dist':      p_props,
#         'per_neuron':       per_neuron,
#         'n_spikes_analyzed': total_spikes_analyzed,
#     }


# def sigma_vs_binsize(spike_dict: dict, T_ms: float, warmup_ms: float,
#                      bin_sizes_ms: list, Ne: int = None) -> dict:
#     """
#     Barre tamaños de bin sobre sigma_global.
#     Útil para elegir el bin óptimo (buscar plateau de σ).
#     """
#     neuron_mask = set(range(Ne)) if Ne is not None else None
#     results = {}
#     for bin_ms in bin_sizes_ms:
#         A   = build_population_activity(spike_dict, bin_ms, T_ms, warmup_ms, neuron_mask)
#         sig = sigma_global(A)
#         results[bin_ms] = {
#             'sigma':         sig,
#             'mean_activity': float(np.mean(A)),
#             'mean_rate_hz':  float(np.mean(A)) / bin_ms * 1000,
#         }
#     return results


# # ─────────────────────────────────────────────────────────────────────────────
# # 4. ANÁLISIS COMPLETO DE UNA SIMULACIÓN
# # ─────────────────────────────────────────────────────────────────────────────

# def analyze_single_sim(raw_data: dict, cfg: dict,
#                        A_baseline_mean:  float = None,
#                        baseline_fr_exc:  float = None,
#                        causal_weighting: bool  = False,
#                        baseline_correction: str = None) -> dict:
#     """
#     Análisis completo de una simulación raw.

#     Parameters
#     ----------
#     raw_data           : {'spike_times', 'spike_indices', 'synapses': {i, j, w}}
#     cfg                : REANALYSIS_CONFIG
#     A_baseline_mean    : media de A(t) del caso k=0 para el mismo rate_hz
#     baseline_fr_exc    : FR excitatorio medio (Hz) del caso k=0 para el mismo rate_hz
#                          (usado para calcular p_base en corrección Poisson)
#     causal_weighting   : V7-style 1/n_parents weighting
#     baseline_correction: None | 'poisson'
#     """
#     Ne      = cfg['Ne']
#     Ni      = cfg['Ni']
#     warmup  = cfg['warmup_ms']
#     T_ms    = cfg['T_ms']
#     dt_bin  = cfg['dt_bin_ms']
#     window  = cfg['window_ms']
#     min_spk = cfg['min_spikes_ancestor']

#     spike_times   = np.asarray(raw_data['spike_times'],   dtype=np.float64)
#     spike_indices = np.asarray(raw_data['spike_indices'], dtype=np.int32)
#     syn_i = np.asarray(raw_data['synapses']['i'], dtype=np.int32)
#     syn_j = np.asarray(raw_data['synapses']['j'], dtype=np.int32)
#     syn_w = np.asarray(raw_data['synapses']['w'], dtype=np.float32)

#     spike_dict = build_spike_dict(spike_times, spike_indices, warmup)

#     # ── Actividad poblacional E (para sigma_global) ──────────────────────────
#     exc_set = set(range(Ne))
#     A = build_population_activity(spike_dict, dt_bin, T_ms, warmup, exc_set)

#     # ── Firing rates ─────────────────────────────────────────────────────────
#     T_analysis   = T_ms - warmup
#     n_spk_exc    = sum(len(t) for nid, t in spike_dict.items() if nid < Ne)
#     n_spk_inh    = sum(len(t) for nid, t in spike_dict.items() if nid >= Ne)
#     fr_exc = (n_spk_exc / Ne / T_analysis) * 1000.0
#     fr_inh = (n_spk_inh / Ni / T_analysis) * 1000.0 if Ni > 0 else 0.0
#     fr_all = ((n_spk_exc + n_spk_inh) / (Ne + Ni) / T_analysis) * 1000.0

#     # ── Sigma global ─────────────────────────────────────────────────────────
#     sig_global_val = sigma_global(A)

#     # ── Sigma corregido (fórmula Wilting & Priesemann) ───────────────────────
#     sig_corr = (sigma_corrected_k0(A, A_baseline_mean)
#                 if A_baseline_mean is not None else np.nan)

#     # ── p_base para corrección Poisson ───────────────────────────────────────
#     if baseline_correction == 'poisson' and baseline_fr_exc is not None:
#         window_s = window / 1000.0
#         p_base = float(1.0 - np.exp(-baseline_fr_exc * window_s))
#     else:
#         p_base = 0.0

#     # ── Conectividades ───────────────────────────────────────────────────────
#     nb_EE   = build_neighbors(syn_i, syn_j, syn_w, Ne, Ni, mode='EE')
#     nb_Eall = build_neighbors(syn_i, syn_j, syn_w, Ne, Ni, mode='Eall')

#     # ── Sigma microscópico ───────────────────────────────────────────────────
#     micro_EE = sigma_micro(
#         spike_dict, nb_EE, window, T_ms, warmup, min_spk,
#         causal_weighting=causal_weighting,
#         baseline_correction=baseline_correction,
#         p_base=p_base,
#     )
#     micro_Eall = sigma_micro(
#         spike_dict, nb_Eall, window, T_ms, warmup, min_spk,
#         causal_weighting=causal_weighting,
#         baseline_correction=baseline_correction,
#         p_base=p_base,
#     )

#     return {
#         'firing_rate_exc':        fr_exc,
#         'firing_rate_inh':        fr_inh,
#         'firing_rate_all':        fr_all,
#         'n_spikes_exc':           n_spk_exc,
#         'n_spikes_inh':           n_spk_inh,
#         'sigma_global':           sig_global_val,
#         'sigma_global_corr':      sig_corr,
#         'A_mean':                 float(np.mean(A)),
#         'sigma_micro_EE':         micro_EE['sigma'],
#         'p_prop_EE':              micro_EE['p_prop_mean'],
#         'n_spikes_analyzed_EE':   micro_EE['n_spikes_analyzed'],
#         'sigma_micro_Eall':       micro_Eall['sigma'],
#         'p_prop_Eall':            micro_Eall['p_prop_mean'],
#         'n_spikes_analyzed_Eall': micro_Eall['n_spikes_analyzed'],
#         'p_base':                 p_base,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # WORKER MULTIPROCESSING (función global, pickleable en fork y spawn)
# # ─────────────────────────────────────────────────────────────────────────────

# def _worker_analyze_sim(args):
#     (k, rate, trial), fp, cfg, baseline_a, baseline_fr, cw, bc = args
#     try:
#         raw     = load_raw_sim(fp)
#         metrics = analyze_single_sim(raw, cfg,
#                                      A_baseline_mean=baseline_a,
#                                      baseline_fr_exc=baseline_fr,
#                                      causal_weighting=cw,
#                                      baseline_correction=bc)
#         return {'k': k, 'rate_hz': rate, 'trial': trial, **metrics}
#     except Exception as e:
#         print(f"  [ERROR] k={k:.2f}, rate={rate:.1f}, trial={trial}: {e}")
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. PIPELINE PRINCIPAL
# # ─────────────────────────────────────────────────────────────────────────────

# def reanalyze_sweep(sweep_dir: Path, cfg: dict,
#                     run_bin_sweep: bool    = False,
#                     causal_weighting: bool = False,
#                     baseline_correction: str = None,
#                     n_processes: int = None) -> pd.DataFrame:
#     """
#     Re-analiza todos los raw data del sweep 2D en dos pasadas.

#     Pasada 1 → k=0: construye tabla de baselines (A_mean, FR_exc) por rate_hz
#     Pasada 2 → todos los k: análisis completo en paralelo

#     Parameters
#     ----------
#     causal_weighting    : V7-style 1/n_parents weighting en sigma_micro
#     baseline_correction : None | 'poisson'  (corrección probabilística Poisson)
#     n_processes         : número de procesos (None = cpu_count - 1, max 31)
#     """
#     print(f"\n{'='*60}")
#     print(f"  Re-análisis de propagación de spikes  (v2)")
#     print(f"  causal_weighting   : {causal_weighting}")
#     print(f"  baseline_correction: {baseline_correction}")
#     print(f"  Sweep: {sweep_dir}")
#     print(f"{'='*60}\n")

#     index_data  = load_sweep_index(sweep_dir)
#     file_paths  = index_data['file_paths']
#     K_values    = index_data['K_values']
#     rate_values = index_data['rate_hz_values']
#     n_trials    = index_data['n_trials']

#     print(f"  K values   : {len(K_values)}  ({min(K_values):.1f} – {max(K_values):.1f})")
#     print(f"  rate_hz    : {len(rate_values)} ({min(rate_values):.1f} – {max(rate_values):.1f})")
#     print(f"  Trials     : {n_trials}")
#     print(f"  Total sims : {len(file_paths)}\n")

#     # ── Pasada 1: baselines k=0 ──────────────────────────────────────────────
#     print("── Pasada 1: calculando baselines k=0 ──")
#     baseline_table = defaultdict(lambda: {'A_means': [], 'fr_exc': []})
#     k0_keys = [(k, r, t) for (k, r, t) in file_paths if k == 0.0]

#     for (k, rate, trial) in tqdm(k0_keys, desc='k=0 baseline'):
#         raw = load_raw_sim(file_paths[(k, rate, trial)])
#         sd  = build_spike_dict(
#             np.asarray(raw['spike_times']),
#             np.asarray(raw['spike_indices']),
#             cfg['warmup_ms'],
#         )
#         exc_set    = set(range(cfg['Ne']))
#         A          = build_population_activity(sd, cfg['dt_bin_ms'],
#                                                cfg['T_ms'], cfg['warmup_ms'], exc_set)
#         T_analysis = cfg['T_ms'] - cfg['warmup_ms']
#         n_spk_exc  = sum(len(t) for nid, t in sd.items() if nid < cfg['Ne'])
#         fr_exc     = (n_spk_exc / cfg['Ne'] / T_analysis) * 1000.0

#         baseline_table[rate]['A_means'].append(float(np.mean(A)))
#         baseline_table[rate]['fr_exc'].append(fr_exc)

#     baseline_A_mean  = {r: np.mean(v['A_means']) for r, v in baseline_table.items()}
#     baseline_FR_mean = {r: np.mean(v['fr_exc'])  for r, v in baseline_table.items()}
#     print(f"  Baselines listos para {len(baseline_A_mean)} valores de rate_hz\n")

#     # ── Barrido de bins (opcional) ───────────────────────────────────────────
#     if run_bin_sweep:
#         print("── Barrido de bin temporal ──")
#         k_mid  = K_values[len(K_values) // 2]
#         r_mid  = rate_values[len(rate_values) // 2]
#         key_s  = min(file_paths.keys(),
#                      key=lambda x: abs(x[0] - k_mid) + abs(x[1] - r_mid))
#         raw_s  = load_raw_sim(file_paths[key_s])
#         sd_s   = build_spike_dict(
#             np.asarray(raw_s['spike_times']),
#             np.asarray(raw_s['spike_indices']),
#             cfg['warmup_ms'],
#         )
#         bin_res = sigma_vs_binsize(sd_s, cfg['T_ms'], cfg['warmup_ms'],
#                                    cfg['bin_sweep'], Ne=cfg['Ne'])
#         print(f"  Muestra: k={key_s[0]:.1f}, rate={key_s[1]:.1f} Hz")
#         print(f"  {'bin_ms':>8} {'sigma':>8} {'mean_A':>8} {'FR_exc':>10}")
#         for bms, bres in sorted(bin_res.items()):
#             print(f"  {bms:>8.1f} {bres['sigma']:>8.4f} "
#                   f"{bres['mean_activity']:>8.2f} {bres['mean_rate_hz']:>10.2f}")
#         print()

#     # ── Pasada 2: análisis completo en paralelo ──────────────────────────────
#     print("── Pasada 2: análisis completo (multiprocessing) ──")
#     if n_processes is None:
#         n_processes = min(mp.cpu_count() - 1, 31)

#     tasks = [
#         ((k, rate, trial), fp, cfg,
#          baseline_A_mean.get(rate),
#          baseline_FR_mean.get(rate),
#          causal_weighting,
#          baseline_correction)
#         for (k, rate, trial), fp in file_paths.items()
#     ]

#     all_results = []
#     with mp.Pool(processes=n_processes) as pool:
#         for result in tqdm(
#             pool.imap_unordered(_worker_analyze_sim, tasks, chunksize=4),
#             total=len(tasks), desc='Simulaciones',
#         ):
#             if result is not None:
#                 all_results.append(result)

#     df = pd.DataFrame(all_results)
#     df = df.sort_values(['k', 'rate_hz', 'trial']).reset_index(drop=True)
#     print(f"\n  Simulaciones procesadas: {len(df)}")
#     print(f"  Columnas: {list(df.columns)}\n")
#     return df


# # ─────────────────────────────────────────────────────────────────────────────
# # 6. AGREGACIÓN Y ΔP
# # ─────────────────────────────────────────────────────────────────────────────

# def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Agrega sobre trials: media ± std para cada (k, rate_hz).
#     Calcula ΔP = métrica(k>0) - métrica(k=0, mismo rate_hz).
#     """
#     metrics = [
#         'firing_rate_exc', 'firing_rate_inh', 'firing_rate_all',
#         'sigma_global', 'sigma_global_corr',
#         'sigma_micro_EE', 'sigma_micro_Eall',
#         'p_prop_EE', 'p_prop_Eall', 'A_mean', 'p_base',
#     ]
#     agg_dict = {m: ['mean', 'std'] for m in metrics if m in df.columns}
#     df_agg   = df.groupby(['k', 'rate_hz']).agg(agg_dict)
#     df_agg.columns = ['_'.join(c) for c in df_agg.columns]
#     df_agg   = df_agg.reset_index()

#     k0 = df_agg[df_agg['k'] == 0.0].set_index('rate_hz')
#     for metric in ['p_prop_EE', 'p_prop_Eall', 'sigma_micro_EE', 'sigma_micro_Eall']:
#         col = f'{metric}_mean'
#         if col in df_agg.columns:
#             df_agg[f'delta_{metric}'] = df_agg.apply(
#                 lambda row: row[col] - k0.loc[row['rate_hz'], col]
#                 if row['rate_hz'] in k0.index else np.nan,
#                 axis=1,
#             )
#     return df_agg


# # ─────────────────────────────────────────────────────────────────────────────
# # 7. GUARDADO
# # ─────────────────────────────────────────────────────────────────────────────

# def save_results(df_raw: pd.DataFrame, df_agg: pd.DataFrame,
#                  sweep_dir: Path, cfg: dict, tag: str = ''):
#     """Guarda resultados del re-análisis. tag permite versionar runs."""
#     suffix = f'_{tag}' if tag else ''
#     out = {'df_raw': df_raw, 'df_aggregated': df_agg, 'config': cfg}
#     pkl_path = sweep_dir / f'reanalysis_propagation{suffix}.pkl'
#     csv_path = sweep_dir / f'reanalysis_aggregated{suffix}.csv'
#     with open(pkl_path, 'wb') as f:
#         pickle.dump(out, f, protocol=4)
#     df_agg.to_csv(csv_path, index=False)
#     print(f"  Guardado: {pkl_path}")
#     print(f"  CSV:      {csv_path}\n")


# # ─────────────────────────────────────────────────────────────────────────────
# # 8. VISUALIZACIÓN
# # ─────────────────────────────────────────────────────────────────────────────

# def plot_sigma_comparison(df_agg: pd.DataFrame, sweep_dir: Path = None):
#     """Compara estimadores de sigma vs K para distintos rate_hz."""
#     import matplotlib.pyplot as plt

#     rates  = sorted(df_agg['rate_hz'].unique())
#     rates  = rates[::max(1, len(rates) // 5)]
#     colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rates)))

#     cols_labels = [
#         ('sigma_global_mean',       'σ global (Beggs-Plenz)'),
#         ('sigma_global_corr_mean',  'σ global corr. (W&P 2018)'),
#         ('sigma_micro_EE_mean',     'σ micro E→E (continuo)'),
#     ]
#     fig, axes = plt.subplots(1, len(cols_labels), figsize=(6 * len(cols_labels), 5))

#     for ax, (col, label) in zip(axes, cols_labels):
#         if col not in df_agg.columns:
#             ax.text(0.5, 0.5, f'{col}\nno disponible',
#                     ha='center', va='center', transform=ax.transAxes)
#             continue
#         for rate, color in zip(rates, colors):
#             sub = df_agg[df_agg['rate_hz'] == rate].sort_values('k')
#             ax.plot(sub['k'], sub[col], 'o-', color=color,
#                     label=f'{rate:.0f} Hz', markersize=4, lw=1.5)
#         ax.axhline(1.0, color='red', ls='--', lw=1, label='σ=1 (crítico)')
#         ax.set_xlabel('K (acoplamiento)', fontsize=11)
#         ax.set_ylabel(label, fontsize=10)
#         ax.set_title(label, fontsize=11)
#         ax.legend(fontsize=7, loc='upper left')
#         ax.grid(True, alpha=0.3)

#     plt.suptitle('Comparación de estimadores — branching ratio', fontsize=13, y=1.01)
#     plt.tight_layout()
#     if sweep_dir:
#         p = sweep_dir / 'reanalysis_sigma_comparison.png'
#         plt.savefig(p, dpi=200, bbox_inches='tight')
#         print(f"  Figura: {p}")
#     plt.show()


# def plot_delta_p(df_agg: pd.DataFrame, sweep_dir: Path = None):
#     """Heatmap ΔP_EE = P(k>0) − P_baseline(k=0)."""
#     import matplotlib.pyplot as plt

#     if 'delta_p_prop_EE' not in df_agg.columns:
#         print("  delta_p_prop_EE no disponible")
#         return

#     K_vals    = sorted(df_agg['k'].unique())
#     rate_vals = sorted(df_agg['rate_hz'].unique())
#     pivot     = df_agg.pivot(index='k', columns='rate_hz', values='delta_p_prop_EE')

#     fig, ax = plt.subplots(figsize=(10, 7))
#     vmax = np.nanpercentile(np.abs(pivot.values), 95)
#     im   = ax.imshow(pivot.values, aspect='auto', origin='lower',
#                      cmap='RdBu_r', vmin=-vmax, vmax=vmax)
#     ax.set_xticks(range(len(rate_vals)))
#     ax.set_xticklabels([f'{r:.0f}' for r in rate_vals], rotation=45)
#     ax.set_yticks(range(len(K_vals)))
#     ax.set_yticklabels([f'{k:.1f}' for k in K_vals])
#     ax.set_xlabel('rate_hz (Hz)', fontsize=12)
#     ax.set_ylabel('K (acoplamiento)', fontsize=12)
#     ax.set_title('ΔP_EE = P(k>0) − P_baseline(k=0)\n'
#                  '(Contribución neta de la red)', fontsize=12)
#     plt.colorbar(im, ax=ax, label='ΔP')
#     plt.tight_layout()
#     if sweep_dir:
#         p = sweep_dir / 'reanalysis_delta_p_heatmap.png'
#         plt.savefig(p, dpi=200, bbox_inches='tight')
#         print(f"  Figura: {p}")
#     plt.show()


"""
spike_propagation_reanalysis.py  — v3 (optimized)
==================================================
Re-análisis de propagación de spikes sobre raw data del barrido 2D existente.

Optimizaciones respecto a v2:
    [Fase 1] Formato CSR para spike_dict y neighbors:
        - Elimina overhead de dict lookup dentro de bucles críticos
        - Contiguous memory → mejor cache locality
        - Permite slicing numpy directo en lugar de list conversion
    [Fase 2] Kernels Numba @njit:
        - Compilación JIT del triple bucle (pre × spike × post)
        - cache=True: evita recompilación entre llamadas
        - Fallback transparente a Python puro si Numba no está disponible
        - Speedup esperado: 8–20× respecto a v2

Misma API pública que v2. Cambios internos transparentes al pipeline.

Métricas:
    1. sigma_global        : <A(t+1)/A(t)>              — Beggs & Plenz
    2. sigma_global_corr   : <(A(t+1)-h) / A(t)>        — Wilting & Priesemann
    3. sigma_micro_EE      : tracking E→E tiempo continuo
    4. sigma_micro_Eall    : tracking E→(E+I) tiempo continuo
    5. p_prop_*            : P(vecino activado | spike ancestro)

Referencias:
    - Beggs & Plenz (2003) J. Neurosci. 23:11167
    - Wilting & Priesemann (2018) Nat. Commun. 9:2325
"""

import numpy as np
import pandas as pd
import pickle
import gzip
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Numba: import con fallback transparente ───────────────────────────────────
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):          # decorator no-op
        def _wrap(fn): return fn
        return _wrap
    print("[WARN] Numba no disponible — usando Python puro (más lento)")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_DIR = Path('results/spike_propagation_2d/sweep_2d_XXXXXXXX_XXXXXX')  # ← EDITAR

REANALYSIS_CONFIG = {
    'Ne': 800, 'Ni': 200,
    'dt_sim_ms': 0.1,
    'T_ms': 4000, 'warmup_ms': 500,
    'dt_bin_ms': 1.0,          # solo para sigma_global y A(t)
    'window_ms': 4.0,          # ventana monosináptica
    'min_spikes_ancestor': 3,
    'bin_sweep': [0.5, 1.0, 2.0, 4.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_sweep_index(sweep_dir: Path) -> dict:
    with open(sweep_dir / 'results.pkl', 'rb') as f:
        return pickle.load(f)

def load_raw_sim(filepath: str) -> dict:
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUCCIÓN DE ESTRUCTURAS INTERNAS
# ─────────────────────────────────────────────────────────────────────────────

def build_neighbors(syn_i, syn_j, syn_w, Ne, Ni,
                    mode='EE', min_weight=0.0) -> dict:
    syn_i = np.asarray(syn_i, dtype=np.int32)
    syn_j = np.asarray(syn_j, dtype=np.int32)
    syn_w = np.asarray(syn_w, dtype=np.float32)

    anc_mask  = (syn_i < Ne)  if mode in ('EE', 'Eall') else np.ones(len(syn_i), bool)
    desc_mask = (syn_j < Ne)  if mode == 'EE'            else np.ones(len(syn_j), bool)
    final     = anc_mask & desc_mask & (syn_w >= min_weight)

    nb = defaultdict(list)
    for pre, post in zip(syn_i[final], syn_j[final]):
        nb[int(pre)].append(int(post))
    return {k: np.array(v, dtype=np.int32) for k, v in nb.items()}


def build_spike_dict(spike_times: np.ndarray, spike_indices: np.ndarray,
                     warmup_ms: float) -> dict:
    """
    Organiza spikes por neurona filtrando warmup.
    float64 necesario para precisión sub-ms en searchsorted.
    """
    mask = spike_times >= warmup_ms
    t    = spike_times[mask].astype(np.float64)
    idx  = spike_indices[mask].astype(np.int32)

    sd = defaultdict(list)
    for time, nid in zip(t, idx):
        sd[int(nid)].append(time)
    return {k: np.sort(v) for k, v in sd.items()}


def build_population_activity(spike_dict: dict, dt_bin_ms: float,
                               T_ms: float, warmup_ms: float,
                               neuron_mask=None) -> np.ndarray:
    T_analysis = T_ms - warmup_ms
    n_bins     = int(T_analysis / dt_bin_ms)
    A          = np.zeros(n_bins, dtype=np.float64)

    for nid, times in spike_dict.items():
        if neuron_mask is not None and nid not in neuron_mask:
            continue
        shifted  = times - warmup_ms
        bins_idx = (shifted / dt_bin_ms).astype(np.int32)
        valid    = (bins_idx >= 0) & (bins_idx < n_bins)
        np.add.at(A, bins_idx[valid], 1)
    return A


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1: ESTRUCTURAS CSR
# Convierte dicts a arrays contiguos para acceso O(1) sin overhead Python.
# ─────────────────────────────────────────────────────────────────────────────

def build_spike_csr(spike_dict: dict, n_neurons: int):
    """
    Comprime spike_dict en dos arrays contiguos (formato CSR).

        spike_data[spike_ptr[i] : spike_ptr[i+1]] = spike times de neurona i

    Ventajas sobre dict:
      - Sin overhead de hashing en cada acceso
      - Memoria contigua → mejor cache locality en el kernel Numba
      - slice = view numpy, sin copia
    """
    counts   = np.zeros(n_neurons, dtype=np.int64)
    for nid, times in spike_dict.items():
        if 0 <= nid < n_neurons:
            counts[nid] = len(times)

    spike_ptr  = np.zeros(n_neurons + 1, dtype=np.int64)
    np.cumsum(counts, out=spike_ptr[1:])

    spike_data = np.empty(int(spike_ptr[-1]), dtype=np.float64)
    for nid, times in spike_dict.items():
        if 0 <= nid < n_neurons:
            s, e = int(spike_ptr[nid]), int(spike_ptr[nid + 1])
            spike_data[s:e] = times           # ya ordenados en build_spike_dict

    return spike_data, spike_ptr


def build_neighbor_csr(neighbors: dict, n_neurons: int):
    """
    Comprime neighbors dict en formato CSR.

        nb_data[nb_ptr[i] : nb_ptr[i+1]] = post_ids de neurona i
    """
    counts = np.zeros(n_neurons, dtype=np.int64)
    for pre_id, post_ids in neighbors.items():
        if 0 <= pre_id < n_neurons:
            counts[pre_id] = len(post_ids)

    nb_ptr  = np.zeros(n_neurons + 1, dtype=np.int64)
    np.cumsum(counts, out=nb_ptr[1:])

    nb_data = np.empty(int(nb_ptr[-1]), dtype=np.int32)
    for pre_id, post_ids in neighbors.items():
        if 0 <= pre_id < n_neurons:
            s, e = int(nb_ptr[pre_id]), int(nb_ptr[pre_id + 1])
            nb_data[s:e] = post_ids

    return nb_data, nb_ptr


def build_reverse_neighbor_csr(nb_data: np.ndarray, nb_ptr: np.ndarray,
                                n_neurons: int):
    """
    Construye CSR inverso post → [pre] a partir del CSR forward.
    Completamente vectorizado (sin bucles Python).
    """
    # Reconstruir array de pre_ids correspondiente a cada entrada de nb_data
    pre_ids_rep = np.repeat(
        np.arange(n_neurons, dtype=np.int32),
        np.diff(nb_ptr).astype(np.int64)
    )
    post_ids_rep = nb_data.astype(np.int32)

    # Filtrar post_ids fuera de rango (seguridad)
    valid    = (post_ids_rep >= 0) & (post_ids_rep < n_neurons)
    pre_ids_rep  = pre_ids_rep[valid]
    post_ids_rep = post_ids_rep[valid]

    # Ordenar por post_id para construir CSR
    order    = np.argsort(post_ids_rep, kind='stable')
    rev_data = pre_ids_rep[order].astype(np.int32)
    post_sorted = post_ids_rep[order]

    counts   = np.bincount(post_sorted.astype(np.intp), minlength=n_neurons).astype(np.int64)
    rev_ptr  = np.zeros(n_neurons + 1, dtype=np.int64)
    np.cumsum(counts, out=rev_ptr[1:])

    return rev_data, rev_ptr


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2: KERNELS NUMBA
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _kernel_sigma_micro(spike_data, spike_ptr,
                         nb_data,    nb_ptr,
                         pre_ids,
                         window_ms, min_spikes,
                         p_base, do_poisson):
    """
    Kernel Numba — sin causal weighting.

    Recorre el triple bucle (pre_id × t_spike × post_id) en código compilado.
    searchsorted sobre slices del array contiguo spike_data.

    Returns
    -------
    sigma        : float64
    p_prop_mean  : float64
    n_analyzed   : int64
    """
    total_desc    = 0.0
    total_spikes  = 0
    p_prop_sum    = 0.0
    n_neurons_ok  = 0

    for ii in range(len(pre_ids)):
        pre_id = pre_ids[ii]

        # Rango de spikes del pre en spike_data
        ps = spike_ptr[pre_id]
        pe = spike_ptr[pre_id + 1]
        if (pe - ps) < min_spikes:
            continue

        # Rango de vecinos del pre en nb_data
        ns   = nb_ptr[pre_id]
        ne_n = nb_ptr[pre_id + 1]
        n_nb = ne_n - ns
        if n_nb == 0:
            continue

        neuron_desc = 0.0
        n_spike_ok  = 0

        for si in range(ps, pe):
            t_spike = spike_data[si]
            t_max   = t_spike + window_ms
            n_act   = 0.0

            for ni in range(ns, ne_n):
                post_id  = nb_data[ni]
                qs       = spike_ptr[post_id]
                qe       = spike_ptr[post_id + 1]
                n_post   = qe - qs
                if n_post == 0:
                    continue

                post_spikes = spike_data[qs:qe]
                idx = np.searchsorted(post_spikes, t_spike, side='right')
                if idx < n_post and post_spikes[idx] <= t_max:
                    n_act += 1.0

            P_obs = n_act / n_nb
            if do_poisson and p_base > 0.0:
                P_net = (P_obs - p_base) / (1.0 - p_base)
                if P_net < 0.0:
                    P_net = 0.0
            else:
                P_net = P_obs

            desc         = P_net * n_nb
            total_desc  += desc
            neuron_desc += desc
            n_spike_ok  += 1
            total_spikes += 1

        if n_spike_ok > 0:
            mean_d       = neuron_desc / n_spike_ok
            p_prop_sum  += mean_d / n_nb
            n_neurons_ok += 1

    if total_spikes == 0:
        return np.nan, np.nan, 0

    sigma       = total_desc / total_spikes
    p_prop_mean = p_prop_sum / n_neurons_ok if n_neurons_ok > 0 else np.nan
    return sigma, p_prop_mean, total_spikes


@njit(cache=True)
def _kernel_sigma_micro_causal(spike_data, spike_ptr,
                                nb_data,    nb_ptr,
                                rev_data,   rev_ptr,
                                pre_ids,
                                window_ms, min_spikes,
                                p_base, do_poisson):
    """
    Kernel Numba — con causal weighting (1/n_padres_disparados).

    Para cada post activado busca cuántos de sus padres
    dispararon en (t_post - window_ms, t_post] y pondera por 1/n.
    Todos los accesos son O(log N) via searchsorted sobre CSR.
    """
    total_desc   = 0.0
    total_spikes = 0
    p_prop_sum   = 0.0
    n_neurons_ok = 0

    for ii in range(len(pre_ids)):
        pre_id = pre_ids[ii]

        ps = spike_ptr[pre_id]
        pe = spike_ptr[pre_id + 1]
        if (pe - ps) < min_spikes:
            continue

        ns   = nb_ptr[pre_id]
        ne_n = nb_ptr[pre_id + 1]
        n_nb = ne_n - ns
        if n_nb == 0:
            continue

        neuron_desc = 0.0
        n_spike_ok  = 0

        for si in range(ps, pe):
            t_spike = spike_data[si]
            t_max   = t_spike + window_ms
            n_act   = 0.0

            for ni in range(ns, ne_n):
                post_id  = nb_data[ni]
                qs       = spike_ptr[post_id]
                qe       = spike_ptr[post_id + 1]
                n_post   = qe - qs
                if n_post == 0:
                    continue

                post_spikes = spike_data[qs:qe]
                idx_post = np.searchsorted(post_spikes, t_spike, side='right')
                if idx_post >= n_post or post_spikes[idx_post] > t_max:
                    continue

                t_post = post_spikes[idx_post]

                # Contar padres de post_id que dispararon en (t_post-window, t_post]
                rs = rev_ptr[post_id]
                re = rev_ptr[post_id + 1]
                n_parents_fired = 0

                for ri in range(rs, re):
                    parent_id    = rev_data[ri]
                    ks           = spike_ptr[parent_id]
                    ke           = spike_ptr[parent_id + 1]
                    n_par        = ke - ks
                    if n_par == 0:
                        continue
                    par_spikes   = spike_data[ks:ke]
                    # Spikes de parent en (t_post - window_ms, t_post]
                    i0 = np.searchsorted(par_spikes, t_post - window_ms, side='right')
                    i1 = np.searchsorted(par_spikes, t_post,             side='right')
                    if i1 > i0:
                        n_parents_fired += 1

                weight = 1.0 / n_parents_fired if n_parents_fired > 0 else 1.0
                n_act += weight

            P_obs = n_act / n_nb
            if do_poisson and p_base > 0.0:
                P_net = (P_obs - p_base) / (1.0 - p_base)
                if P_net < 0.0:
                    P_net = 0.0
            else:
                P_net = P_obs

            desc         = P_net * n_nb
            total_desc  += desc
            neuron_desc += desc
            n_spike_ok  += 1
            total_spikes += 1

        if n_spike_ok > 0:
            mean_d       = neuron_desc / n_spike_ok
            p_prop_sum  += mean_d / n_nb
            n_neurons_ok += 1

    if total_spikes == 0:
        return np.nan, np.nan, 0

    sigma       = total_desc / total_spikes
    p_prop_mean = p_prop_sum / n_neurons_ok if n_neurons_ok > 0 else np.nan
    return sigma, p_prop_mean, total_spikes


# ─────────────────────────────────────────────────────────────────────────────
# 3. ESTIMADORES DE SIGMA
# ─────────────────────────────────────────────────────────────────────────────

def sigma_global(A: np.ndarray, min_active: int = 1) -> float:
    """Estimador Beggs-Plenz: <A(t+1)/A(t)>."""
    mask = A[:-1] >= min_active
    if not mask.any():
        return np.nan
    return float(np.mean(A[1:][mask] / A[:-1][mask]))


def sigma_corrected_k0(A_coupled: np.ndarray, A_baseline_mean: float,
                        min_active: int = 1) -> float:
    """
    Sigma corregido con drive externo (Wilting & Priesemann 2018).
    Formula derivada del branching process: E[A(t+1)] = m·A(t) + h
        m̂ = <(A(t+1) - h) / A(t)>

    Solo el numerador se corrige; el denominador permanece completo
    porque todos los spikes en t generan despolarización real en t+1.
    """
    mask = A_coupled[:-1] >= min_active
    if not mask.any():
        return np.nan
    num = A_coupled[1:][mask] - A_baseline_mean
    den = A_coupled[:-1][mask]
    return float(np.mean(num / den))


def sigma_micro(spike_dict: dict, neighbors: dict,
                window_ms: float,
                T_ms: float, warmup_ms: float,
                min_spikes: int = 3,
                causal_weighting: bool = False,
                baseline_correction: str = None,
                p_base: float = 0.0,
                n_neurons: int = None,
                dt_bin_ms: float = None) -> dict:   # dt_bin_ms: solo compatibilidad de firma
    """
    Wrapper del kernel Numba.

    Convierte spike_dict y neighbors a CSR y delega en el kernel compilado.
    API idéntica a v2; internamente ya no hay bucles Python.

    Parameters
    ----------
    n_neurons : Ne + Ni. Si None se infiere del máximo id encontrado.
    """
    if n_neurons is None:
        n_neurons = max(
            max(spike_dict.keys(), default=0),
            max(neighbors.keys(),  default=0),
        ) + 1

    # ── Fase 1: construir CSR ────────────────────────────────────────────────
    spike_data, spike_ptr = build_spike_csr(spike_dict, n_neurons)
    nb_data,    nb_ptr    = build_neighbor_csr(neighbors, n_neurons)

    # pre_ids: solo los que tienen vecinos
    pre_ids    = np.array(sorted(neighbors.keys()), dtype=np.int32)
    do_poisson = (baseline_correction == 'poisson') and (p_base > 0.0)

    # ── Fase 2: kernel Numba ─────────────────────────────────────────────────
    if causal_weighting:
        rev_data, rev_ptr = build_reverse_neighbor_csr(nb_data, nb_ptr, n_neurons)
        sigma_val, p_prop_mean, n_analyzed = _kernel_sigma_micro_causal(
            spike_data, spike_ptr,
            nb_data,    nb_ptr,
            rev_data,   rev_ptr,
            pre_ids, window_ms, min_spikes, p_base, do_poisson,
        )
    else:
        sigma_val, p_prop_mean, n_analyzed = _kernel_sigma_micro(
            spike_data, spike_ptr,
            nb_data,    nb_ptr,
            pre_ids, window_ms, min_spikes, p_base, do_poisson,
        )

    return {
        'sigma':             float(sigma_val)    if not np.isnan(sigma_val)    else np.nan,
        'p_prop_mean':       float(p_prop_mean)  if not np.isnan(p_prop_mean)  else np.nan,
        'p_prop_dist':       np.array([]),        # no calculado a nivel neurona en kernel
        'per_neuron':        {},
        'n_spikes_analyzed': int(n_analyzed),
    }


def sigma_vs_binsize(spike_dict: dict, T_ms: float, warmup_ms: float,
                     bin_sizes_ms: list, Ne: int = None) -> dict:
    """Barre bin sizes para sigma_global. Útil para elegir bin óptimo."""
    neuron_mask = set(range(Ne)) if Ne is not None else None
    results     = {}
    for bin_ms in bin_sizes_ms:
        A   = build_population_activity(spike_dict, bin_ms, T_ms, warmup_ms, neuron_mask)
        sig = sigma_global(A)
        results[bin_ms] = {
            'sigma':         sig,
            'mean_activity': float(np.mean(A)),
            'mean_rate_hz':  float(np.mean(A)) / bin_ms * 1000,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANÁLISIS COMPLETO DE UNA SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def analyze_single_sim(raw_data: dict, cfg: dict,
                       A_baseline_mean: float  = None,
                       baseline_fr_exc: float  = None,
                       causal_weighting: bool  = False,
                       baseline_correction: str = None) -> dict:
    """
    Análisis completo de una simulación raw.

    Parameters
    ----------
    baseline_fr_exc    : FR excitatorio medio (Hz) del caso k=0 para este rate_hz.
                         Usado para calcular p_base = 1 - exp(-FR × window_s).
                         Más preciso que derivarlo de A_mean/Ne/dt_bin.
    baseline_correction: None | 'poisson'
    """
    Ne      = cfg['Ne']
    Ni      = cfg['Ni']
    warmup  = cfg['warmup_ms']
    T_ms    = cfg['T_ms']
    dt_bin  = cfg['dt_bin_ms']
    window  = cfg['window_ms']
    min_spk = cfg['min_spikes_ancestor']
    N       = Ne + Ni

    spike_times   = np.asarray(raw_data['spike_times'],   dtype=np.float64)
    spike_indices = np.asarray(raw_data['spike_indices'], dtype=np.int32)
    syn_i = np.asarray(raw_data['synapses']['i'], dtype=np.int32)
    syn_j = np.asarray(raw_data['synapses']['j'], dtype=np.int32)
    syn_w = np.asarray(raw_data['synapses']['w'], dtype=np.float32)

    spike_dict = build_spike_dict(spike_times, spike_indices, warmup)

    # ── Actividad poblacional E (para sigma_global) ──────────────────────────
    exc_set = set(range(Ne))
    A       = build_population_activity(spike_dict, dt_bin, T_ms, warmup, exc_set)

    # ── Firing rates ─────────────────────────────────────────────────────────
    T_analysis = T_ms - warmup
    n_spk_exc  = sum(len(t) for nid, t in spike_dict.items() if nid < Ne)
    n_spk_inh  = sum(len(t) for nid, t in spike_dict.items() if nid >= Ne)
    fr_exc = (n_spk_exc / Ne / T_analysis) * 1000.0
    fr_inh = (n_spk_inh / Ni / T_analysis) * 1000.0 if Ni > 0 else 0.0
    fr_all = ((n_spk_exc + n_spk_inh) / N / T_analysis) * 1000.0

    # ── Sigma global ─────────────────────────────────────────────────────────
    sig_global_val = sigma_global(A)

    # ── Sigma corregido (W&P formula) ────────────────────────────────────────
    sig_corr = (sigma_corrected_k0(A, A_baseline_mean)
                if A_baseline_mean is not None else np.nan)

    # ── p_base para corrección Poisson (desde FR directo de spikes) ──────────
    if baseline_correction == 'poisson' and baseline_fr_exc is not None:
        p_base = float(1.0 - np.exp(-baseline_fr_exc * (window / 1000.0)))
    else:
        p_base = 0.0

    # ── Conectividades ───────────────────────────────────────────────────────
    nb_EE   = build_neighbors(syn_i, syn_j, syn_w, Ne, Ni, mode='EE')
    nb_Eall = build_neighbors(syn_i, syn_j, syn_w, Ne, Ni, mode='Eall')

    # ── Sigma microscópico (via kernel Numba) ────────────────────────────────
    kw = dict(window_ms=window, T_ms=T_ms, warmup_ms=warmup,
              min_spikes=min_spk, causal_weighting=causal_weighting,
              baseline_correction=baseline_correction, p_base=p_base,
              n_neurons=N)

    micro_EE   = sigma_micro(spike_dict, nb_EE,   **kw)
    micro_Eall = sigma_micro(spike_dict, nb_Eall, **kw)

    return {
        'firing_rate_exc':        fr_exc,
        'firing_rate_inh':        fr_inh,
        'firing_rate_all':        fr_all,
        'n_spikes_exc':           n_spk_exc,
        'n_spikes_inh':           n_spk_inh,
        'sigma_global':           sig_global_val,
        'sigma_global_corr':      sig_corr,
        'A_mean':                 float(np.mean(A)),
        'sigma_micro_EE':         micro_EE['sigma'],
        'p_prop_EE':              micro_EE['p_prop_mean'],
        'n_spikes_analyzed_EE':   micro_EE['n_spikes_analyzed'],
        'sigma_micro_Eall':       micro_Eall['sigma'],
        'p_prop_Eall':            micro_Eall['p_prop_mean'],
        'n_spikes_analyzed_Eall': micro_Eall['n_spikes_analyzed'],
        'p_base':                 p_base,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORKER MULTIPROCESSING (función global, pickleable en fork y spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_analyze_sim(args):
    (k, rate, trial), fp, cfg, baseline_a, baseline_fr, cw, bc = args
    try:
        raw     = load_raw_sim(fp)
        metrics = analyze_single_sim(raw, cfg,
                                     A_baseline_mean=baseline_a,
                                     baseline_fr_exc=baseline_fr,
                                     causal_weighting=cw,
                                     baseline_correction=bc)
        return {'k': k, 'rate_hz': rate, 'trial': trial, **metrics}
    except Exception as e:
        print(f"  [ERROR] k={k:.2f}, rate={rate:.1f}, trial={trial}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def reanalyze_sweep(sweep_dir: Path, cfg: dict,
                    run_bin_sweep: bool     = False,
                    causal_weighting: bool  = False,
                    baseline_correction: str = None,
                    n_processes: int        = None) -> pd.DataFrame:
    """
    Re-analiza todos los raw data del sweep 2D en dos pasadas.

    Pasada 1 → k=0: baselines (A_mean, FR_exc) por rate_hz
    Pasada 2 → todos los k: análisis completo en paralelo (multiprocessing)
    """
    print(f"\n{'='*60}")
    print(f"  Re-análisis de propagación de spikes  (v3 — Numba:{NUMBA_AVAILABLE})")
    print(f"  causal_weighting   : {causal_weighting}")
    print(f"  baseline_correction: {baseline_correction}")
    print(f"  Sweep: {sweep_dir}")
    print(f"{'='*60}\n")

    index_data  = load_sweep_index(sweep_dir)
    file_paths  = index_data['file_paths']
    K_values    = index_data['K_values']
    rate_values = index_data['rate_hz_values']
    n_trials    = index_data['n_trials']

    print(f"  K values   : {len(K_values)}  ({min(K_values):.1f} – {max(K_values):.1f})")
    print(f"  rate_hz    : {len(rate_values)} ({min(rate_values):.1f} – {max(rate_values):.1f})")
    print(f"  Trials     : {n_trials}")
    print(f"  Total sims : {len(file_paths)}\n")

    # ── Pasada 1: baselines k=0 ──────────────────────────────────────────────
    print("── Pasada 1: calculando baselines k=0 ──")
    baseline_table = defaultdict(lambda: {'A_means': [], 'fr_exc': []})
    k0_keys = [(k, r, t) for (k, r, t) in file_paths if k == 0.0]

    for (k, rate, trial) in tqdm(k0_keys, desc='k=0 baseline'):
        raw = load_raw_sim(file_paths[(k, rate, trial)])
        sd  = build_spike_dict(
            np.asarray(raw['spike_times']),
            np.asarray(raw['spike_indices']),
            cfg['warmup_ms'],
        )
        exc_set    = set(range(cfg['Ne']))
        A          = build_population_activity(sd, cfg['dt_bin_ms'],
                                               cfg['T_ms'], cfg['warmup_ms'], exc_set)
        T_analysis = cfg['T_ms'] - cfg['warmup_ms']
        n_spk_exc  = sum(len(t) for nid, t in sd.items() if nid < cfg['Ne'])
        fr_exc     = (n_spk_exc / cfg['Ne'] / T_analysis) * 1000.0

        baseline_table[rate]['A_means'].append(float(np.mean(A)))
        baseline_table[rate]['fr_exc'].append(fr_exc)

    baseline_A_mean  = {r: np.mean(v['A_means']) for r, v in baseline_table.items()}
    baseline_FR_mean = {r: np.mean(v['fr_exc'])  for r, v in baseline_table.items()}
    print(f"  Baselines listos para {len(baseline_A_mean)} valores de rate_hz\n")

    # ── Barrido de bins (opcional) ───────────────────────────────────────────
    if run_bin_sweep:
        print("── Barrido de bin temporal ──")
        k_mid = K_values[len(K_values) // 2]
        r_mid = rate_values[len(rate_values) // 2]
        key_s = min(file_paths.keys(),
                    key=lambda x: abs(x[0] - k_mid) + abs(x[1] - r_mid))
        raw_s = load_raw_sim(file_paths[key_s])
        sd_s  = build_spike_dict(
            np.asarray(raw_s['spike_times']),
            np.asarray(raw_s['spike_indices']),
            cfg['warmup_ms'],
        )
        bin_res = sigma_vs_binsize(sd_s, cfg['T_ms'], cfg['warmup_ms'],
                                   cfg['bin_sweep'], Ne=cfg['Ne'])
        print(f"  Muestra: k={key_s[0]:.1f}, rate={key_s[1]:.1f} Hz")
        print(f"  {'bin_ms':>8} {'sigma':>8} {'mean_A':>8} {'FR_exc':>10}")
        for bms, bres in sorted(bin_res.items()):
            print(f"  {bms:>8.1f} {bres['sigma']:>8.4f} "
                  f"{bres['mean_activity']:>8.2f} {bres['mean_rate_hz']:>10.2f}")
        print()

    # ── Pasada 2: análisis completo en paralelo ──────────────────────────────
    print("── Pasada 2: análisis completo (multiprocessing) ──")
    if n_processes is None:
        n_processes = min(mp.cpu_count() - 1, 31)

    tasks = [
        ((k, rate, trial), fp, cfg,
         baseline_A_mean.get(rate),
         baseline_FR_mean.get(rate),
         causal_weighting, baseline_correction)
        for (k, rate, trial), fp in file_paths.items()
    ]

    all_results = []
    with mp.Pool(processes=n_processes) as pool:
        for result in tqdm(
            pool.imap_unordered(_worker_analyze_sim, tasks, chunksize=4),
            total=len(tasks), desc='Simulaciones',
        ):
            if result is not None:
                all_results.append(result)

    df = pd.DataFrame(all_results)
    df = df.sort_values(['k', 'rate_hz', 'trial']).reset_index(drop=True)
    print(f"\n  Simulaciones procesadas: {len(df)}")
    print(f"  Columnas: {list(df.columns)}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. AGREGACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega sobre trials. Calcula ΔP = métrica(k>0) − métrica(k=0, mismo rate)."""
    metrics  = [
        'firing_rate_exc', 'firing_rate_inh', 'firing_rate_all',
        'sigma_global', 'sigma_global_corr',
        'sigma_micro_EE', 'sigma_micro_Eall',
        'p_prop_EE', 'p_prop_Eall', 'A_mean', 'p_base',
    ]
    agg_dict = {m: ['mean', 'std'] for m in metrics if m in df.columns}
    df_agg   = df.groupby(['k', 'rate_hz']).agg(agg_dict)
    df_agg.columns = ['_'.join(c) for c in df_agg.columns]
    df_agg   = df_agg.reset_index()

    k0 = df_agg[df_agg['k'] == 0.0].set_index('rate_hz')
    for metric in ['p_prop_EE', 'p_prop_Eall', 'sigma_micro_EE', 'sigma_micro_Eall']:
        col = f'{metric}_mean'
        if col in df_agg.columns:
            df_agg[f'delta_{metric}'] = df_agg.apply(
                lambda row: row[col] - k0.loc[row['rate_hz'], col]
                if row['rate_hz'] in k0.index else np.nan,
                axis=1,
            )
    return df_agg


# ─────────────────────────────────────────────────────────────────────────────
# 7. GUARDADO
# ─────────────────────────────────────────────────────────────────────────────

def save_results(df_raw: pd.DataFrame, df_agg: pd.DataFrame,
                 sweep_dir: Path, cfg: dict, tag: str = ''):
    suffix   = f'_{tag}' if tag else ''
    pkl_path = sweep_dir / f'reanalysis_propagation{suffix}.pkl'
    csv_path = sweep_dir / f'reanalysis_aggregated{suffix}.csv'
    with open(pkl_path, 'wb') as f:
        pickle.dump({'df_raw': df_raw, 'df_aggregated': df_agg, 'config': cfg},
                    f, protocol=4)
    df_agg.to_csv(csv_path, index=False)
    print(f"  Guardado: {pkl_path}")
    print(f"  CSV:      {csv_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def plot_sigma_comparison(df_agg: pd.DataFrame, sweep_dir: Path = None):
    import matplotlib.pyplot as plt
    rates  = sorted(df_agg['rate_hz'].unique())
    rates  = rates[::max(1, len(rates) // 5)]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rates)))
    cols   = [
        ('sigma_global_mean',      'σ global (Beggs-Plenz)'),
        ('sigma_global_corr_mean', 'σ global corr. (W&P 2018)'),
        ('sigma_micro_EE_mean',    'σ micro E→E (Numba)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, label) in zip(axes, cols):
        if col not in df_agg.columns:
            ax.text(0.5, 0.5, 'no disponible', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        for rate, color in zip(rates, colors):
            sub = df_agg[df_agg['rate_hz'] == rate].sort_values('k')
            ax.plot(sub['k'], sub[col], 'o-', color=color,
                    label=f'{rate:.0f} Hz', markersize=4, lw=1.5)
        ax.axhline(1.0, color='red', ls='--', lw=1)
        ax.set_xlabel('K'); ax.set_ylabel(label)
        ax.set_title(label); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    if sweep_dir:
        p = sweep_dir / 'reanalysis_sigma_comparison.png'
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f"  Figura: {p}")
    plt.show()


def plot_delta_p(df_agg: pd.DataFrame, sweep_dir: Path = None):
    import matplotlib.pyplot as plt
    if 'delta_p_prop_EE' not in df_agg.columns:
        return
    K_vals    = sorted(df_agg['k'].unique())
    rate_vals = sorted(df_agg['rate_hz'].unique())
    pivot     = df_agg.pivot(index='k', columns='rate_hz', values='delta_p_prop_EE')
    fig, ax   = plt.subplots(figsize=(10, 7))
    vmax = np.nanpercentile(np.abs(pivot.values), 95)
    im   = ax.imshow(pivot.values, aspect='auto', origin='lower',
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(rate_vals)))
    ax.set_xticklabels([f'{r:.0f}' for r in rate_vals], rotation=45)
    ax.set_yticks(range(len(K_vals)))
    ax.set_yticklabels([f'{k:.1f}' for k in K_vals])
    ax.set_xlabel('rate_hz (Hz)'); ax.set_ylabel('K')
    ax.set_title('ΔP_EE = P(k>0) − P_baseline(k=0)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if sweep_dir:
        p = sweep_dir / 'reanalysis_delta_p_heatmap.png'
        plt.savefig(p, dpi=200, bbox_inches='tight')
        print(f"  Figura: {p}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9. BENCHMARK  (python spike_propagation_reanalysis_v3.py --bench)
# ─────────────────────────────────────────────────────────────────────────────

def _run_benchmark():
    """
    Compara v2 (Python puro) vs v3 (Numba) en datos sintéticos.
    Ejecutar una vez para compilar el JIT, luego mide de verdad.
    """
    import time
    from collections import defaultdict

    print("\n── Benchmark v2 (Python) vs v3 (Numba) ──")
    np.random.seed(0)
    Ne, Ni, T_ms, warmup = 800, 200, 4000.0, 500.0
    N = Ne + Ni

    # Datos sintéticos realistas (~10 Hz por neurona)
    n_spikes = int(Ne * (T_ms - warmup) / 1000.0 * 10)
    times    = np.random.uniform(warmup, T_ms, n_spikes)
    idx      = np.random.randint(0, Ne, n_spikes)
    sd       = defaultdict(list)
    for t, n in zip(times, idx):
        sd[int(n)].append(t)
    spike_dict = {k: np.sort(v) for k, v in sd.items()}

    # Red esparsa: cada E tiene ~100 vecinos E
    syn_i = np.repeat(np.arange(Ne, dtype=np.int32), 100)
    syn_j = np.array([np.random.choice(Ne, 100, replace=False)
                      for _ in range(Ne)], dtype=np.int32).ravel()
    nb    = defaultdict(list)
    for i, j in zip(syn_i, syn_j):
        nb[int(i)].append(int(j))
    nb_np = {k: np.array(v, dtype=np.int32) for k, v in nb.items()}

    # ── Compilar Numba (primera llamada siempre más lenta) ───────────────────
    print("  Compilando kernel Numba... ", end='', flush=True)
    t0 = time.perf_counter()
    sigma_micro(spike_dict, nb_np, 4.0, T_ms, warmup,
                min_spikes=1, n_neurons=N)
    t_compile = time.perf_counter() - t0
    print(f"{t_compile:.2f}s (JIT warmup, solo ocurre una vez por sesión)")

    # ── Python puro (simula lo que hacía v2) ─────────────────────────────────
    def sigma_micro_python(spike_dict, neighbors, window_ms, T_ms, warmup_ms, min_spikes=1):
        total_desc = 0.0; total_spk = 0
        for pre_id, post_ids in neighbors.items():
            if pre_id not in spike_dict: continue
            pre_times = spike_dict[pre_id]
            if len(pre_times) < min_spikes: continue
            n_nb = len(post_ids)
            for t_spike in pre_times:
                n_act = 0.0
                for post_id in post_ids:
                    if post_id not in spike_dict: continue
                    post_times = spike_dict[post_id]
                    idx = np.searchsorted(post_times, t_spike, side='right')
                    if idx < len(post_times) and post_times[idx] <= t_spike + window_ms:
                        n_act += 1.0
                total_desc += n_act / n_nb * n_nb; total_spk += 1
        return total_desc / total_spk if total_spk > 0 else np.nan

    # Subset para Python (sería demasiado lento con todo)
    nb_subset = {k: v for k, v in list(nb_np.items())[:50]}
    t0 = time.perf_counter()
    s_py = sigma_micro_python(spike_dict, nb_subset, 4.0, T_ms, warmup)
    t_py = time.perf_counter() - t0

    # Numba con el mismo subset 
    t0 = time.perf_counter()
    r_nb = sigma_micro(spike_dict, nb_subset, 4.0, T_ms, warmup,
                       min_spikes=1, n_neurons=N)
    t_nb = time.perf_counter() - t0

    # Numba red completa
    t0 = time.perf_counter()
    r_full = sigma_micro(spike_dict, nb_np, 4.0, T_ms, warmup,
                         min_spikes=1, n_neurons=N)
    t_full = time.perf_counter() - t0

    print(f"\n  {'Modo':<30} {'Tiempo':>10} {'Speedup':>10} {'Sigma':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Python puro (50 neuronas)':<30} {t_py:>9.3f}s {'1.0×':>10} {s_py:>10.4f}")
    print(f"  {'Numba (50 neuronas)':<30} {t_nb:>9.3f}s {t_py/t_nb:>9.1f}× {r_nb['sigma']:>10.4f}")
    print(f"  {'Numba (800 neuronas, red completa)':<30} {t_full:>9.3f}s {'—':>10} {r_full['sigma']:>10.4f}")
    print(f"\n  Resultados Python vs Numba: consistentes = {abs(s_py - r_nb['sigma']) < 1e-6}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if '--bench' in sys.argv:
        _run_benchmark()
    else:
        df_raw = reanalyze_sweep(
            sweep_dir=SWEEP_DIR,
            cfg=REANALYSIS_CONFIG,
            run_bin_sweep=True,
            causal_weighting=False,
            baseline_correction=None,
        )
        df_agg = aggregate_results(df_raw)
        save_results(df_raw, df_agg, SWEEP_DIR, REANALYSIS_CONFIG, tag='v3')
        plot_sigma_comparison(df_agg, SWEEP_DIR)
        plot_delta_p(df_agg, SWEEP_DIR)
