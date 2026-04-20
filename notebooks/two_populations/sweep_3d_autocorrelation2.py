#!/usr/bin/env python
# coding: utf-8

# # Barrido 3D: K_intra × K_inter_ratio × Delay
# **Objetivo:** Mapear funciones de autocorrelación e intrinsic timescales
# 
# **Estrategia:**
# - Solo guardar spikes (no voltage)
# - Calcular autocorrelación on-the-fly
# - Extraer timescales con 2 métodos (exponential + integrated)
# - 3 trials por configuración
# - Plots progresivos cada 5 batches
# - Checkpointing cada 10 batches

# In[1]:


import os
import sys
import gc
# =============================================================================
# 🛑 1. CRÍTICO: CONFIGURACIÓN DE HILOS (ANTES DE CUALQUIER IMPORT NUMÉRICO)
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

os.chdir('../..')
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import json
import pickle
from datetime import datetime
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

from brian2 import *

from src.two_populations.model import IzhikevichNetwork
from src.two_populations.metrics import (
    spikes_to_population_rate,
    cross_correlation_analysis,
    intrinsic_timescale_analysis
)
from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="sweep_3d_autocorr_gamma",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=True
)

logger.info(f"Working directory: {Path.cwd()}")

# =============================================================================
# CONFIGURACIÓN DEL BARRIDO
# =============================================================================

# 1. Definición de la Grilla (CORREGIDO: Usando linspace para puntos exactos)
# Queremos 50 puntos entre 20.0 y 25.0
K_INTRA_VALUES = np.linspace(0.0, 20.0, 20)
K_INTER_RATIOS = np.linspace(0.0, 1.0, 20)        # paso 0.05
DELAY_VALUES   = np.linspace(0.0, 125.0, 20)      # paso 2.5ms

POPULATION_PARAMS = {
    'Ne': 800, 'Ni': 200,
    'noise_exc': 0.88, 'noise_inh': 0.6,
    'p_intra': 0.1, 'p_inter': 0.02,
    'delay_intra': 0.0,
    'rate_hz': 10.0,
    'stim_base': 1.0
}

SIM_CONFIG = {
    'batch_size': 200,
    'n_jobs': 63,
    'dt_ms': 0.1,
    'T_ms': 3500,       # 10s útiles tras warmup
    'warmup_ms': 500,
    'n_trials': 3,
    'checkpoint_every': 5,
    'plot_every': 5
}

AC_CONFIG = {
    'max_lag_ms': 300,   # cubre dos períodos alfa completos
    'analysis_dt': 0.5,
}

LFP_DT_MS = 0.5  # debe coincidir con record_v_dt

# Crear todas las configuraciones
configs = []
for k_intra in K_INTRA_VALUES:
    for k_inter_ratio in K_INTER_RATIOS:
        for delay in DELAY_VALUES:
            configs.append({
                'k_intra': k_intra,
                'k_inter_ratio': k_inter_ratio,
                'delay': delay
            })

n_configs = len(configs)
n_total_sims = n_configs * SIM_CONFIG['n_trials']

logger.info(f"Grid: {len(K_INTRA_VALUES)} × {len(K_INTER_RATIOS)} × {len(DELAY_VALUES)}")
logger.info(f"Total configs: {n_configs}")
logger.info(f"Total sims: {n_total_sims} ({SIM_CONFIG['n_trials']} trials/config)")

# =============================================================================
# SETUP DE ALMACENAMIENTO
# =============================================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = Path(f"./results/sweep_3d_autocorr_gamma_{timestamp}")
# output_dir.mkdir(parents=True, exist_ok=True)

# (output_dir / "raw_spikes").mkdir(exist_ok=True)
# (output_dir / "metrics_3d").mkdir(exist_ok=True)
# (output_dir / "plots_progress").mkdir(exist_ok=True)
# (output_dir / "casos_particulares").mkdir(exist_ok=True)
# (output_dir / "diagnostics").mkdir(exist_ok=True)

# =============================================================================
# SETUP DE ALMACENAMIENTO (MODO RESUME)
# =============================================================================

# # 🛑 IMPORTANTE: Pon aquí la ruta EXACTA del job que falló al 75%
output_dir = Path("./results/sweep_3d_autocorr_gamma_20260416_024714") 

# Aseguramos que existan las subcarpetas (por si acaso)
(output_dir / "raw_spikes").mkdir(parents=True, exist_ok=True)
(output_dir / "metrics_3d").mkdir(exist_ok=True)
(output_dir / "plots_progress").mkdir(exist_ok=True)

logger.info(f"📂 RESUMIENDO TRABAJO EN: {output_dir}")

# Guardar configuración
config_dict = {
    'timestamp': timestamp,
    'K_INTRA_VALUES': K_INTRA_VALUES.tolist(),
    'K_INTER_RATIOS': K_INTER_RATIOS.tolist(),
    'DELAY_VALUES': DELAY_VALUES.tolist(),
    'population_params': POPULATION_PARAMS,
    'sim_config': SIM_CONFIG,
    'ac_config': AC_CONFIG,
    'n_configs': n_configs,
    'n_total_sims': n_total_sims
}

with open(output_dir / "config.json", 'w') as f:
    json.dump(config_dict, f, indent=2)

# Arrays 3D (SOLO tau_int)
shape_3d = (len(K_INTRA_VALUES), len(K_INTER_RATIOS), len(DELAY_VALUES))

arrays_3d = {
    # Existentes
    'tau_int_A': np.full(shape_3d, np.nan),
    'tau_int_A_std': np.full(shape_3d, np.nan),
    'mean_rate_A': np.full(shape_3d, np.nan),
    # Nuevos
    'freq_dominant_A': np.full(shape_3d, np.nan),
    'freq_dominant_A_std': np.full(shape_3d, np.nan),
    # Idem para B
    'tau_int_B': np.full(shape_3d, np.nan),
    'tau_int_B_std': np.full(shape_3d, np.nan),
    'mean_rate_B': np.full(shape_3d, np.nan),
    'freq_dominant_B': np.full(shape_3d, np.nan),
    'freq_dominant_B_std': np.full(shape_3d, np.nan),
    # Añadir al dict arrays_3d junto a los otros campos:
    'tau_int_lfp_A':     np.full(shape_3d, np.nan),
    'tau_int_lfp_A_std': np.full(shape_3d, np.nan),
    'tau_int_lfp_B':     np.full(shape_3d, np.nan),
    'tau_int_lfp_B_std': np.full(shape_3d, np.nan),
}


# Metadata para índices
metadata = {
    'k_intra_values': K_INTRA_VALUES.tolist(),
    'k_inter_ratios': K_INTER_RATIOS.tolist(),
    'delay_values': DELAY_VALUES.tolist(),
    'shape': shape_3d
}
with open(output_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

logger.success(f"Output directory: {output_dir}")


def compute_lfp_signal(v_traces, warmup_ms, dt_ms):
    """
    v_traces: array (n_exc_sampled, n_timepoints) — solo excitatorias
    Devuelve LFP proxy como promedio de potenciales de membrana excitatorios.
    """
    lfp = np.mean(v_traces, axis=0)          # promedio sobre neuronas
    n_warmup = int(warmup_ms / dt_ms)
    return lfp[n_warmup:]                     # descarta warmup


def compute_psd_from_lfp(lfp, dt_ms, freq_min=1.0, freq_max=80.0):
    """
    PSD Welch sobre LFP. Usa ventana de min(3s, T_útil) para resolución df≤0.33Hz.
    Devuelve (freqs, psd) recortado a [freq_min, freq_max].
    """
    from scipy.signal import welch
    fs = 1000.0 / dt_ms                          # sampling rate en Hz
    # ventana de 3s para df=0.33Hz; si señal más corta usa lo que haya
    nperseg = min(int(fs * 3.0), len(lfp))
    # con 10s útiles: nperseg=6000 (3s), df=0.167Hz — resolución cómoda
    freqs, psd = welch(lfp, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                       window='hann', scaling='density')
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    return freqs[mask], psd[mask]


def get_dominant_freq(freqs, psd, f_low=5.0, f_high=15.0):
    """Frecuencia del pico focal en [f_low, f_high] Hz sobre el PSD del LFP."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask) or np.all(psd[mask] == 0):
        return np.nan
    return float(freqs[mask][np.argmax(psd[mask])])


def compute_int_from_lfp(lfp, dt_ms, max_lag_ms=250.0, threshold=0.1):
    """
    INT calculada sobre LFP: área bajo la autocorrelación normalizada hasta
    primer cruce del umbral (equivalente a tau_int sobre spike rate).
    """
    from scipy.signal import correlate
    n = len(lfp)
    lfp_z = lfp - np.mean(lfp)                  # centrar
    # autocorrelación completa, normalizada
    ac_full = correlate(lfp_z, lfp_z, mode='full')
    ac_full = ac_full / ac_full[n - 1]           # normalizar en lag=0
    ac_pos = ac_full[n - 1:]                     # lags positivos
    
    max_lag_pts = int(max_lag_ms / dt_ms)
    ac_pos = ac_pos[:max_lag_pts]
    lags = np.arange(len(ac_pos)) * dt_ms
    
    # primer cruce del umbral
    cross = np.where(ac_pos <= threshold)[0]
    end_idx = cross[0] if len(cross) > 0 else len(ac_pos)
    
    tau_int = float(np.trapezoid(ac_pos[:end_idx], lags[:end_idx]))
    return tau_int, ac_pos, lags

# =============================================================================
# FUNCIONES DE CÁLCULO ON-THE-FLY
# =============================================================================

def compute_autocorr_and_timescales(spike_times, spike_neurons, N_neurons, 
                                     warmup_ms, T_total, analysis_dt, max_lag_ms):
    """Calcula autocorrelación y tau_int."""
    
    # Population rate
    time, rate = spikes_to_population_rate(
        type('obj', (), {'t': spike_times*ms, 'i': spike_neurons})(),
        N_neurons,
        smooth_window=1,
        analysis_dt=analysis_dt,
        T_total=T_total
    )
    
    # Filtrar warmup
    mask = time >= warmup_ms
    time_filt = time[mask]
    rate_filt = rate[mask]
    
    mean_rate = np.mean(rate_filt)
    
    if len(rate_filt) < 10:
        return None
    
    # Autocorrelación
    ac_result = cross_correlation_analysis(
        rate_filt, rate_filt,
        max_lag_ms=max_lag_ms,
        dt=analysis_dt
    )
    
    # Timescales (tau_int es robusto)
    ts_result = intrinsic_timescale_analysis(
        rate_filt,
        max_lag_ms=max_lag_ms,
        dt=analysis_dt
    )
    
    return {
        'tau_int': ts_result['tau_int'],
        'ac_peak': ac_result['peak_value'],
        'ac_lags': ac_result['lags'],
        'ac_corr': ac_result['correlation'],
        'mean_rate': mean_rate,
        'quality': ts_result['quality']
    }


def run_single_simulation(config, trial, base_seed=100):
    """Simula una configuración y retorna spikes + métricas."""
    
    # 🛑 1. HACK MEMORIA
    gc.disable()
    
    # 🛑 2. HACK ANTI-BLOQUEO (CRÍTICO: ESTO FALTABA)
    # Fuerza modo Python puro. Sin esto, 32 procesos intentan compilar C++ a la vez y se bloquean.
    prefs.codegen.target = 'numpy'  
    
    start_scope() # Inicia el ámbito de Brian2
    
    try:
        k_intra = config['k_intra']
        k_inter_ratio = config['k_inter_ratio']
        delay = config['delay']
        k_inter = k_inter_ratio * k_intra
        
        # Red
        net = IzhikevichNetwork(
            dt_val=SIM_CONFIG['dt_ms'],
            T_total=SIM_CONFIG['T_ms'],
            fixed_seed=base_seed,
            variable_seed=base_seed + trial * 1000
        )
        
        # Poblaciones
        for pop_name in ['A', 'B']:
            net.create_population2(
                name=pop_name,
                Ne=POPULATION_PARAMS['Ne'],
                Ni=POPULATION_PARAMS['Ni'],
                k_exc=k_intra,
                k_inh=k_intra * 3.9,
                noise_exc=POPULATION_PARAMS['noise_exc'],
                noise_inh=POPULATION_PARAMS['noise_inh'],
                p_intra=POPULATION_PARAMS['p_intra'],
                delay=POPULATION_PARAMS['delay_intra'],
                rate_hz=POPULATION_PARAMS['rate_hz'],
                stim_start_ms=None,
                stim_duration_ms=None,
                stim_base=1.0,
                stim_elevated=None
            )
        
        # Conexiones inter
        if k_inter > 0:
            for src, tgt in [('A', 'B')]: #,  , , ('B', 'A')
                net.connect_populations( 
                    src, tgt,
                    p_inter=POPULATION_PARAMS['p_inter'],
                    weight_scale=k_inter,
                    delay_value=delay,
                    delay_dist='constant'
            )
            
            for src, tgt in [('B', 'A')]: #,  , , ('B', 'A')
                net.connect_populations( 
                    src, tgt,
                    p_inter=POPULATION_PARAMS['p_inter'],
                    weight_scale=k_inter,
                    delay_value=delay,
                    delay_dist='constant'
                )
            
        
        # Monitores (solo spikes)
        net.setup_monitors(['A', 'B'], record_v_dt=0.5, sample_fraction=0.1, 
                        monitor_conductance=False)
        
                # Simular
        results = net.run_simulation()

        # Extraer spikes
        spikes_A = {'times': results['A']['spike_times'], 'neurons': results['A']['spike_indices']}
        spikes_B = {'times': results['B']['spike_times'], 'neurons': results['B']['spike_indices']}

        def _compute_all_metrics(spikes, results_pop, pop_name):
            N = POPULATION_PARAMS['Ne'] + POPULATION_PARAMS['Ni']

            # --- Spike rate (idéntico al sweep anterior) ---
            time, rate = spikes_to_population_rate(
                type('obj', (), {'t': spikes['times'] * ms, 'i': spikes['neurons']})(),
                N, smooth_window=1, analysis_dt=AC_CONFIG['analysis_dt'],
                T_total=SIM_CONFIG['T_ms']
            )
            mask = time >= SIM_CONFIG['warmup_ms']
            rate_filt = rate[mask]
            mean_rate = float(np.mean(rate_filt))

            # Métricas de spikes — conservadas para compatibilidad
            ac_result = cross_correlation_analysis(
                rate_filt, rate_filt,
                max_lag_ms=AC_CONFIG['max_lag_ms'],
                dt=AC_CONFIG['analysis_dt']
            )
            ts_result = intrinsic_timescale_analysis(
                rate_filt,
                max_lag_ms=AC_CONFIG['max_lag_ms'],
                dt=AC_CONFIG['analysis_dt']
            )

            out = {
                # ── campos legacy (compatibles con notebook anterior) ──
                'tau_int':  ts_result['tau_int'],
                'ac_peak':  ac_result['peak_value'],
                'ac_lags':  ac_result['lags'],
                'ac_corr':  ac_result['correlation'],
                'mean_rate': mean_rate,
                'quality':  ts_result['quality'],
            }

            # --- LFP proxy (nuevo, adicional) ---
            v_mon = results_pop.get('voltage_monitor', None)
            if v_mon is not None and hasattr(v_mon, 'v') and v_mon.v.shape[0] > 0:
                v_traces = np.array(v_mon.v)
                lfp = compute_lfp_signal(v_traces, SIM_CONFIG['warmup_ms'], LFP_DT_MS)

                tau_int_lfp, ac_pos_lfp, lags_lfp = compute_int_from_lfp(
                    lfp, dt_ms=LFP_DT_MS,
                    max_lag_ms=AC_CONFIG['max_lag_ms'],
                    threshold=0.1
                )
                psd_freqs, psd_power = compute_psd_from_lfp(lfp, dt_ms=LFP_DT_MS)
                freq_dom = get_dominant_freq(psd_freqs, psd_power, f_low=5.0, f_high=15.0)

                out.update({
                    # ── campos nuevos (LFP-based) ──
                    'tau_int_lfp':    tau_int_lfp,
                    'freq_dominant':  freq_dom,
                    'lfp_mean':       lfp.astype(np.float32),
                    'psd_freqs':      psd_freqs.astype(np.float32),
                    'psd_power':      psd_power.astype(np.float32),
                    'ac_lags_lfp':    lags_lfp.astype(np.float32),
                    'ac_corr_lfp':    ac_pos_lfp.astype(np.float32),
                })
            else:
                out.update({
                    'tau_int_lfp':   np.nan,
                    'freq_dominant': np.nan,
                })

            return out

        metrics_A = _compute_all_metrics(spikes_A, results['A'], 'A')
        metrics_B = _compute_all_metrics(spikes_B, results['B'], 'B')

        return {
            'config': config, 'trial': trial,
            'spikes_A': spikes_A, 'spikes_B': spikes_B,
            'metrics_A': metrics_A, 'metrics_B': metrics_B
        }
        
        
    except Exception as e:
        # Capturar error sin romper el worker
        return {'error': str(e), 'config': config, 'trial': trial}
    
    finally:
        # 🛑 REACTIVAR GC SIEMPRE (Vital para que no explote la RAM)
        gc.enable()
        gc.collect()


# 2. Análisis post-hoc de casos fallidos
def analyze_failed_cases(output_dir):
    """Analizar simulaciones con tau_exp < 1ms"""
    failed_cases = []
    
    for h5_file in sorted((output_dir / "raw_spikes").glob("batch_*.h5")):
        with h5py.File(h5_file, 'r') as f:
            for key in f.keys():
                grp = f[key]
                tau_exp_A = grp['metrics/A'].attrs.get('tau_exp', 0)
                tau_int_A = grp['metrics/A'].attrs.get('tau_int', 0)
                
                if tau_int_A > 5.0:  # Discrepancia tau_exp_A < 1.0 and t
                    failed_cases.append({
                        'file': h5_file.name,
                        'key': key,
                        'k_intra': grp.attrs['k_intra'],
                        'k_inter_ratio': grp.attrs['k_inter_ratio'],
                        'delay_ms': grp.attrs['delay_ms'],
                        'tau_exp': tau_exp_A,
                        'tau_int': tau_int_A,
                        'mean_rate': grp['metrics/A'].attrs.get('mean_rate', 0)
                    })
    
    # Ordenar por discrepancia
    failed_cases.sort(key=lambda x: x['tau_int'] - x['tau_exp'], reverse=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CASOS FALLIDOS (tau_exp < 1ms, tau_int > 5ms): {len(failed_cases)}")
    logger.info(f"{'='*60}")
    
    for case in failed_cases[:10]:  # Top 10
        logger.info(f"K={case['k_intra']:.1f}, ratio={case['k_inter_ratio']:.1f}, d={case['delay_ms']:.0f}ms")
        logger.info(f"  tau_exp={case['tau_exp']:.3f}, tau_int={case['tau_int']:.1f}, rate={case['mean_rate']:.1f}Hz")
    
    return failed_cases

def plot_ac_diagnostic(h5_file, sim_key, output_dir):
    """Plot detallado de AC con ambos métodos"""
    with h5py.File(h5_file, 'r') as f:
        grp = f[sim_key]
        
        # Config info
        k_intra = grp.attrs['k_intra']
        k_inter_ratio = grp.attrs['k_inter_ratio']
        delay_ms = grp.attrs['delay_ms']
        
        # Cargar spikes
        spike_times = grp['spikes_A_times'][:]
        spike_neurons = grp['spikes_A_neurons'][:]
        
        # Cargar métricas guardadas
        # tau_exp_saved = grp['metrics/A'].attrs['tau_exp']
        tau_int_saved = grp['metrics/A'].attrs['tau_int']
        mean_rate_saved = grp['metrics/A'].attrs['mean_rate']
    
    # Recalcular population rate
    time, rate = spikes_to_population_rate(
        type('obj', (), {'t': spike_times*ms, 'i': spike_neurons})(),
        1000, smooth_window=0, analysis_dt=1.0, T_total=4000
    )
    
    # Filtrar warmup
    mask = time >= 500
    time_filt, rate_filt = time[mask], rate[mask]
    
    # Recalcular AC y timescales
    ac_result = cross_correlation_analysis(rate_filt, rate_filt, max_lag_ms=500, dt=1.0)
    ts_result = intrinsic_timescale_analysis(rate_filt, max_lag_ms=500, dt=1.0)
    
    lags = ac_result['lags']
    corr = ac_result['correlation']
    corr_norm = corr / np.max(np.abs(corr))
    
    pos_mask = lags >= 0
    lags_pos, corr_pos = lags[pos_mask], corr_norm[pos_mask]
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Population rate
    axes[0,0].plot(time_filt, rate_filt, 'k-', linewidth=1)
    axes[0,0].set_xlabel('Time (ms)')
    axes[0,0].set_ylabel('Rate (Hz)')
    axes[0,0].set_title(f'Pop Rate (mean={mean_rate_saved:.1f}Hz)')
    axes[0,0].grid(alpha=0.3)
    
    # 2. AC completa
    axes[0,1].plot(lags, corr_norm, 'k-', linewidth=2)
    axes[0,1].axhline(1/np.e, color='gray', ls=':', label='1/e threshold')
    axes[0,1].axhline(0, color='k', ls='--', alpha=0.3)
    axes[0,1].axvline(0, color='k', ls='-', alpha=0.3)
    axes[0,1].set_xlabel('Lag (ms)')
    axes[0,1].set_ylabel('Autocorrelation (normalized)')
    axes[0,1].set_title('Full AC')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # 4. Área integrada
    zero_cross = np.where(corr_pos <= 0.1)[0]
    end_idx = zero_cross[0] if len(zero_cross) > 0 else len(corr_pos)
    
    axes[1,1].fill_between(lags_pos[:end_idx], 0, corr_pos[:end_idx], alpha=0.3, color='blue')
    axes[1,1].plot(lags_pos, corr_pos, 'k-', linewidth=2)
    axes[1,1].axvline(tau_int_saved, color='blue', ls='--', linewidth=2, label=f'τ_int={tau_int_saved:.1f}ms')
    axes[1,1].axhline(1/np.e, color='gray', ls=':', alpha=0.5)
    axes[1,1].set_xlabel('Lag (ms)')
    axes[1,1].set_ylabel('AC (normalized)')
    axes[1,1].set_title('Integrated Method')
    axes[1,1].legend()
    axes[1,1].set_xlim(0, 100)
    axes[1,1].set_ylim(0, 1.1)
    axes[1,1].grid(alpha=0.3)
    
    # Título general
    fig.suptitle(f'AC Diagnostic: K={k_intra:.1f}, ratio={k_inter_ratio:.1f}, delay={delay_ms:.0f}ms', 
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"diagnostic_{sim_key.replace('/', '_')}.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Diagnostic saved: {sim_key}")

# =============================================================================
# FUNCIONES DE GUARDADO
# =============================================================================

def save_batch_to_hdf5(batch_results, batch_idx, output_dir):
    """Guarda batch en HDF5 - SOLO tau_int."""
    
    h5_path = output_dir / "raw_spikes" / f"batch_{batch_idx:04d}.h5"
    
    with h5py.File(h5_path, 'w') as f:
        for res in batch_results:
            config = res['config']
            trial = res['trial']
            
            # Índices globales
            k_idx = np.where(K_INTRA_VALUES == config['k_intra'])[0][0]
            r_idx = np.where(K_INTER_RATIOS == config['k_inter_ratio'])[0][0]
            d_idx = np.where(DELAY_VALUES == config['delay'])[0][0]
            global_idx = k_idx * len(K_INTER_RATIOS) * len(DELAY_VALUES) + \
                        r_idx * len(DELAY_VALUES) + d_idx
            
            grp = f.create_group(f"sim_{global_idx}_trial{trial}")
            
            # Attrs
            grp.attrs['k_intra'] = config['k_intra']
            grp.attrs['k_inter_ratio'] = config['k_inter_ratio']
            grp.attrs['delay_ms'] = config['delay']
            grp.attrs['trial'] = trial
            grp.attrs['k_intra_idx'] = k_idx
            grp.attrs['k_inter_ratio_idx'] = r_idx
            grp.attrs['delay_idx'] = d_idx
            
            # Spikes
            grp.create_dataset('spikes_A_times', data=res['spikes_A']['times'], 
                             dtype='float32', compression='gzip')
            grp.create_dataset('spikes_A_neurons', data=res['spikes_A']['neurons'], 
                             dtype='int16', compression='gzip')
            grp.create_dataset('spikes_B_times', data=res['spikes_B']['times'], 
                             dtype='float32', compression='gzip')
            grp.create_dataset('spikes_B_neurons', data=res['spikes_B']['neurons'], 
                             dtype='int16', compression='gzip')
            
            # Métricas
            metrics_grp = grp.create_group('metrics')
            for pop, metrics in [('A', res['metrics_A']), ('B', res['metrics_B'])]:
                if metrics is None:
                    continue
                
                pop_grp = metrics_grp.create_group(pop)  
                pop_grp.attrs['tau_int']       = metrics['tau_int']       # legacy
                pop_grp.attrs['ac_peak']       = metrics['ac_peak']       # legacy
                pop_grp.attrs['mean_rate']     = metrics['mean_rate']     # legacy
                pop_grp.attrs['quality']       = metrics['quality']       # legacy
                pop_grp.attrs['tau_int_lfp']   = metrics['tau_int_lfp']   # nuevo
                pop_grp.attrs['freq_dominant'] = metrics['freq_dominant'] # nuevo

                pop_grp.create_dataset('ac_lags', data=metrics['ac_lags'], dtype='float32')
                pop_grp.create_dataset('ac_corr', data=metrics['ac_corr'], dtype='float32')

                # Solo guardar trazas LFP si están disponibles
                if not np.isnan(metrics['tau_int_lfp']):
                    pop_grp.create_dataset('lfp_mean',    data=metrics['lfp_mean'],  dtype='float32', compression='gzip')
                    pop_grp.create_dataset('psd_freqs',   data=metrics['psd_freqs'], dtype='float32')
                    pop_grp.create_dataset('psd_power',   data=metrics['psd_power'], dtype='float32')
                    pop_grp.create_dataset('ac_lags_lfp', data=metrics['ac_lags_lfp'], dtype='float32')
                    pop_grp.create_dataset('ac_corr_lfp', data=metrics['ac_corr_lfp'], dtype='float32')


def update_3d_arrays(batch_results, arrays_3d):
    """Actualiza arrays 3D - SOLO tau_int."""
    
    config_accumulator = {}
    
    for res in batch_results:
        config = res['config']
        config_key = (config['k_intra'], config['k_inter_ratio'], config['delay'])
        
        if config_key not in config_accumulator:
            config_accumulator[config_key] = {'A': [], 'B': []}
        
        if res['metrics_A'] is not None:
            config_accumulator[config_key]['A'].append(res['metrics_A'])
        if res['metrics_B'] is not None:
            config_accumulator[config_key]['B'].append(res['metrics_B'])
    
    # Promediar
    for config_key, metrics_dict in config_accumulator.items():
        k_intra, k_inter_ratio, delay = config_key
        k_idx = np.where(K_INTRA_VALUES == k_intra)[0][0]
        r_idx = np.where(K_INTER_RATIOS == k_inter_ratio)[0][0]
        d_idx = np.where(DELAY_VALUES == delay)[0][0]
        
        for pop in ['A', 'B']:
            if len(metrics_dict[pop]) == 0:
                continue
            trials_data = metrics_dict[pop]
            tau_ints    = [m['tau_int']        for m in trials_data if m is not None]
            freqs       = [m['freq_dominant']  for m in trials_data if m is not None]
            rates       = [m['mean_rate']      for m in trials_data if m is not None]

            # Añadir junto a las otras líneas de actualización dentro del bucle for pop:
            tau_ints_lfp = [m['tau_int_lfp'] for m in trials_data if m is not None]
            arrays_3d[f'tau_int_lfp_{pop}'][k_idx, r_idx, d_idx]     = np.nanmean(tau_ints_lfp)
            arrays_3d[f'tau_int_lfp_{pop}_std'][k_idx, r_idx, d_idx] = np.nanstd(tau_ints_lfp)
            arrays_3d[f'tau_int_{pop}'][k_idx, r_idx, d_idx]      = np.nanmean(tau_ints)
            arrays_3d[f'tau_int_{pop}_std'][k_idx, r_idx, d_idx]  = np.nanstd(tau_ints)
            arrays_3d[f'freq_dominant_{pop}'][k_idx, r_idx, d_idx]     = np.nanmean(freqs)
            arrays_3d[f'freq_dominant_{pop}_std'][k_idx, r_idx, d_idx] = np.nanstd(freqs)
            arrays_3d[f'mean_rate_{pop}'][k_idx, r_idx, d_idx]    = np.nanmean(rates)


def save_checkpoint(batch_idx, arrays_3d, output_dir):
    """Guardar checkpoint."""
    checkpoint_path = output_dir / f"checkpoint_batch{batch_idx}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'batch_idx': batch_idx, 'arrays_3d': arrays_3d}, f)
    logger.info(f"Checkpoint saved: batch {batch_idx}")

# =============================================================================
# FUNCIONES DE PLOTTING PROGRESIVO
# =============================================================================

# def plot_progress_heatmaps(arrays_3d, delay_idx, output_dir, batch_idx):
#     """Plot 2D heatmaps - SOLO tau_int."""
    
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     delay_val = DELAY_VALUES[delay_idx]
    
#     # tau_int A
#     ax = axes[0]
#     data = arrays_3d['tau_int_A'][:, :, delay_idx]
#     im = ax.imshow(data.T, origin='lower', aspect='auto', cmap='viridis',
#                    extent=[K_INTRA_VALUES[0], K_INTRA_VALUES[-1],
#                            K_INTER_RATIOS[0], K_INTER_RATIOS[-1]])
#     ax.set_xlabel('K_intra', fontsize=12)
#     ax.set_ylabel('K_inter_ratio', fontsize=12)
#     ax.set_title(f'τ_int Pop A (delay={delay_val}ms)', fontsize=13, weight='bold')
#     plt.colorbar(im, ax=ax, label='Timescale (ms)')
    
#     # tau_int B
#     ax = axes[1]
#     data = arrays_3d['tau_int_B'][:, :, delay_idx]
#     im = ax.imshow(data.T, origin='lower', aspect='auto', cmap='viridis',
#                    extent=[K_INTRA_VALUES[0], K_INTRA_VALUES[-1],
#                            K_INTER_RATIOS[0], K_INTER_RATIOS[-1]])
#     ax.set_xlabel('K_intra', fontsize=12)
#     ax.set_ylabel('K_inter_ratio', fontsize=12)
#     ax.set_title(f'τ_int Pop B (delay={delay_val}ms)', fontsize=13, weight='bold')
#     plt.colorbar(im, ax=ax, label='Timescale (ms)')
    
#     plt.suptitle(f'Progress: Batch {batch_idx}', fontsize=15, weight='bold')
#     plt.tight_layout()
    
#     save_path = output_dir / "plots_progress" / f"heatmap_delay{delay_val}ms_batch{batch_idx}.png"
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.show()
    
#     logger.info(f"Progress plot saved: {save_path.name}")


# def plot_kinter_vs_tau(arrays_3d, output_dir):
#     """Heatmap K_inter_ratio vs Delay."""
    
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     for idx, pop in enumerate(['A', 'B']):
#         ax = axes[idx]
#         data = np.nanmean(arrays_3d[f'tau_int_{pop}'], axis=0)
        
#         im = ax.imshow(data.T, origin='lower', aspect='auto', cmap='viridis',
#                        extent=[K_INTER_RATIOS[0], K_INTER_RATIOS[-1],
#                                DELAY_VALUES[0], DELAY_VALUES[-1]])
#         ax.set_xlabel('K_inter_ratio', fontsize=12, weight='bold')
#         ax.set_ylabel('Delay (ms)', fontsize=12, weight='bold')
#         ax.set_title(f'τ_int Pop {pop} (avg over K_intra)', fontsize=13, weight='bold')
#         plt.colorbar(im, ax=ax, label='Timescale (ms)')
    
#     plt.suptitle('K_inter_ratio vs Delay', fontsize=15, weight='bold')
#     plt.tight_layout()
    
#     save_path = output_dir / "kinter_vs_delay_tau.png"
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     plt.show()
    
#     logger.success(f"K_inter vs tau plot saved")

def plot_progress_heatmaps(arrays_3d, delay_idx, output_dir, batch_idx):
    """Plot 2D heatmaps de tau_int para un delay dado."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    delay_val = DELAY_VALUES[delay_idx]

    n_kintra = len(K_INTRA_VALUES)
    n_ratio  = len(K_INTER_RATIOS)
    xticks   = range(n_kintra)
    yticks   = np.linspace(0, n_ratio - 1, 5).astype(int)

    for ax, pop in zip(axes, ['A', 'B']):
        data = arrays_3d[f'tau_int_{pop}'][:, :, delay_idx]
        im = ax.imshow(data.T, origin='lower', aspect='auto', cmap='viridis')

        # Eje X: K_intra (no uniforme → ticks explícitos)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{v:.2f}' for v in K_INTRA_VALUES],
                           rotation=45, fontsize=8)
        # Eje Y: K_inter_ratio (uniforme → linspace de etiquetas)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{K_INTER_RATIOS[i]:.2f}' for i in yticks])

        ax.set_xlabel('K_intra', fontsize=12)
        ax.set_ylabel('K_inter_ratio', fontsize=12)
        ax.set_title(f'τ_int Pop {pop} (delay={delay_val:.1f}ms)',
                     fontsize=13, weight='bold')
        plt.colorbar(im, ax=ax, label='Timescale (ms)')

    plt.suptitle(f'Progress: Batch {batch_idx}', fontsize=15, weight='bold')
    plt.tight_layout()

    save_path = output_dir / "plots_progress" / \
                f"heatmap_delay{delay_val:.1f}ms_batch{batch_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Progress plot saved: {save_path.name}")


def plot_kinter_vs_tau(arrays_3d, output_dir):
    """Heatmap K_inter_ratio vs Delay (promediado sobre K_intra)."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_ratio = len(K_INTER_RATIOS)
    n_delay = len(DELAY_VALUES)
    # K_inter_ratio es uniforme → extent directo
    # Delay es uniforme → extent directo
    ext = [K_INTER_RATIOS[0], K_INTER_RATIOS[-1],
           DELAY_VALUES[0],   DELAY_VALUES[-1]]

    for ax, pop in zip(axes, ['A', 'B']):
        data = np.nanmean(arrays_3d[f'tau_int_{pop}'], axis=0)  # (n_ratio, n_delay)
        im = ax.imshow(data.T, origin='lower', aspect='auto',
                       cmap='viridis', extent=ext)
        ax.set_xlabel('K_inter_ratio', fontsize=12, weight='bold')
        ax.set_ylabel('Delay (ms)',    fontsize=12, weight='bold')
        ax.set_title(f'τ_int Pop {pop} (avg over K_intra)',
                     fontsize=13, weight='bold')
        plt.colorbar(im, ax=ax, label='Timescale (ms)')

    plt.suptitle('K_inter_ratio vs Delay', fontsize=15, weight='bold')
    plt.tight_layout()

    save_path = output_dir / "kinter_vs_delay_tau.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.success(f"K_inter vs tau plot saved")
    
# =============================================================================
# EJECUTAR BARRIDO CON MULTIPROCESSING
# =============================================================================
import multiprocessing
from multiprocessing import Pool

N_JOBS = SIM_CONFIG['n_jobs']

def run_task_wrapper(params):
    """Wrapper para multiprocessing."""
    config, trial = params
    try:
        return run_single_simulation(config, trial)
    except Exception as e:
        logger.error(f"Error in {config}, trial {trial}: {e}")
        return None


def run_sweep():
    """Barrido 3D con multiprocessing, limpieza de RAM y Resume."""
    
    n_batches = int(np.ceil(n_configs / SIM_CONFIG['batch_size']))
    
    logger.info(f"Starting sweep: {n_batches} batches, {N_JOBS} cores")
    logger.info(f"Total simulations: {n_total_sims}")
    
    # 🛑 FIX 1: Reiniciar workers cada 10 tareas para purgar RAM
    with Pool(N_JOBS, maxtasksperchild=50) as pool:
        
        pbar = tqdm(total=n_batches, desc="Batches")
        
        for batch_start in range(0, n_configs, SIM_CONFIG['batch_size']):
            
            # Índice real del batch basado en la posición actual
            current_batch_idx = batch_start // SIM_CONFIG['batch_size']
            
            # --- 🛑 FIX 2: LOGICA DE RESUME ---
            h5_path = output_dir / "raw_spikes" / f"batch_{current_batch_idx:04d}.h5"
            
            if h5_path.exists():
                try:
                    # Chequeo integridad básico
                    with h5py.File(h5_path, 'r') as f: pass
                    logger.info(f"⏭️ Batch {current_batch_idx} ya existe. Saltando...")
                    pbar.update(1)
                    continue # <--- SALTA AL SIGUIENTE BUCLE
                except:
                    logger.warning(f"⚠️ Batch {current_batch_idx} corrupto. Recalculando...")
                    h5_path.unlink()
            # ---------------------------------

            batch_end = min(batch_start + SIM_CONFIG['batch_size'], n_configs)
            batch_configs = configs[batch_start:batch_end]
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Batch {current_batch_idx}: configs {batch_start}-{batch_end}")
            
            # Crear tareas
            tasks = [(config, trial) for config in batch_configs
                     for trial in range(SIM_CONFIG['n_trials'])]
            
            # Ejecutar lote
            batch_results = list(tqdm(
                pool.imap(run_task_wrapper, tasks),
                total=len(tasks),
                desc=f"Run Batch {current_batch_idx}",
                leave=False
            ))
            
            # Filtrar errores
            batch_results = [r for r in batch_results if r is not None]
            
            if len(batch_results) > 0:
                # Guardar con el índice correcto
                save_batch_to_hdf5(batch_results, current_batch_idx, output_dir)
                update_3d_arrays(batch_results, arrays_3d)
            
            # Checkpoint Arrays
            if (current_batch_idx + 1) % SIM_CONFIG['checkpoint_every'] == 0:
                save_checkpoint(current_batch_idx, arrays_3d, output_dir)
                
            # Plot
            if (current_batch_idx + 1) % SIM_CONFIG['plot_every'] == 0:
                delay_idx_0 = 0
                plot_progress_heatmaps(arrays_3d, delay_idx_0, output_dir, current_batch_idx)
                
            pbar.update(1)
            
        pbar.close()
    
    # Guardar final
    for key, arr in arrays_3d.items():
        np.save(output_dir / "metrics_3d" / f"{key}.npy", arr)
    
    logger.success("Sweep completed!")
    return arrays_3d

# EJECUTAR
arrays_3d = run_sweep()

output_dir

# =============================================================================
# GUARDAR ARRAYS 3D FINALES
# =============================================================================
# output_dir = Path(f"./results/sweep_3d_autocorr_20251202_223002")
for name, array in arrays_3d.items():
    np.save(output_dir / "metrics_3d" / f"{name}.npy", array)

logger.success(f"3D arrays saved in {output_dir / 'metrics_3d'}")
