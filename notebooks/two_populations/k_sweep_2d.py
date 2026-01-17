#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
# CRÍTICO: Backend 'Agg' para que no falle en el nodo sin pantalla
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from brian2 import *
from datetime import datetime
import json
import pickle
import gzip
from tqdm.auto import tqdm
from multiprocessing import Pool
import pandas as pd

# Ajusta el path para importar tus módulos src
# sys.path.append(os.path.abspath('../..')) 
from src.two_populations.model import IzhikevichNetwork
from src.two_populations.metrics import analyze_simulation_results
from src.two_populations.helpers.logger import setup_logger

# Logger limpio
logger = setup_logger("sweep_prod", console_level="INFO", log_to_file=False)

# ==========================================
# 1. CONFIGURACIÓN DEL BARRIDO
# ==========================================

# Grilla de Alta Densidad: 100 x 50 x 5 = 25.000 Sims
K_INTRA_VALUES = np.linspace(0.1, 25.0, 100)
K_INTER_RATIOS = np.linspace(0.0, 1.0, 50)

# Parámetros originales del notebook
POPULATION_PARAMS = {
    'Ne': 800, 'Ni': 200,
    'noise_exc': 0.88, 'noise_inh': 0.6,
    'p_intra': 0.1, 'delay': 0.0,
    'rate_hz': 10.0, 'stim_base': 1.0
}

SIM_CONFIG = {
    'dt_ms': 0.1,
    'T_ms': 3000,
    'warmup_ms': 750,  # 750ms de descarte
    'n_trials': 5,     # 5 trials por punto
    'fixed_seed': 42,
    'variable_seed_base': 500
}

# ==========================================
# 2. FUNCIONES MATEMÁTICAS (RECUPERADAS)
# ==========================================
# Estas funciones estaban en el notebook y se perdieron. Son vitales.

def compute_ei_ratio(spike_mon, N_exc, N_total, warmup_ms, T_total):
    """Calcula el ratio de firing rate Excitatorio vs Inhibitorio."""
    times = np.array(spike_mon.t/ms)
    indices = np.array(spike_mon.i)
    mask = (times >= warmup_ms) & (times < T_total)
    
    # Tiempo efectivo en segundos
    T_sec = (T_total - warmup_ms) / 1000.0
    if T_sec <= 0: return 0.0
    
    indices_filt = indices[mask]
    exc_spikes = np.sum(indices_filt < N_exc)
    inh_spikes = np.sum(indices_filt >= N_exc)
    
    exc_rate = exc_spikes / (N_exc * T_sec)
    inh_rate = inh_spikes / ((N_total - N_exc) * T_sec)
    
    return exc_rate / max(inh_rate, 0.01)

def compute_burst_ratio(spike_times, warmup_ms, T_total, bin_ms=50):
    """Calcula Burstiness (Std/Mean del histograma de actividad)."""
    mask = (spike_times >= warmup_ms) & (spike_times < T_total)
    spikes_filt = spike_times[mask]
    
    if len(spikes_filt) < 10: return np.nan
    
    bins = np.arange(warmup_ms, T_total, bin_ms)
    counts, _ = np.histogram(spikes_filt, bins)
    
    mean_counts = counts.mean()
    if mean_counts == 0: return 0.0
    return counts.std() / mean_counts

# ==========================================
# 3. WORKER (SIMULACIÓN + GUARDADO)
# ==========================================

def run_single_simulation(args):
    k_intra, k_inter_ratio, trial_idx = args
    
    # Recompilación segura para multiprocessing
    prefs.codegen.target = 'numpy'
    start_scope()
    
    k_inter = k_intra * k_inter_ratio
    
    # --- A. Configuración Red ---
    network = IzhikevichNetwork(
        dt_val=SIM_CONFIG['dt_ms'],
        T_total=SIM_CONFIG['T_ms'],
        fixed_seed=SIM_CONFIG['fixed_seed'],
        variable_seed=SIM_CONFIG['variable_seed_base'],
        trial=trial_idx
    )
    
    # Crear poblaciones y conexiones (Idéntico al notebook)
    network.create_population2('A', k_exc=k_intra, k_inh=k_intra*3.9, **POPULATION_PARAMS)
    network.create_population2('B', k_exc=k_intra, k_inh=k_intra*3.9, **POPULATION_PARAMS)
    
    network.connect_populations('A', 'B', p_inter=0.02, weight_scale=k_inter, delay_value=0.0)
    network.connect_populations('B', 'A', p_inter=0.02, weight_scale=k_inter, delay_value=0.0)
    
    # Monitores Ligeros: Solo spikes y 5% de LFP
    network.setup_monitors(['A', 'B'], record_v_dt=1.0, sample_fraction=0.05)
    
    results = network.run_simulation()
    
    try:
        raw_file_path = None
        
        # --- B. GUARDADO RAW (SOLO TRIAL 0) ---
        if trial_idx == 0:
            # Extraer datos ligeros (float32)
            raw_data = {
                't': np.array(results['A']['spike_monitor'].t/ms, dtype=np.float32),
                'i': np.array(results['A']['spike_monitor'].i, dtype=np.int32),
                # LFP Promedio
                'lfp_A': np.mean(results['A']['state_monitor'].v, axis=0).astype(np.float32),
                'lfp_B': np.mean(results['B']['state_monitor'].v, axis=0).astype(np.float32)
            }
            
            # Carpeta organizada
            save_dir = Path(f"./results/experiments/k_sweep_high_res/raw_data/k_{k_intra:.2f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"raw_r{k_inter_ratio:.3f}_t{trial_idx}.pkl.gz"
            full_path = save_dir / filename
            
            with gzip.open(full_path, 'wb') as f:
                pickle.dump(raw_data, f)
            
            raw_file_path = str(full_path)

        # --- C. ANÁLISIS MÉTRICAS (COMPLETE) ---
        
        # 1. Métricas estándar (src module)
        conn = analyze_simulation_results(
            results['A']['spike_monitor'], results['B']['spike_monitor'],
            1000, "dummy", warmup=SIM_CONFIG['warmup_ms'],
            state_monitors={'A': network.monitors['A'], 'B': network.monitors['B']},
            signal_mode='lfp',
            plotting=False 
        )
        
        # 2. Métricas Extra (¡AQUÍ ESTÁ LO QUE FALTABA!)
        spikes_t = np.array(results['A']['spike_monitor'].t/ms)
        
        ei_ratio = compute_ei_ratio(results['A']['spike_monitor'], 
                                  POPULATION_PARAMS['Ne'], 
                                  POPULATION_PARAMS['Ne']+POPULATION_PARAMS['Ni'],
                                  SIM_CONFIG['warmup_ms'], SIM_CONFIG['T_ms'])
                                  
        burstiness = compute_burst_ratio(spikes_t, 
                                       SIM_CONFIG['warmup_ms'], SIM_CONFIG['T_ms'])

        return {
            'k_intra': k_intra,
            'k_inter_ratio': k_inter_ratio,
            'k_inter': k_inter,
            'trial': trial_idx,
            # --- Métricas Clave ---
            'rate_A': np.mean(conn['time_series']['fr_A']),
            'beta_A': conn['power_A']['beta_power'],
            'gamma_A': conn['power_A']['gamma_power'],
            'cc_peak': conn['cross_corr_peak'],
            'plv_alpha': conn['plv_alpha'],
            'coherence_peak': conn['coherence_peak'],
            # --- Métricas Restauradas ---
            'ei_ratio': ei_ratio,     
            'burstiness': burstiness, 
            # --- Metadata ---
            'raw_path': raw_file_path
        }
        
    except Exception as e:
        return {
            'k_intra': k_intra, 'k_inter_ratio': k_inter_ratio, 'trial': trial_idx,
            'error': str(e)
        }

# ==========================================
# 4. EJECUCIÓN PARALELA
# ==========================================

if __name__ == "__main__":
    
    tasks = [
        (k, r, t) 
        for k in K_INTRA_VALUES 
        for r in K_INTER_RATIOS 
        for t in range(SIM_CONFIG['n_trials'])
    ]
    
    print(f"🚀 Iniciando Barrido Producción: {len(tasks)} simulaciones")
    print(f"   Grid: {len(K_INTRA_VALUES)}x{len(K_INTER_RATIOS)}")
    print(f"   Incluye: CSV completo + RAW Data (Trial 0)")
    
    n_cores = 32
    with Pool(n_cores) as pool:
        # chunksize=10 para optimizar el paso de tareas
        results = list(tqdm(pool.imap(run_single_simulation, tasks, chunksize=10), total=len(tasks)))
    
    # --- Guardado Final ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path("./results/experiments/k_sweep_high_res")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # 1. CSV Maestro (Todas las trials)
    csv_path = out_dir / f"sweep_results_raw_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # 2. CSV Agregado (Promedios)
    try:
        # Agrupamos por condición para tener medias y desviaciones
        aggregated = df.groupby(['k_intra', 'k_inter_ratio']).agg({
            'rate_A': ['mean', 'std'],
            'beta_A': ['mean', 'std'],
            'cc_peak': ['mean', 'std'],
            'plv_alpha': ['mean', 'std'],
            'ei_ratio': ['mean', 'std'],     # <--- Ahora sí podemos agregar esto
            'burstiness': ['mean', 'std']    # <--- Y esto
        }).reset_index()
        
        aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
        agg_path = out_dir / f"sweep_results_aggregated_{timestamp}.csv"
        aggregated.to_csv(agg_path, index=False)
        print(f"✅ CSV Agregado generado correctamente.")
        
    except Exception as e:
        print(f"⚠️ Aviso: No se pudo generar el CSV agregado ({e}). Revisa el Raw.")
        
    print(f"\n✅ PROCESO COMPLETADO.")
    print(f"   Datos Raw: {csv_path}")