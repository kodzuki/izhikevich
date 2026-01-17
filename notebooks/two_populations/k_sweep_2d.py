#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
# CRITICO: Backend no interactivo para cluster
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from brian2 import *
from datetime import datetime
import json
import pickle
from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Pool
import pandas as pd

# Ajustar path si es necesario
os.chdir('../..') 

from src.two_populations.model import IzhikevichNetwork
from src.two_populations.metrics import analyze_simulation_results, print_network_statistics_table
from src.two_populations.helpers.logger import setup_logger

# Configuración de Logging minimalista para no llenar el disco
logger = setup_logger(
    experiment_name="k_sweep_high_res",
    console_level="INFO",
    file_level="WARNING",
    log_to_file=True
)

# =============================================================================
# 1. CONFIGURACIÓN DEL BARRIDO (ESCALADO A 16k)
# =============================================================================

POPULATION_PARAMS = {
    'Ne': 800, 'Ni': 200,
    'noise_exc': 0.88, 'noise_inh': 0.6,
    'p_intra': 0.1, 'delay': 0.0,
    'rate_hz': 10.0,
    'stim_base': 1.0
}

# ALTA DENSIDAD: 80x40x5 = 16,000 simulaciones
# Tiempo estimado en 32 cores: ~3-4 horas (si cada sim tarda ~20s)
K_INTRA_VALUES = np.linspace(0.1, 25.0, 80)     # Mayor rango y densidad
K_INTER_RATIOS = np.linspace(0.0, 1.0, 40)      # Densidad fina en el coupling

SIM_CONFIG = {
    'dt_ms': 0.1,
    'T_ms': 3000,
    'warmup_ms': 1000, # Subimos warmup a 1s para asegurar estabilidad
    'n_trials': 5,
    'fixed_seed': 42,
    'variable_seed_base': 500
}

# =============================================================================
# 2. HELPER FUNCTIONS (Optimizadas)
# =============================================================================

def compute_ei_ratio(spike_mon, N_exc, N_total, warmup_ms, T_total):
    times = np.array(spike_mon.t/ms)
    indices = np.array(spike_mon.i)
    mask = (times >= warmup_ms) & (times < T_total)
    times_filt = times[mask]
    indices_filt = indices[mask]
    T_analysis = (T_total - warmup_ms) / 1000
    
    if T_analysis <= 0: return 0
    
    exc_rate = np.sum(indices_filt < N_exc) / (N_exc * T_analysis)
    inh_rate = np.sum(indices_filt >= N_exc) / ((N_total - N_exc) * T_analysis)
    return exc_rate / max(inh_rate, 0.01)

def compute_burst_ratio(spike_times, warmup_ms, T_total, bin_ms=50):
    spike_times_filt = spike_times[(spike_times >= warmup_ms) & (spike_times < T_total)]
    if len(spike_times_filt) < 10: return np.nan
    bins = np.arange(warmup_ms, T_total, bin_ms)
    counts, _ = np.histogram(spike_times_filt, bins)
    return counts.std() / max(counts.mean(), 0.01)

def compute_cv_isi(spike_monitor, warmup_ms=500):
    spike_times = np.array(spike_monitor.t/ms)
    spike_indices = np.array(spike_monitor.i)
    valid_mask = spike_times >= warmup_ms
    spike_times = spike_times[valid_mask]
    spike_indices = spike_indices[valid_mask]
    
    if len(spike_times) < 50: return np.nan # Filtro rápido
    
    # Vectorizado (más rápido que bucle puro si es posible, pero mantenemos lógica segura)
    cvs = []
    unique_neurons = np.unique(spike_indices)
    # Muestreo si hay demasiadas neuronas para ahorrar CPU
    if len(unique_neurons) > 200: 
        unique_neurons = np.random.choice(unique_neurons, 200, replace=False)
        
    for neuron_id in unique_neurons:
        neuron_spikes = spike_times[spike_indices == neuron_id]
        if len(neuron_spikes) >= 3:
            isis = np.diff(neuron_spikes)
            m = np.mean(isis)
            if m > 0: cvs.append(np.std(isis) / m)
            
    return np.mean(cvs) if cvs else np.nan

# =============================================================================
# 3. CORE SIMULATION
# =============================================================================

def run_single_simulation(k_intra, k_inter_ratio, trial_idx):
    # start_scope() es crucial en multiprocessing
    start_scope()
    
    # Brian2 settings para velocidad
    prefs.codegen.target = 'numpy' # O 'cython' si está configurado, pero numpy es más seguro en paralelo
    
    k_inter = k_intra * k_inter_ratio
    
    network = IzhikevichNetwork(
        dt_val=SIM_CONFIG['dt_ms'],
        T_total=SIM_CONFIG['T_ms'],
        fixed_seed=SIM_CONFIG['fixed_seed'],
        variable_seed=SIM_CONFIG['variable_seed_base'],
        trial=trial_idx
    )
    
    # Setup Network
    pop_A = network.create_population2('A', k_exc=k_intra, k_inh=k_intra * 3.9, **POPULATION_PARAMS)
    pop_B = network.create_population2('B', k_exc=k_intra, k_inh=k_intra * 3.9, **POPULATION_PARAMS)
    
    network.connect_populations('A', 'B', p_inter=0.02, weight_scale=k_inter, delay_value=0.0)
    network.connect_populations('B', 'A', p_inter=0.02, weight_scale=k_inter, delay_value=0.0)
    
    # Monitores Ligeros (Solo spikes para rate, state monitor solo si es necesario para LFP)
    # Reducimos sample_fraction para ahorrar memoria
    network.setup_monitors(['A', 'B'], record_v_dt=0.5, sample_fraction=0.1) 
    
    results = network.run_simulation()
    
    try:
        # Análisis numérico (sin plots)
        conn = analyze_simulation_results(
            results['A']['spike_monitor'],
            results['B']['spike_monitor'],
            1000,
            f"sim", # Nombre dummy, no guardamos plots
            warmup=SIM_CONFIG['warmup_ms'],
            state_monitors={'A': network.monitors['A'], 'B': network.monitors['B']},
            signal_mode='lfp',
            plotting=False # ¡IMPORTANTE! Modifica tu función analyze para aceptar este flag si existe, o ignora los plots
        )
        
        # Extracción de métricas
        cv_A = compute_cv_isi(results['A']['spike_monitor'], SIM_CONFIG['warmup_ms'])
        ei_ratio_A = compute_ei_ratio(results['A']['spike_monitor'], 
                                    POPULATION_PARAMS['Ne'], 
                                    POPULATION_PARAMS['Ne'] + POPULATION_PARAMS['Ni'],
                                    SIM_CONFIG['warmup_ms'], SIM_CONFIG['T_ms'])
        burst_A = compute_burst_ratio(np.array(results['A']['spike_monitor'].t/ms),
                                    SIM_CONFIG['warmup_ms'], SIM_CONFIG['T_ms'])

        # NOTA: He eliminado toda la sección de PLOTS. 
        # Razón: Generar 16,000 PNGs colapsará el I/O del cluster.
        # Guardaremos los datos y plotearás los 5 mejores casos luego.

        return {
            'k_intra': k_intra,
            'k_inter': k_inter,
            'k_inter_ratio': k_inter_ratio,
            'trial': trial_idx,
            'mean_rate_A': np.mean(conn['time_series']['fr_A']),
            'mean_rate_B': np.mean(conn['time_series']['fr_B']),
            'cv_A': cv_A,
            'beta_power_A': conn['power_A']['beta_power'],
            'alpha_power_A': conn['power_A']['alpha_power'],
            'gamma_power_A': conn['power_A']['gamma_power'],
            'cc_peak': conn['cross_corr_peak'],
            'cc_lag': conn['cross_corr_lag'],
            'plv_alpha': conn['plv_alpha'],
            'coherence_peak': conn['coherence_peak'],
            'ei_ratio_A': ei_ratio_A,
            'burst_ratio_A': burst_A
        }
            
    except Exception as e:
        # Retorno seguro en caso de fallo numérico
        return {
            'k_intra': k_intra, 'k_inter_ratio': k_inter_ratio, 'trial': trial_idx,
            'mean_rate_A': np.nan, 'cv_A': np.nan, 'cc_peak': np.nan, 'plv_alpha': np.nan
        }

# Wrapper para multiprocessing
def run_task_wrapper(params):
    return run_single_simulation(*params)

# =============================================================================
# 4. EXECUTION & SAVING
# =============================================================================

if __name__ == "__main__":
    
    # Generar tareas
    tasks = [
        (k_intra, k_inter_ratio, trial)
        for k_intra in K_INTRA_VALUES
        for k_inter_ratio in K_INTER_RATIOS
        for trial in range(SIM_CONFIG['n_trials'])
    ]
    
    print(f"🚀 Iniciando Barrido Masivo: {len(tasks)} simulaciones")
    print(f"   Grid: {len(K_INTRA_VALUES)}x{len(K_INTER_RATIOS)} Intra/Inter")
    print(f"   Cores: 32 (aprox)")

    # Ejecución Paralela
    N_JOBS = 32 # Usar todos los cores del nodo
    with Pool(N_JOBS) as pool:
        # Chunksize mayor para reducir overhead en 16k tareas
        all_results = list(tqdm(pool.imap(run_task_wrapper, tasks, chunksize=4), 
                               total=len(tasks), desc="Simulando"))

    # Procesamiento y Guardado
    print("💾 Guardando resultados...")
    
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./results/experiments/k_sweep_high_res")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Guardar CSV Crudo (Lo más importante)
    csv_path = output_dir / f"sweep_results_raw_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # 2. Guardar Agregado (Mean/Std)
    aggregated = df.groupby(['k_intra', 'k_inter_ratio']).agg(['mean', 'std']).reset_index()
    # Aplanar columnas
    aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
    agg_path = output_dir / f"sweep_results_aggregated_{timestamp}.csv"
    aggregated.to_csv(agg_path, index=False)
    
    # 3. Guardar Config JSON (Serialización simple manual para evitar errores)
    config = {
        'timestamp': timestamp,
        'k_intra_range': [K_INTRA_VALUES.min(), K_INTRA_VALUES.max(), len(K_INTRA_VALUES)],
        'k_inter_ratio_range': [K_INTER_RATIOS.min(), K_INTER_RATIOS.max(), len(K_INTER_RATIOS)],
        'population_params': POPULATION_PARAMS,
        'sim_config': SIM_CONFIG
    }
    with open(output_dir / f"config_{timestamp}.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Barrido completado.")
    print(f"   Raw Data: {csv_path}")
    print(f"   Aggregated: {agg_path}")