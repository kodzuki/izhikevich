#!/usr/bin/env python
# coding: utf-8

"""
Script Definitivo para Barrido 3D en Cluster HPC
Configuración: 50 (K_intra) x 50 (Ratio) x 5 (Delay)
Duración: 4000ms | Trials: 5
Estimación Datos: ~12-15 GB (Comprimidos)
"""

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

import argparse
from pathlib import Path
import numpy as np
import matplotlib
# 🛑 2. CRÍTICO: BACKEND SIN VENTANAS
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import h5py
import json
import pickle
import datetime
from tqdm.auto import tqdm
from multiprocessing import Pool
import warnings

# Silenciar advertencias no críticas
warnings.filterwarnings("ignore")

from brian2 import *

# =============================================================================
# SETUP DE RUTAS (ROBUSTO)
# =============================================================================
current_file = Path(__file__).resolve()
# Intentamos subir niveles hasta encontrar la carpeta 'src'
project_root = None
for parent in [current_file.parent] + list(current_file.parents):
    if (parent / 'src').exists():
        project_root = parent
        break

if project_root and str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.two_populations.model import IzhikevichNetwork
    from src.two_populations.metrics import (
        spikes_to_population_rate,
        cross_correlation_analysis,
        intrinsic_timescale_analysis
    )
    from src.two_populations.helpers.logger import setup_logger
except ImportError:
    # Fallback de emergencia
    sys.path.append(os.getcwd())
    from src.two_populations.model import IzhikevichNetwork
    from src.two_populations.metrics import (
        spikes_to_population_rate,
        cross_correlation_analysis,
        intrinsic_timescale_analysis
    )
    from src.two_populations.helpers.logger import setup_logger

# =============================================================================
# ARGUMENTOS & DIRECTORIOS
# =============================================================================
parser = argparse.ArgumentParser(description='Sweep 3D Production')
parser.add_argument('--n_jobs', type=int, default=4, help='Cores (CPUs) a utilizar')
parser.add_argument('--batch_size', type=int, default=16, help='Simulaciones por archivo HDF5')
parser.add_argument('--output_dir', type=str, default=None, help='Directorio de salida')
args = parser.parse_args()

# Setup Directorios
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    output_dir = Path(f"./results/sweep_3d_prod_{timestamp}")

for d in ["raw_spikes", "metrics_3d", "plots_progress", "casos_particulares", "diagnostics"]:
    (output_dir / d).mkdir(parents=True, exist_ok=True)

logger = setup_logger("sweep_prod", results_dir=output_dir, log_to_file=True)
logger.info(f"🚀 INICIO BARRIDO PRODUCCIÓN. Grid 50x50x5. Cores: {args.n_jobs}")

# =============================================================================
# CONFIGURACIÓN CIENTÍFICA (DEFINITIVA)
# =============================================================================

# 1. Definición de la Grilla (CORREGIDO: Usando linspace para puntos exactos)
# Queremos 50 puntos entre 0.1 y 25.0
K_INTRA_VALUES = np.linspace(0.1, 25.0, 2)  
# Queremos 50 puntos entre 0.0 y 1.0
K_INTER_RATIOS = np.linspace(0.0, 1.0, 2)   
# Queremos 5 delays representativos (ej: 0, 5, 10, 20, 40 ms)
# O si prefieres lineal: np.linspace(0, 40, 5) -> [0, 10, 20, 30, 40]
DELAY_VALUES = np.linspace(0.0, 125.0, 2) # Ajusta según preferencia

POPULATION_PARAMS = {
    'Ne': 800, 'Ni': 200,
    'noise_exc': 0.88, 'noise_inh': 0.6,
    'p_intra': 0.1, 'p_inter': 0.02,
    'delay_intra': 0.0,
    'rate_hz': 10.0,
    'stim_base': 1.0
}

SIM_CONFIG = {
    'dt_ms': 0.1,
    'T_ms': 3000,       # Aumentado a 4s como pediste
    'warmup_ms': 500,  # Aumentamos warmup a 1s para asegurar estado estable
    'n_trials': 2,      # 5 Trials por punto
    'checkpoint_every': 20,  # Guardar arrays cada X batches
    'plot_every': 20    # Plotear progreso cada X batches
}

AC_CONFIG = {
    'max_lag_ms': 200,
    'analysis_dt': 0.5,
}

# Generar lista de configuraciones
configs = []
for k_intra in K_INTRA_VALUES:
    for k_inter_ratio in K_INTER_RATIOS:
        for delay in DELAY_VALUES:
            configs.append({
                'k_intra': float(k_intra),
                'k_inter_ratio': float(k_inter_ratio),
                'delay': float(delay)
            })

n_configs = len(configs)
n_total_sims = n_configs * SIM_CONFIG['n_trials']

logger.info(f"📊 Estadísticas del Grid:")
logger.info(f"   K_Intra: {len(K_INTRA_VALUES)} puntos [{K_INTRA_VALUES[0]:.2f} - {K_INTRA_VALUES[-1]:.2f}]")
logger.info(f"   Ratio:   {len(K_INTER_RATIOS)} puntos [{K_INTER_RATIOS[0]:.2f} - {K_INTER_RATIOS[-1]:.2f}]")
logger.info(f"   Delays:  {len(DELAY_VALUES)} puntos {DELAY_VALUES}")
logger.info(f"   Total Configs: {n_configs}")
logger.info(f"   Total Sims:    {n_total_sims} ({SIM_CONFIG['n_trials']} trials/config)")

# Guardar Metadata
with open(output_dir / "config.json", 'w') as f:
    json.dump({
        'K_INTRA_VALUES': K_INTRA_VALUES.tolist(),
        'K_INTER_RATIOS': K_INTER_RATIOS.tolist(),
        'DELAY_VALUES': DELAY_VALUES.tolist(),
        'POPULATION_PARAMS': POPULATION_PARAMS,
        'SIM_CONFIG': SIM_CONFIG
    }, f, indent=2)

# Inicializar Arrays 3D en Memoria
shape_3d = (len(K_INTRA_VALUES), len(K_INTER_RATIOS), len(DELAY_VALUES))
metrics_keys = ['tau_int_A', 'tau_int_B', 'ac_peak_A', 'ac_peak_B', 'mean_rate_A', 'mean_rate_B']
arrays_3d = {k: np.full(shape_3d, np.nan) for k in metrics_keys}
# Matriz auxiliar para llevar la cuenta de cuántos trials se han promediado en cada celda
arrays_count = np.zeros(shape_3d, dtype=int)

# =============================================================================
# FUNCIONES DEL WORKER
# =============================================================================

def compute_metrics(spike_times, spike_neurons, N, warmup, T_total):
    """Calcula métricas ligeras (Rates, Tau, AC)"""
    class DummySpikes:
        def __init__(self, t, i):
            self.t = t * ms
            self.i = i
            
    try:
        # 1. Rate
        time, rate = spikes_to_population_rate(
            DummySpikes(spike_times, spike_neurons), N, 
            smooth_window=1, analysis_dt=AC_CONFIG['analysis_dt'], T_total=T_total
        )
        
        # 2. Cut Warmup
        mask = time >= warmup
        rate_filt = rate[mask]
        
        if len(rate_filt) < 20: return None # Demasiado silencio
        
        # 3. Analysis
        ac = cross_correlation_analysis(rate_filt, rate_filt, max_lag_ms=AC_CONFIG['max_lag_ms'], dt=AC_CONFIG['analysis_dt'])
        ts = intrinsic_timescale_analysis(rate_filt, max_lag_ms=AC_CONFIG['max_lag_ms'], dt=AC_CONFIG['analysis_dt'])
        
        return {
            'tau_int': ts.get('tau_int', np.nan),
            'ac_peak': ac.get('peak_value', np.nan),
            'mean_rate': np.mean(rate_filt)
        }
    except Exception:
        return None

def run_single_simulation(args_task):
    """Ejecuta una simulación completa con PROTECCIÓN DE MEMORIA."""
    config, trial = args_task
    
    # 🛑 HACK: Desactivar Garbage Collector temporalmente
    # Esto evita que Python borre las sinapsis 'locales' de model.py
    # antes de que Brian2 pueda construir la red C++.
    gc.disable()
    
    # Reconfiguración Brian2
    prefs.codegen.target = 'numpy'
    start_scope()
    
    try:
        k_intra = config['k_intra']
        k_inter = config['k_inter_ratio'] * k_intra
        delay = config['delay']
        
        # Seed Determinista
        seed_val = int((k_intra*1000) + (config['k_inter_ratio']*100) + delay + (trial*10000))
        
        net = IzhikevichNetwork(
            dt_val=SIM_CONFIG['dt_ms'],
            T_total=SIM_CONFIG['T_ms'],
            fixed_seed=100,
            variable_seed=seed_val
        )
        
        # Crear Poblaciones
        for pop in ['A', 'B']:
            net.create_population2(
                pop, Ne=POPULATION_PARAMS['Ne'], Ni=POPULATION_PARAMS['Ni'],
                k_exc=k_intra, k_inh=k_intra*3.9,
                noise_exc=POPULATION_PARAMS['noise_exc'], noise_inh=POPULATION_PARAMS['noise_inh'],
                p_intra=POPULATION_PARAMS['p_intra'], delay=POPULATION_PARAMS['delay_intra'],
                rate_hz=POPULATION_PARAMS['rate_hz'], stim_base=POPULATION_PARAMS['stim_base']
            )
            
        # Conexión Inter
        if k_inter > 0:
            net.connect_populations('A', 'B', p_inter=POPULATION_PARAMS['p_inter'], 
                                  weight_scale=k_inter, delay_value=delay, delay_dist='constant')
            
        # Monitores
        net.setup_monitors(['A', 'B'], record_v_dt=None, sample_fraction=0, monitor_conductance=False)
        
        # Ejecutar Simulación
        results = net.run_simulation()
        
        output = {
            'config': config,
            'trial': trial,
            'data': {}
        }
        
        # Procesar resultados
        for pop in ['A', 'B']:
            spikes = results[pop]['spike_times']
            indices = results[pop]['spike_indices']
            
            metrics = compute_metrics(
                spikes, indices, 
                POPULATION_PARAMS['Ne'] + POPULATION_PARAMS['Ni'],
                SIM_CONFIG['warmup_ms'], SIM_CONFIG['T_ms']
            )
            
            output['data'][pop] = {
                'spikes': spikes,  
                'indices': indices, 
                'metrics': metrics
            }
            
        return output

    except Exception as e:
        logger.warning(f"Sim Error {config}: {e}")
        return {'error': str(e), 'config': config, 'trial': trial}
    
    finally:
        # 🛑 CRÍTICO: Reactivar GC y limpiar memoria
        gc.enable()
        gc.collect()

# =============================================================================
# GESTIÓN DE SALIDA & ACTUALIZACIÓN
# =============================================================================

def save_batch(batch_results, batch_idx):
    """Guarda HDF5 y actualiza matrices en memoria."""
    
    h5_path = output_dir / "raw_spikes" / f"batch_{batch_idx:04d}.h5"
    
    with h5py.File(h5_path, 'w') as f:
        for res in batch_results:
            if 'error' in res: continue
                
            cfg = res['config']
            trial = res['trial']
            
            # Índices
            ix = np.searchsorted(K_INTRA_VALUES, cfg['k_intra'])
            iy = np.searchsorted(K_INTER_RATIOS, cfg['k_inter_ratio'])
            iz = np.searchsorted(DELAY_VALUES, cfg['delay'])
            
            # Guardar en HDF5
            grp = f.create_group(f"sim_{ix}_{iy}_{iz}_trial{trial}")
            grp.attrs['k_intra'] = cfg['k_intra']
            grp.attrs['k_inter_ratio'] = cfg['k_inter_ratio']
            grp.attrs['delay'] = cfg['delay']
            
            for pop in ['A', 'B']:
                d = res['data'][pop]
                grp.create_dataset(f's_{pop}_t', data=d['spikes'], compression='gzip')
                grp.create_dataset(f's_{pop}_i', data=d['indices'], compression='gzip')
                
                # Actualizar Arrays en Memoria (Promedio Acumulativo Correcto)
                if d['metrics']:
                    m = d['metrics']
                    
                    # Usamos arrays_count para sincronizar
                    # Importante: Esto asume que actualizamos A y B al mismo tiempo
                    # Para evitar contar dos veces, incrementamos count solo una vez al final del loop de pops
                    # O usamos un flag. Aquí lo haremos simple:
                    
                    for key in metrics_keys:
                        # key ej: 'tau_int_A'
                        metric_name, pop_suffix = key.rsplit('_', 1) 
                        if pop_suffix == pop and metric_name in m:
                            val = m[metric_name]
                            if not np.isnan(val):
                                count = arrays_count[ix, iy, iz] # Cuenta actual ANTES de incrementar
                                # Nota: count aquí es compartido para A y B.
                                # Si es el primer trial (count=0), asignamos.
                                # Si no, promedio ponderado.
                                
                                # Para simplificar la lógica de A y B teniendo nans distintos:
                                # Usaremos la fórmula iterativa ignorando Nans.
                                # Pero 'arrays_count' es por celda (config).
                                # Asumimos que si la simulación corre, tenemos datos para A y B.
                                
                                curr_avg = arrays_3d[key][ix, iy, iz]
                                if np.isnan(curr_avg) or count == 0:
                                    arrays_3d[key][ix, iy, iz] = val
                                else:
                                    # Formula: new_avg = old_avg + (val - old_avg) / (count + 1)
                                    arrays_3d[key][ix, iy, iz] = curr_avg + (val - curr_avg) / (count + 1)
            
            # Incrementar contador de trials procesados para esta celda
            arrays_count[ix, iy, iz] += 1

def plot_status(batch_idx):
    """Heatmap rápido de progreso (Delay index 0)"""
    try:
        d_idx = 0
        data = arrays_3d['tau_int_A'][:, :, d_idx].T # Transponer para ejes correctos
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data, origin='lower', aspect='auto', cmap='viridis',
                      extent=[K_INTRA_VALUES[0], K_INTRA_VALUES[-1], 
                              K_INTER_RATIOS[0], K_INTER_RATIOS[-1]])
        plt.colorbar(im, label='Tau Int A (ms)')
        ax.set_title(f"Batch {batch_idx} | Delay {DELAY_VALUES[d_idx]}ms | Trials Avg")
        ax.set_xlabel("K Intra")
        ax.set_ylabel("K Inter Ratio")
        plt.savefig(output_dir / "plots_progress" / f"status_batch_{batch_idx:04d}.png")
        plt.close()
    except Exception:
        pass

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # Lista de tareas
    tasks = []
    for cfg in configs:
        for t in range(SIM_CONFIG['n_trials']):
            tasks.append((cfg, t))
            
    # Batches
    n_batches = int(np.ceil(len(tasks) / args.batch_size))
    
    logger.info(f"Procesando {len(tasks)} simulaciones en {n_batches} lotes.")

    with Pool(args.n_jobs) as pool:
        for i in tqdm(range(n_batches), desc="Processing Batches"):
            batch_tasks = tasks[i*args.batch_size : (i+1)*args.batch_size]
            
            results = list(pool.imap(run_single_simulation, batch_tasks))
            
            save_batch(results, i)
            
            if i % SIM_CONFIG['checkpoint_every'] == 0:
                # Guardar arrays npy
                for k, arr in arrays_3d.items():
                    np.save(output_dir / "metrics_3d" / f"{k}.npy", arr)
                np.save(output_dir / "metrics_3d" / "counts.npy", arrays_count)
                
            if i % SIM_CONFIG['plot_every'] == 0:
                plot_status(i)

    # Guardado Final
    logger.success("Guardando resultados finales...")
    for k, arr in arrays_3d.items():
        np.save(output_dir / "metrics_3d" / f"{k}_final.npy", arr)
    np.save(output_dir / "metrics_3d" / "counts_final.npy", arrays_count)
    
    logger.info("✅ PROCESO COMPLETADO")