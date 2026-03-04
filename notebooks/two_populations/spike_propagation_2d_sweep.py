# # Análisis de Propagación de Spikes: Barrido 2D (K, rate_hz)
# 
# ## Objetivo
# 
# Estudiar las probabilidades de activación de vecinos y el firing rate en función de:
# - **K (acoplamiento recurrente)**: Factor de escalado de pesos sinápticos
# - **rate_hz (input externo)**: Tasa de estímulo talámico
# 
# ## Estrategia
# 
# 1. **Barrido 2D**: Simular todas las combinaciones (K, rate_hz) → obtener (FR, P, σ)
# 2. **Matching por FR**: Para cada (K≠0, FR_target), encontrar K=0 con FR≈FR_target
# 3. **Contribución de red**: ΔP = P_coupled - P_baseline (mismo FR, diferente origen)
# 4. **Visualización**: Heatmaps, cortes 1D, análisis de ΔP(K, FR)
# 
# ## Hipótesis
# 
# - FR ≈ a·rate_hz (relación casi lineal)
# - K=0 define actividad espúrea (baseline)
# - ΔP(K>0) captura la dinámica de red pura
# 
# ---

# ## 1. Setup y Configuración

# =============================================================================
# IMPORTS
# =============================================================================
import sys
import numpy as np
import os
import pickle
from pathlib import Path
from brian2 import *

# # Navegación al directorio raíz del proyecto

os.chdir('../..')
sys.path.insert(0, str(Path.cwd()))

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from loguru import logger
import gzip

# Imports del proyecto
from src.two_populations.model import IzhikevichNetwork
from src.two_populations.metrics import analyze_simulation_results
from src.two_populations.helpers.logger import setup_logger
import multiprocessing as mp

# Configurar logger
logger = setup_logger(
    experiment_name="spike_propagation_2d",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

logger.info(f"Working directory: {Path.cwd()}")
logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Estilo de plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# 🛑 1. CRÍTICO: CONFIGURACIÓN DE HILOS (ANTES DE CUALQUIER IMPORT NUMÉRICO)
# =============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ## 2. Parámetros del Barrido

# =============================================================================
# CONFIGURACIÓN DEL BARRIDO
# =============================================================================

# Tamaño de red
Ne = 800
Ni = 200

# Parámetros de simulación
SIM_CONFIG = {
    'dt_ms': 0.1,
    'T_ms': 4000,
    'warmup_ms': 500
}

# Parámetros fijos de red
NETWORK_PARAMS = {
    'Ne': Ne,
    'Ni': Ni,
    'noise_exc': 0.884,
    'noise_inh': 0.60,
    'p_intra': 0.1,
    'delay': 0.0,
    'stim_start_ms': None,
    'stim_duration_ms': SIM_CONFIG['T_ms'],
    'stim_base': 1.0,
    'stim_elevated': None
}


# Rango de parámetros a barrer
K_VALUES = np.linspace(0.0, 15.0, 25) #np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])  # 17 valores
RATE_HZ_VALUES = np.linspace(2.0, 25.0, 25) #np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20])  # 11 valores
N_TRIALS = 4
N_PROCESSES = 31

# Parámetros del análisis de propagación
PROPAGATION_CONFIG = {
    'window_ms': 4.0,         # Monosynaptic (correcto)
    'min_weight': 0.00,        # Peso mínimo para considerar conexión
    'min_spikes': 1,         # Mínimo de spikes para incluir neurona
}

# Seeds
FIXED_SEED = 100
VARIABLE_SEED = 200

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = Path('results/spike_propagation_2d') / f'sweep_2d_{timestamp}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"SWEEP OPTIMIZADO:")
logger.info(f"  K: {len(K_VALUES)} × rate: {len(RATE_HZ_VALUES)} × trials: {N_TRIALS}")
logger.info(f"  Total: {len(K_VALUES)*len(RATE_HZ_VALUES)*N_TRIALS} sims")
logger.info(f"  Tiempo estimado: ~{len(K_VALUES)*len(RATE_HZ_VALUES)*N_TRIALS*0.35:.0f}min")
logger.info(f"  Rate range: {RATE_HZ_VALUES.min():.1f}-{RATE_HZ_VALUES.max():.1f}Hz (FR matching extendido)")


# ## 3. Clase de Análisis de Propagación

# =============================================================================
# PROPAGATION ANALYZER
# =============================================================================

class PropagationAnalyzer:
    """
    Analiza propagación forward E→E:
    Cuando neurona i dispara, ¿cuántos vecinos j responden en ventana temporal?
    
    Métricas:
        - P_transmission: probabilidad de activar vecino por spike
        - σ (sigma): branching ratio = <n_activados>
        - firing_rate: tasa de disparo poblacional (Hz)
    """
    
    def __init__(self, window_ms=5.0, min_weight=0.0, min_spikes=5):
        self.window = window_ms
        self.min_weight = min_weight
        self.min_spikes = min_spikes
        
    def extract_connectivity_E2E(self, synapses_intra, Ne, verbose=False):
        """
        Extrae grafo de conectividad E→E desde sinapsis Brian2.
        
        Returns:
            neighbors: dict {pre_idx: [post_idx_1, post_idx_2, ...]}
            weights: dict {(pre, post): weight}
        """
        neighbors = defaultdict(list)
        weights = {}
        
        pre_indices = np.array(synapses_intra.i)
        post_indices = np.array(synapses_intra.j)
        syn_weights = np.array(synapses_intra.w)
        
        # Filtro: E→E con peso > threshold
        E_to_any_mask = (pre_indices < Ne) & (syn_weights >= 0.0)  # Solo pre < Ne
        mask = E_to_any_mask & (syn_weights >= self.min_weight)
        
        for pre, post, w in zip(pre_indices[mask], post_indices[mask], syn_weights[mask]):
            neighbors[int(pre)].append(int(post))
            weights[(int(pre), int(post))] = float(w)
        
        if verbose:
            degrees = [len(v) for v in neighbors.values()]
            logger.debug(f"  E→E connections: {np.sum(mask)} (w>{self.min_weight})")
            logger.debug(f"  Out-degree: mean={np.mean(degrees):.1f}, max={np.max(degrees)}")
        
        return dict(neighbors), weights
    
    def organize_spike_times(self, spike_times_arr, spike_indices_arr):
        """
        Organiza spikes por neurona.
        
        Returns:
            spike_dict: {neuron_idx: sorted_spike_times_array}
        """
        spike_dict = defaultdict(list)
        
        for t, idx in zip(spike_times_arr, spike_indices_arr):
            spike_dict[int(idx)].append(float(t))
        
        spike_dict = {k: np.sort(v) for k, v in spike_dict.items()}
        return dict(spike_dict)
    
    def count_responses_single_spike(self, pre_spike_time, post_neuron_spikes):
        """
        Verifica si neurona post respondió en ventana [t, t+window).
        """
        if len(post_neuron_spikes) == 0:
            return False
        
        responses = post_neuron_spikes[
            (post_neuron_spikes > pre_spike_time) & 
            (post_neuron_spikes < pre_spike_time + self.window)
        ]
        return len(responses) > 0
    
    def analyze(self, spike_dict, neighbors, T_total, warmup=0.0):
        """
        Análisis principal de propagación.
        
        Args:
            spike_dict: {neuron_idx: spike_times}
            neighbors: {pre_idx: [post_idx_list]}
            T_total: duración total (ms)
            warmup: tiempo de warmup a excluir (ms)
            
        Returns:
            dict con métricas: P_transmission, sigma, firing_rate, stats
        """
        # Filtrar spikes por warmup
        spike_dict_filtered = {
            nid: times[times >= warmup] 
            for nid, times in spike_dict.items()
        }
        
        T_analysis = T_total - warmup
        
        ratios_per_spike = []
        activated_counts = []
        per_neuron_stats = {}
        
        total_spikes_analyzed = 0
        neurons_analyzed = 0
        
        for pre_idx in neighbors.keys():
            if pre_idx not in spike_dict_filtered:
                continue
            
            pre_spikes = spike_dict_filtered[pre_idx]
            
            if len(pre_spikes) < self.min_spikes:
                continue
            
            post_neighbors = neighbors[pre_idx]
            n_neighbors = len(post_neighbors)
            
            if n_neighbors == 0:
                continue
            
            neuron_ratios = []
            neuron_activated = []
            
            for spike_time in pre_spikes:
                n_activated = 0
                
                for post_idx in post_neighbors:
                    if post_idx not in spike_dict_filtered:
                        continue
                    
                    post_spikes = spike_dict_filtered[post_idx]
                    
                    if self.count_responses_single_spike(spike_time, post_spikes):
                        n_activated += 1
                
                ratio = n_activated / n_neighbors
                
                ratios_per_spike.append(ratio)
                activated_counts.append(n_activated)
                neuron_ratios.append(ratio)
                neuron_activated.append(n_activated)
                
                total_spikes_analyzed += 1
            
            per_neuron_stats[pre_idx] = {
                'n_spikes': len(pre_spikes),
                'n_neighbors': n_neighbors,
                'mean_ratio': np.mean(neuron_ratios),
                'mean_activated': np.mean(neuron_activated)
            }
            neurons_analyzed += 1
        
        # Calcular firing rate poblacional
        total_spikes = sum([len(times) for times in spike_dict_filtered.values()]) if spike_dict_filtered else 0
        n_neurons = len(spike_dict_filtered)
        firing_rate = (total_spikes / n_neurons / T_analysis) * 1000.0  # Hz
        
        ratios_per_spike = np.array(ratios_per_spike)
        activated_counts = np.array(activated_counts)
        
        results = {
            'P_transmission': np.mean(ratios_per_spike) if len(ratios_per_spike) > 0 else 0.0,
            'P_transmission_std': np.std(ratios_per_spike) if len(ratios_per_spike) > 0 else 0.0,
            'sigma': np.mean(activated_counts) if len(activated_counts) > 0 else 0.0,
            'sigma_std': np.std(activated_counts) if len(activated_counts) > 0 else 0.0,
            'firing_rate': firing_rate,
            'ratio_distribution': ratios_per_spike,
            'activated_counts': activated_counts,
            'per_neuron': per_neuron_stats,
            'stats': {
                'n_neurons_analyzed': neurons_analyzed,
                'total_spikes_analyzed': total_spikes_analyzed,
                'total_spikes': total_spikes,
                'n_neurons_active': n_neurons,
                'T_analysis': T_analysis
            }
        }
        
        return results

logger.success("PropagationAnalyzer class defined")


# ## 4. Función de Simulación Parametrizada
from brian2 import ms, mV

# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_single_simulation(k_factor, rate_hz, trial=0, verbose=False):
    """
    Ejecuta una simulación con parámetros (k_factor, rate_hz).
    
    Returns:
        dict con spike_times, spike_indices, synapses (i, j, w)
    """
    start_scope()
    
    # Crear red
    network = IzhikevichNetwork(
        dt_val=SIM_CONFIG['dt_ms'],
        T_total=SIM_CONFIG['T_ms'],
        fixed_seed=FIXED_SEED,
        variable_seed=VARIABLE_SEED + trial,
        trial=trial
    )
    
    # Parámetros de población
    params = {
        **NETWORK_PARAMS,
        'k_exc': k_factor,
        'k_inh': k_factor * 3.9,
        'rate_hz': rate_hz
    }
    
    # Crear población A
    pop_A = network.create_population2(name='A', **params)
    
    # IMPORTANTE: record_v_dt debe ser un número válido, no None
    # Setup monitors
    
    network.setup_monitors(['A'], record_v_dt=None, sample_fraction=0.0, monitor_conductance=False, monitor_input=False)
    
    # IMPORTANTE: Extraer conectividad ANTES de run_simulation
    # porque después de la simulación los objetos pueden estar en scope diferente
    
    syn = network.populations['A']['syn_intra']
    syn_i = np.array(syn.i[:])
    syn_j = np.array(syn.j[:])
    syn_w = np.array(syn.w[:])
    
    # Ejecutar simulación
    results = network.run_simulation()
    
    # Extraer spikes INMEDIATAMENTE
    spike_times = np.array(results['A']['spike_times'])
    spike_indices = np.array(results['A']['spike_indices'])
    
    # AÑADIR: Extraer voltajes
    # v_mon = network.monitors['A']['voltage']
    # v_times = np.array(v_mon.t / ms)
    # v_values = np.array(v_mon.v / mV)
    # v_neuron_ids = v_mon.record
        
    if verbose:
        logger.info(f"  Simulation completed: {len(spike_times)} spikes")
    
    # Retornar solo arrays numpy (pickleable)
    return {
        'k': k_factor,
        'rate_hz': rate_hz,
        'trial': trial,
        'spike_times': spike_times,
        'spike_indices': spike_indices,
        # 'v_times': v_times,
        # 'v_values': v_values,
        # 'v_neuron_ids': np.array(v_neuron_ids),
        'synapses': {
            'i': syn_i,
            'j': syn_j,
            'w': syn_w
        }
    }

logger.success("Simulation runner function defined")


# ## 5. Barrido 2D: (K, rate_hz) → (FR, P, σ)
# =============================================================================
# HELPER: PROCESS SIMULATION RESULTS
# =============================================================================

def process_simulation_results(sim_data):
    """
    Procesa resultados de simulación: extrae conectividad y analiza propagación.
    
    Args:
        sim_data: dict con spike_times, spike_indices, synapses, k, rate_hz, trial
        
    Returns:
        dict con métricas de propagación
    """
    # Crear analyzer
    analyzer = PropagationAnalyzer(
        window_ms=PROPAGATION_CONFIG['window_ms'],
        min_weight=PROPAGATION_CONFIG['min_weight'],
        min_spikes=PROPAGATION_CONFIG['min_spikes']
    )
    
    # Extraer conectividad E→E desde arrays
    neighbors = defaultdict(list)
    weights = {}
    
    pre_indices = sim_data['synapses']['i']
    post_indices = sim_data['synapses']['j']
    syn_weights = sim_data['synapses']['w']
    
    E2E_mask = (pre_indices < Ne) & (post_indices < Ne) & (syn_weights >= 0.0)
    mask = E2E_mask & (syn_weights >= analyzer.min_weight)
    
    for pre, post, w in zip(pre_indices[mask], post_indices[mask], syn_weights[mask]):
        neighbors[int(pre)].append(int(post))
        weights[(int(pre), int(post))] = float(w)
    
    neighbors = dict(neighbors)
    
    # Organizar spikes
    spike_dict = analyzer.organize_spike_times(
        sim_data['spike_times'],
        sim_data['spike_indices']
    )
    
    # Analizar propagación
    prop_results = analyzer.analyze(
        spike_dict=spike_dict,
        neighbors=neighbors,
        T_total=SIM_CONFIG['T_ms'],
        warmup=SIM_CONFIG['warmup_ms']
    )
    
    # Resultado condensado
    return {
        'k': sim_data['k'],
        'rate_hz': sim_data['rate_hz'],
        'trial': sim_data['trial'],
        'firing_rate': prop_results['firing_rate'],
        'P_transmission': prop_results['P_transmission'],
        'P_transmission_std': prop_results['P_transmission_std'],
        'sigma': prop_results['sigma'],
        'sigma_std': prop_results['sigma_std'],
        'n_neurons_analyzed': prop_results['stats']['n_neurons_analyzed'],
        'total_spikes': prop_results['stats']['total_spikes']
    }
    

# =============================================================================
# TASK RUNNER (GLOBAL FUNCTION FOR MULTIPROCESSING)
# =============================================================================

def run_single_task(args):
    """Wrapper global para multiprocessing con métricas + datos raw"""
    k_val, rate_val, trial = args
    try:
        # Simulación completa
        sim_data = run_single_simulation(
            k_factor=k_val,
            rate_hz=rate_val,
            trial=trial,
            verbose=False
        )
        
        # Métricas de propagación (análisis actual)
        metrics = process_simulation_results(sim_data)
        
        # Combinar métricas + raw data
        return {
            **metrics,
            'raw_data': {
                'spike_times': sim_data['spike_times'].astype(np.float32),
                'spike_indices': sim_data['spike_indices'].astype(np.int16),
                'synapses': {
                    'i': sim_data['synapses']['i'].astype(np.int16),
                    'j': sim_data['synapses']['j'].astype(np.int16),
                    'w': sim_data['synapses']['w'].astype(np.float32)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error K={k_val}, rate={rate_val}, trial={trial}: {e}")
        return None 

logger.success("Helper functions defined")

# =============================================================================
# HELPER: CARGAR BATCH ESPECÍFICO
# =============================================================================
def load_batch(batch_idx, sweep_dir=OUTPUT_DIR):
    """Carga raw_data de un batch específico"""
    import gzip
    batch_file = sweep_dir / f'batch_{batch_idx:03d}.pkl.gz'
    with gzip.open(batch_file, 'rb') as f:
        return pickle.load(f)

def load_specific_sim(k, rate, trial, sweep_dir=OUTPUT_DIR):
    with open(sweep_dir / 'results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    filepath = data['file_paths'][(k, rate, trial)]
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)

# =============================================================================
# 2D SWEEP WITH MULTIPROCESSING
# =============================================================================

import hashlib

def generate_file_hash(k, rate, trial):
    """Genera hash único para nombre de archivo"""
    seed = f"{k:.2f}_{rate:.1f}_{trial}"
    return hashlib.md5(seed.encode()).hexdigest()[:8]

def run_2d_sweep(K_values, rate_hz_values, n_trials=5, n_processes=32, save_results=True):
    import gc, shutil, hashlib
    
    tasks = [(k, r, t) for k in K_values for r in rate_hz_values for t in range(n_trials)]
    total_sims = len(tasks)
    
    # Validación disco
    disk_stats = shutil.disk_usage(OUTPUT_DIR.parent)
    required_gb = total_sims * 0.00064  # 0.64 MB comprimido
    if disk_stats.free / 1024**3 < required_gb * 1.2:
        logger.error(f"Espacio insuficiente")
        raise RuntimeError("Espacio en disco insuficiente")
    
    logger.info(f"High-res sweep: {total_sims} sims, ~{required_gb:.1f} GB")
    
    batch_size = 128
    n_batches = (len(tasks) - 1) // batch_size + 1
    all_metrics = []
    file_paths = {}
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tasks))
        batch = tasks[start_idx:end_idx]
        
        logger.info(f"Batch {batch_idx+1}/{n_batches} ({len(batch)} sims)")
        
        with mp.Pool(processes=n_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(run_single_task, batch),
                total=len(batch),
                desc=f"Batch {batch_idx+1}/{n_batches}"
            ))
        
        batch_results = [r for r in batch_results if r is not None]
        
        # Guardar archivos individuales comprimidos
        for sim in batch_results:
            k, rate, trial = sim['k'], sim['rate_hz'], sim['trial']
            seed = f"{k:.2f}_{rate:.1f}_{trial}"
            file_hash = hashlib.md5(seed.encode()).hexdigest()[:8]
            filename = f'raw_data_k{k:.2f}_r{rate}_t{trial}_{file_hash}.pkl.gz'
            filepath = OUTPUT_DIR / filename
            
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(sim['raw_data'], f, protocol=4)
            
            file_paths[(k, rate, trial)] = str(filepath)
        
        # Extraer métricas
        batch_metrics = [{k: v for k, v in r.items() if k != 'raw_data'} 
                        for r in batch_results]
        all_metrics.extend(batch_metrics)
        
        del batch_results, batch_metrics
        gc.collect()
        
        logger.success(f"Batch {batch_idx+1} done: {len(batch)} files")
    
    df_results = pd.DataFrame(all_metrics)
    
    if save_results:
        # Guardar en formato compatible
        results_data = {
            'df_metrics': df_results,
            'file_paths': file_paths,
            'K_values': K_values.tolist(),  # Convertir a lista
            'rate_hz_values': rate_hz_values.tolist(),
            'n_trials': n_trials,
            'config': {
                'SIM_CONFIG': SIM_CONFIG,
                'NETWORK_PARAMS': NETWORK_PARAMS,
                'PROPAGATION_CONFIG': PROPAGATION_CONFIG
            }
        }
        
        with open(OUTPUT_DIR / 'results.pkl', 'wb') as f:
            pickle.dump(results_data, f, protocol=4)
            
        logger.success(f"Results saved: {OUTPUT_DIR / 'results.pkl'}")
    
    return df_results

# # =============================================================================
# # EJECUTAR BARRIDO 2D
# # =============================================================================

# CONFIGURACIÓN DEL BARRIDO

# NOTA: Este bloque tarda ~30-60 minutos dependiendo del hardware y n_trials
# Puedes comentar esta celda y cargar resultados previos en la siguiente sección

logger.info("Starting 2D sweep...")
logger.info(f"Total simulations: {len(K_VALUES) * len(RATE_HZ_VALUES) * N_TRIALS}")
logger.info(f"Estimated time: ~{len(K_VALUES) * len(RATE_HZ_VALUES) * N_TRIALS * 0.3:.1f} min")


df_sweep = run_2d_sweep(
    K_values=K_VALUES,
    rate_hz_values=RATE_HZ_VALUES,
    n_trials=N_TRIALS,
    n_processes=N_PROCESSES,
    save_results=True
)

# =============================================================================
# AUTOMATIC PLOTTING
# =============================================================================

def generate_summary_plots(df, output_dir):
    """Genera y guarda figuras resumen del barrido"""
    
    # Agregar por (K, rate_hz)
    df_mean = df.groupby(['k', 'rate_hz']).agg({
        'firing_rate': 'mean',
        'P_transmission': 'mean',
        'sigma': 'mean'
    }).reset_index()
    
    # Crear grids para heatmaps
    K_unique = np.sort(df_mean['k'].unique())
    rate_unique = np.sort(df_mean['rate_hz'].unique())
    
    FR_grid = df_mean.pivot(index='k', columns='rate_hz', values='firing_rate').values
    P_grid = df_mean.pivot(index='k', columns='rate_hz', values='P_transmission').values
    sigma_grid = df_mean.pivot(index='k', columns='rate_hz', values='sigma').values
    
    # 1. Heatmaps principales
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = axes[0].imshow(FR_grid, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Firing Rate (Hz)', fontsize=14)
    axes[0].set_xlabel('rate_hz (Hz)')
    axes[0].set_ylabel('K (coupling)')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(P_grid, aspect='auto', origin='lower', cmap='plasma')
    axes[1].set_title('P_transmission', fontsize=14)
    axes[1].set_xlabel('rate_hz (Hz)')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(sigma_grid, aspect='auto', origin='lower', cmap='coolwarm')
    axes[2].set_title('σ (branching ratio)', fontsize=14)
    axes[2].set_xlabel('rate_hz (Hz)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Curvas 1D: P vs K
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for rate in [4, 8, 12, 20]:
        mask = df_mean['rate_hz'] == rate
        axes[0].plot(df_mean[mask]['k'], df_mean[mask]['P_transmission'], 
                     'o-', label=f'rate={rate}Hz', markersize=6)
    
    axes[0].set_xlabel('K (coupling)', fontsize=12)
    axes[0].set_ylabel('P_transmission', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # σ vs K
    for rate in [4, 8, 12, 20]:
        mask = df_mean['rate_hz'] == rate
        axes[1].plot(df_mean[mask]['k'], df_mean[mask]['sigma'], 
                     'o-', label=f'rate={rate}Hz', markersize=6)
    
    axes[1].set_xlabel('K (coupling)', fontsize=12)
    axes[1].set_ylabel('σ (branching ratio)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'curves_vs_K.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Figures saved to {output_dir}")

# Generar plots automáticamente
generate_summary_plots(df_sweep, OUTPUT_DIR)

# Mostrar resumen
print("\n" + "="*80)
print("SWEEP SUMMARY")
print("="*80)
print(df_sweep.describe())
print("\n" + "="*80)

# Resumen por trial
if N_TRIALS > 1:
    print("\n" + "="*80)
    print("TRIAL VARIABILITY")
    print("="*80)
    trial_stats = df_sweep.groupby('trial')[['firing_rate', 'P_transmission', 'sigma']].agg(['mean', 'std'])
    print(trial_stats)
    print("\n" + "="*80)

# =============================================================================
# GLOBAL FILTER: ONLY EXCITATORY AS SENDERS
# =============================================================================

def filter_excitatory_senders(spike_times, spike_indices, Ne=800):
    """
    Filtra spikes: solo neuronas excitatorias (idx < Ne) como EMISORAS.
    Post-sinápticas pueden ser cualquiera (E o I).
    """
    mask = spike_indices < Ne
    return spike_times[mask], spike_indices[mask]

logger.success("Excitatory filter defined")


