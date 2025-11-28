
import numpy as np
from brian2 import *
from datetime import datetime as datetim
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import os, json, pickle, gc
import traceback
from scipy import stats

from src.two_populations.helpers.helpers import create_readable_config_names, estimate_memory_mb, print_memory_snapshot
from src.two_populations.helpers.validator import add_validation_to_analysis, plot_population_validation_dashboard, print_validation_summary
from src.two_populations.metrics import analyze_simulation_results
from src.two_populations.plots.dashboard_plots import plot_population_dashboard, plot_connectivity_dashboard
from src.two_populations.plots.basic_plots import plot_raster_results, plot_spectrogram, plot_correlation_matrix, plot_interpop_correlation, plot_thalamic_drive, plot_synaptic_currents

from src.two_populations.metrics import palmigiano_analysis, extract_lfp
from src.two_populations.plots.dashboard_plots import plot_palmigiano_dashboard

from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

def create_sweep_folder(base_dir: str, tag: str, timestamp: str = None) -> str:
    """Crea carpeta results/experiments/.../<TAG>__<timestamp>/"""
    if timestamp is None:
        timestamp = datetim.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(base_dir, f"{tag}__{timestamp}")
    os.makedirs(os.path.join(sweep_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(sweep_dir, "figures", "dashboards"), exist_ok=True)
    return sweep_dir

def _to_jsonable(obj):
    """Convierte np.* y tipos no serializables a tipos JSON-friendly."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    return obj

def save_config_json(cfg: dict, path: str):
    """Guarda config en JSON (con conversión segura)."""
    safe = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            safe[k] = {kk: _to_jsonable(vv) for kk, vv in v.items()} if v else {}
        else:
            safe[k] = _to_jsonable(v)
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)

def delay_distribution_sweep(sweep_dir, dt, time, simulator_class, base_params, delay_configs, directionality = 0, n_trials=3,device_mode='runtime'):
    """
    Barrido sistemático de distribuciones de delay con monitoreo de memoria
    """
    logger.info(">>> sweep.py - device inicial:", get_device(), " build_on_run:", get_device().build_on_run)

    results_database = {}
    p_inter = base_params['connection']['p_inter']
    weight_scale = base_params['connection']['weight_scale']
    
    for config_name, delay_config in delay_configs.items():
        
        logger.info(f"\n=== EJECUTANDO: {config_name} ===")
    
        trial_results = []
        delay_samples = []
        trial_meta = []
        
        try:
            
            # Crear subcarpeta por config
            config_dir = os.path.join(sweep_dir, "figures", config_name)
            os.makedirs(config_dir, exist_ok=True)
            
            for trial in range(n_trials):
                
                gc.collect() 
                
                # Brian2 cleanup
                start_scope()  # Fresh Brian2 scope
                set_device('runtime')  # Force runtime mode
                
                seed_val = 100 # misma seed
                variable_seed_val = 200 + trial  # variable seed
                
                # if device_mode in ['cpp_standalone', 'cuda_standalone']:
                #     get_device().reinit()
                #     get_device().activate()
                #     set_device(device_mode, directory='output', build_on_run=False)
                    # logger.info(">>> sweep.py - device inicial:", get_device(), " build_on_run:", get_device().build_on_run)
                
                logger.info(f"Trial {trial+1}/{n_trials}")
                logger.info(f"Trial Params: {dt=}, {time=}, {seed_val=}, {variable_seed_val=}")
                
                # Crear simulación
                sim = simulator_class(dt_val=dt, T_total=time, fixed_seed=seed_val, variable_seed=variable_seed_val, warmup=None, device_mode=device_mode)
                
                # Poblaciones con parámetros base
                pop_A = sim.create_population('A', **base_params['pop_A'])
                pop_B = sim.create_population('B', **base_params['pop_B'])
                
                # MEMORIA CHECK 1: Después de crear poblaciones
                sim_mem = estimate_memory_mb(sim, "simulator")
                logger.info(f"[MEM] Después de crear poblaciones: simulator = {sim_mem:.2f} MB")
                
                # Conectar con distribución de delay específica
                if delay_config['type'] == 'constant':
                    syn_AB = sim.connect_populations('A', 'B', 
                                                p_inter=base_params['connection']['p_inter'],
                                                weight_scale=base_params['connection']['weight_scale'],
                                                delay_value=delay_config['value'])
                    
                    if directionality == 1: 
                        syn_BA = sim.connect_populations('B', 'A',
                                                    p_inter=base_params['connection']['p_inter'],
                                                    weight_scale=base_params['connection']['weight_scale'],
                                                    delay_value=delay_config['value'])
                                
                        
                else:
                    syn_AB = sim.connect_populations('A', 'B',
                                                p_inter=base_params['connection']['p_inter'],
                                                weight_scale=base_params['connection']['weight_scale'],
                                                delay_dist=delay_config['type'],
                                                delay_params=delay_config['params'])
                    
                    if directionality == 1: 
                        syn_BA = sim.connect_populations('B', 'A',
                                                    p_inter=base_params['connection']['p_inter'],
                                                    weight_scale=base_params['connection']['weight_scale'],
                                                    delay_dist=delay_config['type'],
                                                    delay_params=delay_config['params'])

                if directionality == 0: 
                    current_delays = np.array(syn_AB.delay/ms)
                    
                else:
                    current_delays = np.array([np.array(syn_AB.delay/ms), np.array(syn_BA.delay/ms)])
                    
                if trial == 0:  # Solo guardar estadísticas en el primer trial
                        delay_samples = [current_delays]  # Reset list
                    
                else:
                    delay_samples.append(current_delays)
                
                # MEMORIA CHECK 2: Después de delays
                logger.info(f"[MEM] Delays acumulados: {len(delay_samples)} arrays, {estimate_memory_mb(delay_samples):.2f} MB")
        
                # Ejecutar simulación
                sim.setup_monitors(['A', 'B'], 0.5, 0.5)
                
                try:
                    results = sim.run_simulation()
                except Exception as e:
                    logger.info(f"\n{'#'*60}")
                    logger.info(f"[ERROR] Trial {trial+1}/{n_trials} falló")
                    logger.info(f"Config: {config_name}")
                    logger.info(f"Error: {e}")
                    logger.info(f"{'#'*60}\n")
                    
                    traceback.print_exc()
                    
                    # Guardar metadata del trial fallido
                    trial_meta.append({
                        'seed': seed_val, 
                        'variable_seed': variable_seed_val,
                        'trial': trial+1, 
                        'status': 'FAILED',
                        'error': str(e)
                    })
                    
                    continue
                
                # En standalone, sim_result es el simulador mismo
                if device_mode in ['cpp_standalone', 'cuda_standalone']:
                    
                    # Compilar y ejecutar el binario standalone
                    # get_device().build(directory='output', compile=True, run=True, clean=True)
        
                    # Extraer resultados DESPUÉS de run()
                    results = results.get_results()


                conn = analyze_simulation_results(
                    results['A']['spike_monitor'], 
                    results['B']['spike_monitor'], 
                    1000, "Baseline", warmup=500,
                    state_monitors={'A': sim.monitors['A'], 'B': sim.monitors['B']}, delays = {'AB': np.array(results['delays_AB']), 'BA': None}, signal_mode='lfp')    

                if 'delays_A_B' in results:
                    conn['delays_AB'] = results['delays_A_B']
                if 'delays_B_A' in results:
                    conn['delays_BA'] = results['delays_B_A']
                
                if conn is None: 
                    continue
                
                metrics_only = {
                    'cross_correlation': {'peak_value': conn['cross_correlation']['peak_value'], 
                                        'peak_lag': conn['cross_correlation']['peak_lag']},
                    'plv_pli': {'alpha': {'plv': conn['plv_pli']['alpha']['plv'], 
                                        'pli': conn['plv_pli']['alpha']['pli']},
                            'gamma': {'plv': conn['plv_pli']['gamma']['plv'],
                                        'pli': conn['plv_pli']['gamma']['pli']}},
                    'coherence': {'peak_coherence': conn['coherence']['peak_coherence'],
                                'peak_freq': conn['coherence']['peak_freq']},
                    'tau_A': conn['tau_A'],  # ← Cambiar de int_A
                    'tau_B': conn['tau_B'],  # ← Cambiar de int_B
                    'timescale_A': conn['timescale_A'],  # ← Añadir dict completo
                    'timescale_B': conn['timescale_B']   # ← Para exp/int en plots
                }

                # ===== GENERAR PLOTS INMEDIATAMENTE =====
                trial_name = f"{config_name}_trial{trial+1}"
                conn_dict = {trial_name: conn}
                
                if trial == 0:
                    
                    palmi_metrics = palmigiano_analysis(results, start_ms=500)
                    lfp_A = extract_lfp(results['A']['voltage_monitor'], 500, results['dt'])
                    lfp_B = extract_lfp(results['B']['voltage_monitor'], 500, results['dt'])
                    
                    fig_palmi = plot_palmigiano_dashboard(palmi_metrics, lfp_A, lfp_B, fs=1000/results['dt'])
                    fig_palmi.savefig(os.path.join(config_dir, f"palmigiano_{trial_name}.png"), dpi=150, bbox_inches='tight')
                    plt.close()
                          
                    # Corrientes sinapticas
                    fig_syn = plot_synaptic_currents(conn_dict, results)  
                    fig_syn.savefig(os.path.join(config_dir, f"syn_currents_{trial_name}.png"), dpi=150)
                    plt.close()
                    
                    # Drive externo
                    fig_syn = plot_thalamic_drive(conn_dict, results)  
                    fig_syn.savefig(os.path.join(config_dir, f"thalamic_currents_{trial_name}.png"), dpi=150)
                    plt.close()
                    
                    # Espectrogramas
                    fig_spec = plot_spectrogram(conn_dict) 
                    fig_spec.savefig(os.path.join(config_dir, f"spectrogram_{trial_name}.png"), dpi=150)
                    plt.close()
                    
                    fig_corr = plot_correlation_matrix(results)  # ✅ OK
                    fig_corr.savefig(os.path.join(config_dir, f"correlations_{trial_name}.png"), dpi=150)
                    plt.close()
                    
                    fig_inter = plot_interpop_correlation(results, n_sample=50)  # ✅ OK
                    fig_inter.savefig(os.path.join(config_dir, f"interpop_{trial_name}.png"), dpi=150)
                    plt.close()
                    
                    # Con tus results_dict existentes
                    validation_results = add_validation_to_analysis(conn_dict)
                    fig = plot_population_validation_dashboard(validation_results)
                    fig.savefig(os.path.join(config_dir, f"validation_{trial_name}.png"), 
                                dpi=150, bbox_inches='tight')
                    print_validation_summary(validation_results)

                    # Plot conectividad
                    fig1 = plot_connectivity_dashboard(conn_dict)
                    fig1.savefig(os.path.join(config_dir, f"connectivity_{trial_name}.png"), 
                                dpi=150, bbox_inches='tight')
                    plt.close(fig1)
                    
                    # Plot población  
                    fig2 = plot_population_dashboard(conn_dict)
                    fig2.savefig(os.path.join(config_dir, f"population_{trial_name}.png"), 
                                dpi=150, bbox_inches='tight')
                    plt.close(fig2)
                    
                    # Plot raster (usa results directamente)
                    fig3 = plot_raster_results(results)
                    fig3.savefig(os.path.join(config_dir, f"raster_{trial_name}.png"), 
                                dpi=150, bbox_inches='tight')
                    plt.close(fig3)
                    
                    logger.info(f"[PLOTS] Guardados para {trial_name}")
                    
                # Pickle completo para los primeros 5 trials
                if trial < 5:
                    trial_pickle = os.path.join(config_dir, f"trial_{trial+1}_full.pkl")
                    with open(trial_pickle, 'wb') as f:
                        pickle.dump(clean_single_trial_result(conn), f)
                        
                    logger.info(f"[PICKLE] Trial {trial+1} completo guardado")

                trial_results.append(metrics_only)

                # Guardar métricas
                trial_meta.append({
                    'seed': seed_val,
                    'variable_seed': variable_seed_val, 
                    'trial': trial+1, 
                    'dt_ms': float(dt), 
                    'T_total_ms': float(time),
                    'p_inter': float(p_inter), 
                    'weight_scale': float(weight_scale),
                    'directionality': int(directionality)
                })
                
                # MEMORIA CHECK 4: Después de análisis
                conn_mem = estimate_memory_mb(conn, "conn_result")
                trial_mem = estimate_memory_mb(trial_results, "trial_results")
                logger.info(f"[MEM] Conn analysis: {conn_mem:.2f} MB, Total trials: {trial_mem:.2f} MB")
                    
                # ===== LIMPIEZA INMEDIATA =====
                plt.close('all')
                try:
                    del sim, pop_A, pop_B, syn_AB, results, conn
                    try:
                        if directionality == 1:
                            del syn_BA
                    except:
                        pass
                except:
                    pass
                gc.collect()
                
                # Memory check cada 3 trials
                if (trial + 1) % 3 == 0:

                    process = psutil.Process()
                    actual_mb = process.memory_info().rss / (1024**2)
                    
                    print_memory_snapshot(trial_results, delay_samples, None, results_database, trial+1, config_name)
                    logger.info(f"[MEM] Después de trial {trial+1}: {actual_mb:.1f} MB")
                    
            # Calcular estadísticas agregadas
            if trial_results:
                aggregated_metrics = aggregate_trial_metrics(trial_results)
                delay_statistics = calculate_delay_statistics(delay_samples)
                
                # Identificar trial representativo para CSV 
                lags = [tr['cross_correlation']['peak_lag'] for tr in trial_results]
                rep_idx = np.argsort(lags)[len(lags)//2]
                rep_trial = rep_idx + 1
                
                results_database[config_name] = {
                    'delay_config': delay_config,
                    'delay_statistics': delay_statistics,
                    'trials': trial_results,
                    'trials_meta': trial_meta,
                    'aggregated': aggregated_metrics,
                    'n_trials': len(trial_results),
                    'run_meta': {
                        'dt_ms': float(dt), 'T_total_ms': float(time),
                        'p_inter': float(p_inter), 'weight_scale': float(weight_scale),
                        'directionality': int(directionality), 'device_mode': str(device_mode)
                    },
                    'representative_trial': {
                        'index': rep_idx,
                        'trial_number': rep_trial,
                        'lag': lags[rep_idx]
                    }
                }
                
                logger.info(f"[CONFIG] {config_name} completado: {len(trial_results)} trials, trial representativo: {rep_trial}")
            
            else:
                # Config completó 0 trials exitosos
                results_database[config_name] = {
                    'status': 'NO_SUCCESSFUL_TRIALS',
                    'n_trials': 0,
                    'delay_config': delay_config
                }
                
                logger.info(f"[CONFIG] {config_name}: 0 trials exitosos")
                
            # Limpieza entre configs
            plt.close('all')
            try:
                del trial_results, delay_samples, trial_meta, aggregated_metrics, delay_statistics
            except:
                pass
            
            start_scope()  # Forzar limpieza Brian2
            gc.collect()
        
        except Exception as e:
            logger.info(f"\n{'!'*60}")
            logger.info(f"[ERROR CRÍTICO] Config {config_name} falló completamente")
            logger.info(f"Error: {e}")
            logger.info(f"{'!'*60}\n")

            traceback.print_exc()
            
            # Guardar error en results_database
            results_database[config_name] = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'status': 'FAILED',
                'n_trials_attempted': trial + 1 if 'trial' in locals() else 0,  # Cuántos trials intentó antes de fallar
                'n_trials_succeeded': len(trial_results)  # Cuántos completó exitosamente
            }
            
            error_log = os.path.join(config_dir, "ERROR.txt")
            with open(error_log, 'w') as f:
                f.write(traceback.format_exc())
                                    
            # Limpieza entre configs
            plt.close('all')
            try:
                del trial_results, delay_samples, trial_meta, aggregated_metrics, delay_statistics
            except:
                pass
            
            start_scope()  # Forzar limpieza Brian2
            gc.collect()
            
            # Continuar con siguiente config
            continue
        
    return results_database    

def aggregate_trial_metrics(trial_results):
    metrics = {
        'cross_corr_peak': {'mean': 0, 'std': 0, 'values': []},
        'cross_corr_lag':  {'mean': 0, 'std': 0, 'values': []},
        'plv_alpha':       {'mean': 0, 'std': 0, 'values': []},
        'pli_alpha':       {'mean': 0, 'std': 0, 'values': []},
        'plv_gamma':       {'mean': 0, 'std': 0, 'values': []},
        'coherence_peak':  {'mean': 0, 'std': 0, 'values': []},
        'tau_A':           {'mean': 0, 'std': 0, 'values': []},
        'tau_B':           {'mean': 0, 'std': 0, 'values': []},
    }
    for tr in trial_results:
        metrics['cross_corr_peak']['values'].append(abs(tr['cross_correlation']['peak_value']))
        metrics['cross_corr_lag']['values'].append(tr['cross_correlation']['peak_lag'])
        metrics['plv_alpha']['values'].append(tr['plv_pli']['alpha']['plv'])
        metrics['pli_alpha']['values'].append(tr['plv_pli']['alpha']['pli'])
        metrics['plv_gamma']['values'].append(tr['plv_pli']['gamma']['plv'])
        metrics['coherence_peak']['values'].append(tr['coherence']['peak_coherence'])
        tau_A = tr.get('tau_A', 0)
        tau_B = tr.get('tau_B', 0)
        metrics['tau_A']['values'].append(tau_A)
        metrics['tau_B']['values'].append(tau_B)

    for name, data in metrics.items():
        v = np.array(data['values'])
        data['mean']   = float(np.mean(v)) if len(v) else None
        data['std']    = float(np.std(v)) if len(v) else None
        data['median'] = float(np.median(v)) if len(v) else None
        data['min']    = float(np.min(v)) if len(v) else None
        data['max']    = float(np.max(v)) if len(v) else None
    return metrics

"""
Versión corregida de calculate_delay_statistics con TODAS las métricas estadísticas.
Reemplazar en sweep.py
"""

import numpy as np
from scipy.stats import skew, kurtosis

def calculate_delay_statistics(delay_samples_list):
    """
    Calcula estadísticas completas de distribuciones de delays.
    
    Incluye todas las métricas necesarias para análisis de correlaciones:
    - Centralidad: mean, median
    - Dispersión: std, cv, iqr, range
    - Forma: skewness, kurtosis
    - Límites: min, max, cuartiles
    
    Args:
        delay_samples_list: Lista de arrays con samples de delays
        
    Returns:
        Dict con todas las estadísticas
    """
    all_delays = np.concatenate([s for s in delay_samples_list if len(s) > 0]) if len(delay_samples_list) else np.array([])
    
    if len(all_delays) == 0:
        stats = {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'q25': 0.0,
            'q75': 0.0,
            'iqr': 0.0,
            'range': 0.0,
            'cv': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'n_connections': 0,
            'distribution_type': 'empty'
        }
    else:
        mean_val = float(np.mean(all_delays))
        std_val = float(np.std(all_delays))
        q25 = float(np.percentile(all_delays, 25))
        q75 = float(np.percentile(all_delays, 75))
        
        stats = {
            # Centralidad
            'mean': mean_val,
            'median': float(np.median(all_delays)),
            
            # Dispersión absoluta
            'std': std_val,
            'iqr': q75 - q25,
            'range': float(np.max(all_delays) - np.min(all_delays)),
            
            # Dispersión relativa
            'cv': std_val / mean_val if mean_val > 1e-10 else 0.0,
            
            # Forma de la distribución
            'skewness': float(skew(all_delays)),
            'kurtosis': float(kurtosis(all_delays)),
            
            # Límites
            'min': float(np.min(all_delays)),
            'max': float(np.max(all_delays)),
            'q25': q25,
            'q75': q75,
            
            # Metadata
            'n_connections': int(len(all_delays)),
            'distribution_type': 'unknown'  # Se puede inferir del config
        }
    
    delay_samples_list.clear()  # Limpiar lista
    return stats

def save_results(results_db, filename_prefix="./results/delay_sweep"):
    """Guardar resultados en pickle"""
    timestamp = datetim.now().strftime("%Y_%m_%d_%H%M")
    filename = f"{filename_prefix}_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(results_db, f)
    
    logger.info(f"Resultados guardados en: {filename}")
    return filename

# Modificar save_results_with_csv para limpiar antes de pickle
def save_results_with_csv(results_db, out_dir, tag="sweep", per_trial_csv=True):
    
    os.makedirs(out_dir, exist_ok=True)
    ts = datetim.now().strftime("%Y%m%d_%H%M%S")  # Corregido typo

    # Limpiar resultados antes de pickle
    cleaned_results_db = clean_results_for_pickle(results_db)
    
    pkl_path = os.path.join(out_dir, f"results_{tag}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(cleaned_results_db, f)

    # Generar nombres legibles
    delay_configs = {}
    for cfg, entry in results_db.items():
        delay_config = entry.get("delay_config", {})
        if delay_config:
            delay_configs[cfg] = delay_config
    
    readable_names = create_readable_config_names(delay_configs) if delay_configs else {}

    # CSV de configuraciones con nombres legibles
    rows = []
    for cfg, entry in results_db.items():
        rep_trial = entry.get('representative_trial', {})
        rep_idx = rep_trial.get('index', 0)
        rep_trial_num = rep_trial.get('trial_number', 1)
        
        agg = entry.get("aggregated", {})
        dstat = entry.get("delay_statistics", {}) or {}
        rm = entry.get("run_meta", {}) or {}
        delay_config = entry.get("delay_config", {})
        
        def g(m, k): return agg.get(m, {}).get(k, None)
        
        rows.append({
            "config": cfg,
            "readable_name": readable_names.get(cfg, cfg),  # ← NUEVO
            "delay_type": delay_config.get("type", "unknown"),
            "delay_config_full": json.dumps(delay_config),  # ← NUEVO: config completo
            "delay_params": json.dumps(delay_config.get("params", delay_config.get("value", {}))),
            "n_trials": entry.get("n_trials", 0),
            
            # Métricas de conectividad
            "cc_peak_mean": g("cross_corr_peak", "mean"),
            "cc_peak_std":  g("cross_corr_peak", "std"),
            "cc_lag_mean_ms": g("cross_corr_lag", "mean"),
            "cc_lag_std_ms":  g("cross_corr_lag", "std"),
            "cc_lag_median_ms": g("cross_corr_lag", "median"),
            
            # PLV/PLI por bandas
            "plv_alpha_mean": g("plv_alpha", "mean"),
            "plv_alpha_std":  g("plv_alpha", "std"),
            "pli_alpha_mean": g("pli_alpha", "mean"),  # ← AÑADIDO
            "pli_alpha_std":  g("pli_alpha", "std"),   # ← AÑADIDO
            "plv_gamma_mean": g("plv_gamma", "mean"),
            "plv_gamma_std":  g("plv_gamma", "std"),
            "pli_gamma_mean": g("pli_gamma", "mean"),  # ← AÑADIDO
            "pli_gamma_std":  g("pli_gamma", "std"),   # ← AÑADIDO
            
            # Coherencia
            "coh_peak_mean":  g("coherence_peak", "mean"),
            "coh_peak_std":   g("coherence_peak", "std"),
            "alpha_coherence_mean": g("alpha_coherence", "mean"),
            "gamma_coherence_mean": g("gamma_coherence", "mean"),
            
            # Timescales intrínsecos
            "tauA_mean_ms":   g("tau_A", "mean"),
            "tauA_std_ms":    g("tau_A", "std"),
            "tauB_mean_ms":   g("tau_B", "mean"),
            "tauB_std_ms":    g("tau_B", "std"),
            
            # Estadísticas de delay
            "delay_mean_ms":  dstat.get("mean"),
            "delay_std_ms":   dstat.get("std"),
            "delay_median_ms":dstat.get("median"),
            "delay_q25_ms":   dstat.get("q25"),
            "delay_q75_ms":   dstat.get("q75"),
            "n_connections":  dstat.get("n_connections"),
            
            # Metadatos del experimento
            "dt_ms":          rm.get("dt_ms"),
            "T_total_ms":     rm.get("T_total_ms"),
            "p_inter":        rm.get("p_inter"),
            "weight_scale":   rm.get("weight_scale"),
            "directionality": rm.get("directionality"),
            "device_mode":    rm.get("device_mode"),
            
            # Rutas a figuras
            "connectivity_png": f"figures/{cfg}/connectivity_{cfg}_trial{rep_trial_num}.png",
            "population_png": f"figures/{cfg}/population_{cfg}_trial{rep_trial_num}.png", 
            "raster_png": f"figures/{cfg}/raster_{cfg}_trial{rep_trial_num}.png",
            "representative_trial": rep_trial_num
        })
        
    df_cfg = pd.DataFrame(rows)
    csv_cfg_path = os.path.join(out_dir, f"summary_config_{tag}.csv")
    df_cfg.to_csv(csv_cfg_path, index=False)

    # CSV por trials (extendido)
    csv_trials_path = None
    if per_trial_csv:
        trial_rows = []
        for cfg, entry in results_db.items():
            metas = entry.get("trials_meta", []) or []
            trials = entry.get("trials", []) or []
            delay_config = entry.get("delay_config", {})
            readable_name = readable_names.get(cfg, cfg)
            
            for k, (m, res) in enumerate(zip(metas, trials)):
                trial_rows.append({
                    "config": cfg, 
                    "readable_name": readable_name,  # ← AÑADIDO
                    "trial_idx": k,
                    "delay_type": delay_config.get("type", "unknown"),
                    "delay_params": json.dumps(delay_config.get("params", {})),
                    
                    # Métricas por trial
                    "cc_peak": float(abs(res['cross_correlation']['peak_value'])),
                    "cc_lag_ms": float(res['cross_correlation']['peak_lag']),
                    "plv_alpha": float(res['plv_pli']['alpha']['plv']),
                    "pli_alpha": float(res['plv_pli']['alpha']['pli']),
                    "plv_gamma": float(res['plv_pli']['gamma']['plv']),
                    "pli_gamma": float(res['plv_pli']['gamma']['pli']),
                    "coh_peak": float(res['coherence']['peak_coherence']),
                    "coh_peak_freq": float(res['coherence']['peak_freq']),
                    "tau_A_ms": float(res.get('tau_A', 0)),
                    "tau_B_ms": float(res.get('tau_B', 0)),
                    
                    # Metadata del trial
                    "fixed_seed": m.get("seed"),
                    "variable_seed": m.get("variable_seed"),
                    "trial_number": m.get("trial"),
                    "dt_ms": m.get("dt_ms"),
                    "T_total_ms": m.get("T_total_ms"),
                    "p_inter": m.get("p_inter"),
                    "weight_scale": m.get("weight_scale"),
                    "directionality": m.get("directionality"),
                })
        df_trials = pd.DataFrame(trial_rows)
        csv_trials_path = os.path.join(out_dir, f"trials_{tag}.csv")
        df_trials.to_csv(csv_trials_path, index=False)

    logger.info(f"[OK] Saved: {pkl_path} (cleaned version)")
    logger.info(f"[OK] Saved: {csv_cfg_path}")
    if csv_trials_path: 
        logger.info(f"[OK] Saved: {csv_trials_path}")
    
    return {"pkl": pkl_path, "csv_config": csv_cfg_path, "csv_trials": csv_trials_path}

def clean_results_for_pickle(results_db):
    """
    Limpia results_db removiendo objetos no serializables (Brian2 monitors)
    manteniendo solo datos numéricos y arrays numpy.
    """
    cleaned_db = {}
    
    for config_name, config_data in results_db.items():
        cleaned_config = {}
        
        # Copiar metadatos (son serializables)
        for key in ['delay_config', 'delay_statistics', 'n_trials', 'run_meta', 'representative_trial']:
            if key in config_data:
                cleaned_config[key] = config_data[key]
        
        # Limpiar trials individuales
        if 'trials' in config_data:
            cleaned_trials = []
            for trial in config_data['trials']:
                cleaned_trial = clean_single_trial_result(trial)
                cleaned_trials.append(cleaned_trial)
            cleaned_config['trials'] = cleaned_trials
        
        # Copiar métricas agregadas (son serializables)
        if 'aggregated' in config_data:
            cleaned_config['aggregated'] = config_data['aggregated']
            
        # Copiar trials_meta (son serializables)
        if 'trials_meta' in config_data:
            cleaned_config['trials_meta'] = config_data['trials_meta']
            
        cleaned_db[config_name] = cleaned_config
    
    return cleaned_db

def clean_single_trial_result(trial_result):
    """
    Limpia un resultado individual removiendo monitors y manteniendo solo datos numpy.
    """
    cleaned = {}
    
    # Copiar datos que son seguros para pickle
    safe_keys = [
        'time', 'rate_A', 'rate_B', 't0_ms',
        'cross_correlation', 'autocorr_A', 'autocorr_B', 
        'int_A', 'int_B', 'plv_pli', 'coherence', 'psd_A', 'psd_B'
    ]
    
    for key in safe_keys:
        if key in trial_result:
            cleaned[key] = trial_result[key]
    
    # Para spike times/neurons, convertir a numpy arrays simples
    if 'spike_times_A' in trial_result:
        cleaned['spike_times_A'] = np.array(trial_result['spike_times_A'])
    if 'spike_neurons_A' in trial_result:
        cleaned['spike_neurons_A'] = np.array(trial_result['spike_neurons_A'])
    if 'spike_times_B' in trial_result:
        cleaned['spike_times_B'] = np.array(trial_result['spike_times_B'])  
    if 'spike_neurons_B' in trial_result:
        cleaned['spike_neurons_B'] = np.array(trial_result['spike_neurons_B'])
    
    # NO copiar state_monitor_A, state_monitor_B (son los problemáticos)
    
    return cleaned

def plot_delay_distributions(results_db, ax=None, show_legend=True):
    """Plot delay distributions for each configuration"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract delay info from config names
    def parse_delay_from_config_name(config_name):
        parts = config_name.split('_')
        
        for i, part in enumerate(parts):
            if part == 'delta':
                # Look for numeric value after delta
                for j in range(i+1, len(parts)):
                    try:
                        value = float(parts[j].replace('p', '.'))
                        return 'delta', {'value': value}
                    except:
                        continue
                        
            elif part.startswith('lognorm'):
                # lognorm_1p0-0p3
                for j in range(i+1, len(parts)):
                    if '-' in parts[j]:
                        try:
                            alpha, beta = parts[j].split('-')
                            return 'lognormal', {'alpha': float(alpha.replace('p','.')), 
                                                'beta': float(beta.replace('p','.'))}
                        except:
                            continue

            elif part.startswith('gamma'):
                # gamma_3-1
                for j in range(i+1, len(parts)):
                    if '-' in parts[j]:
                        try:
                            shape, scale = parts[j].split('-')
                            return 'gamma', {'shape': float(shape.replace('p','.')), 
                                            'scale': float(scale.replace('p','.'))}
                        except:
                            continue
                    
            elif part.startswith('uniform'):
                # Handle uniform_0-16 format
                if '-' in part:
                    try:
                        _, params_str = part.split('_', 1)
                        low, high = params_str.split('-')
                        return 'uniform', {'low': float(low), 'high': float(high)}
                    except:
                        pass
                # Also check next parts for uniform_0-16 as separate elements
                for j in range(i+1, len(parts)):
                    if '-' in parts[j]:
                        try:
                            low, high = parts[j].split('-')
                            return 'uniform', {'low': float(low), 'high': float(high)}
                        except:
                            continue
                            
            elif part == 'beta':
                # Map descriptive beta names to parameters
                beta_presets = {
                    'slow_bias': {'alpha': 2, 'beta': 5, 'scale': 20, 'shift': 0},
                    'fast_bias': {'alpha': 5, 'beta': 2, 'scale': 15, 'shift': 0},
                    'centered_narrow': {'alpha': 8, 'beta': 8, 'scale': 12, 'shift': 4},
                    'centered_wide': {'alpha': 2, 'beta': 2, 'scale': 16, 'shift': 2},
                    'very_sharp': {'alpha': 10, 'beta': 10, 'scale': 8, 'shift': 6},
                    'very_slow': {'alpha': 1, 'beta': 4, 'scale': 20, 'shift': 0},
                    'very_fast': {'alpha': 4, 'beta': 1, 'scale': 10, 'shift': 2},
                    'uniform': {'alpha': 1, 'beta': 1, 'scale': 18, 'shift': 2},
                    'mild_fast': {'alpha': 3, 'beta': 2, 'scale': 14, 'shift': 2},
                    'mild_slow': {'alpha': 2, 'beta': 3, 'scale': 16, 'shift': 2}
                }
                
                # Look for preset name after beta
                if i+1 < len(parts) and parts[i+1] in beta_presets:
                    return 'beta', beta_presets[parts[i+1]]
                
                # Look for compound names like slow_bias
                for j in range(i+1, min(i+3, len(parts))):
                    compound_name = '_'.join(parts[i+1:j+1])
                    if compound_name in beta_presets:
                        return 'beta', beta_presets[compound_name]
                        
        return None, None
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    x = np.linspace(0, 30, 1000)
    color_idx = 0
    
    for config, data in results_db.items():
        # Parse delay configuration
        delay_type, params = parse_delay_from_config_name(config)
        
        if delay_type is None:
            continue  # Skip unparseable configs
        
        if delay_type == 'delta':
            value = params.get('value', 4)
            # More visible delta distribution
            y = stats.norm.pdf(x, value, 0.3) * 3
            label = f"δ({value:.0f}ms)"
            
        elif delay_type == 'lognormal':
            alpha = params.get('alpha', 1.0)
            beta = params.get('beta', 0.5)
            y = stats.lognorm.pdf(x, beta, scale=np.exp(alpha))
            label = f"LogN({alpha:.1f},{beta:.1f})"

        elif delay_type == 'gamma':
            shape = params.get('shape', 3.0)
            scale = params.get('scale', 1.0)
            y = stats.gamma.pdf(x, shape, scale=scale)
            label = f"Γ({shape:.1f},{scale:.1f})"
            
        elif delay_type == 'uniform':
            low = params.get('low', 2)
            high = params.get('high', 6)
            y = np.where((x >= low) & (x <= high), 1/(high-low), 0)
            label = f"U({low:.0f},{high:.0f})"
            
        elif delay_type == 'beta':
            alpha = params.get('alpha', 2)
            beta_param = params.get('beta', 5)
            scale = params.get('scale', 10)
            shift = params.get('shift', 0)
            
            # Beta distribution scaled and shifted
            x_norm = (x - shift) / scale
            y = np.where((x_norm >= 0) & (x_norm <= 1), 
                        stats.beta.pdf(x_norm, alpha, beta_param) / scale, 0)
            
            # Create descriptive label from config name
            config_parts = config.split('_')
            if 'beta' in config_parts:
                beta_idx = config_parts.index('beta')
                if beta_idx + 1 < len(config_parts):
                    preset_name = '_'.join(config_parts[beta_idx+1:beta_idx+3])
                    label = f"β({preset_name})"
                else:
                    label = f"β({alpha},{beta_param})"
            else:
                label = f"β({alpha},{beta_param})"
            
        else:
            continue
        
        # Plot with enhanced visibility
        ax.plot(x, y, color=colors[color_idx], linewidth=4, label=label, alpha=0.8)
        ax.fill_between(x, 0, y, color=colors[color_idx], alpha=0.4)
        color_idx += 1
    
    ax.set_xlabel('Delay (ms)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_title('Delay Distributions', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)
    ax.set_ylim(bottom=0)
    
    # Enhanced legend
    if show_legend and color_idx > 0:
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, 
                loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    return ax

def extract_parameters_for_sorting(config_name):
    """Extract numeric parameters from config name for sorting"""
    parts = config_name.split('_')
    params = []
    
    # Find distribution type and extract following parameters
    for i, part in enumerate(parts):
        if part in ['delta', 'gamma', 'lognormal', 'uniform', 'beta', 'exponential']:
            # Extract numeric values from remaining parts
            for j in range(i+1, len(parts)):
                try:
                    # Handle different parameter formats
                    if parts[j].replace('.', '').replace('p', '').isdigit():
                        value = float(parts[j].replace('p', '.'))
                        params.append(value)
                    elif 'mu' in parts[j]:
                        value = float(parts[j].replace('mu', ''))
                        params.append(value)
                    elif 'sigma' in parts[j]:
                        value = float(parts[j].replace('sigma', ''))
                        params.append(value)
                except:
                    continue
            break
    
    # Ensure at least 2 parameters for sorting
    while len(params) < 2:
        params.append(0)
    
    return params[:2]  # Return first two parameters

def plot_delay_comparison_with_distributions(results_db, save_path=None):
    """Grid 2x3 con métricas principales + distribuciones de delay"""
    if not results_db:
        logger.info("[WARNING] results_db vacío")
        return None
        
    # Sort configs by parameters
    configs = list(results_db.keys())
    configs_with_params = [(config, extract_parameters_for_sorting(config)) for config in configs]
    configs_with_params.sort(key=lambda x: (x[1][0], x[1][1]))  # Sort by first, then second parameter
    configs = [item[0] for item in configs_with_params]
    
    # Seleccionar 5 métricas principales
    key_metrics = ['cross_corr_peak', 'cross_corr_lag', 'plv_alpha', 'coherence_peak', 'tau_A']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Simplificar nombres
    def simplify_config_name(config_name):
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if part in ['delta', 'lognormal', 'gamma', 'uniform', 'beta', 'exponential']:
                return '_'.join(parts[i:]).replace('_input_', '_')
        return '_'.join(parts[-3:])
    
    simplified_names = [simplify_config_name(config) for config in configs]
    x = np.arange(len(configs))
    
    # Plot primeras 5 métricas
    for i, metric in enumerate(key_metrics):
        ax = axes[i]
        
        means = []
        stds = []
        
        for c in configs:
            agg = results_db[c].get('aggregated', {})
            metric_data = agg.get(metric, {})
            
            mean_val = metric_data.get('mean', 0)
            std_val = metric_data.get('std', 0)
            
            if mean_val is None or np.isnan(mean_val):
                mean_val = 0
            if std_val is None or np.isnan(std_val):
                std_val = 0
                
            means.append(mean_val)
            stds.append(std_val)
        
        if len(means) > 0:
            means = np.array(means)
            stds = np.array(stds)
            
            ax.plot(x, means, 'o-', linewidth=2.5, markersize=7, 
                color='steelblue', markerfacecolor='darkblue', 
                markeredgecolor='white', markeredgewidth=1.5)
            
            ax.fill_between(x, means - stds, means + stds, 
                        alpha=0.25, color='steelblue')
            
            ax.set_xticks(x)
            ax.set_xticklabels(simplified_names, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric.replace('_',' ').title(), fontsize=10)
            ax.set_title(f'{metric.replace("_"," ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Normalize y-axis for comparable relative error visualization
            mean_val = np.mean(means)
            std_val = np.mean(stds)
            if mean_val > 0 and std_val > 0:
                cv = std_val / mean_val  # Coefficient of variation
                y_range = mean_val * 0.8  # Show ±40% around mean for consistency
                ax.set_ylim(max(0, mean_val - y_range), mean_val + y_range)
    
    # Sexto plot: distribuciones de delay
    plot_delay_distributions(results_db, ax=axes[5], show_legend=True)
    
    plt.suptitle('Connectivity Metrics & Delay Distributions', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"[PLOT] Saved 2x3 comparison plot: {save_path}")
    
    return fig

