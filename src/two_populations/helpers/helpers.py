import pandas as pd
import numpy as np
import os
import psutil
import sys
import json
import pickle
from scipy.signal import butter, sosfilt, hilbert

from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

def _agg(results_db, config, key, default=0.0):
    return results_db.get(config, {}).get('aggregated', {}).get(key, {}).get('mean', default)

def _phase_diff_band(rate_A, rate_B, fs_hz, band=(8,12)):
    sos = butter(4, band, btype='band', fs=fs_hz, output='sos')
    fA = sosfilt(sos, rate_A); fB = sosfilt(sos, rate_B)
    hA, hB = hilbert(fA), hilbert(fB)
    return np.unwrap(np.angle(hA)) - np.unwrap(np.angle(hB))

def load_trials_data_for_dashboard(sweep_dir, configs):
    """Load individual trial data for box plots"""
    trials_data = {}
    
    for config in configs:
        trials_csv = os.path.join(sweep_dir, f"config_{config}", f"trials_{config}.csv")
        if os.path.exists(trials_csv):
            df = pd.read_csv(trials_csv)
            trials_data[config] = df
    
    return trials_data

def load_raw_timeseries(sweep_dir, config_name, trial=1):
    """Load raw timeseries with multiple fallback strategies"""
    
    # Strategy 1: Standard pickle structure
    trial_file = os.path.join(sweep_dir, 'figures', config_name, f'trial_{trial}_full.pkl')
    
    if not os.path.exists(trial_file):
        # Strategy 2: Try config directory structure
        trial_file = os.path.join(sweep_dir, f'config_{config_name}', f'trial_{trial}_full.pkl')
    
    if not os.path.exists(trial_file):
        # Strategy 3: Search for any pickle files in config directory
        config_dir = os.path.join(sweep_dir, f'config_{config_name}')
        if os.path.exists(config_dir):
            pickle_files = [f for f in os.listdir(config_dir) if f.endswith('.pkl')]
            if pickle_files:
                trial_file = os.path.join(config_dir, pickle_files[0])
    
    if not os.path.exists(trial_file):
        logger.info(f"No raw data found for {config_name}")
        return None
        
    try:
        with open(trial_file, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        logger.info(f"Failed to load {trial_file}: {e}")
        return None

# úsalo en todos los sitios: _agg(results_db, c, 'plv_alpha')
def _adapt_raw_timeseries(raw, default_dt_ms=0.5):
    
    time = raw.get('time_ms', raw.get('time', None))
    
    if time is None and 'rate_A' in raw:
        time = np.arange(len(raw['rate_A'])) * default_dt_ms
    rates = (raw.get('rate_A', None), raw.get('rate_B', None))

    lags, corr = None, None
    cc = raw.get('cross_correlation', None)
    
    if isinstance(cc, dict):
        lags = cc.get('lags') or cc.get('lags_ms')
        
        corr = cc.get('values') or cc.get('correlation')
    elif isinstance(cc, np.ndarray):
        corr = cc
        
    return time, rates, lags, corr


def load_results_from_csvs(sweep_dir, completed_configs, filter_outliers=False):
    """
    Reconstruir results_db desde CSVs para plotting con filtrado opcional de outliers
    """
    
    results_db_light = {}
    
    for config_name in completed_configs:
        config_dir = os.path.join(sweep_dir, f"config_{config_name}")
        summary_csv = os.path.join(config_dir, f"summary_config_{config_name}.csv")
        trials_csv = os.path.join(config_dir, f"trials_{config_name}.csv")
        
        if not os.path.exists(summary_csv):
            continue
            
        df_summary = pd.read_csv(summary_csv)
        row = df_summary.iloc[0]
        
        # Reconstruir delay_config
        delay_config = {}
        if 'delay_config_full' in row and pd.notna(row['delay_config_full']):
            try:
                delay_config = json.loads(row['delay_config_full'])
            except:
                pass
        
        # If trials CSV exists and we want to filter outliers, recalculate
        if filter_outliers and os.path.exists(trials_csv):
            df_trials = pd.read_csv(trials_csv)
            
            if len(df_trials) > 0:
                # Extract expected delay range from config name
                delay_range = extract_delay_range_from_config(config_name)
                max_expected_delay = delay_range[1] if delay_range else 20
                
                # Simple outlier filter: remove lags > 2.5 * max_expected_delay
                lag_threshold = 2.5 * max_expected_delay
                valid_mask = np.abs(df_trials['cc_lag_ms']) <= lag_threshold
                df_filtered = df_trials[valid_mask]
                
                if len(df_filtered) > 0:
                    # Recalculate aggregated metrics from filtered data
                    aggregated = {
                        'cross_corr_peak': {
                            'mean': df_filtered['cc_peak'].mean(),
                            'std': df_filtered['cc_peak'].std()
                        },
                        'cross_corr_lag': {
                            'mean': df_filtered['cc_lag_ms'].mean(),
                            'std': df_filtered['cc_lag_ms'].std()
                        },
                        'plv_alpha': {
                            'mean': df_filtered['plv_alpha'].mean(),
                            'std': df_filtered['plv_alpha'].std()
                        },
                        'coherence_peak': {
                            'mean': df_filtered['coh_peak'].mean(),
                            'std': df_filtered['coh_peak'].std()
                        },
                        'tau_A': {
                            'mean': df_filtered['tau_A_ms'].mean(),
                            'std': df_filtered['tau_A_ms'].std()
                        },
                        'tau_B': {
                            'mean': df_filtered['tau_B_ms'].mean(),
                            'std': df_filtered['tau_B_ms'].std()
                        }
                    }
                    
                    n_filtered = len(df_trials) - len(df_filtered)
                    if n_filtered > 0:
                        logger.info(f"[FILTER] {config_name}: removed {n_filtered}/{len(df_trials)} outlier trials")
                    
                else:
                    logger.info(f"[WARNING] All trials filtered out for {config_name}")
                    continue
            else:
                logger.info(f"[WARNING] Empty trials CSV for {config_name}")
                continue
        else:
            # Use pre-computed aggregated metrics from summary
            aggregated = {
                'cross_corr_peak': {
                    'mean': row.get('cc_peak_mean', 0),
                    'std': row.get('cc_peak_std', 0)
                },
                'cross_corr_lag': {
                    'mean': row.get('cc_lag_mean_ms', 0),
                    'std': row.get('cc_lag_std_ms', 0)
                },
                'plv_alpha': {
                    'mean': row.get('plv_alpha_mean', 0),
                    'std': row.get('plv_alpha_std', 0)
                },
                'coherence_peak': {
                    'mean': row.get('coh_peak_mean', 0),
                    'std': row.get('coh_peak_std', 0)
                }
            }
        
        results_db_light[config_name] = {
            'aggregated': aggregated,
            'n_trials': row.get('n_trials', 0),
            'delay_config': delay_config,
            'delay_statistics': {  # ← FALTA
                'mean': row.get('delay_mean_ms', 0),
                'std': row.get('delay_std_ms', 0),
                'n_connections': row.get('n_connections', 0)
            },
            'readable_name': row.get('readable_name', config_name)
        }
    
    return results_db_light

def extract_delay_range_from_config(config_name):
    """Extract delay range from config name"""
    if 'uniform' in config_name:
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if 'uniform' in part and i+1 < len(parts):
                range_str = parts[i+1]  # e.g., "4-14", "0-18"
                if '-' in range_str:
                    try:
                        low, high = map(float, range_str.split('-'))
                        return (low, high)
                    except:
                        pass
    elif 'lognorm' in config_name:
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if 'lognorm' in part and i+1 < len(parts):
                params = parts[i+1].split('-')
                try:
                    alpha = float(params[0].replace('p', '.'))
                    beta = float(params[1].replace('p', '.')) if len(params) > 1 else 0.5
                    # Lognormal: mean ≈ exp(alpha), use heuristic range
                    mean = np.exp(alpha)
                    return (0, mean * 3)
                except:
                    pass

    elif 'gamma' in config_name:
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if 'gamma' in part and i+1 < len(parts):
                params = parts[i+1].split('-')
                try:
                    shape = float(params[0].replace('p', '.'))
                    scale = float(params[1].replace('p', '.')) if len(params) > 1 else 1.0
                    # Gamma: mean = shape*scale
                    mean = shape * scale
                    return (0, mean * 2.5)
                except:
                    pass
                
    elif 'delta' in config_name:
        parts = config_name.split('_')
        for part in parts:
            try:
                if part.replace('p', '.').replace('.', '', 1).isdigit():  # ✓
                    val = float(part.replace('p', '.'))
                    return (val, val)
            except:
                pass
    
    return (0, 20)  # Default fallback


def extract_delay_parameters(results_db):
    """Extract delay parameters - works with empty delay_config"""
    rows = []
    
    for config, data in results_db.items():
        delay_config = data.get('delay_config', {})
        
        # If empty, parse from name
        if not delay_config or not delay_config.get('type'):
            delay_config = parse_delay_config_from_name(config)
        
        if not delay_config:
            continue
            
        row = {'config': config}
        delay_type = delay_config.get('type', 'unknown')
        row['type'] = delay_type
        
        # Add type-specific parameters
        if delay_type == 'uniform':
            params = delay_config.get('params', delay_config)  # params puede estar en raíz
            row['uniform_low'] = params.get('low', 0)
            row['uniform_high'] = params.get('high', 0)
            row['uniform_mean'] = (params.get('low', 0) + params.get('high', 0)) / 2
            row['uniform_width'] = params.get('high', 0) - params.get('low', 0)
            
        elif delay_type == 'lognormal':
            params = delay_config.get('params', {})
            row['lognorm_alpha'] = params.get('alpha', 0)
            row['lognorm_beta'] = params.get('beta', 0)
            row['lognorm_mean'] = np.exp(params.get('alpha', 0) + 0.5*params.get('beta', 0)**2)

        elif delay_type == 'gamma':
            params = delay_config.get('params', {})
            row['gamma_shape'] = params.get('shape', 0)
            row['gamma_scale'] = params.get('scale', 0)
            row['gamma_mean'] = params.get('shape', 0) * params.get('scale', 0)
            
        elif delay_type == 'constant':
            row['delta_value'] = delay_config.get('value', 0)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def parse_delay_config_from_name(config_name):
    """Parse delay configuration from config name"""
    if 'uniform' in config_name:
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if 'uniform' in part and i+1 < len(parts):
                range_str = parts[i+1]
                if '-' in range_str:
                    try:
                        low_str, high_str = range_str.split('-')
                        # Handle 'p' notation for decimals
                        low = float(low_str.replace('p', '.'))
                        high = float(high_str.replace('p', '.'))
                        return {'type': 'uniform', 'params': {'low': low, 'high': high}}
                    except:
                        pass
    
    elif 'lognorm' in config_name:
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if 'lognorm' in part and i+1 < len(parts):
                try:
                    param_str = parts[i+1].split('-')
                    alpha = float(param_str[0].replace('p', '.'))
                    beta = float(param_str[1].replace('p', '.')) if len(param_str) > 1 else 0.5
                    return {'type': 'lognormal', 'params': {'alpha': alpha, 'beta': beta}}
                except:
                    pass

    elif 'gamma' in config_name:
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if 'gamma' in part and i+1 < len(parts):
                try:
                    param_str = parts[i+1].split('-')
                    shape = float(param_str[0].replace('p', '.'))
                    scale = float(param_str[1].replace('p', '.')) if len(param_str) > 1 else 1.0
                    return {'type': 'gamma', 'params': {'shape': shape, 'scale': scale}}
                except:
                    pass
    
    elif 'delta' in config_name:
        parts = config_name.split('_')
        for part in parts:
            # BUG: '3p5' no es .isdigit()
            if part.replace('p', '.').replace('.', '', 1).isdigit():  # Fix
                value = float(part.replace('p', '.'))
                return {'type': 'constant', 'value': value}
        # Default fallback
    return {'type': 'constant', 'value': 0}


#  Configuraciones de distribución beta para retrasos temporales entre ROIs
def create_readable_config_names(delay_configs):
    """Convert technical config names to human-readable format"""
    readable_names = {}
    for old_name, config in delay_configs.items():
        
        if config['type'] == 'constant':
            new_name = f"Delta_{config['value']}ms"
            
        elif config['type'] == 'uniform':
            low, high = config['params']['low'], config['params']['high']
            new_name = f"Uniform_{low}-{high}"
            
        elif config['type'] == 'lognormal':
            alpha, beta = config['params']['alpha'], config['params']['beta']
            new_name = f"LogN_α{alpha:.1f}β{beta:.1f}"
            
        elif config['type'] == 'gamma':
            shape, scale = config['params']['shape'], config['params']['scale']
            new_name = f"Gamma_k{shape:.1f}θ{scale:.1f}"
            
        elif config['type'] == 'beta':
            alpha, beta = config['params']['alpha'], config['params']['beta']
            scale = config['params']['scale']
            new_name = f"Beta_α{alpha}β{beta}_s{scale}"
        else:
            new_name = old_name
        
        readable_names[old_name] = new_name
    return readable_names

def get_display_names(results_db, use_readable_names=True):
    """Get display names for configs"""
    configs = list(results_db.keys())
    
    if not use_readable_names:
        return configs
    
    # Reconstruct delay_configs from results_db
    delay_configs = {}
    for cfg in configs:
        delay_config = results_db[cfg].get('delay_config', {})
        if delay_config:
            delay_configs[cfg] = delay_config
    
    if delay_configs:
        readable_names = create_readable_config_names(delay_configs)
        return [readable_names.get(cfg, cfg) for cfg in configs]
    
    return configs


def find_best_predictors(corr_matrix, p_matrix):
    """Find which delay parameters best predict each metric"""
    predictors = {}
    
    for metric in corr_matrix.columns:
        metric_corrs = corr_matrix[metric].abs()
        metric_pvals = p_matrix[metric]
        
        # Significant correlations only
        significant = metric_pvals < 0.05
        if significant.any():
            best_predictor = metric_corrs[significant].idxmax()
            best_corr = corr_matrix.loc[best_predictor, metric]
            best_p = p_matrix.loc[best_predictor, metric]
            
            predictors[metric] = {
                'predictor': best_predictor,
                'correlation': best_corr,
                'p_value': best_p
            }
    
    return predictors


parse_delay_from_name = parse_delay_config_from_name

def estimate_memory_mb(obj, name="object"):
    """Estima el uso de memoria de un objeto en MB"""
    
    def get_size(obj):
        
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        
        elif isinstance(obj, (list, tuple)):
            
            if len(obj) == 0:
                return sys.getsizeof(obj)
            
            # Estimar basado en el primer elemento
            first_size = get_size(obj[0]) if len(obj) > 0 else 8
            return sys.getsizeof(obj) + len(obj) * first_size
        
        elif isinstance(obj, dict):
            
            total = sys.getsizeof(obj)
            
            for k, v in obj.items():
                total += get_size(k) + get_size(v)
            return total
        else:
            return sys.getsizeof(obj)
    
    size_bytes = get_size(obj)
    size_mb = size_bytes / (1024**2)
    return size_mb

def print_memory_snapshot(trial_results, delay_samples, all_results_raw, results_database, trial_num, config_name):
    """Print snapshot de uso de memoria"""
    
    logger.info(f"\n--- MEMORIA SNAPSHOT (Trial {trial_num}, Config: {config_name}) ---")
    
    # Trial results (lista de dicts con métricas)
    trial_mem = estimate_memory_mb(trial_results, "trial_results") 
    logger.info(f"trial_results: {trial_mem:.2f} MB ({len(trial_results)} items)")
    
    # Delay samples (lista de arrays numpy)
    delay_mem = estimate_memory_mb(delay_samples, "delay_samples")
    logger.info(f"delay_samples: {delay_mem:.2f} MB ({len(delay_samples)} arrays)")
    
    # Database total
    db_mem = estimate_memory_mb(results_database, "results_database") 
    logger.info(f"results_database: {db_mem:.2f} MB ({len(results_database)} configs)")
    
    # Memory total estimado
    total_est = trial_mem + delay_mem + db_mem
    logger.info(f"TOTAL ESTIMADO: {total_est:.2f} MB")
    
    # System memory actual

    process = psutil.Process()
    actual_mb = process.memory_info().rss / (1024**2)
    logger.info(f"PROCESO ACTUAL: {actual_mb:.1f} MB")
    logger.info("--- END SNAPSHOT ---\n")

# =============================================================================
# DELAY-METRICS CORRELATION ANALYSIS (FIXED)
# =============================================================================

# def parse_delay_from_name(config_name):
#     """Extract delay parameters from config name"""
#     params = {}
    
#     if 'uniform' in config_name.lower():
#         # Extract range like "uniform_8-10" or "uniform_2-16"
#         parts = config_name.split('_')
#         for i, part in enumerate(parts):
#             if 'uniform' in part.lower() and i+1 < len(parts):
#                 range_str = parts[i+1]
#                 if '-' in range_str:
#                     try:
#                         low, high = range_str.split('-')
#                         low = float(low.replace('p', '.'))
#                         high = float(high.replace('p', '.'))
#                         params['type'] = 'uniform'
#                         params['low'] = low
#                         params['high'] = high
#                         params['mean'] = (low + high) / 2
#                         params['width'] = high - low
#                         return params
#                     except:
#                         pass
    
#     elif 'lognorm' in config_name.lower():
#         parts = config_name.split('_')
#         for i, part in enumerate(parts):
#             if 'lognorm' in part.lower() and i+1 < len(parts):
#                 param_str = parts[i+1].split('-')
#                 try:
#                     alpha = float(param_str[0].replace('p', '.'))
#                     beta = float(param_str[1].replace('p', '.')) if len(param_str) > 1 else 0.5
#                     params['type'] = 'lognormal'
#                     params['alpha'] = alpha
#                     params['beta'] = beta
#                     params['mean'] = np.exp(alpha + 0.5*beta**2)
#                     return params
#                 except:
#                     pass

#     elif 'gamma' in config_name.lower():
#         parts = config_name.split('_')
#         for i, part in enumerate(parts):
#             if 'gamma' in part.lower() and i+1 < len(parts):
#                 param_str = parts[i+1].split('-')
#                 try:
#                     shape = float(param_str[0].replace('p', '.'))
#                     scale = float(param_str[1].replace('p', '.')) if len(param_str) > 1 else 1.0
#                     params['type'] = 'gamma'
#                     params['shape'] = shape
#                     params['scale'] = scale
#                     params['mean'] = shape * scale
#                     return params
#                 except:
#                     pass
        
#     elif 'delta' in config_name.lower():
#         parts = config_name.split('_')
#         for part in parts:
#             try:
#                 val = float(part.replace('p', '.'))
#                 params['type'] = 'constant'
#                 params['value'] = val
#                 return params
#             except:
#                 continue
    
#     return params