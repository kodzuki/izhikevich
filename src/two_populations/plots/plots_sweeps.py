import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

from datetime import datetime

from src.two_populations.helpers.helpers import load_raw_timeseries, load_results_from_csvs, load_trials_data_for_dashboard, _phase_diff_band
from src.two_populations.helpers.helpers import get_display_names
from src.two_populations.helpers.helpers import (
    _phase_diff_band,
    load_raw_timeseries,
    load_trials_data_for_dashboard
)

from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)


# ============================================================================
# EXTRACCIÓN DE MÉTRICAS GENERALIZADAS
# ============================================================================

def extract_delay_statistics(results_db: Dict) -> pd.DataFrame:
    """
    Extrae métricas estadísticas generalizadas de las distribuciones de delays.
    
    Estas métricas permiten comparar distribuciones de diferentes familias:
    - Centralidad: mean, median
    - Dispersión: std, cv, iqr, range
    - Forma: skewness, kurtosis
    - Rango: min, max
    
    Args:
        results_db: Diccionario con resultados del sweep
        
    Returns:
        DataFrame con métricas generalizadas por configuración
    """
    rows = []
    
    for config, data in results_db.items():
        delay_stats = data.get('delay_statistics', {})
        
        if not delay_stats or 'mean' not in delay_stats:
            continue
        
        row = {
            'config': config,
            'type': delay_stats.get('distribution_type', 'unknown'),
            
            # Centralidad
            'mean': delay_stats.get('mean', np.nan),
            'median': delay_stats.get('median', np.nan),
            
            # Dispersión absoluta
            'std': delay_stats.get('std', np.nan),
            'iqr': delay_stats.get('q75', np.nan) - delay_stats.get('q25', np.nan),
            'range': delay_stats.get('max', np.nan) - delay_stats.get('min', np.nan),
            
            # Dispersión relativa (adimensional)
            'cv': delay_stats.get('cv', np.nan),
            
            # Forma de la distribución
            'skewness': delay_stats.get('skewness', np.nan),
            'kurtosis': delay_stats.get('kurtosis', np.nan),
            
            # Límites
            'min': delay_stats.get('min', np.nan),
            'max': delay_stats.get('max', np.nan),
            
            # Cuartiles (útil para distribuciones asimétricas)
            'q25': delay_stats.get('q25', np.nan),
            'q75': delay_stats.get('q75', np.nan),
            
            # Info adicional
            'n_connections': delay_stats.get('n_connections', 0),
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Calcular CV si no viene calculado
    if 'cv' in df.columns and df['cv'].isna().any():
        mask = df['cv'].isna() & (df['mean'] > 0)
        df.loc[mask, 'cv'] = df.loc[mask, 'std'] / df.loc[mask, 'mean']
    
    return df


def plot_advanced_metrics_dashboard(results_db, trials_data=None, figsize=(18, 14), 
                                readable_names=True, sweep_dir=None, auto_save=True, dpi=300):
    """Dashboard avanzado con heatmaps, scatter plots y box plots"""
    
    # Preparar datos
    configs = list(results_db.keys())
    display_names = get_display_names(results_db, readable_names)
    n_configs = len(configs)
    
    if n_configs == 0:
        logger.info("No data to plot")
        return None
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # 1. Heatmap: Métricas vs Configuraciones  
    ax1 = fig.add_subplot(gs[0, :2])
    
    metrics_names = ['CC Peak', 'CC Lag', 'PLV α', 'PLI α', 'PLV γ', 'PLI γ', 'Coherence']
    heatmap_data = []
    
    for config in configs:
        agg = results_db[config]['aggregated']
        row = [
            agg['cross_corr_peak']['mean'],
            abs(agg['cross_corr_lag']['mean']),  # Use absolute lag
            agg.get('plv_alpha', {}).get('mean', 0),
            agg.get('pli_alpha', {}).get('mean', 0), 
            agg.get('plv_gamma', {}).get('mean', 0),
            agg.get('pli_gamma', {}).get('mean', 0),
            agg['coherence_peak']['mean']
        ]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    # Normalize each metric column to [0,1] for visualization
    heatmap_norm = np.zeros_like(heatmap_data)
    for j in range(heatmap_data.shape[1]):
        col = heatmap_data[:, j]
        if col.max() > col.min():
            heatmap_norm[:, j] = (col - col.min()) / (col.max() - col.min())
    
    im1 = ax1.imshow(heatmap_norm, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(len(metrics_names)))
    ax1.set_xticklabels(metrics_names, rotation=45, ha='right')
    # 1. Heatmap con nombres legibles
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels([name[:25] + "..." if len(name) > 25 else name for name in display_names], fontsize=8)
    ax1.set_title('Metrics Heatmap (Normalized)')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6)
    cbar1.set_label('Normalized Value')
    
    # 2. PLV vs Coherence scatter
    ax2 = fig.add_subplot(gs[0, 2])
    
    plv_alpha = [results_db[c]['aggregated'].get('plv_alpha', {}).get('mean', 0) for c in configs]
    coh_peak = [results_db[c]['aggregated']['coherence_peak']['mean'] for c in configs]
    
    scatter = ax2.scatter(plv_alpha, coh_peak, c=range(len(configs)), 
                         cmap='tab10', s=60, alpha=0.7)
    ax2.set_xlabel('PLV Alpha')
    ax2.set_ylabel('Coherence Peak')
    ax2.set_title('PLV vs Coherence')
    ax2.grid(True, alpha=0.3)
    
    # 3. Tau vs CC Lag scatter
    ax3 = fig.add_subplot(gs[0, 3])
    
    tau_avg = [(results_db[c]['aggregated'].get('tau_A', {}).get('mean', 0) + 
               results_db[c]['aggregated'].get('tau_B', {}).get('mean', 0)) / 2 for c in configs]
    cc_lag = [results_db[c]['aggregated']['cross_corr_lag']['mean'] for c in configs]
    
    ax3.scatter(tau_avg, cc_lag, c=range(len(configs)), cmap='tab10', s=60, alpha=0.7)
    ax3.set_xlabel('Avg Intrinsic Timescale (ms)')
    ax3.set_ylabel('CC Lag (ms)')
    ax3.set_title('Timescale vs Lag')
    ax3.grid(True, alpha=0.3)
    
    # 4-7. Box plots for key metrics (if trial data available)
    box_metrics = ['cc_peak', 'cc_lag_ms', 'plv_alpha', 'coh_peak']
    box_titles = ['Cross-Correlation Peak', 'Cross-Correlation Lag (ms)', 'PLV Alpha', 'Coherence Peak']
    
    if trials_data is not None:
        for i, (metric, title) in enumerate(zip(box_metrics, box_titles)):
            ax = fig.add_subplot(gs[1, i])
            
            # Prepare data for box plot
            box_data = []
            box_labels = []
            
            for config, display_name in zip(configs, display_names):
                if config in trials_data:
                    config_trials = trials_data[config]
                    if metric in config_trials.columns:
                        values = config_trials[metric].dropna()
                        if len(values) > 0:
                            if metric == 'cc_lag_ms':
                                values = np.abs(values)  # Use absolute lag
                            box_data.append(values)
                            box_labels.append(display_name[:15] + "..." if len(display_name) > 15 else display_name)
            
            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                
                # Color boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No trial data\navailable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
                ax.set_title(title)
    
    # 8. PLV/PLI Comparison (Gamma vs Alpha)
    ax8 = fig.add_subplot(gs[2, 0])

    plv_alpha = [results_db[c]['aggregated'].get('plv_alpha', {}).get('mean', 0) for c in configs]
    pli_alpha = [results_db[c]['aggregated'].get('pli_alpha', {}).get('mean', 0) for c in configs]

    ax8.scatter(plv_alpha, pli_alpha, c=range(len(configs)), cmap='tab10', s=60, alpha=0.7)
    ax8.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='PLV=PLI')
    ax8.set_xlabel('PLV Alpha')
    ax8.set_ylabel('PLI Alpha')
    ax8.set_title('PLV vs PLI (Alpha Band)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
        
    # 9. Error bars comparison
    ax9 = fig.add_subplot(gs[2, 1])
    
    x_pos = np.arange(len(configs))
    cc_means = [results_db[c]['aggregated']['cross_corr_peak']['mean'] for c in configs]
    cc_stds = [results_db[c]['aggregated']['cross_corr_peak']['std'] for c in configs]
    
    ax9.errorbar(x_pos, cc_means, yerr=cc_stds, fmt='o', capsize=3, alpha=0.7)
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels([c.replace('_', ' ')[:10] for c in configs], rotation=45, ha='right')
    ax9.set_ylabel('Cross-Correlation Peak')
    ax9.set_title('CC Peak with Error Bars')
    ax9.grid(True, alpha=0.3)
    
    # 10. Coherence bands comparison
    ax10 = fig.add_subplot(gs[2, 2])
    
    alpha_coh = [results_db[c]['aggregated'].get('alpha_coherence', {}).get('mean', 0) for c in configs]
    gamma_coh = [results_db[c]['aggregated'].get('gamma_coherence', {}).get('mean', 0) for c in configs]
    
    x_pos = np.arange(len(configs))
    width = 0.35
    
    ax10.bar(x_pos - width/2, alpha_coh, width, label='Alpha', alpha=0.7)
    ax10.bar(x_pos + width/2, gamma_coh, width, label='Gamma', alpha=0.7)
    ax10.set_xticks(x_pos)
    ax10.set_xticklabels([c.replace('_', ' ')[:10] for c in configs], rotation=45, ha='right')
    ax10.set_ylabel('Coherence')
    ax10.set_title('Band-Specific Coherence')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Summary statistics
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')
    
    # Calculate summary stats
    all_plv_alpha = [v for v in plv_alpha if v > 0]
    all_cc_peaks = [abs(results_db[c]['aggregated']['cross_corr_peak']['mean']) for c in configs]
    
    summary_text = f"""SUMMARY STATISTICS
    
        Configs analyzed: {len(configs)}
            
        PLV Alpha:
        Mean: {np.mean(all_plv_alpha):.3f}
        Range: {np.min(all_plv_alpha):.3f} - {np.max(all_plv_alpha):.3f}
            
        CC Peak:
        Mean: {np.mean(all_cc_peaks):.3f}
        Range: {np.min(all_cc_peaks):.3f} - {np.max(all_cc_peaks):.3f}
            
        Strong coupling: {sum(1 for v in all_plv_alpha if v > 0.3)}
        Weak coupling: {sum(1 for v in all_plv_alpha if v < 0.1)}
        """
    
    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    fig.suptitle(f'Advanced Metrics Dashboard (n={n_configs} configs)', fontsize=16)
    
    plt.tight_layout()
    
    if auto_save and sweep_dir:
        save_dir = os.path.join(sweep_dir, 'figures')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(os.path.join(save_dir, filename), dpi=dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"[SAVED] {os.path.join(save_dir, filename)} ({dpi} DPI)")
    
    return fig


def select_extreme_cases(results_db: Dict, metric: str = 'plv_alpha', 
                        n_cases: int = 3) -> Dict[str, List[str]]:
    """Selecciona casos extremos (high/low) para una métrica específica"""
    
    metric_values = []
    config_names = []
    
    for config_name, data in results_db.items():
        if metric in data['aggregated']:
            value = data['aggregated'][metric]['mean']
            metric_values.append(value)
            config_names.append(config_name)
    
    if len(metric_values) == 0:
        return {'high': [], 'low': []}
    
    # Ordenar por valor de métrica
    sorted_indices = np.argsort(metric_values)
    
    return {
        'high': [config_names[i] for i in sorted_indices[-n_cases:]],
        'low': [config_names[i] for i in sorted_indices[:n_cases]],
        'values': {config_names[i]: metric_values[i] for i in range(len(config_names))}
    }

# def plot_timeseries_overlay_comparison(sweep_dir, results_db, metric='plv_alpha',
#                                       dt_ms=0.25, figsize=(14, 8), auto_save=True, dpi=300) -> plt.Figure:
#     """Plot overlay de timeseries para casos extremos de una métrica"""
    
#     # Seleccionar casos extremos
#     extreme_cases = select_extreme_cases(results_db, metric, n_cases=2)
    
#     if len(extreme_cases['high']) == 0 or len(extreme_cases['low']) == 0:
#         logger.info(f"No hay suficientes datos para métrica {metric}")
#         return None
    
#     fig, axes = plt.subplots(2, 2, figsize=figsize)
#     fig.suptitle(f'Time Series Comparison: {metric.upper()}', fontsize=16)
    
#     # Casos high y low (tomar el más extremo)
#     cases = {
#         'HIGH': extreme_cases['high'][-1],
#         'LOW': extreme_cases['low'][0]
#     }
    
#     colors = {'HIGH': 'red', 'LOW': 'blue'}
    
#     for col, (case_type, config_name) in enumerate(cases.items()):
        
#         # Cargar datos raw
#         raw_data = load_raw_timeseries(sweep_dir, config_name, 1)
        
#         if raw_data is None:
#             logger.info(f"No data found for {config_name}")
#             continue
            
#         metric_val = extreme_cases['values'][config_name]
#         color = colors[case_type]
        
#         # Debug: print available keys
#         logger.info(f"Available keys in {case_type}: {list(raw_data.keys())[:10]}")
        
#         # 1. Firing rates plot
#         ax_rates = axes[0, col]
        
#         # Use the correct keys from your data structure
#         if 'rate_A' in raw_data and 'rate_B' in raw_data:
#             rate_A = raw_data['rate_A']
#             rate_B = raw_data['rate_B']
#             time_axis = raw_data.get('time', np.arange(len(rate_A)) * dt_ms)  # Use actual time or fallback
            
#             if len(rate_A) > 100 and len(rate_B) > 100:
#                 # Plot middle section for better visibility
#                 start_idx = len(rate_A) // 4
#                 end_idx = 3 * len(rate_A) // 4
                
#                 ax_rates.plot(time_axis[start_idx:end_idx], rate_A[start_idx:end_idx], 
#                              color=color, alpha=0.8, linewidth=1.5, label=f'Pop A')
#                 ax_rates.plot(time_axis[start_idx:end_idx], rate_B[start_idx:end_idx], 
#                              color=color, alpha=0.8, linewidth=1.5, linestyle='--', label=f'Pop B')
        
#         ax_rates.set_title(f'{case_type}: {metric}={metric_val:.3f}')
#         ax_rates.set_xlabel('Time (ms)')
#         ax_rates.set_ylabel('Rate (Hz)')
#         ax_rates.legend()
#         ax_rates.grid(True, alpha=0.3)
        
#         # 2. Cross-correlation plot (instead of raster since no spike data)
#         ax_xcorr = axes[1, col]
        
#         if 'cross_correlation' in raw_data:
#             xcorr_data = raw_data['cross_correlation']
#             if isinstance(xcorr_data, dict) and 'lags' in xcorr_data and 'values' in xcorr_data:
#                 lags = xcorr_data['lags']
#                 values = xcorr_data['values']
#                 ax_xcorr.plot(lags, values, color=color, linewidth=2, alpha=0.8)
#             elif isinstance(xcorr_data, np.ndarray):
#                 # If it's just the correlation values, create lag axis
#                 n_lags = len(xcorr_data)
#                 lags = np.arange(-n_lags//2, n_lags//2 + 1) * dt_ms  # Assuming dt=0.05ms
#                 ax_xcorr.plot(lags[:len(xcorr_data)], xcorr_data, color=color, linewidth=2, alpha=0.8)
#         else:
#             # Calculate cross-correlation if not pre-computed
#             if 'rate_A' in raw_data and 'rate_B' in raw_data:
#                 rate_A = raw_data['rate_A'][5000:15000]
#                 rate_B = raw_data['rate_B'][5000:15000]
#                 xcorr = np.correlate(rate_A - rate_A.mean(), rate_B - rate_B.mean(), mode='full')
#                 xcorr = xcorr / np.max(np.abs(xcorr) + 1e-9)
#                 lags = np.arange(-len(rate_A) + 1, len(rate_A)) * dt_ms
#                 ax_xcorr.plot(lags, xcorr, color=color, linewidth=2, alpha=0.8)

#                 # línea vertical en el lag detectado si viene del pickle
#                 lag_ms = None
#                 cc = raw_data.get('cross_correlation', {})
#                 if isinstance(cc, dict) and 'peak_lag' in cc:
#                     lag_ms = cc['peak_lag']
#                 if lag_ms is not None:
#                     ax_xcorr.axvline(lag_ms, linestyle='--', color=color, alpha=0.6)
        
#         ax_xcorr.set_title(f'Cross-Correlation - {case_type}')
#         ax_xcorr.set_xlabel('Lag (ms)')
#         ax_xcorr.set_ylabel('Correlation')
#         ax_xcorr.axvline(0, color='k', linestyle=':', alpha=0.5)
#         ax_xcorr.grid(True, alpha=0.3)
        
#         delay_params = results_db[config_name].get('delay_statistics', {})
        
#         if delay_params:
#             stats_text = f"n={delay_params.get('n_connections', 0)}\n" \
#                         f"μ={delay_params.get('mean', 0):.2f}ms\n" \
#                         f"σ={delay_params.get('std', 0):.2f}ms"
#             ax_rates.text(0.02, 0.98, stats_text, transform=ax_rates.transAxes,
#                         verticalalignment='top', bbox=dict(boxstyle='round', 
#                         facecolor='wheat', alpha=0.5), fontsize=8)
    
#     plt.tight_layout()
    
#     if auto_save:
#         save_dir = os.path.join(sweep_dir, 'figures')
#         os.makedirs(save_dir, exist_ok=True)
#         filename = f"timeseries_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#         fig.savefig(os.path.join(save_dir, filename), dpi=dpi, bbox_inches='tight', facecolor='white')
#         logger.info(f"[SAVED] {os.path.join(save_dir, filename)} ({dpi} DPI)")
        
#     return fig

# def plot_delay_distribution_comparison(sweep_dir: str, results_db: Dict,
#                                     delay_types: List[str] = ['DELTA', 'LOGNORMAL', 'GAMMA', 'UNIFORM', 'BETA'],
#                                     figsize: Tuple[int, int] = (15, 8), auto_save=True, dpi=300) -> plt.Figure:
#     """Compara timeseries para diferentes tipos de distribución de delays"""
    
#     fig, axes = plt.subplots(2, len(delay_types), figsize=figsize)
#     fig.suptitle('Delay Distribution Comparison', fontsize=16)
    
#     colors = {'DELTA': 'blue', 'LOGNORMAL': 'orange', 'GAMMA': 'red', 'BETA': 'green', 'UNIFORM': 'purple'}
    
#     for col, delay_type in enumerate(delay_types):
        
#         # Buscar configuración representativa de este tipo
#         config_name = None
#         for name in results_db.keys():
#             if delay_type in name.upper():
#                 config_name = name
#                 break
                
#         if config_name is None:
#             continue
            
#         # Cargar datos
#         raw_data = load_raw_timeseries(sweep_dir, config_name, 1)
#         if raw_data is None:
#             continue
            
#         color = colors.get(delay_type, 'black')
        
#         delay_params = results_db[config_name]['delay_statistics']
        
#         plv = results_db[config_name]['aggregated']['plv_alpha']['mean']
#         cc = results_db[config_name]['aggregated']['cross_corr_peak']['mean']
        
#         # Firing rates
#         ax_rates = axes[0, col]
#         if 'rate_A' in raw_data and 'rate_B' in raw_data:
#             time_axis = raw_data.get('time_ms', np.arange(len(raw_data['rate_A']))) / 1000
            
#             ax_rates.plot(time_axis, raw_data['rate_A'], color=color, 
#                         alpha=0.8, linewidth=1.5, label=f'Pop A (PLV α={plv:.2f})')
#             ax_rates.plot(time_axis, raw_data['rate_B'], color=color, 
#                         alpha=0.8, linewidth=1.5, linestyle='--', label=f'Pop B (CC={cc:.2f})')
            
#         # Añadir textbox en cada subplot con estadísticas:
#         stats_text = f"n={delay_params['n_connections']}\n" \
#                     f"μ={delay_params['mean']:.2f}ms\n" \
#                     f"σ={delay_params['std']:.2f}ms"
#         ax_rates.text(0.02, 0.98, stats_text, transform=ax_rates.transAxes,
#                     verticalalignment='top', bbox=dict(boxstyle='round', 
#                     facecolor='wheat', alpha=0.5), fontsize=8)

    
#         title = f"{delay_type} (μ={delay_params['mean']:.1f}±{delay_params['std']:.1f}ms)"
#         ax_rates.set_title(title, fontsize=10, fontweight='bold')
#         ax_rates.set_xlim(1, 3)
#         ax_rates.set_ylabel('Rate (Hz)')
#         ax_rates.legend(loc='upper right', fontsize=9, framealpha=0.9)
#         ax_rates.grid(True, alpha=0.3)
        
#         # Phase relationship (usando transformada de Hilbert aproximada)
#         ax_phase = axes[1, col]
#         if 'rate_A' in raw_data and 'rate_B' in raw_data:
            
#             rate_A = raw_data['rate_A'][5000:15000]  # Ventana estable
#             rate_B = raw_data['rate_B'][5000:15000]
            
#             # Aproximación simple de diferencia de fase
#             phase_diff = _phase_diff_band(rate_A, rate_B, 1000, band=(8,12))
            
#             freqs = np.fft.fftfreq(len(rate_A), d=0.0001)  # dt = 0.5ms
            
#             # Focus en rango de frecuencias relevante
#             freq_mask = (freqs >= 8) & (freqs <= 50)
            
#             ax_phase.scatter(freqs[freq_mask], phase_diff[freq_mask], 
#                            c=color, alpha=0.6, s=10)
            
#         ax_phase.set_title(f'Phase Relationship - {delay_type}')
#         ax_phase.set_xlabel('Frequency (Hz)')
#         ax_phase.set_ylabel('Phase Diff (rad)')
#         ax_phase.grid(True, alpha=0.3)
#         ax_phase.axhline(0, color='k', linestyle=':', alpha=0.5)
    
#     plt.tight_layout()
    
#     if auto_save:
#         save_dir = os.path.join(sweep_dir, 'figures')
#         os.makedirs(save_dir, exist_ok=True)
        
#         filename = f"delay_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#         save_path = os.path.join(save_dir, filename)
        
#         fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
#         logger.info(f"[SAVED] {save_path} ({dpi} DPI)")
        
#     return fig

# Función principal de alto nivel
# def generate_representative_timeseries_plots(sweep_dir: str, results_db: Dict,
#                                            metrics: List[str] = ['plv_alpha', 'cross_corr_peak'], dt=0.25) -> Dict[str, plt.Figure]:
#     """Genera todos los plots de timeseries representativos"""
    
#     figures = {}
    
#     # 1. Comparación por métrica
#     for metric in metrics:
#         fig = plot_timeseries_overlay_comparison(sweep_dir, results_db, metric, dt)
#         if fig is not None:
#             figures[f'timeseries_comparison_{metric}'] = fig
    
#     # 2. Comparación por tipo de delay
#     fig_delays = plot_delay_distribution_comparison(sweep_dir, results_db)
#     if fig_delays is not None:
#         figures['delay_distribution_comparison'] = fig_delays
    
#     return figures


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
            try:
                if part.replace('.','').replace('p','.').isdigit():
                    value = float(part.replace('p', '.'))
                    return {'type': 'constant', 'value': value}
            except:
                pass
    
    # Default fallback
    return {'type': 'constant', 'value': 0}

def extract_metrics_data(results_db: Dict) -> pd.DataFrame:
    """Extrae métricas de conectividad funcional"""
    metrics_data = []
    
    for config, data in results_db.items():
        agg = data['aggregated']
        
        metrics = {
            'config': config,
            
            # Phase metrics
            'plv_alpha': agg.get('plv_alpha', {}).get('mean', np.nan),
            'pli_alpha': agg.get('pli_alpha', {}).get('mean', np.nan),
            'plv_gamma': agg.get('plv_gamma', {}).get('mean', np.nan),
            'pli_gamma': agg.get('pli_gamma', {}).get('mean', np.nan),
            
            # Cross-correlation
            'cc_peak': abs(agg['cross_corr_peak']['mean']),
            'cc_lag': abs(agg['cross_corr_lag']['mean']),
            
            # Spectral
            'coherence_peak': agg['coherence_peak']['mean'],
            'alpha_coherence': agg.get('alpha_coherence', {}).get('mean', np.nan),
            'gamma_coherence': agg.get('gamma_coherence', {}).get('mean', np.nan),
            
            # Intrinsic timescales
            'tau_A': agg.get('tau_A', {}).get('mean', np.nan),
            'tau_B': agg.get('tau_B', {}).get('mean', np.nan),
            'tau_avg': (agg.get('tau_A', {}).get('mean', 0) + 
                       agg.get('tau_B', {}).get('mean', 0)) / 2,
        }
        
        metrics_data.append(metrics)
    
    return pd.DataFrame(metrics_data)

# ============================================================================
# ANÁLISIS DE CORRELACIONES
# ============================================================================

def calculate_correlations(delay_df: pd.DataFrame, 
                          metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcula correlaciones entre métricas estadísticas de delays y métricas de conectividad.
    
    Args:
        delay_df: DataFrame con métricas generalizadas de delays
        metrics_df: DataFrame con métricas de conectividad
        
    Returns:
        (corr_matrix, p_matrix, combined_df)
    """
    # Merge dataframes
    combined = pd.merge(delay_df, metrics_df, on='config')
    
    # Seleccionar columnas de estadísticas de delays
    delay_cols = ['mean', 'std', 'cv', 'skewness', 'kurtosis', 
                  'iqr', 'range', 'median', 'min', 'max']
    delay_cols = [col for col in delay_cols if col in combined.columns]
    
    # Filtrar columnas constantes (sin variabilidad)
    valid_delay_cols = []
    for col in delay_cols:
        if combined[col].std() > 1e-10:
            valid_delay_cols.append(col)
    
    # Métricas de conectividad
    metric_cols = ['plv_alpha', 'pli_alpha', 'plv_gamma', 'pli_gamma',
                   'cc_peak', 'cc_lag', 'coherence_peak', 
                   'alpha_coherence', 'gamma_coherence', 'tau_avg']
    metric_cols = [col for col in metric_cols if col in combined.columns]
    
    # Calcular correlaciones
    correlations = {}
    p_values = {}
    
    for delay_param in valid_delay_cols:
        correlations[delay_param] = {}
        p_values[delay_param] = {}
        
        for metric in metric_cols:
            # Remover NaN
            mask = ~(np.isnan(combined[delay_param]) | np.isnan(combined[metric]))
            
            if mask.sum() > 2:  # Mínimo 3 puntos
                try:
                    corr, p_val = pearsonr(combined[delay_param][mask], 
                                          combined[metric][mask])
                    correlations[delay_param][metric] = corr
                    p_values[delay_param][metric] = p_val
                except:
                    correlations[delay_param][metric] = np.nan
                    p_values[delay_param][metric] = np.nan
            else:
                correlations[delay_param][metric] = np.nan
                p_values[delay_param][metric] = np.nan
    
    return pd.DataFrame(correlations).T, pd.DataFrame(p_values).T, combined


def plot_correlation_analysis(delay_df: pd.DataFrame, 
                              metrics_df: pd.DataFrame,
                              figsize: Tuple[int, int] = (18, 14),
                              sweep_dir: str = None,
                              auto_save: bool = True,
                              dpi: int = 300) -> Tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Dashboard completo de análisis de correlaciones con métricas generalizadas.
    """
    corr_matrix, p_matrix, combined = calculate_correlations(delay_df, metrics_df)
    
    if len(corr_matrix) == 0:
        print("No valid correlations found")
        return None, None, None
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # ========================================================================
    # 1. HEATMAP PRINCIPAL DE CORRELACIONES
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    im = ax1.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', 
                    vmin=-1, vmax=1)
    
    ax1.set_xticks(range(len(corr_matrix.columns)))
    ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax1.set_yticks(range(len(corr_matrix.index)))
    ax1.set_yticklabels([param.replace('_', ' ').title() 
                         for param in corr_matrix.index])
    ax1.set_title(f'Delay Statistics vs Connectivity Metrics (n={len(delay_df)})', 
                  fontsize=14, fontweight='bold')
    
    # Añadir valores y significancia
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]
            
            if not np.isnan(corr_val):
                stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                
                ax1.text(j, i, f'{corr_val:.2f}\n{stars}', 
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(corr_val) > 0.5 else 'black',
                        fontweight='bold' if p_val < 0.05 else 'normal')
    
    plt.colorbar(im, ax=ax1, shrink=0.8, label='Pearson r')
    
    # ========================================================================
    # 2-7. SCATTER PLOTS DE CORRELACIONES MÁS FUERTES
    # ========================================================================
    significant_corrs = []
    for delay_param in corr_matrix.index:
        for metric in corr_matrix.columns:
            corr_val = corr_matrix.loc[delay_param, metric]
            p_val = p_matrix.loc[delay_param, metric]
            if not np.isnan(corr_val) and p_val < 0.05:
                significant_corrs.append((delay_param, metric, abs(corr_val), 
                                        p_val, corr_val))
    
    # Top 6 por magnitud
    significant_corrs.sort(key=lambda x: x[2], reverse=True)
    top_corrs = significant_corrs[:6]
    
    for plot_idx, (delay_param, metric, _, p_val, corr_val) in enumerate(top_corrs):
        ax = fig.add_subplot(gs[1 + plot_idx//3, plot_idx%3])
        
        x = combined[delay_param]
        y = combined[metric]
        
        mask = ~(np.isnan(x) | np.isnan(y))
        
        if mask.sum() > 2:
            # Scatter coloreado por tipo de distribución
            if 'type' in combined.columns:
                types = combined['type'][mask]
                for dist_type in types.unique():
                    type_mask = types == dist_type
                    ax.scatter(x[mask][type_mask], y[mask][type_mask], 
                             alpha=0.7, s=60, label=dist_type)
                ax.legend(fontsize=8, loc='best')
            else:
                ax.scatter(x[mask], y[mask], alpha=0.7, s=60)
            
            # Línea de tendencia
            try:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x[mask])
                ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)
            except:
                pass
            
            # Info de correlación
            direction = "↑" if corr_val > 0 else "↓"
            strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.5 else "Weak"
            
            ax.text(0.05, 0.95, f'{direction} r={corr_val:.3f}\n{strength}\np={p_val:.4f}', 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
                   verticalalignment='top')
        
        ax.set_xlabel(delay_param.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Llenar espacios vacíos
    for plot_idx in range(len(top_corrs), 6):
        ax = fig.add_subplot(gs[1 + plot_idx//3, plot_idx%3])
        ax.text(0.5, 0.5, 'No additional\nsignificant\ncorrelations', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=11, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # ========================================================================
    # 8. SUMMARY BOX
    # ========================================================================
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Estadísticas de resumen
    n_significant = len(significant_corrs)
    n_strong = sum(1 for _, _, mag, _, _ in significant_corrs if mag > 0.7)
    n_moderate = sum(1 for _, _, mag, _, _ in significant_corrs if 0.5 < mag <= 0.7)
    
    # Top 3 correlaciones
    top3_text = "\n".join([
        f"  {i+1}. {param.upper()} ↔ {metric.upper()}: r={corr:.3f} (p={p:.4f})"
        for i, (param, metric, _, p, corr) in enumerate(top_corrs[:3])
    ])
    
    summary_text = f"""CORRELATION SUMMARY
    
        Total configurations: {len(delay_df)}
        Significant correlations (p<0.05): {n_significant}
        • Strong (|r|>0.7): {n_strong}
        • Moderate (0.5<|r|≤0.7): {n_moderate}
        • Weak (|r|≤0.5): {n_significant - n_strong - n_moderate}

        Top 3 Strongest Correlations:
        {top3_text if top3_text else "  None found"}

        Key Insights:
        • Mean delay: Primary predictor of {corr_matrix.loc['mean'].abs().idxmax() if 'mean' in corr_matrix.index else 'N/A'}
        • CV (variability): Primary predictor of {corr_matrix.loc['cv'].abs().idxmax() if 'cv' in corr_matrix.index else 'N/A'}
        • Skewness: Primary predictor of {corr_matrix.loc['skewness'].abs().idxmax() if 'skewness' in corr_matrix.index else 'N/A'}
        """
            
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Generalized Delay Statistics → Connectivity Metrics Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if auto_save and sweep_dir:
        save_dir = os.path.join(sweep_dir, 'figures')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"correlation_generalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(os.path.join(save_dir, filename), dpi=dpi, 
                   bbox_inches='tight', facecolor='white')
        print(f"[SAVED] {os.path.join(save_dir, filename)}")
    
    return fig, corr_matrix, p_matrix


def print_correlation_summary(corr_matrix: pd.DataFrame, 
                              p_matrix: pd.DataFrame, 
                              threshold: float = 0.05):
    """Imprime resumen textual de correlaciones significativas"""
    print("\n" + "="*80)
    print("SIGNIFICANT CORRELATIONS (p < 0.05)")
    print("="*80)
    
    for delay_param in corr_matrix.index:
        has_significant = False
        param_text = []
        
        for metric in corr_matrix.columns:
            corr = corr_matrix.loc[delay_param, metric]
            p_val = p_matrix.loc[delay_param, metric]
            
            if not np.isnan(corr) and p_val < threshold:
                has_significant = True
                strength = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.5 else "weak"
                direction = "↑" if corr > 0 else "↓"
                param_text.append(
                    f"  {direction} {metric.upper()}: r={corr:+.3f} ({strength}, p={p_val:.4f})"
                )
        
        if has_significant:
            print(f"\n{delay_param.upper().replace('_', ' ')}:")
            print("\n".join(param_text))
    
    print("\n" + "="*80)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

from src.two_populations.sweep import plot_delay_comparison_with_distributions
from src.two_populations.plots.plots_sweeps import (
            extract_metrics_data, plot_correlation_analysis, print_correlation_summary
        )
        
from src.two_populations.helpers.helpers import extract_delay_parameters

    
def save_all_sweep_plots(sweep_dir, results_db, trials_data=None, dpi=300, auto_save=True):
    """Auto-genera y guarda todos los plots del sweep
        Args:
            results_db: Diccionario con resultados del sweep
            sweep_dir: Directorio para guardar figuras
            auto_save: Si guardar automáticamente
            dpi: Resolución de figuras
    """
    
    figures = {}
    
    # Dashboard (se auto-guarda internamente)
    figures['dashboard'] = plot_advanced_metrics_dashboard(
        results_db, trials_data, sweep_dir=sweep_dir, auto_save=auto_save, dpi=dpi)
    
    # Extraer datos
    delay_df = extract_delay_statistics(results_db)
    metrics_df = extract_metrics_data(results_db)
    
    if delay_df.empty or metrics_df.empty:
        print("Insufficient data for analysis")
        return None, None, None, None, None
    
    logger.info(f"\nAnalyzing {len(delay_df)} configurations")
    logger.info(f"Delay statistics: {[c for c in delay_df.columns if c not in ['config', 'type', 'n_connections']]}")
    logger.info(f"Connectivity metrics: {[c for c in metrics_df.columns if c != 'config']}")
    
    # Análisis de correlaciones
    fig, corr_matrix, p_matrix = plot_correlation_analysis(
        delay_df, metrics_df, 
        sweep_dir=sweep_dir, 
        auto_save=auto_save, 
        dpi=dpi
    )
    
    # Imprimir resumen
    if corr_matrix is not None:
        print_correlation_summary(corr_matrix, p_matrix)
    
    # Delay comparison (ya tiene auto_save=True por defecto)
    figures['delay_comparison'] = plot_delay_comparison_with_distributions(
        results_db, sweep_dir if auto_save else None)
    
    figures[f'correlation_analysis'] = fig
    
    logger.info(f"[SAVED] {len(figures)} figures total")
    return figures, corr_matrix, p_matrix, delay_df, metrics_df