import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit
import seaborn as sns
from brian2 import ms

import os
from scipy import signal as sg

import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit
import seaborn as sns


def plot_raster_results(results, N_exc=800, N_total=1000):
    """Graficar raster plots usando resultados de la clase"""
    fig = plt.figure(figsize=(14, 8))  # Asignar a fig

    # Grupo A
    plt.subplot(1, 2, 1)
    spike_mon_A = results['A']['spike_monitor']
    
    # Separar exc/inh por índices
    exc_mask_A = spike_mon_A.i < N_exc
    inh_mask_A = spike_mon_A.i >= N_exc
    
    plt.plot(spike_mon_A.t[exc_mask_A]/ms, spike_mon_A.i[exc_mask_A], '.k', markersize=0.7)
    plt.plot(spike_mon_A.t[inh_mask_A]/ms, spike_mon_A.i[inh_mask_A], '.k', markersize=0.7)
    plt.axhline(y=N_exc, color='r', linestyle='-', linewidth=1)
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Neuronas A')
    plt.title('Raster Plot A - Estilo Paper')
    plt.ylim(0, N_total)

    # Grupo B
    plt.subplot(1, 2, 2)
    spike_mon_B = results['B']['spike_monitor']
    
    exc_mask_B = spike_mon_B.i < N_exc
    inh_mask_B = spike_mon_B.i >= N_exc
    
    plt.plot(spike_mon_B.t[exc_mask_B]/ms, spike_mon_B.i[exc_mask_B], '.k', markersize=0.7)
    plt.plot(spike_mon_B.t[inh_mask_B]/ms, spike_mon_B.i[inh_mask_B], '.k', markersize=0.7)
    plt.axhline(y=N_exc, color='r', linestyle='-', linewidth=1)
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Neuronas B')
    plt.title('Raster Plot B - Estilo Paper')
    plt.ylim(0, N_total)
    
    #plt.show()
    return fig

def plot_cross_correlation_detailed(results_dict, figsize=(15, 10)):
    """Cross-correlation analysis detallado"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Cross-correlation curves
    ax1 = axes[0, 0]
    for condition, results in valid_results.items():
        cc = results['cross_correlation']
        ax1.plot(cc['lags'], cc['correlation'], label=condition, linewidth=2)
        # Mark peak
        ax1.axvline(cc['peak_lag'], color='red', linestyle='--', alpha=0.7)
        ax1.axhline(cc['peak_value'], color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Lag (ms)')
    ax1.set_ylabel('Cross-correlation')
    ax1.set_title('Cross-correlation Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Peak values comparison
    ax2 = axes[0, 1]
    conditions = list(valid_results.keys())
    peak_values = [valid_results[c]['cross_correlation']['peak_value'] for c in conditions]
    peak_lags = [valid_results[c]['cross_correlation']['peak_lag'] for c in conditions]
    
    bars = ax2.bar(conditions, peak_values, alpha=0.7)
    ax2.set_ylabel('Peak Cross-correlation')
    ax2.set_title('Peak Cross-correlation Values')
    ax2.grid(True, alpha=0.3)
    
    # Add lag annotations
    for bar, lag in zip(bars, peak_lags):
        ax2.annotate(f'{lag:.1f}ms', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center')
    
    # 3. Autocorrelation comparison
    ax3 = axes[1, 0]
    for condition, results in valid_results.items():
        ac_A = results['autocorr_A']
        ac_B = results['autocorr_B']
        ax3.plot(ac_A['lags'], ac_A['correlation'], 
                label=f"{condition} Pop A", alpha=0.8, linewidth=2)
        ax3.plot(ac_B['lags'], ac_B['correlation'], 
                label=f"{condition} Pop B", alpha=0.8, linewidth=2, linestyle='--')
    ax3.set_xlabel('Lag (ms)')
    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('Autocorrelation Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cross-correlation zero-lag distribution
    ax4 = axes[1, 1]
    zero_lag_values = []
    for condition, results in valid_results.items():
        cc = results['cross_correlation']
        lags = cc['lags']
        corr = cc['correlation']
        # Find value at zero lag
        zero_idx = np.argmin(np.abs(lags))
        zero_lag_val = corr[zero_idx]
        zero_lag_values.append(zero_lag_val)
    
    ax4.bar(conditions, zero_lag_values, alpha=0.7, color='orange')
    ax4.set_ylabel('Zero-lag Cross-correlation')
    ax4.set_title('Instantaneous Correlation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_phase_analysis(results_dict, figsize=(15, 8)):
    """Análisis de fases detallado"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    bands = ['alpha', 'beta', 'gamma']
    
    # PLV vs PLI comparison por banda
    for i, band in enumerate(bands):
        ax = axes[0, i]
        for condition, results in valid_results.items():
            plv = results['plv_pli'][band]['plv']
            pli = results['plv_pli'][band]['pli']
            ax.scatter(plv, pli, s=100, label=condition, alpha=0.7)
        
        ax.set_xlabel('PLV')
        ax.set_ylabel('PLI')
        ax.set_title(f'{band.title()} Band')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Phase stability comparison
    ax1 = axes[1, 0]
    conditions = list(valid_results.keys())
    x_pos = np.arange(len(conditions))
    width = 0.25
    
    for i, band in enumerate(bands):
        stabilities = [valid_results[c]['plv_pli'][band]['phase_stability'] for c in conditions]
        ax1.bar(x_pos + i*width, stabilities, width, label=band, alpha=0.8)
    
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Phase Stability')
    ax1.set_title('Phase Stability by Band')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(conditions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PLV summary heatmap
    ax2 = axes[1, 1]
    plv_matrix = []
    for condition in conditions:
        row = [valid_results[condition]['plv_pli'][band]['plv'] for band in bands]
        plv_matrix.append(row)
    
    im = ax2.imshow(plv_matrix, cmap='viridis', aspect='auto')
    ax2.set_xticks(range(len(bands)))
    ax2.set_xticklabels(bands)
    ax2.set_yticks(range(len(conditions)))
    ax2.set_yticklabels(conditions)
    ax2.set_title('PLV Heatmap')
    plt.colorbar(im, ax=ax2)
    
    # Add text annotations
    for i in range(len(conditions)):
        for j in range(len(bands)):
            text = ax2.text(j, i, f'{plv_matrix[i][j]:.3f}',
                           ha="center", va="center", color="white")
    
    # PLI summary heatmap  
    ax3 = axes[1, 2]
    pli_matrix = []
    for condition in conditions:
        row = [valid_results[condition]['plv_pli'][band]['pli'] for band in bands]
        pli_matrix.append(row)
    
    im = ax3.imshow(pli_matrix, cmap='plasma', aspect='auto')
    ax3.set_xticks(range(len(bands)))
    ax3.set_xticklabels(bands)
    ax3.set_yticks(range(len(conditions)))
    ax3.set_yticklabels(conditions)
    ax3.set_title('PLI Heatmap')
    plt.colorbar(im, ax=ax3)
    
    # Add text annotations
    for i in range(len(conditions)):
        for j in range(len(bands)):
            text = ax3.text(j, i, f'{pli_matrix[i][j]:.3f}',
                           ha="center", va="center", color="white")
    
    plt.tight_layout()
    return fig


def plot_spectral_detailed(results_dict, figsize=(15, 10)):
    """Análisis espectral detallado"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Power spectra comparison
    ax1 = axes[0, 0]
    for condition, results in valid_results.items():
        psd_A = results['psd_A']
        psd_B = results['psd_B']
        ax1.semilogy(psd_A['freqs'], psd_A['psd'], 
                    label=f"{condition} Pop A", alpha=0.8, linewidth=2)
        ax1.semilogy(psd_B['freqs'], psd_B['psd'], 
                    label=f"{condition} Pop B", alpha=0.8, linewidth=2, linestyle='--')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title('Power Spectra')
    ax1.set_xlim(0, 100)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Coherence with confidence intervals
    ax2 = axes[0, 1]
    for condition, results in valid_results.items():
        coh = results['coherence']
        ax2.plot(coh['freqs'], coh['coherence'], label=condition, linewidth=2)
        # Mark peak
        ax2.axvline(coh['peak_freq'], color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Coherence')
    ax2.set_title('Spectral Coherence')
    ax2.set_xlim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Band power comparison
    ax3 = axes[1, 0]
    conditions = list(valid_results.keys())
    x_pos = np.arange(len(conditions))
    width = 0.35
    
    alpha_power_A = [valid_results[c]['psd_A']['alpha_power'] for c in conditions]
    alpha_power_B = [valid_results[c]['psd_B']['alpha_power'] for c in conditions]
    gamma_power_A = [valid_results[c]['psd_A']['gamma_power'] for c in conditions]
    gamma_power_B = [valid_results[c]['psd_B']['gamma_power'] for c in conditions]
    
    ax3.bar(x_pos - width/2, alpha_power_A, width/2, label='Pop A Alpha', alpha=0.8)
    ax3.bar(x_pos, alpha_power_B, width/2, label='Pop B Alpha', alpha=0.8)
    ax3.bar(x_pos + width/2, gamma_power_A, width/2, label='Pop A Gamma', alpha=0.6)
    ax3.bar(x_pos + width, gamma_power_B, width/2, label='Pop B Gamma', alpha=0.6)
    
    ax3.set_xlabel('Condition')
    ax3.set_ylabel('Band Power')
    ax3.set_title('Alpha/Gamma Power by Population')
    ax3.set_xticks(x_pos + width/4)
    ax3.set_xticklabels(conditions)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Coherence band analysis
    ax4 = axes[1, 1]
    alpha_coh = [valid_results[c]['coherence']['alpha_coherence'] for c in conditions]
    gamma_coh = [valid_results[c]['coherence']['gamma_coherence'] for c in conditions]
    
    ax4.bar(x_pos - width/2, alpha_coh, width, label='Alpha Coherence', alpha=0.8)
    ax4.bar(x_pos + width/2, gamma_coh, width, label='Gamma Coherence', alpha=0.8)
    
    ax4.set_xlabel('Condition')
    ax4.set_ylabel('Coherence')
    ax4.set_title('Band-specific Coherence')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(conditions)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_timescales_detailed(results_dict, figsize=(12, 8)):
    """Análisis de escalas temporales intrínsecas detallado"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Intrinsic timescales comparison
    ax1 = axes[0, 0]
    conditions = list(valid_results.keys())
    x_pos = np.arange(len(conditions))
    width = 0.35
    
    tau_A = [valid_results[c]['int_A']['tau'] for c in conditions]
    tau_B = [valid_results[c]['int_B']['tau'] for c in conditions]
    quality_A = [valid_results[c]['int_A']['fit_quality'] for c in conditions]
    quality_B = [valid_results[c]['int_B']['fit_quality'] for c in conditions]
    
    bars_A = ax1.bar(x_pos - width/2, tau_A, width, label='Pop A', alpha=0.8)
    bars_B = ax1.bar(x_pos + width/2, tau_B, width, label='Pop B', alpha=0.8)
    
    # Color code by quality
    quality_colors = {'good': 'green', 'moderate': 'orange', 'poor': 'red', 'very_poor': 'darkred'}
    for bar, quality in zip(bars_A, quality_A):
        bar.set_edgecolor(quality_colors.get(quality, 'black'))
        bar.set_linewidth(2)
    for bar, quality in zip(bars_B, quality_B):
        bar.set_edgecolor(quality_colors.get(quality, 'black'))
        bar.set_linewidth(2)
    
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Intrinsic Timescale (ms)')
    ax1.set_title('Intrinsic Timescales')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(conditions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Threshold crossing analysis
    ax2 = axes[0, 1]
    threshold_lags_A = [valid_results[c]['int_A']['threshold_lag'] for c in conditions]
    threshold_lags_B = [valid_results[c]['int_B']['threshold_lag'] for c in conditions]
    
    ax2.bar(x_pos - width/2, threshold_lags_A, width, label='Pop A', alpha=0.8)
    ax2.bar(x_pos + width/2, threshold_lags_B, width, label='Pop B', alpha=0.8)
    
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Threshold Crossing Lag (ms)')
    ax2.set_title('Autocorrelation Threshold Crossing')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conditions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Quality distribution
    ax3 = axes[1, 0]
    all_qualities = quality_A + quality_B
    quality_counts = {q: all_qualities.count(q) for q in set(all_qualities)}
    
    ax3.pie(quality_counts.values(), labels=quality_counts.keys(), autopct='%1.1f%%')
    ax3.set_title('Fit Quality Distribution')
    
    # 4. Timescale vs Cross-correlation
    ax4 = axes[1, 1]
    for condition, results in valid_results.items():
        tau_avg = (results['int_A']['tau'] + results['int_B']['tau']) / 2
        cc_peak = abs(results['cross_correlation']['peak_value'])
        ax4.scatter(tau_avg, cc_peak, s=100, label=condition, alpha=0.7)
    
    ax4.set_xlabel('Average Intrinsic Timescale (ms)')
    ax4.set_ylabel('|Cross-correlation Peak|')
    ax4.set_title('Timescale vs Cross-correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_raster_comparison(results_dict, figsize=(15, 10)):
    """Raster plots comparativos entre condiciones"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None
    
    n_conditions = len(valid_results)
    fig, axes = plt.subplots(n_conditions, 2, figsize=figsize)
    if n_conditions == 1:
        axes = axes.reshape(1, -1)
    
    neuron_limit = 100
    time_window = (0, 2000)  # ms
    
    for i, (condition, results) in enumerate(valid_results.items()):
        t0 = float(results.get('t0_ms', 0.0))
        
        # Pop A
        ax_A = axes[i, 0]
        tA = results['spike_times_A']
        nA = results['spike_neurons_A']
        maskA = (tA >= t0 + time_window[0]) & (tA < t0 + time_window[1]) & (nA < neuron_limit)
        
        ax_A.scatter(tA[maskA] - t0, nA[maskA], s=0.5, alpha=0.6, color='blue')
        ax_A.set_xlim(time_window)
        ax_A.set_ylim(0, neuron_limit)
        ax_A.set_ylabel('Neuron ID')
        ax_A.set_title(f'{condition} - Population A')
        ax_A.grid(True, alpha=0.3)
        
        # Pop B  
        ax_B = axes[i, 1]
        tB = results['spike_times_B']
        nB = results['spike_neurons_B']
        maskB = (tB >= t0 + time_window[0]) & (tB < t0 + time_window[1]) & (nB < neuron_limit)
        
        ax_B.scatter(tB[maskB] - t0, nB[maskB], s=0.5, alpha=0.6, color='red')
        ax_B.set_xlim(time_window)
        ax_B.set_ylim(0, neuron_limit)
        ax_B.set_ylabel('Neuron ID')
        ax_B.set_title(f'{condition} - Population B')
        ax_B.grid(True, alpha=0.3)
        
        if i == n_conditions - 1:
            ax_A.set_xlabel('Time (ms)')
            ax_B.set_xlabel('Time (ms)')
    
    plt.tight_layout()
    return fig


# Función wrapper para llamar todos los plots
def plot_all_analysis(results_dict, save_prefix=None):
    """Generar todos los plots de análisis"""
    plots = {}
    
    plots['cross_correlation'] = plot_cross_correlation_detailed(results_dict)
    plots['phase_analysis'] = plot_phase_analysis(results_dict)
    plots['spectral_detailed'] = plot_spectral_detailed(results_dict)
    plots['timescales'] = plot_timescales_detailed(results_dict)
    plots['raster_comparison'] = plot_raster_comparison(results_dict)
    
    if save_prefix:
        for name, fig in plots.items():
            if fig is not None:
                fig.savefig(f"{save_prefix}_{name}.png", dpi=300, bbox_inches='tight')
    
    return plots



######



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_advanced_metrics_dashboard(results_db, trials_data=None, figsize=(18, 14)):
    """Dashboard avanzado con heatmaps, scatter plots y box plots"""
    
    # Preparar datos
    configs = list(results_db.keys())
    n_configs = len(configs)
    
    if n_configs == 0:
        print("No data to plot")
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
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels([c.replace('_', ' ')[:20] for c in configs], fontsize=8)
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
            
            for config in configs:
                if config in trials_data:
                    config_trials = trials_data[config]
                    if metric in config_trials.columns:
                        values = config_trials[metric].dropna()
                        if len(values) > 0:
                            if metric == 'cc_lag_ms':
                                values = np.abs(values)  # Use absolute lag
                            box_data.append(values)
                            box_labels.append(config.replace('_', ' ')[:15])
            
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
    
    plt.tight_layout()
    return fig

def load_trials_data_for_dashboard(sweep_dir, configs):
    """Load individual trial data for box plots"""
    trials_data = {}
    
    for config in configs:
        trials_csv = os.path.join(sweep_dir, f"config_{config}", f"trials_{config}.csv")
        if os.path.exists(trials_csv):
            df = pd.read_csv(trials_csv)
            trials_data[config] = df
    
    return trials_data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple, Optional

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

def load_raw_timeseries(sweep_dir: str, config_name: str, trial: int = 1) -> Optional[Dict]:
    """Carga datos raw de timeseries para un trial específico"""
    
    # Try the actual structure: figures/config_name/trial_X_full.pkl
    trial_file = os.path.join(sweep_dir, 'figures', config_name, f'trial_{trial}_full.pkl')
    
    if not os.path.exists(trial_file):
        return None
        
    try:
        with open(trial_file, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {trial_file}: {e}")
        return None

def plot_timeseries_overlay_comparison(sweep_dir: str, results_db: Dict, 
                                     metric: str = 'plv_alpha',
                                     figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """Plot overlay de timeseries para casos extremos de una métrica"""
    
    # Seleccionar casos extremos
    extreme_cases = select_extreme_cases(results_db, metric, n_cases=2)
    
    if len(extreme_cases['high']) == 0 or len(extreme_cases['low']) == 0:
        print(f"No hay suficientes datos para métrica {metric}")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Time Series Comparison: {metric.upper()}', fontsize=16)
    
    # Casos high y low (tomar el más extremo)
    cases = {
        'HIGH': extreme_cases['high'][-1],
        'LOW': extreme_cases['low'][0]
    }
    
    colors = {'HIGH': 'red', 'LOW': 'blue'}
    
    for col, (case_type, config_name) in enumerate(cases.items()):
        
        # Cargar datos raw
        raw_data = load_raw_timeseries(sweep_dir, config_name, 0)
        
        if raw_data is None:
            print(f"No data found for {config_name}")
            continue
            
        metric_val = extreme_cases['values'][config_name]
        color = colors[case_type]
        
        # Debug: print available keys
        print(f"Available keys in {case_type}: {list(raw_data.keys())[:10]}")
        
        # 1. Firing rates plot
        ax_rates = axes[0, col]
        
        # Use the correct keys from your data structure
        if 'rate_A' in raw_data and 'rate_B' in raw_data:
            rate_A = raw_data['rate_A']
            rate_B = raw_data['rate_B']
            time_axis = raw_data.get('time', np.arange(len(rate_A)) * 0.05)  # Use actual time or fallback
            
            if len(rate_A) > 100 and len(rate_B) > 100:
                # Plot middle section for better visibility
                start_idx = len(rate_A) // 4
                end_idx = 3 * len(rate_A) // 4
                
                ax_rates.plot(time_axis[start_idx:end_idx], rate_A[start_idx:end_idx], 
                             color=color, alpha=0.8, linewidth=1.5, label=f'Pop A')
                ax_rates.plot(time_axis[start_idx:end_idx], rate_B[start_idx:end_idx], 
                             color=color, alpha=0.8, linewidth=1.5, linestyle='--', label=f'Pop B')
        
        ax_rates.set_title(f'{case_type}: {metric}={metric_val:.3f}')
        ax_rates.set_xlabel('Time (ms)')
        ax_rates.set_ylabel('Rate (Hz)')
        ax_rates.legend()
        ax_rates.grid(True, alpha=0.3)
        
        # 2. Cross-correlation plot (instead of raster since no spike data)
        ax_xcorr = axes[1, col]
        
        if 'cross_correlation' in raw_data:
            xcorr_data = raw_data['cross_correlation']
            if isinstance(xcorr_data, dict) and 'lags' in xcorr_data and 'values' in xcorr_data:
                lags = xcorr_data['lags']
                values = xcorr_data['values']
                ax_xcorr.plot(lags, values, color=color, linewidth=2, alpha=0.8)
            elif isinstance(xcorr_data, np.ndarray):
                # If it's just the correlation values, create lag axis
                n_lags = len(xcorr_data)
                lags = np.arange(-n_lags//2, n_lags//2 + 1) * 0.05  # Assuming dt=0.05ms
                ax_xcorr.plot(lags[:len(xcorr_data)], xcorr_data, color=color, linewidth=2, alpha=0.8)
        else:
            # Calculate cross-correlation if not pre-computed
            if 'rate_A' in raw_data and 'rate_B' in raw_data:
                rate_A = raw_data['rate_A'][5000:15000]  # Stable window
                rate_B = raw_data['rate_B'][5000:15000]
                
                xcorr = np.correlate(rate_A - rate_A.mean(), 
                                   rate_B - rate_B.mean(), mode='full')
                xcorr = xcorr / np.max(np.abs(xcorr))
                
                lags = np.arange(-len(rate_A) + 1, len(rate_A)) * 0.05  # dt = 0.05ms
                ax_xcorr.plot(lags, xcorr, color=color, linewidth=2, alpha=0.8)
        
        ax_xcorr.set_title(f'Cross-Correlation - {case_type}')
        ax_xcorr.set_xlabel('Lag (ms)')
        ax_xcorr.set_ylabel('Correlation')
        ax_xcorr.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax_xcorr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_delay_distribution_comparison(sweep_dir: str, results_db: Dict,
                                     delay_types: List[str] = ['DELTA', 'GAUSSIAN', 'BETA'],
                                     figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """Compara timeseries para diferentes tipos de distribución de delays"""
    
    fig, axes = plt.subplots(2, len(delay_types), figsize=figsize)
    fig.suptitle('Delay Distribution Comparison', fontsize=16)
    
    colors = {'DELTA': 'blue', 'GAUSSIAN': 'orange', 'BETA': 'green'}
    
    for col, delay_type in enumerate(delay_types):
        
        # Buscar configuración representativa de este tipo
        config_name = None
        for name in results_db.keys():
            if delay_type in name.upper():
                config_name = name
                break
                
        if config_name is None:
            continue
            
        # Cargar datos
        raw_data = load_raw_timeseries(sweep_dir, config_name, 0)
        if raw_data is None:
            continue
            
        color = colors.get(delay_type, 'black')
        
        # Firing rates
        ax_rates = axes[0, col]
        if 'rate_A' in raw_data and 'rate_B' in raw_data:
            time_axis = raw_data.get('time_ms', np.arange(len(raw_data['rate_A']))) / 1000
            
            ax_rates.plot(time_axis, raw_data['rate_A'], color=color, 
                         alpha=0.8, linewidth=1.5, label='Pop A')
            ax_rates.plot(time_axis, raw_data['rate_B'], color=color, 
                         alpha=0.8, linewidth=1.5, linestyle='--', label='Pop B')
            
        ax_rates.set_title(f'{delay_type} Delays')
        ax_rates.set_xlim(1, 3)
        ax_rates.set_ylabel('Rate (Hz)')
        ax_rates.legend()
        ax_rates.grid(True, alpha=0.3)
        
        # Phase relationship (usando transformada de Hilbert aproximada)
        ax_phase = axes[1, col]
        if 'rate_A' in raw_data and 'rate_B' in raw_data:
            
            rate_A = raw_data['rate_A'][5000:15000]  # Ventana estable
            rate_B = raw_data['rate_B'][5000:15000]
            
            # Aproximación simple de diferencia de fase
            phase_diff = np.angle(np.fft.fft(rate_A)) - np.angle(np.fft.fft(rate_B))
            freqs = np.fft.fftfreq(len(rate_A), d=0.0005)  # dt = 0.5ms
            
            # Focus en rango de frecuencias relevante
            freq_mask = (freqs >= 8) & (freqs <= 50)
            
            ax_phase.scatter(freqs[freq_mask], phase_diff[freq_mask], 
                           c=color, alpha=0.6, s=10)
            
        ax_phase.set_title(f'Phase Relationship - {delay_type}')
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.set_ylabel('Phase Diff (rad)')
        ax_phase.grid(True, alpha=0.3)
        ax_phase.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    return fig

# Función principal de alto nivel
def generate_representative_timeseries_plots(sweep_dir: str, results_db: Dict,
                                           metrics: List[str] = ['plv_alpha', 'cross_corr_peak']) -> Dict[str, plt.Figure]:
    """Genera todos los plots de timeseries representativos"""
    
    figures = {}
    
    # 1. Comparación por métrica
    for metric in metrics:
        fig = plot_timeseries_overlay_comparison(sweep_dir, results_db, metric)
        if fig is not None:
            figures[f'timeseries_comparison_{metric}'] = fig
    
    # 2. Comparación por tipo de delay
    fig_delays = plot_delay_distribution_comparison(sweep_dir, results_db)
    if fig_delays is not None:
        figures['delay_distribution_comparison'] = fig_delays
    
    return figures


###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr



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
    
    elif 'gauss' in config_name:
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if 'gauss' in part and i+1 < len(parts):
                try:
                    param_str = parts[i+1].split('-')
                    mu = float(param_str[0])
                    sigma = float(param_str[1]) if len(param_str) > 1 else mu * 0.3
                    return {'type': 'gaussian', 'params': {'mu': mu, 'sigma': sigma}}
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

def extract_metrics_data(results_db):
    """Extract connectivity metrics"""
    metrics_data = []
    
    for config, data in results_db.items():
        agg = data['aggregated']
        
        metrics = {
            'config': config,
            'plv_alpha': agg.get('plv_alpha', {}).get('mean', 0),
            'pli_alpha': agg.get('pli_alpha', {}).get('mean', 0),
            'plv_gamma': agg.get('plv_gamma', {}).get('mean', 0),
            'cc_peak': abs(agg['cross_corr_peak']['mean']),
            'cc_lag': abs(agg['cross_corr_lag']['mean']),
            'coherence_peak': agg['coherence_peak']['mean'],
            'tau_avg': (agg.get('tau_A', {}).get('mean', 0) + agg.get('tau_B', {}).get('mean', 0)) / 2
        }
        
        metrics_data.append(metrics)
    
    return pd.DataFrame(metrics_data)

def calculate_correlations(delay_df, metrics_df):
    """Calculate correlation coefficients using native parameters"""
    # Merge dataframes
    combined = pd.merge(delay_df, metrics_df, on='config')
    
    # Get all delay parameter columns (exclude config and type)
    delay_cols = [col for col in delay_df.columns if col not in ['config', 'type']]
    metric_cols = ['plv_alpha', 'pli_alpha', 'cc_peak', 'cc_lag', 'coherence_peak', 'tau_avg']
    
    # Filter out constant columns
    valid_delay_cols = []
    for col in delay_cols:
        if col in combined.columns and combined[col].std() > 1e-10:
            valid_delay_cols.append(col)
        elif col in combined.columns:
            print(f"Skipping constant column: {col}")
    
    correlations = {}
    p_values = {}
    
    for delay_param in valid_delay_cols:
        correlations[delay_param] = {}
        p_values[delay_param] = {}
        
        for metric in metric_cols:
            # Remove any NaN values
            mask = ~(np.isnan(combined[delay_param]) | np.isnan(combined[metric]))
            if mask.sum() > 2:  # Need at least 3 points
                try:
                    corr, p_val = pearsonr(combined[delay_param][mask], combined[metric][mask])
                    correlations[delay_param][metric] = corr
                    p_values[delay_param][metric] = p_val
                except:
                    correlations[delay_param][metric] = np.nan
                    p_values[delay_param][metric] = np.nan
            else:
                correlations[delay_param][metric] = np.nan
                p_values[delay_param][metric] = np.nan
    
    return pd.DataFrame(correlations).T, pd.DataFrame(p_values).T, combined

def plot_correlation_analysis(delay_df, metrics_df, figsize=(16, 12)):
    """Complete correlation analysis dashboard with native parameters"""
    corr_matrix, p_matrix, combined = calculate_correlations(delay_df, metrics_df)
    
    if len(corr_matrix) == 0:
        print("No valid correlations found")
        return None, None, None
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Correlation heatmap
    ax1 = fig.add_subplot(gs[0, :])
    mask = np.isnan(corr_matrix.values)
    
    im = ax1.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(corr_matrix.columns)))
    ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax1.set_yticks(range(len(corr_matrix.index)))
    ax1.set_yticklabels([param.replace('_', ' ').title() for param in corr_matrix.index])
    ax1.set_title('Native Delay Parameters vs Metrics Correlations')
    
    # Add correlation values and significance stars
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]
            
            if not np.isnan(corr_val):
                # Significance stars
                stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                
                ax1.text(j, i, f'{corr_val:.3f}\n{stars}', 
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(corr_val) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax1, shrink=0.8, label='Correlation Coefficient')
    
    # 2-7. Key scatter plots - select strongest correlations
    significant_corrs = []
    for delay_param in corr_matrix.index:
        for metric in corr_matrix.columns:
            corr_val = corr_matrix.loc[delay_param, metric]
            p_val = p_matrix.loc[delay_param, metric]
            if not np.isnan(corr_val) and p_val < 0.05 and abs(corr_val) > 0.3:
                significant_corrs.append((delay_param, metric, abs(corr_val), p_val))
    
    # Sort by correlation strength and take top 6
    significant_corrs.sort(key=lambda x: x[2], reverse=True)
    top_corrs = significant_corrs[:6]
    
    for plot_idx, (delay_param, metric, _, p_val) in enumerate(top_corrs):
        if plot_idx < 6:
            ax = fig.add_subplot(gs[(plot_idx)//3 + 1, (plot_idx)%3])
            
            x = combined[delay_param]
            y = combined[metric]
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                ax.scatter(x[mask], y[mask], alpha=0.7, s=60)
                
                # Add trend line
                try:
                    z = np.polyfit(x[mask], y[mask], 1)
                    p = np.poly1d(z)
                    ax.plot(x[mask], p(x[mask]), "r--", alpha=0.8)
                except:
                    pass
                
                # Add correlation info
                corr_val = corr_matrix.loc[delay_param, metric]
                ax.text(0.05, 0.95, f'r={corr_val:.3f}\np={p_val:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_xlabel(delay_param.replace('_', ' ').title())
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
    
    # Fill remaining subplots if less than 6 significant correlations
    for plot_idx in range(len(top_corrs), 6):
        ax = fig.add_subplot(gs[(plot_idx)//3 + 1, (plot_idx)%3])
        ax.text(0.5, 0.5, 'No significant\ncorrelation', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig, corr_matrix, p_matrix

def print_correlation_summary(corr_matrix, p_matrix, threshold=0.05):
    """Print significant correlations"""
    print("\n=== SIGNIFICANT CORRELATIONS (p < 0.05) ===")
    
    for delay_param in corr_matrix.index:
        print(f"\n{delay_param.replace('_', ' ').title()}:")
        
        for metric in corr_matrix.columns:
            corr = corr_matrix.loc[delay_param, metric]
            p_val = p_matrix.loc[delay_param, metric]
            
            if not np.isnan(corr) and p_val < threshold:
                strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                direction = "positive" if corr > 0 else "negative"
                
                print(f"  → {metric}: {corr:.3f} ({strength} {direction}, p={p_val:.3f})")

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


####


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

def extract_delay_parameters(results_db):
    """Extract native delay parameters by distribution type"""
    delay_data = []
    
    for config, data in results_db.items():
        # Get delay config if available, otherwise parse from name
        if 'delay_config' in data:
            delay_config = data['delay_config']
        else:
            delay_config = parse_delay_config_from_name(config)
        
        params = {
            'config': config,
            'type': delay_config['type']
        }
        
        # Add type-specific parameters
        if delay_config['type'] == 'uniform':
            params.update({
                'uniform_low': delay_config['params']['low'],
                'uniform_high': delay_config['params']['high'],
                'uniform_width': delay_config['params']['high'] - delay_config['params']['low']
            })
        elif delay_config['type'] == 'gaussian':
            params.update({
                'gauss_mu': delay_config['params']['mu'],
                'gauss_sigma': delay_config['params']['sigma'],
                'gauss_cv': delay_config['params']['sigma'] / delay_config['params']['mu'] if delay_config['params']['mu'] > 0 else 0
            })
        elif delay_config['type'] == 'constant':
            params.update({
                'delta_value': delay_config['value']
            })
        elif delay_config['type'] == 'beta':
            params.update({
                'beta_alpha': delay_config['params']['alpha'],
                'beta_beta': delay_config['params']['beta'], 
                'beta_scale': delay_config['params']['scale']
            })
        
        delay_data.append(params)
    
    return pd.DataFrame(delay_data)

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
    
    elif 'gauss' in config_name:
        parts = config_name.split('_')
        for i, part in enumerate(parts):
            if 'gauss' in part and i+1 < len(parts):
                try:
                    param_str = parts[i+1].split('-')
                    mu = float(param_str[0])
                    sigma = float(param_str[1]) if len(param_str) > 1 else mu * 0.3
                    return {'type': 'gaussian', 'params': {'mu': mu, 'sigma': sigma}}
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


# =============================================================================
# PLOTTING FUNCTIONS (SEPARATED FROM ANALYSIS)
# =============================================================================

def _band_order(phase_locking_dict):
    pref = ['theta','alpha','beta','gamma','broadband']
    bands = list(phase_locking_dict.keys())
    # mantiene el orden preferido y añade el resto al final
    ordered = [b for b in pref if b in bands] + [b for b in bands if b not in pref]
    return ordered

def plot_connectivity_dashboard(results_dict, figsize=(18, 12)):
    """Main connectivity dashboard (all PLV/PLI bands + coherence-by-band + cc peak/lag)."""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None

    conditions = list(valid_results.keys())
    n_conditions = len(conditions)

    # === detectar bandas disponibles dinámicamente ===
    # (tomamos las del primer resultado)
    some = valid_results[conditions[0]]
    plv_bands = list(some.get('phase_locking', {}).keys() or
                     some.get('plv_pli', {}).keys())
    # ordenar de forma “neuro-friendly” si existen
    pref = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broadband']
    plv_bands = sorted(plv_bands, key=lambda b: (pref.index(b) if b in pref else 999, b))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # 1) Cross-correlation curve (con línea vertical en peak lag)
    ax1 = fig.add_subplot(gs[0, 0])
    for condition, res in valid_results.items():
        cc = res.get('cross_correlation', res.get('crosscorr', {}))
        if 'lags' in cc and len(cc['lags']) > 0:
            ax1.plot(cc['lags'], cc['correlation'], label=condition, linewidth=2)
            # línea vertical en el lag detectado
            peak_lag = cc.get('peak_lag', None)
            if peak_lag is not None:
                ax1.axvline(x=peak_lag, color='k', linestyle='--', alpha=0.6)

    ax1.set_xlabel('Lag (ms)')
    ax1.set_ylabel('Cross-correlation')
    ax1.set_title('Cross-correlation Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    # 2) PLV & PLI de TODAS las bandas (barras agrupadas por condición)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(plv_bands))
    # ancho pequeño para poder meter PLV y PLI por condición sin solapar
    width = 0.35 / max(1, n_conditions)

    for i, (condition, res) in enumerate(valid_results.items()):
        src = res.get('phase_locking', res.get('plv_pli', {}))
        plv_vals = [float(src.get(b, {}).get('plv', 0)) for b in plv_bands]
        pli_vals = [float(src.get(b, {}).get('pli', 0)) for b in plv_bands]
        ax2.bar(x + (i*2)*width - width*(n_conditions-1), plv_vals, width,
                label=f'{condition} PLV', alpha=0.85)
        ax2.bar(x + ((i*2)+1)*width - width*(n_conditions-1), pli_vals, width,
                label=f'{condition} PLI', alpha=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(plv_bands)
    ax2.set_xlabel('Frequency Bands')
    ax2.set_ylabel('Phase Locking')
    ax2.set_title('PLV & PLI across Bands')
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) Coherence spectrum (como antes)
    ax3 = fig.add_subplot(gs[0, 2])
    for condition, res in valid_results.items():
        coh = res.get('coherence', {})
        if 'freqs' in coh and len(coh['freqs']) > 0:
            ax3.plot(coh['freqs'], coh['coherence'], label=condition, linewidth=2)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Coherence')
    ax3.set_title('Spectral Coherence')
    ax3.set_xlim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4) Intrinsic Timescales (etiquetas A/B explícitas)
    ax4 = fig.add_subplot(gs[1, 0])
    xAB = np.arange(2)  # A, B
    width_ts = 0.8 / max(1, n_conditions)
    for i, (condition, res) in enumerate(valid_results.items()):
        tau_A = float(res.get('tau_A', res.get('timescale_A', {}).get('tau', 0)))
        tau_B = float(res.get('tau_B', res.get('timescale_B', {}).get('tau', 0)))
        ax4.bar(xAB + i*width_ts, [tau_A, tau_B], width_ts, label=condition, alpha=0.85)
    ax4.set_xticks(xAB + width_ts*(n_conditions-1)/2)
    ax4.set_xticklabels(['A', 'B'])
    ax4.set_xlabel('Population')
    ax4.set_ylabel('Intrinsic Timescale (ms)')
    ax4.set_title('Intrinsic Timescales')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5) “Coherence per band” + CC peak/lag (summary compacto)
    ax5 = fig.add_subplot(gs[1, 1])
    summary_names = ['|CC peak|', 'PLV α', 'PLV γ', 'Coh α', 'Coh γ']
    base = np.arange(len(summary_names))
    width_sum = 0.8 / max(1, n_conditions)

    for i, (condition, res) in enumerate(valid_results.items()):
        # CC
        cc = res.get('cross_correlation', {})
        cc_peak = abs(float(cc.get('peak_value', 0)))
        # cc_lag = abs(float(cc.get('peak_lag', 0)))
        # PLV bands
        plv_src = res.get('phase_locking', res.get('plv_pli', {}))
        plv_alpha = float(plv_src.get('alpha', {}).get('plv', 0))
        plv_gamma = float(plv_src.get('gamma', {}).get('plv', 0))
        # Coherence en bandas (tu función ya devuelve alpha/gamma)
        coh = res.get('coherence', {})
        coh_alpha = float(coh.get('alpha_coherence', 0))
        coh_gamma = float(coh.get('gamma_coherence', 0))

        vals = [cc_peak, plv_alpha, plv_gamma, coh_alpha, coh_gamma]
        ax5.bar(base + i*width_sum, vals, width_sum, label=condition, alpha=0.85)

    ax5.set_xticks(base + width_sum*(n_conditions-1)/2)
    ax5.set_xticklabels(summary_names, rotation=20)
    ax5.set_ylabel('Value')
    ax5.set_title('Summary: CC / PLV / Coherence')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6) PLV vs PLI scatter para TODAS las bandas (no solo alfa)
    ax6 = fig.add_subplot(gs[1, 2])
    for condition, res in valid_results.items():
        src = res.get('phase_locking', res.get('plv_pli', {}))
        for b in plv_bands:
            plv = float(src.get(b, {}).get('plv', 0))
            pli = float(src.get(b, {}).get('pli', 0))
            ax6.scatter(plv, pli, s=60, alpha=0.7, label=f'{condition} {b}')
    # línea identidad
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax6.set_xlabel('PLV')
    ax6.set_ylabel('PLI')
    ax6.set_title('PLV vs PLI across Bands')
    # para no repetir mil leyendas, cogemos únicas y limitamos tamaño
    handles, labels = ax6.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax6.legend(uniq.values(), uniq.keys(), fontsize=7, ncol=2)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_population_dashboard(results_dict, figsize=(16, 9)):
    """
    Dashboard de dinámicas poblacionales (post-corte) centrado en:
      (1) Autocorrelaciones A/B
      (2) PSD A/B
      (3) Series temporales A/B (ventana común)
    """
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        print("Error: No valid results to plot")
        return None

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    # 1) Autocorrelaciones (arriba izquierda)
    ax1 = fig.add_subplot(gs[0, 0])
    for condition, res in valid_results.items():
        ac_A = res.get('autocorr_A', {})
        ac_B = res.get('autocorr_B', {})
        if len(ac_A.get('lags', [])) > 0:
            ax1.plot(ac_A['lags'], ac_A['correlation'],
                     label=f"{condition} Pop A", alpha=0.9, lw=2)
        if len(ac_B.get('lags', [])) > 0:
            ax1.plot(ac_B['lags'], ac_B['correlation'],
                     label=f"{condition} Pop B", alpha=0.9, lw=2, ls='--')
    ax1.set_xlabel('Lag (ms)')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Autocorrelation Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2) PSD (arriba derecha)
    ax2 = fig.add_subplot(gs[0, 1])
    for condition, res in valid_results.items():
        psd_A = res.get('psd_A', res.get('power_A', {}))
        psd_B = res.get('psd_B', res.get('power_B', {}))
        if len(psd_A.get('freqs', [])) > 0:
            ax2.plot(psd_A['freqs'], psd_A['psd'],
                     label=f"{condition} Pop A", alpha=0.9, lw=2)
        if len(psd_B.get('freqs', [])) > 0:
            ax2.plot(psd_B['freqs'], psd_B['psd'],
                     label=f"{condition} Pop B", alpha=0.9, lw=2, ls='--')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Power Spectra')
    ax2.set_xlim(0, 60)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3) Series temporales (abajo a lo ancho)
    ax3 = fig.add_subplot(gs[1, :])

    # ventana común (mínimo tiempo disponible en todas las condiciones)
    def _last_time(res):
        t = res.get('time_series', {}).get('time', res.get('time', np.array([])))
        return float(t[-1]) if len(t) else 0.0

    view_ms_common = min((_last_time(r) for r in valid_results.values()), default=0.0)

    for condition, res in valid_results.items():
        ts = res.get('time_series', {})
        t = ts.get('time', res.get('time', np.array([])))
        rA = ts.get('rate_A', res.get('rate_A', np.array([])))
        rB = ts.get('rate_B', res.get('rate_B', np.array([])))
        if len(t) == 0:
            continue
        end_idx = np.searchsorted(t, view_ms_common, side='right')
        ax3.plot(t[:end_idx], rA[:end_idx], label=f"{condition} Pop A", alpha=0.75, lw=1.2)
        ax3.plot(t[:end_idx], rB[:end_idx], label=f"{condition} Pop B", alpha=0.75, lw=1.2, ls='--')

    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Population Rate (Hz)')
    ax3.set_title(f'Population Activity Time Series (first {int(view_ms_common)} ms post-cut)')
    ax3.legend(ncol=2)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

