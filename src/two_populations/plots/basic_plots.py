
###### BASIC PLOTS FOR TWO POPULATION COMPARISON AND EVALUATION ######

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from brian2 import *

from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

def plot_raster_results(results, N_exc=800, N_total=1000, warmup_ms=500, neuron_limit=None):
    fig = plt.figure(figsize=(14, 8))
    for idx, pop in enumerate(['A', 'B'], start=1):
        
        if pop not in results:
            continue
        plt.subplot(1, 2, idx)
        sm = results[pop]['spike_monitor']
        t = sm.t/ms
        i = sm.i
        m = t >= warmup_ms
        t, i = t[m], i[m]
        if neuron_limit is not None:
            m2 = i < neuron_limit
            t, i = t[m2], i[m2]

        exc = i < N_exc; inh = i >= N_exc
        plt.plot(t[exc], i[exc], '.', color="#100533", markersize=0.75, alpha=1.0)
        plt.plot(t[inh], i[inh], '.', color="#330507", markersize=0.75, alpha=1.0)
        plt.axhline(y=N_exc, color='grey', linestyle=':', linewidth=1)
        plt.xlabel('Time (ms)'); plt.ylabel(f'Neurons {pop}')
        plt.title(f'Raster {pop}')
        plt.ylim(0, N_total)
    return fig


def plot_voltage_traces(results_dict, raw_results=None, 
                        time_window=(500, 1500),
                        n_exc_examples=3, 
                        n_inh_examples=3,
                        show_spikes=True,
                        ncols=3):  # Número de columnas en la grid
    """
    Plot de trazas de voltaje organizado en grid de ncols columnas
    """
    
    valid_results = {k: v for k, v in results_dict.items() 
                    if 'voltage_monitor_A' in v}
    
    if not valid_results:
        print("❌ No voltage data")
        return None
    
    n_conditions = len(valid_results)
    n_neurons_total = n_exc_examples + n_inh_examples
    
    # Calcular grid: ncols columnas, nrows filas
    nrows = int(np.ceil(n_neurons_total / ncols))
    
    colors_exc = plt.cm.Reds(np.linspace(0.4, 0.8, n_exc_examples))
    colors_inh = plt.cm.Blues(np.linspace(0.5, 0.8, n_inh_examples))
    
    # Una figura por condición
    figs = []
    
    for condition, results in valid_results.items():
        volt_mon = results['voltage_monitor_A']
        spike_mon_times = results.get('spike_times_A')
        spike_mon_neurons = results.get('spike_neurons_A')
        
        # Obtener índices
        if raw_results and 'A' in raw_results:
            sample_indices = raw_results['A'].get('v_sample_indices', [])
            N_exc_sampled = raw_results['A'].get('v_n_exc_sampled', 400)
        else:
            n_monitored = volt_mon.v.shape[0]
            sample_indices = np.arange(n_monitored)
            N_exc_sampled = int(n_monitored * 0.8)
        
        print(f"{condition}: {len(sample_indices)} monitored, {N_exc_sampled} exc")
        
        # Seleccionar ejemplos
        exc_candidates = sample_indices[:N_exc_sampled]
        inh_candidates = sample_indices[N_exc_sampled:]
        
        exc_selected = []
        if len(exc_candidates) >= n_exc_examples:
            step = len(exc_candidates) // (n_exc_examples + 1)
            exc_selected = [exc_candidates[i * step] for i in range(1, n_exc_examples + 1)]
        else:
            exc_selected = list(exc_candidates[:n_exc_examples])
        
        inh_selected = list(inh_candidates[:n_inh_examples]) if len(inh_candidates) >= n_inh_examples else list(inh_candidates)
        
        all_neurons = exc_selected + inh_selected
        print(f"  Plotting: {all_neurons}")
        
        # Crear figura
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
        
        # Tiempos
        times = np.array(volt_mon.t / ms)
        time_mask = (times >= time_window[0]) & (times <= time_window[1])
        times_cut = times[time_mask]
        
        # Plot cada neurona
        for plot_idx, neuron_idx in enumerate(all_neurons):
            ax = axes[plot_idx]
            
            is_exc = plot_idx < len(exc_selected)
            neuron_type = 'Exc' if is_exc else 'Inh'
            color = colors_exc[plot_idx] if is_exc else colors_inh[plot_idx - len(exc_selected)]
            
            # Encontrar índice en monitoreadas
            monitor_idx = np.where(sample_indices == neuron_idx)[0]
            if len(monitor_idx) == 0:
                ax.text(0.5, 0.5, f'Neuron {neuron_idx}\nnot found', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            monitor_idx = monitor_idx[0]
            v_trace = np.array(volt_mon.v[monitor_idx])[time_mask]
            
            # Plot principal
            ax.plot(times_cut, v_trace, linewidth=1, color=color, alpha=0.85)
            
            # Marcar spikes
            spike_times = []
            if show_spikes and spike_mon_times is not None:
                neuron_mask = spike_mon_neurons == neuron_idx
                neuron_spikes = spike_mon_times[neuron_mask]
                spike_mask = (neuron_spikes >= time_window[0]) & (neuron_spikes <= time_window[1])
                spike_times = neuron_spikes[spike_mask]
                
                # Líneas verticales
                for st in spike_times:
                    ax.axvline(st, color=color, alpha=0.25, linewidth=1.5, linestyle='-', zorder=1)
                
                # Puntos en spikes
                if len(spike_times) > 0:
                    spike_v = [v_trace[np.argmin(np.abs(times_cut - st))] for st in spike_times]
                    ax.scatter(spike_times, spike_v, color=color, s=40, zorder=5, 
                              edgecolors='white', linewidths=1.5, marker='o')
            
            # Referencias
            ax.axhline(30, color='darkgray', linestyle='--', alpha=0.6, linewidth=1, zorder=0)
            ax.axhline(-65, color='lightgray', linestyle='--', alpha=0.5, linewidth=0.8, zorder=0)
            
            # Límites y estética
            ax.set_ylim(-80, 40)
            ax.set_xlim(time_window)
            ax.grid(True, alpha=0.15, linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Título con info
            title = f'{neuron_type} #{neuron_idx}'
            if len(spike_times) > 1:
                isi_mean = np.mean(np.diff(spike_times))
                title += f'\n{len(spike_times)} sp | ISI: {isi_mean:.1f}ms'
            elif len(spike_times) == 1:
                title += f'\n1 spike'
            ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
            
            # Labels
            ax.set_ylabel('mV', fontsize=9)
            if plot_idx >= n_neurons_total - ncols:  # Última fila
                ax.set_xlabel('Time (ms)', fontsize=9)
            
            # Tick params
            ax.tick_params(labelsize=8)
        
        # Ocultar axes sobrantes
        for idx in range(len(all_neurons), len(axes)):
            axes[idx].axis('off')
        
        # Título general
        fig.suptitle(f'{condition} - Voltage Traces (n={n_neurons_total})', 
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        figs.append(fig)
    
    return figs if len(figs) > 1 else figs[0]

def plot_activity_distributions_dict(results_dict, N_exc=800, N_inh=200, warmup=1000, figsize=(15, 8)):
    """Distribuciones de actividad adaptado para results_dict"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None
    
    n_conditions = len(valid_results)
    fig, axes = plt.subplots(2, n_conditions, figsize=figsize)
    if n_conditions == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (condition, results) in enumerate(valid_results.items()):
        t0 = float(results.get('t0_ms', warmup))
        
        # Population A
        ax_A = axes[0, i]
        tA = results['spike_times_A']
        nA = results['spike_neurons_A']
        
        # Filter spikes post-warmup
        mask_A = tA >= t0
        filtered_indices_A = nA[mask_A]
        
        # Count spikes per neuron
        freq_exc_A = [np.sum(filtered_indices_A == j) for j in range(N_exc)]
        freq_inh_A = [np.sum(filtered_indices_A == (j + N_exc)) for j in range(N_inh)]
        
        ax_A.hist(freq_exc_A, bins=30, alpha=0.7, density=True, label='Excitatory', color='blue')
        ax_A.hist(freq_inh_A, bins=30, alpha=0.7, density=True, label='Inhibitory', color='red')
        ax_A.set_xlabel('Spike Count')
        ax_A.set_ylabel('Density')
        ax_A.set_title(f'{condition} - Population A')
        ax_A.legend()
        ax_A.grid(True, alpha=0.3)
        
        # Population B
        ax_B = axes[1, i]
        tB = results['spike_times_B']
        nB = results['spike_neurons_B']
        
        mask_B = tB >= t0
        filtered_indices_B = nB[mask_B]
        
        freq_exc_B = [np.sum(filtered_indices_B == j) for j in range(N_exc)]
        freq_inh_B = [np.sum(filtered_indices_B == (j + N_exc)) for j in range(N_inh)]
        
        ax_B.hist(freq_exc_B, bins=30, alpha=0.7, density=True, label='Excitatory', color='blue')
        ax_B.hist(freq_inh_B, bins=30, alpha=0.7, density=True, label='Inhibitory', color='red')
        ax_B.set_xlabel('Spike Count')
        ax_B.set_ylabel('Density')
        ax_B.set_title(f'{condition} - Population B')
        ax_B.legend()
        ax_B.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_population_activity_dict(results_dict, bin_size_ms=5, figsize=(15, 8)):
    """Actividad poblacional adaptado para results_dict"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None
    
    n_conditions = len(valid_results)
    fig, axes = plt.subplots(2, n_conditions, figsize=figsize)
    if n_conditions == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (condition, results) in enumerate(valid_results.items()):
        t0 = float(results.get('t0_ms', 0))
        
        # Use existing time series data if available
        if 'time' in results and 'rate_A' in results:
            times = results['time']
            rate_A = results['rate_A']
            rate_B = results['rate_B']
            
            min_len = min(len(times), len(rate_A), len(rate_B))
            times, rate_A, rate_B = times[:min_len], rate_A[:min_len], rate_B[:min_len]
            
            # Population A
            ax_A = axes[0, i]
            ax_A.plot(times, rate_A, 'k-', linewidth=1)
            ax_A.set_xlabel('Time (ms)')
            ax_A.set_ylabel('Population Rate (Hz)')
            ax_A.set_title(f'{condition} - Population A Activity')
            ax_A.grid(True, alpha=0.3)
            
            # Population B
            ax_B = axes[1, i]
            ax_B.plot(times, rate_B, 'k-', linewidth=1)
            ax_B.set_xlabel('Time (ms)')
            ax_B.set_ylabel('Population Rate (Hz)')
            ax_B.set_title(f'{condition} - Population B Activity')
            ax_B.grid(True, alpha=0.3)
        else:
            # Fall back to spike times
            tA = results['spike_times_A']
            tB = results['spike_times_B']
            
            # Filter post-warmup
            tA_filt = tA[tA >= t0] - t0
            tB_filt = tB[tB >= t0] - t0
            
            T_max = max(np.max(tA_filt) if len(tA_filt) > 0 else 0,
                       np.max(tB_filt) if len(tB_filt) > 0 else 0)
            
            if T_max > 0:
                time_bins = np.arange(0, T_max, bin_size_ms)
                
                # Population A
                ax_A = axes[0, i]
                activity_A = []
                for t in time_bins:
                    spikes = np.sum((tA_filt >= t) & (tA_filt < t + bin_size_ms))
                    activity_A.append(spikes)
                
                ax_A.plot(time_bins, activity_A, 'k-', linewidth=1)
                ax_A.set_xlabel('Time (ms)')
                ax_A.set_ylabel(f'Spikes/{bin_size_ms}ms')
                ax_A.set_title(f'{condition} - Population A Activity')
                ax_A.grid(True, alpha=0.3)
                
                # Population B
                ax_B = axes[1, i]
                activity_B = []
                for t in time_bins:
                    spikes = np.sum((tB_filt >= t) & (tB_filt < t + bin_size_ms))
                    activity_B.append(spikes)
                
                ax_B.plot(time_bins, activity_B, 'k-', linewidth=1)
                ax_B.set_xlabel('Time (ms)')
                ax_B.set_ylabel(f'Spikes/{bin_size_ms}ms')
                ax_B.set_title(f'{condition} - Population B Activity')
                ax_B.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spectral_analysis_dict(results_dict, bin_size_ms=5, figsize=(15, 8)):
    """Análisis espectral adaptado para results_dict"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None
    
    n_conditions = len(valid_results)
    fig, axes = plt.subplots(2, n_conditions, figsize=figsize)
    if n_conditions == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (condition, results) in enumerate(valid_results.items()):
        # Use existing PSD if available
        if 'psd_A' in results and 'psd_B' in results:
            psd_A = results['psd_A']
            psd_B = results['psd_B']
            
            # Population A
            ax_A = axes[0, i]
            freqs = psd_A['freqs']
            psd = psd_A['psd']
            
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            gamma_mask = (freqs >= 30) & (freqs <= 50)
            
            ax_A.plot(freqs, psd, 'k-', alpha=0.5)
            if np.any(alpha_mask):
                ax_A.plot(freqs[alpha_mask], psd[alpha_mask], 'b-', linewidth=2, label='Alpha (8-12 Hz)')
            if np.any(gamma_mask):
                ax_A.plot(freqs[gamma_mask], psd[gamma_mask], 'r-', linewidth=2, label='Gamma (30-50 Hz)')
            
            ax_A.set_xlabel('Frequency (Hz)')
            ax_A.set_ylabel('PSD')
            ax_A.set_title(f'{condition} - Population A Spectrum')
            ax_A.set_xlim(0, 100)
            ax_A.legend()
            ax_A.grid(True, alpha=0.3)
            
            # Population B
            ax_B = axes[1, i]
            freqs = psd_B['freqs']
            psd = psd_B['psd']
            
            ax_B.plot(freqs, psd, 'k-', alpha=0.5)
            if np.any(alpha_mask):
                ax_B.plot(freqs[alpha_mask], psd[alpha_mask], 'b-', linewidth=2, label='Alpha (8-12 Hz)')
            if np.any(gamma_mask):
                ax_B.plot(freqs[gamma_mask], psd[gamma_mask], 'r-', linewidth=2, label='Gamma (30-50 Hz)')
            
            ax_B.set_xlabel('Frequency (Hz)')
            ax_B.set_ylabel('PSD')
            ax_B.set_title(f'{condition} - Population B Spectrum')
            ax_B.set_xlim(0, 100)
            ax_B.legend()
            ax_B.grid(True, alpha=0.3)
            
        else:
            # Fall back to rate-based analysis
            if 'rate_A' in results and 'rate_B' in results:
                fs = 1000 / results.get('analysis_dt', 0.5)  # Sampling frequency
                
                # Population A
                ax_A = axes[0, i]
                rate_A = results['rate_A']
                if len(rate_A) > 100:
                    freqs, psd = signal.periodogram(rate_A, fs=fs)
                    
                    alpha_mask = (freqs >= 8) & (freqs <= 12)
                    gamma_mask = (freqs >= 30) & (freqs <= 50)
                    
                    ax_A.plot(freqs, psd, 'k-', alpha=0.5)
                    if np.any(alpha_mask):
                        ax_A.plot(freqs[alpha_mask], psd[alpha_mask], 'b-', linewidth=2, label='Alpha (8-12 Hz)')
                    if np.any(gamma_mask):
                        ax_A.plot(freqs[gamma_mask], psd[gamma_mask], 'r-', linewidth=2, label='Gamma (30-50 Hz)')
                    
                    ax_A.set_xlabel('Frequency (Hz)')
                    ax_A.set_ylabel('PSD')
                    ax_A.set_title(f'{condition} - Population A Spectrum')
                    ax_A.set_xlim(0, 100)
                    ax_A.legend()
                    ax_A.grid(True, alpha=0.3)
                
                # Population B
                ax_B = axes[1, i]
                rate_B = results['rate_B']
                if len(rate_B) > 100:
                    freqs, psd = signal.periodogram(rate_B, fs=fs)
                    
                    ax_B.plot(freqs, psd, 'k-', alpha=0.5)
                    if np.any(alpha_mask):
                        ax_B.plot(freqs[alpha_mask], psd[alpha_mask], 'b-', linewidth=2, label='Alpha (8-12 Hz)')
                    if np.any(gamma_mask):
                        ax_B.plot(freqs[gamma_mask], psd[gamma_mask], 'r-', linewidth=2, label='Gamma (30-50 Hz)')
                    
                    ax_B.set_xlabel('Frequency (Hz)')
                    ax_B.set_ylabel('PSD')
                    ax_B.set_title(f'{condition} - Population B Spectrum')
                    ax_B.set_xlim(0, 100)
                    ax_B.legend()
                    ax_B.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_parameter_distributions_dict(results_dict, network_objects=None, N_exc=800, figsize=(12, 8)):
    """Distribuciones de parámetros adaptado para results_dict
    
    Note: Requires access to network objects since parameters aren't stored in results_dict.
    Pass network objects as dict: {'condition_name': network_obj, ...}
    """
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results or network_objects is None:
        logger.info("Parameter distributions require network objects")
        return None
    
    fig = plt.figure(figsize=figsize)
    
    # Collect all parameters
    all_params = {}
    for condition in valid_results.keys():
        if condition in network_objects:
            net = network_objects[condition]
            if hasattr(net, 'populations'):
                group_A = net.populations['A']['group']
                group_B = net.populations['B']['group']
                
                all_params[condition] = {
                    'a_exc_A': group_A.a[:N_exc], 'a_inh_A': group_A.a[N_exc:],
                    'b_exc_A': group_A.b[:N_exc], 'b_inh_A': group_A.b[N_exc:],
                    'c_exc_A': group_A.c[:N_exc], 'c_inh_A': group_A.c[N_exc:],
                    'd_exc_A': group_A.d[:N_exc], 'd_inh_A': group_A.d[N_exc:],
                    'a_exc_B': group_B.a[:N_exc], 'a_inh_B': group_B.a[N_exc:],
                    'b_exc_B': group_B.b[:N_exc], 'b_inh_B': group_B.b[N_exc:],
                    'c_exc_B': group_B.c[:N_exc], 'c_inh_B': group_B.c[N_exc:],
                    'd_exc_B': group_B.d[:N_exc], 'd_inh_B': group_B.d[N_exc:]
                }
    
    if not all_params:
        return None
    
    # Plot parameter distributions
    param_pairs = [('a', 'b'), ('c', 'd')]
    cell_types = ['exc', 'inh']
    populations = ['A', 'B']
    
    for i, (p1, p2) in enumerate(param_pairs):
        for j, cell_type in enumerate(cell_types):
            ax = plt.subplot(2, 2, i*2 + j + 1)
            
            for condition, params in all_params.items():
                for pop in populations:
                    alpha = 0.7 if pop == 'A' else 0.4
                    key1 = f'{p1}_{cell_type}_{pop}'
                    key2 = f'{p2}_{cell_type}_{pop}'
                    
                    if key1 in params and key2 in params:
                        ax.hist(params[key1], alpha=alpha, bins=10, 
                               label=f'{condition} {cell_type} {p1} - {pop}')
                        ax.hist(params[key2], alpha=alpha, bins=10,
                               label=f'{condition} {cell_type} {p2} - {pop}')
            
            ax.set_title(f'{cell_type.title()} Parameters {p1.upper()}, {p2.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_currents_analysis_dict(results_dict, raw_results=None, N_exc=800, figsize=(15, 10)):
    """Plot current analysis for results_dict"""
    if raw_results is None:
        logger.info("Current plots require raw simulation results")
        return None
    
    current_stats = extract_currents_info_dict(results_dict, raw_results, N_exc)
    if not current_stats:
        logger.info("No current data available for plotting")
        return None
    
    conditions = list(current_stats.keys())
    n_conditions = len(conditions)
    
    fig, axes = plt.subplots(3, n_conditions, figsize=figsize)
    if n_conditions == 1:
        axes = axes.reshape(-1, 1)
    
    for i, condition in enumerate(conditions):
        # Thalamic currents
        ax1 = axes[0, i]
        for pop in ['A', 'B']:
            if pop in current_stats[condition]:
                thal_exc = current_stats[condition][pop]['thalamic_exc']['mean_per_neuron']
                ax1.hist(thal_exc, alpha=0.7, label=f'Pop {pop} Exc', bins=20)
        ax1.set_title(f'{condition} - Thalamic Currents')
        ax1.set_xlabel('Current')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Synaptic currents
        ax2 = axes[1, i]
        for pop in ['A', 'B']:
            if pop in current_stats[condition]:
                syn_exc = current_stats[condition][pop]['synaptic_exc']['mean_per_neuron']
                ax2.hist(syn_exc, alpha=0.7, label=f'Pop {pop} Exc', bins=20)
        ax2.set_title(f'{condition} - Synaptic Currents')
        ax2.set_xlabel('Current')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Total currents
        ax3 = axes[2, i]
        for pop in ['A', 'B']:
            if pop in current_stats[condition]:
                tot_exc = current_stats[condition][pop]['total_exc']['mean_per_neuron']
                ax3.hist(tot_exc, alpha=0.7, label=f'Pop {pop} Exc', bins=20)
        ax3.set_title(f'{condition} - Total Currents')
        ax3.set_xlabel('Current')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Wrapper function for all basic plots
def plot_all_basic_analysis(results_dict, network_objects=None, raw_results=None, save_prefix=None):
    """Generate all basic analysis plots"""
    plots = {}
    
    plots['activity_distributions'] = plot_activity_distributions_dict(results_dict)
    plots['population_activity'] = plot_population_activity_dict(results_dict)
    plots['spectral_analysis'] = plot_spectral_analysis_dict(results_dict)
    plots['voltage_traces'] = plot_voltage_traces(results_dict)
    
    if network_objects:
        plots['parameter_distributions'] = plot_parameter_distributions_dict(results_dict, network_objects)
    
    if raw_results:
        plots['currents_analysis'] = plot_currents_analysis_dict(results_dict, raw_results)
    
    if save_prefix:
        for name, fig in plots.items():
            if fig is not None:
                fig.savefig(f"{save_prefix}_{name}.png", dpi=300, bbox_inches='tight')
    
    return plots



def extract_currents_info_dict(results_dict, raw_results=None, N_exc=800):
    """Extract current statistics adapted for results_dict
    
    Args:
        results_dict: Output from analyze_simulation_results
        raw_results: Raw simulation results with structure {'A': {...}, 'B': {...}}
        N_exc: Number of excitatory neurons
    """
    if raw_results is None:
        logger.info("Current analysis requires raw simulation results with current monitors")
        return {}
    
    # Check if current data exists - adjust for actual structure
    has_current_data = False
    for pop in ['A', 'B']:
        if (pop in raw_results and 'I_thalamic' in raw_results[pop]):
            has_current_data = True
            break
    
    if not has_current_data:
        logger.info("No current data found in raw_results. Available keys:", 
            {pop: list(raw_results[pop].keys()) if pop in raw_results else 'missing' 
            for pop in ['A', 'B']})
        return {}
    
    def separate_by_type(data):
        n_recorded = data.shape[0]
        if n_recorded <= N_exc:
            # Solo excitatorias registradas
            return data, np.array([])
        else:
            return data[:N_exc], data[N_exc:]

    def calc_stats(data):
        if data.size == 0:
            return None  # Skip si vacío
        return {
            'mean_per_neuron': np.mean(data, axis=1),
            'std_per_neuron': np.std(data, axis=1),
            'mean_per_timestep': np.mean(data, axis=0),
            'min_per_timestep': np.min(data, axis=0),
            'max_per_timestep': np.max(data, axis=0)
        }
    
    all_stats = {}
    
    # Since results_dict has only one condition, we'll map it to the raw_results
    for condition in results_dict.keys():
        condition_stats = {}
        
        for pop_name in ['A', 'B']:
            if pop_name in raw_results:
                pop_data = raw_results[pop_name]
                
                # Get current data
                if 'I_thalamic' in pop_data and 'I_syn' in pop_data:
                    I_thal = pop_data['I_thalamic']
                    I_syn = pop_data['I_syn']
                    
                    I_thal_exc, I_thal_inh = separate_by_type(I_thal)
                    I_syn_exc, I_syn_inh = separate_by_type(I_syn)
                    
                    I_tot_exc = I_thal_exc + I_syn_exc
                    I_tot_inh = I_thal_inh + I_syn_inh
                    
                    condition_stats[pop_name] = {
                        'synaptic_exc': calc_stats(I_syn_exc),
                        'total_exc': calc_stats(I_tot_exc),
                        'synaptic_inh': calc_stats(I_syn_inh),
                        'total_inh': calc_stats(I_tot_inh)
                    }
                    
                    stats_exc = calc_stats(I_thal_exc)
                    stats_inh = calc_stats(I_thal_inh)

                    if stats_exc:
                        condition_stats[pop_name]['thalamic_exc'] = stats_exc
                    if stats_inh:
                        condition_stats[pop_name]['thalamic_inh'] = stats_inh
                    
                    # Print statistics
                    logger.info(f"\n=== Estadísticas {condition} - Grupo {pop_name} ===")
                    logger.info("=== INPUT TALÁMICO ===")
                    thal_exc = condition_stats[pop_name]['thalamic_exc']
                    logger.info(f"Exc - Media por neurona: {np.mean(thal_exc['mean_per_neuron']):.4f}")
                    logger.info(f"Exc - Std por neurona: {np.mean(thal_exc['std_per_neuron']):.4f}")
                    
                    logger.info("\n=== INPUT SINÁPTICO ===")
                    syn_exc = condition_stats[pop_name]['synaptic_exc']
                    logger.info(f"Exc - Media: {np.mean(syn_exc['mean_per_neuron']):.4f}")
                    logger.info(f"Exc - Std: {np.mean(syn_exc['std_per_neuron']):.4f}")
                    logger.info(f"Exc - Rango típico: [{np.mean(syn_exc['min_per_timestep']):.2f}, {np.mean(syn_exc['max_per_timestep']):.2f}]")
        
        all_stats[condition] = condition_stats
    
    return all_stats




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

def plot_poisson_input(results, raw_results, figsize=(14, 8)):
    """Visualizar conductancia g_exc del PoissonInput"""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    for i, pop in enumerate(['A', 'B']):
        if 'g_exc' not in raw_results[pop]:
            axes[0, i].text(0.5, 0.5, 'g_exc not monitored', 
                          ha='center', va='center')
            continue
        
        g_exc = raw_results[pop]['g_exc']  # [neurons x time] en siemens
        times = raw_results[pop]['times']
        
        # Temporal
        mean_g = g_exc.mean(axis=0) * 1e9  # convertir a nS
        std_g = g_exc.std(axis=0) * 1e9
        
        axes[0, i].plot(times, mean_g, linewidth=1.5)
        axes[0, i].fill_between(times, mean_g - std_g, mean_g + std_g, alpha=0.3)
        axes[0, i].set_title(f'Pop {pop} - g_exc temporal')
        axes[0, i].set_xlabel('Time (ms)')
        axes[0, i].set_ylabel('Conductance (nS)')
        axes[0, i].grid(alpha=0.3)
        
        # Distribución
        mean_per_neuron = g_exc.mean(axis=1) * 1e9
        axes[1, i].hist(mean_per_neuron, bins=30, alpha=0.7, color='C0')
        axes[1, i].set_title(f'Pop {pop} - g_exc distribución')
        axes[1, i].set_xlabel('Mean g_exc (nS)')
        axes[1, i].set_ylabel('Count')
        axes[1, i].grid(alpha=0.3)
    
    plt.suptitle('PoissonInput Conductance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_thalamic_drive(results_dict, raw_results, figsize=(14,8)):
    """Sanity check del drive talámico"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    for condition in results_dict.keys():
        for i, pop in enumerate(['A', 'B']):
            I_thal = raw_results[pop]['I_thalamic']  # [neurons x time]
            times = raw_results[pop]['times']
            
            # 1. Serie temporal promedio (verificar step profile)
            axes[0, i].plot(times, I_thal.mean(axis=0), linewidth=1)
            axes[0, i].fill_between(times, 
                I_thal.mean(axis=0) - I_thal.std(axis=0),
                I_thal.mean(axis=0) + I_thal.std(axis=0), alpha=0.3)
            axes[0, i].set_title(f'Pop {pop} - I_thalamic temporal')
            axes[0, i].set_xlabel('Time (ms)')
            
            # 2. Distribución espacial (verificar heterogeneidad entre neuronas)
            mean_per_neuron = I_thal.mean(axis=1)  # promedio temporal por neurona
            axes[1, i].hist(mean_per_neuron, bins=30, alpha=0.7)
            axes[1, i].set_title(f'Pop {pop} - Distribución espacial')
            axes[1, i].set_xlabel('Mean current')
    
    plt.tight_layout()
    return fig


def plot_synaptic_currents(results_dict, raw_results, figsize=(16, 10), separate_ei=True):
    """Análisis de corrientes sinápticas I_syn
    
    Args:
        results_dict: Output de analyze_simulation_results
        raw_results: {'A': {...}, 'B': {...}} con I_syn
        separate_ei: Si True, separa exc/inh (requiere conocer N_exc)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    for idx, pop in enumerate(['A', 'B']):
        if pop not in raw_results or 'I_syn' not in raw_results[pop]:
            continue
            
        I_syn = raw_results[pop]['I_syn']  # [neurons x time]
        times = raw_results[pop]['times']
        n_neurons = I_syn.shape[0]
        
        # 1. Serie temporal promedio
        ax1 = fig.add_subplot(gs[0, idx])
        mean_current = I_syn.mean(axis=0)
        std_current = I_syn.std(axis=0)
        
        ax1.plot(times, mean_current, linewidth=1.5, color='C0', label='Mean')
        ax1.fill_between(times, mean_current - std_current, 
                         mean_current + std_current, alpha=0.3, color='C0')
        ax1.set_title(f'Pop {pop} - I_syn temporal')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución espacial (mean per neuron)
        ax2 = fig.add_subplot(gs[1, idx])
        mean_per_neuron = I_syn.mean(axis=1)
        
        if separate_ei and n_neurons > 100:  # Asume primeras N_exc son exc
            N_exc = int(0.8 * n_neurons)  # 800 de 1000
            ax2.hist(mean_per_neuron[:N_exc], bins=30, alpha=0.7, 
                    label='Exc', color='C0')
            ax2.hist(mean_per_neuron[N_exc:], bins=30, alpha=0.7, 
                    label='Inh', color='C1')
            ax2.legend()
        else:
            ax2.hist(mean_per_neuron, bins=30, alpha=0.7, color='C0')
        
        ax2.set_title(f'Pop {pop} - Distribución espacial')
        ax2.set_xlabel('Mean I_syn per neuron')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. Heatmap temporal-neuronal (sample)
        ax3 = fig.add_subplot(gs[2, idx])
        sample_neurons = min(50, n_neurons)
        
        im = ax3.imshow(I_syn[:sample_neurons, ::10],  # downsample time
                       aspect='auto', cmap='RdBu_r', 
                       extent=[times[0], times[-1], sample_neurons, 0],
                       vmin=-np.abs(I_syn[:sample_neurons]).max(),
                       vmax=np.abs(I_syn[:sample_neurons]).max())
        ax3.set_title(f'Pop {pop} - I_syn heatmap (first {sample_neurons} neurons)')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Neuron index')
        plt.colorbar(im, ax=ax3, label='Current')
    
    plt.suptitle('Synaptic Currents Analysis', fontsize=14, y=0.995)
    return fig


# src/two_populations/plots/basic_plots.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy.ndimage import gaussian_filter

def plot_spectrogram(results_dict, figsize=(16, 6)):
    """Espectrograma optimizado para sweeps"""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    
    for idx, pop in enumerate(['A', 'B']):
        condition = list(results_dict.keys())[0]
        res = results_dict[condition]
        
        ts = res.get('time_series', {})
        lfp = ts.get(f'signal_{pop}', np.array([]))
        
        if len(lfp) == 0:
            continue
        
        fs = 1000.0
        nperseg = 512
        noverlap = int(nperseg * 0.8)
        
        lfp_proc = (lfp - np.mean(lfp)) / np.std(lfp)
        
        f, t, Sxx = sg.spectrogram(lfp_proc, fs=fs,
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    window='hann',
                                    scaling='density')
        
        mask = f <= 60
        f_masked = f[mask]
        
        Sxx_smooth = gaussian_filter(Sxx[mask, :], sigma=(0.5, 0.5))
        Sxx_db = Sxx_smooth
        
        vmin = np.percentile(Sxx_db, 12)
        vmax = np.percentile(Sxx_db, 88)
        
        ax = axes[idx]
        im = ax.pcolormesh(t, f_masked, Sxx_db,
                          shading='gouraud',
                          cmap='inferno',
                          vmin=vmin, vmax=vmax,
                          rasterized=True)
        
        ax.set_ylabel('Frequency (Hz)' if idx == 0 else '', fontsize=13)
        ax.set_xlabel('Time (s)', fontsize=13)
        ax.set_ylim(2, 60)
        ax.set_title(f'Population {pop}', fontsize=14, weight='bold', pad=10)
        
        cbar = plt.colorbar(im, ax=ax, aspect=25)
        cbar.set_label('Power (dB)', rotation=270, labelpad=18, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        ax.tick_params(labelsize=11)
    
    fig.suptitle('LFP Spectrograms', fontsize=16, y=0.99, weight='bold')
    plt.tight_layout()
    return fig


def compute_correlation_matrix(voltage_monitor, neuron_indices, warmup_idx):
    """Correlación Pearson entre pares de neuronas"""
    V = voltage_monitor.v[:, warmup_idx:]
    V_sample = V[neuron_indices, :]
    return np.corrcoef(V_sample)


def plot_correlation_matrix(results, n_sample=50, warmup_idx=500):
    """Matrices de correlación intra-poblacional"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    N_exc_mon = 400
    N_inh_mon = 100
    
    for idx, pop in enumerate(['A', 'B']):
        v_mon = results[pop]['voltage_monitor']
        
        sample_exc = np.random.choice(N_exc_mon, n_sample//2, replace=False)
        sample_inh = N_exc_mon + np.random.choice(N_inh_mon, n_sample//2, replace=False)
        sample = np.concatenate([sample_exc, sample_inh])
        
        corr = compute_correlation_matrix(v_mon, sample, warmup_idx)
        
        im = axes[idx].imshow(corr, cmap='RdBu_r', vmin=-0.3, vmax=0.5)
        axes[idx].set_title(f'Pop {pop}', fontsize=13, weight='bold')
        axes[idx].axhline(n_sample//2-0.5, color='k', lw=1.5)
        axes[idx].axvline(n_sample//2-0.5, color='k', lw=1.5)
        axes[idx].set_xlabel('Neuron index')
        axes[idx].set_ylabel('Neuron index')
        
        axes[idx].text(n_sample//4, -2, 'Exc', ha='center', fontsize=10)
        axes[idx].text(3*n_sample//4, -2, 'Inh', ha='center', fontsize=10)
        
        mask = ~np.eye(corr.shape[0], dtype=bool)
        mean_corr = np.mean(corr[mask])
        axes[idx].text(0.02, 0.98, f'μ={mean_corr:.3f}', 
                      transform=axes[idx].transAxes, va='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Histograma combinado
    ax_hist = axes[2]
    for idx, pop in enumerate(['A', 'B']):
        v_mon = results[pop]['voltage_monitor']
        sample_exc = np.random.choice(N_exc_mon, n_sample//2, replace=False)
        sample_inh = N_exc_mon + np.random.choice(N_inh_mon, n_sample//2, replace=False)
        sample = np.concatenate([sample_exc, sample_inh])
        corr = compute_correlation_matrix(v_mon, sample, warmup_idx)
        
        mask = ~np.eye(corr.shape[0], dtype=bool)
        ax_hist.hist(corr[mask].flatten(), bins=30, alpha=0.6, 
                    label=f'Pop {pop}', density=True)
    
    ax_hist.set_xlabel('Correlation', fontsize=11)
    ax_hist.set_ylabel('Density', fontsize=11)
    ax_hist.set_title('Correlation Distribution', fontsize=13, weight='bold')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    plt.colorbar(im, ax=axes, label='Correlation')
    plt.tight_layout()
    return fig


def plot_interpop_correlation(results, n_sample=30, warmup_idx=500):
    """Correlación inter-poblacional"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    N_exc_mon = 400
    
    sample_A = np.random.choice(N_exc_mon, n_sample, replace=False)
    sample_B = np.random.choice(N_exc_mon, n_sample, replace=False)
    
    V_A = results['A']['voltage_monitor'].v[sample_A, warmup_idx:]
    V_B = results['B']['voltage_monitor'].v[sample_B, warmup_idx:]
    
    corr_AB = np.corrcoef(V_A, V_B)[:n_sample, n_sample:]
    
    im = ax.imshow(corr_AB, cmap='RdBu_r', vmin=-0.3, vmax=0.5)
    ax.set_xlabel('Pop B neurons')
    ax.set_ylabel('Pop A neurons')
    ax.set_title('Inter-population Correlation', weight='bold')
    plt.colorbar(im, label='Correlation')
    
    mean_cross = np.mean(corr_AB)
    ax.text(0.02, 0.98, f'μ_AB={mean_cross:.3f}', 
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig