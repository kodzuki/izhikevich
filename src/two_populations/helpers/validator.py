import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from src.two_populations.plots.basic_plots import plot_raster_results

from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

class NeuralActivityValidator:
    """Validación y control de calidad de actividad neuronal"""
    
    def __init__(self, N_exc=800, N_inh=200):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_total = N_exc + N_inh
        
        # Umbrales biológicamente razonables
        self.thresholds = {
            'silent_exc': 0.1,      # Hz - neuronas excitatorias silentes
            'silent_inh': 0.5,      # Hz - neuronas inhibitorias silentes  
            'normal_exc_range': (0.5, 15),   # Hz - rango normal exc
            'normal_inh_range': (2, 40),     # Hz - rango normal inh
            'hyperactive_exc': 25,  # Hz - excitatorias hiperactivas
            'hyperactive_inh': 60,  # Hz - inhibitorias hiperactivas
            'burst_threshold': 0.05, # fracción de neuronas para burst
            'min_population_rate': 0.2, # Hz mínimo poblacional
        }
        
    def calculate_cv_isi(self, spike_monitor, warmup_ms=500):
        """Coeficiente de variación inter-spike interval"""
        times = np.array(spike_monitor.t / 1)
        indices = np.array(spike_monitor.i)
        mask = times >= warmup_ms
        
        cv_values = []
        for neuron in np.unique(indices[mask]):
            neuron_times = times[mask][indices[mask] == neuron]
            if len(neuron_times) > 2:
                isis = np.diff(neuron_times)
                cv = isis.std() / isis.mean() if isis.mean() > 0 else 0
                cv_values.append(cv)
        
        return np.array(cv_values)
    
    def calculate_firing_rates(self, spike_monitor, warmup_ms=500, total_time_ms=4000):
        """Calcula firing rates individuales y poblacionales post-warmup"""
        spike_times = np.array(spike_monitor.t / 1)  # ms
        spike_indices = np.array(spike_monitor.i)
        
        # Filtrar post-warmup
        mask = spike_times >= warmup_ms
        filtered_times = spike_times[mask]
        filtered_indices = spike_indices[mask]
        
        analysis_duration_s = (total_time_ms - warmup_ms) / 1000
        
        # Rates individuales
        exc_rates = np.array([
            np.sum(filtered_indices == i) / analysis_duration_s 
            for i in range(self.N_exc)
        ])
        
        inh_rates = np.array([
            np.sum(filtered_indices == (i + self.N_exc)) / analysis_duration_s
            for i in range(self.N_inh)  
        ])
        
        # Rates poblacionales
        exc_pop_rate = np.sum(filtered_indices < self.N_exc) / (self.N_exc * analysis_duration_s)
        inh_pop_rate = np.sum(filtered_indices >= self.N_exc) / (self.N_inh * analysis_duration_s)
        
        return {
            'exc_individual': exc_rates,
            'inh_individual': inh_rates,
            'exc_population': exc_pop_rate,
            'inh_population': inh_pop_rate,
            'analysis_duration_s': analysis_duration_s,
            'total_spikes': len(filtered_times)
        }
    
    def classify_neurons(self, rates):
        """Clasifica neuronas según actividad"""
        exc_rates = rates['exc_individual']
        inh_rates = rates['inh_individual']
        
        classification = {
            'exc_silent': np.sum(exc_rates <= self.thresholds['silent_exc']),
            'exc_normal': np.sum(
                (exc_rates > self.thresholds['normal_exc_range'][0]) & 
                (exc_rates <= self.thresholds['normal_exc_range'][1])
            ),
            'exc_hyperactive': np.sum(exc_rates > self.thresholds['hyperactive_exc']),
            'inh_silent': np.sum(inh_rates <= self.thresholds['silent_inh']),
            'inh_normal': np.sum(
                (inh_rates > self.thresholds['normal_inh_range'][0]) & 
                (inh_rates <= self.thresholds['normal_inh_range'][1])
            ),
            'inh_hyperactive': np.sum(inh_rates > self.thresholds['hyperactive_inh']),
        }
        
        # Percentages
        classification.update({
            'exc_silent_pct': classification['exc_silent'] / self.N_exc * 100,
            'exc_active_pct': (self.N_exc - classification['exc_silent']) / self.N_exc * 100,
            'inh_silent_pct': classification['inh_silent'] / self.N_inh * 100,
            'inh_active_pct': (self.N_inh - classification['inh_silent']) / self.N_inh * 100,
        })
        
        return classification
    
    def detect_population_issues(self, rates, classification):
        """Detecta problemas poblacionales"""
        issues = []
        warnings = []
        
        # Population rates too low
        if rates['exc_population'] < self.thresholds['min_population_rate']:
            issues.append(f"Excitatory pop rate too low: {rates['exc_population']:.2f} Hz")
            
        if rates['inh_population'] < self.thresholds['min_population_rate']:
            issues.append(f"Inhibitory pop rate too low: {rates['inh_population']:.2f} Hz")
        
        # Too many silent neurons
        if classification['exc_silent_pct'] > 50:
            issues.append(f"Too many silent exc neurons: {classification['exc_silent_pct']:.1f}%")
            
        if classification['inh_silent_pct'] > 30:
            issues.append(f"Too many silent inh neurons: {classification['inh_silent_pct']:.1f}%")
        
        # Hyperactivity
        if classification['exc_hyperactive'] > self.N_exc * 0.1:
            warnings.append(f"Many hyperactive exc neurons: {classification['exc_hyperactive']}")
            
        if classification['inh_hyperactive'] > self.N_inh * 0.1:
            warnings.append(f"Many hyperactive inh neurons: {classification['inh_hyperactive']}")
        
        # E/I ratio
        ei_ratio = rates['exc_population'] / max(rates['inh_population'], 0.1)
        if ei_ratio < 0.1 or ei_ratio > 2.0:
            warnings.append(f"Unusual E/I ratio: {ei_ratio:.2f}")
        
        return issues, warnings
    
    def detect_bursts(self, spike_monitor, warmup_ms=500, bin_size_ms=10):
        """Detecta eventos de burst poblacional"""
        spike_times = np.array(spike_monitor.t / 1)
        spike_indices = np.array(spike_monitor.i)
        
        # Filtrar y crear timeline
        mask = spike_times >= warmup_ms
        filtered_times = spike_times[mask] - warmup_ms  # Reset to 0
        filtered_indices = spike_indices[mask]
        
        if len(filtered_times) == 0:
            return {'n_bursts': 0, 'burst_rate': 0, 'burst_durations': [], 'burst_times': []}
        
        max_time = np.max(filtered_times)
        time_bins = np.arange(0, max_time + bin_size_ms, bin_size_ms)
        
        # Count excitatory spikes per bin
        exc_activity = []
        bin_centers = []
        
        for i in range(len(time_bins) - 1):
            t_start, t_end = time_bins[i], time_bins[i + 1]
            mask_bin = (filtered_times >= t_start) & (filtered_times < t_end)
            exc_spikes = np.sum(filtered_indices[mask_bin] < self.N_exc)
            exc_activity.append(exc_spikes)
            bin_centers.append((t_start + t_end) / 2)
        
        exc_activity = np.array(exc_activity)
        bin_centers = np.array(bin_centers)
        
        # Burst threshold: when >5% of exc neurons fire in bin
        burst_threshold = self.N_exc * self.thresholds['burst_threshold']
        is_burst = exc_activity > burst_threshold
        
        # Find burst episodes
        burst_starts = []
        burst_ends = []
        in_burst = False
        
        for i, burst_state in enumerate(is_burst):
            if burst_state and not in_burst:
                burst_starts.append(i)
                in_burst = True
            elif not burst_state and in_burst:
                burst_ends.append(i - 1)
                in_burst = False
        
        # Handle burst extending to end
        if in_burst:
            burst_ends.append(len(is_burst) - 1)
        
        # Calculate burst properties
        burst_durations = [(burst_ends[i] - burst_starts[i] + 1) * bin_size_ms 
                          for i in range(len(burst_starts))]
        burst_times = [bin_centers[start] for start in burst_starts]
        
        analysis_duration_s = (max_time) / 1000
        burst_rate = len(burst_durations) / analysis_duration_s if analysis_duration_s > 0 else 0
        
        return {
            'n_bursts': len(burst_durations),
            'burst_rate': burst_rate,  # bursts/second
            'burst_durations': burst_durations,  # ms
            'burst_times': burst_times,  # ms
            'mean_burst_duration': np.mean(burst_durations) if burst_durations else 0,
            'burst_coverage': np.sum(is_burst) / len(is_burst) * 100,  # % time in bursts
            'timeline': bin_centers,
            'activity': exc_activity,
            'burst_mask': is_burst,
            'burst_threshold': burst_threshold
        }
    
    def create_activity_timeline(self, spike_monitor, warmup_ms=500, window_ms=50):
        """Crea timeline de actividad poblacional post-warmup"""
        spike_times = np.array(spike_monitor.t / 1)
        spike_indices = np.array(spike_monitor.i)
        
        # Filtrar post-warmup
        mask = spike_times >= warmup_ms
        filtered_times = spike_times[mask] - warmup_ms
        filtered_indices = spike_indices[mask]
        
        if len(filtered_times) == 0:
            return {'times': np.array([]), 'exc_rate': np.array([]), 'inh_rate': np.array([])}
        
        max_time = np.max(filtered_times)
        time_bins = np.arange(0, max_time + window_ms, window_ms)
        
        exc_rates = []
        inh_rates = []
        bin_centers = []
        
        for i in range(len(time_bins) - 1):
            t_start, t_end = time_bins[i], time_bins[i + 1]
            mask_bin = (filtered_times >= t_start) & (filtered_times < t_end)
            indices_in_bin = filtered_indices[mask_bin]
            
            exc_spikes = np.sum(indices_in_bin < self.N_exc)
            inh_spikes = np.sum(indices_in_bin >= self.N_exc)
            
            # Convert to Hz
            bin_duration_s = window_ms / 1000
            exc_rate = exc_spikes / (self.N_exc * bin_duration_s)
            inh_rate = inh_spikes / (self.N_inh * bin_duration_s)
            
            exc_rates.append(exc_rate)
            inh_rates.append(inh_rate)
            bin_centers.append((t_start + t_end) / 2)
        
        return {
            'times': np.array(bin_centers),
            'exc_rate': np.array(exc_rates),
            'inh_rate': np.array(inh_rates)
        }
    
    def validate_single_population(self, spike_monitor, pop_name, warmup_ms=500, total_time_ms=4000):
        """Validación completa de una población"""
        rates = self.calculate_firing_rates(spike_monitor, warmup_ms, total_time_ms)
        classification = self.classify_neurons(rates)
        issues, warnings = self.detect_population_issues(rates, classification)
        bursts = self.detect_bursts(spike_monitor, warmup_ms)
        timeline = self.create_activity_timeline(spike_monitor, warmup_ms)
        
        cv_isi = self.calculate_cv_isi(spike_monitor, warmup_ms)
        
        return {
            'population': pop_name,
            'rates': rates,
            'classification': classification,
            'issues': issues,
            'warnings': warnings,
            'bursts': bursts,
            'timeline': timeline,
            'cv_isi': cv_isi
        }


def plot_population_validation_dashboard(validation_results, figsize=(12, 6)):
    """Dashboard de validación para una o múltiples poblaciones"""
    
    if not isinstance(validation_results, dict):
        # Single population
        validation_results = {'single': validation_results}
    
    n_pops = len(validation_results)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, hspace=0.4, wspace=0.3)
    
    # Collect data for comparison plots
    pop_names = list(validation_results.keys())

    colors_A = ['#E74C3C', '#C0392B']  # Rojos
    colors_B = ['#3498DB', '#2980B9']  # Azules
    colors = [colors_A[0], colors_B[0]]  # Para 2 poblaciones
    
    # 1. Firing rate distributions
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (pop_name, result) in enumerate(validation_results.items()):
        rates = result['rates']
        ax1.hist(rates['exc_individual'], bins=30, alpha=0.6, 
                label=f'{pop_name} Exc', color=colors[i], density=True)
        ax1.hist(rates['inh_individual'], bins=30, alpha=0.4, 
                label=f'{pop_name} Inh', color=colors[i], density=True, linestyle='--')
        
        mean_exc = rates['exc_individual'].mean()
        std_exc = rates['exc_individual'].std()
        ax1.text(0.98, 0.95 - i*0.1, f'{pop_name}: μ={mean_exc:.1f}±{std_exc:.1f}Hz',
                transform=ax1.transAxes, ha='right', fontsize=9)
                
        # Mark thresholds
        ax1.axvline(0.1, color='red', linestyle=':', alpha=0.7, linewidth=1)
        ax1.axvline(25, color='red', linestyle=':', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Firing Rate (Hz)')
    ax1.set_ylabel('Density')
    ax1.set_title('Individual Firing Rate Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # ax1.set_xlim(0, 25)
    
    # # 2. Population classifications
    # ax2 = fig.add_subplot(gs[0, 1])
    # categories = ['Silent', 'Normal', 'Hyperactive']
    # x_pos = np.arange(len(categories))
    width = 0.35 / n_pops
    
    # for i, (pop_name, result) in enumerate(validation_results.items()):
    #     cls = result['classification']
    #     exc_vals = [cls['exc_silent_pct'], 
    #                cls['exc_normal'] / 800 * 100, 
    #                cls['exc_hyperactive'] / 800 * 100]
    #     inh_vals = [cls['inh_silent_pct'], 
    #                cls['inh_normal'] / 200 * 100, 
    #                cls['inh_hyperactive'] / 200 * 100]
        
    #     ax2.bar(x_pos + i*width*2, exc_vals, width, 
    #            label=f'{pop_name} Exc', alpha=0.8, color=colors[i])
    #     ax2.bar(x_pos + i*width*2 + width, inh_vals, width, 
    #            label=f'{pop_name} Inh', alpha=0.6, color=colors[i])
    
    # ax2.set_xlabel('Activity Category')
    # ax2.set_ylabel('Percentage of Neurons')
    # ax2.set_title('Neuron Classification')
    # ax2.set_xticks(x_pos + width * (n_pops - 0.5))
    # ax2.set_xticklabels(categories)
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    
    # # 3. Population rates summary
    # ax3 = fig.add_subplot(gs[0, 2])
    # pop_types = ['Exc Pop', 'Inh Pop']
    # x_pos = np.arange(len(pop_types))
    
    # for i, (pop_name, result) in enumerate(validation_results.items()):
    #     rates = result['rates']
    #     values = [rates['exc_population'], rates['inh_population']]
    #     ax3.bar(x_pos + i*width, values, width, 
    #            label=pop_name, alpha=0.8, color=colors[i])
    
    # ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Min threshold')
    # ax3.set_xlabel('Population Type')
    # ax3.set_ylabel('Population Rate (Hz)')
    # ax3.set_title('Population-Level Activity')
    # ax3.set_xticks(x_pos + width * (n_pops - 1) / 2)
    # ax3.set_xticklabels(pop_types)
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)
    
    # 4. Burst analysis
    ax4 = fig.add_subplot(gs[0, 1])
    
    # Check if any bursts exist across all populations
    has_bursts = any(result['bursts']['n_bursts'] > 0 for result in validation_results.values())
    
    if has_bursts:
        burst_metrics = ['N Bursts', 'Burst Rate\n(bursts/s)', 'Mean Duration\n(ms)', 'Coverage\n(%)']
        x_pos = np.arange(len(burst_metrics))
        
        for i, (pop_name, result) in enumerate(validation_results.items()):
            bursts = result['bursts']
            values = [bursts['n_bursts'], bursts['burst_rate'], 
                     bursts['mean_burst_duration'], bursts['burst_coverage']]
            
            # Normalize each metric separately
            max_vals = [max([validation_results[p]['bursts'][k] for p in validation_results.keys()]) 
                       for k in ['n_bursts', 'burst_rate', 'mean_burst_duration', 'burst_coverage']]
            norm_values = [v/max(mv, 1) for v, mv in zip(values, max_vals)]
            
            ax4.bar(x_pos + i*width, norm_values, width, 
                   label=pop_name, alpha=0.8, color=colors[i])
        
        ax4.set_xlabel('Burst Metrics')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Burst Properties')
        ax4.set_xticks(x_pos + width * (n_pops - 1) / 2)
        ax4.set_xticklabels(burst_metrics, fontsize=8)
        ax4.legend()
        ax4.set_ylim(0, 1.1)
    else:
        # Calcular CV ISI
        all_cv = np.concatenate([result['cv_isi'] for result in validation_results.values()])
        ax4.hist(all_cv, bins=30, alpha=0.7, color='gray', label='All neurons')
        ax4.axvline(1.0, ls='--', color='k', label='Poisson-like')
        ax4.set_xlabel('CV ISI')
        ax4.set_title('Firing Regularity')
        ax4.legend()
    
    ax4.grid(True, alpha=0.3)
    
    # # 5. Issue summary
    # ax5 = fig.add_subplot(gs[0, 2:]) 
    # ax5.axis('off')
    
    # y_pos = 0.9
    # ax5.text(0.05, y_pos, 'VALIDATION SUMMARY', fontsize=14, fontweight='bold')
    # y_pos -= 0.15
    
    # for pop_name, result in validation_results.items():
    #     ax5.text(0.05, y_pos, f'{pop_name}:', fontsize=12, fontweight='bold', color='blue')
    #     y_pos -= 0.1
        
    #     # Issues
    #     if result['issues']:
    #         ax5.text(0.1, y_pos, 'ISSUES:', fontsize=10, fontweight='bold', color='red')
    #         y_pos -= 0.05
    #         for issue in result['issues']:
    #             ax5.text(0.15, y_pos, f'• {issue}', fontsize=9, color='red')
    #             y_pos -= 0.05
    #     else:
    #         ax5.text(0.1, y_pos, '✓ No critical issues', fontsize=10, color='green')
    #         y_pos -= 0.05
        
    #     # Warnings
    #     if result['warnings']:
    #         ax5.text(0.1, y_pos, 'WARNINGS:', fontsize=10, fontweight='bold', color='orange')
    #         y_pos -= 0.05
    #         for warning in result['warnings']:
    #             ax5.text(0.15, y_pos, f'• {warning}', fontsize=9, color='orange')
    #             y_pos -= 0.05
        
    #     y_pos -= 0.1
    
    try:
        plt.tight_layout()
    except:
        pass  # Skip if axes incompatible
    return fig


def validate_simulation_results(results_dict, warmup_ms=500, total_time_ms=4000):
    """Validación completa de resultados de simulación"""
    validator = NeuralActivityValidator()
    validation_results = {}
    
    for condition, results in results_dict.items():
        if results is None:
            continue
            
        # Validate both populations
        validation_A = validator.validate_single_population(
            results['spike_times_A'], results['spike_neurons_A'], 
            f'{condition}_A', warmup_ms, total_time_ms
        )
        validation_B = validator.validate_single_population(
            results['spike_times_B'], results['spike_neurons_B'], 
            f'{condition}_B', warmup_ms, total_time_ms
        )
        
        validation_results[f'{condition}_A'] = validation_A
        validation_results[f'{condition}_B'] = validation_B
    
    return validation_results


def print_validation_summary(validation_results):
    """Resumen textual de validación"""
    logger.info("\n" + "="*60)
    logger.info("NEURAL ACTIVITY VALIDATION SUMMARY")
    logger.info("="*60)
    
    for pop_name, result in validation_results.items():
        logger.info(f"\n{pop_name}:")
        logger.info("-" * 40)
        
        rates = result['rates']
        cls = result['classification']
        bursts = result['bursts']
        
        logger.info(f"Population rates: Exc={rates['exc_population']:.2f} Hz, Inh={rates['inh_population']:.2f} Hz")
        logger.info(f"Active neurons: Exc={cls['exc_active_pct']:.1f}%, Inh={cls['inh_active_pct']:.1f}%")
        logger.info(f"Bursts: {bursts['n_bursts']} total, {bursts['burst_rate']:.2f} bursts/s")
        
        if result['issues']:
            logger.info("🚨 ISSUES:")
            for issue in result['issues']:
                logger.info(f"   • {issue}")
        
        if result['warnings']:
            logger.info("⚠️  WARNINGS:")
            for warning in result['warnings']:
                logger.info(f"   • {warning}")
        
        if not result['issues'] and not result['warnings']:
            logger.info("✅ Population looks healthy")


# Wrapper para usar con analyze_simulation_results
def add_validation_to_analysis(results_dict, warmup_ms=500, total_time_ms=4000):
    """Añade validación a resultados existentes de analyze_simulation_results"""
    
    # Adaptar formato de entrada
    adapted_results = {}
    for condition, results in results_dict.items():
        if results is None:
            continue
        
        # Detectar single population
        single_pop = results.get('single_population', False)
        
        # Create pseudo spike monitors for validation
        class PseudoSpikeMonitor:
            def __init__(self, spike_times, spike_indices):
                self.t = spike_times * 1000  # Convert to ms if needed 
                self.i = spike_indices
        
        spike_mon_A = PseudoSpikeMonitor(results['spike_times_A'], results['spike_neurons_A'])
        spike_mon_B = None if single_pop else PseudoSpikeMonitor(results['spike_times_B'], results['spike_neurons_B'])
        
        adapted_results[condition] = {
            'spike_monitor_A': spike_mon_A,
            'spike_monitor_B': spike_mon_B,
            'single_population': single_pop
        }
    
    # Run validation
    validator = NeuralActivityValidator()
    validation_results = {}
    
    for condition, data in adapted_results.items():
        val_A = validator.validate_single_population(
            data['spike_monitor_A'], f'{condition}_A', warmup_ms, total_time_ms
        )
        validation_results[f'{condition}_A'] = val_A
        
        if not data['single_population']:
            val_B = validator.validate_single_population(
                data['spike_monitor_B'], f'{condition}_B', warmup_ms, total_time_ms
            )
            validation_results[f'{condition}_B'] = val_B
    
    return validation_results