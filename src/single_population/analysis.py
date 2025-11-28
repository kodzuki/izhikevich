from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def analysis(sim_results):
    """
    Análisis espectral y estadístico de los resultados
    """
    pop_mon_exc = sim_results['pop_mon_exc']
    spike_mon_exc = sim_results['spike_mon_exc']
    spike_mon_inh = sim_results['spike_mon_inh']
    synapses = sim_results['synapses']
    params = sim_results['params']
    
    N_exc = params['N_exc']
    N_inh = params['N_inh']
    N_total = params['N_total']
    duration = params['duration']
    connectivity = params['connectivity']
    
    # ANÁLISIS ESPECTRAL
    pop_rate = pop_mon_exc.rate/Hz
    stabilization_time = int(len(pop_rate) * 0.2)
    pop_rate_stable = pop_rate[stabilization_time:]
    
    if len(pop_rate_stable) < 100:
        print("Warning: Datos insuficientes para análisis espectral")
        return None
    
    dt_sec = (pop_mon_exc.t[1] - pop_mon_exc.t[0])/second
    fs = 1.0 / dt_sec
    bin_size = 5*ms
    
    time_bins = np.arange(stabilization_time * (1000/len(pop_rate)), 1000, bin_size/ms)
    activity = []
    for t in time_bins:
        total_spikes = (sum((spike_mon_exc.t >= t*ms) & (spike_mon_exc.t < (t + bin_size/ms)*ms)) +
                    sum((spike_mon_inh.t >= t*ms) & (spike_mon_inh.t < (t + bin_size/ms)*ms)))
        activity.append(total_spikes)
    
    freqs, psd = signal.periodogram(activity, fs=1000/(bin_size/ms))
    
    # Definir bandas
    alfa_band = (freqs >= 8) & (freqs <= 12)
    gamma_band = (freqs >= 30) & (freqs <= 50)
    
    # Calcular métricas
    alfa_power = np.sum(psd[alfa_band]) if np.any(alfa_band) else 0
    gamma_power = np.sum(psd[gamma_band]) if np.any(gamma_band) else 0
    total_power = np.sum(psd)
    
    alfa_peak_freq = freqs[alfa_band][np.argmax(psd[alfa_band])] if np.any(alfa_band) else 0
    gamma_peak_freq = freqs[gamma_band][np.argmax(psd[gamma_band])] if np.any(gamma_band) else 0
    alfa_peak_power = np.max(psd[alfa_band]) if np.any(alfa_band) else 0
    gamma_peak_power = np.max(psd[gamma_band]) if np.any(gamma_band) else 0
    
    # Calcular spikes por neurona
    duration_sec = float(duration/second)
    exc_spike_counts = np.zeros(N_exc)
    inh_spike_counts = np.zeros(N_inh)
    
    for i in range(N_exc):
        exc_spike_counts[i] = np.sum(spike_mon_exc.i == i)
    for i in range(N_inh):
        inh_spike_counts[i] = np.sum(spike_mon_inh.i == i)
    
    # ESTADÍSTICAS
    total_synapses = (len(synapses['syn_ee'].i) + len(synapses['syn_ii'].i) + 
                     len(synapses['syn_ie'].i) + len(synapses['syn_ei'].i))
    conn_prob_real = total_synapses / (N_total * N_total)
    
    total_spikes_exc = len(spike_mon_exc.t)
    total_spikes_inh = len(spike_mon_inh.t)
    mean_freq_exc = total_spikes_exc / (N_exc * duration_sec)
    mean_freq_inh = total_spikes_inh / (N_inh * duration_sec)
    
    asynchrony = np.std(pop_rate_stable) / np.mean(pop_rate_stable)
    active_total = sum(exc_spike_counts > 0) + sum(inh_spike_counts > 0)
    
    # PRINT ESTADÍSTICAS
    print("DATOS GENERALES DE LA SIMULACIÓN")
    print(f"{'='*60}")
    print(f"Arquitectura:")
    print(f"  Neuronas: {N_exc} exc + {N_inh} inh (ratio 4:1 ✓)")
    print(f"  Conexiones: {total_synapses} totales")
    print(f"  Prob. conexión: {conn_prob_real:.3f} (esperada: {connectivity:.3f})")
            
    print(f"\nActividad:")
    print(f"  Freq. excitatorias: {mean_freq_exc:.1f} Hz (paper: ~8Hz)")
    print(f"  Freq. inhibitorias: {mean_freq_inh:.1f} Hz")
    print(f"  Spikes totales: {total_spikes_exc + total_spikes_inh}")
    
    print(f"\nComportamiento emergente:")
    print(f"  Índice asincronía: {asynchrony:.2f} (>1 = asíncrono ✓)")
    print(f"  Neuronas activas: {active_total}/{N_total} ({100*active_total/N_total:.1f}%)")
    print("---------------------------------------------")
    
    print(f"\n=== DIAGNÓSTICO DE LA SIMULACIÓN ===")
    print(f"Duración: {duration}")
    print(f"Spikes excitatorios: {total_spikes_exc}")
    print(f"Spikes inhibitorios: {total_spikes_inh}")
    print(f"Tasa promedio exc: {mean_freq_exc:.1f} Hz/neurona")
    print(f"Tasa promedio inh: {mean_freq_inh:.1f} Hz/neurona")
    print(f"Tasa poblacional media: {np.mean(pop_rate_stable):.1f} Hz")
    print(f"Proporción Alfa: {alfa_power/total_power:.4f}")
    print(f"Proporción Gamma: {gamma_power/total_power:.4f}")
    if alfa_peak_freq > 0:
        print(f"Pico Alfa: {alfa_peak_freq:.2f} Hz")
    if gamma_peak_freq > 0:
        print(f"Pico Gamma: {gamma_peak_freq:.2f} Hz")
        
    return {
        'freqs': freqs,
        'psd': psd,
        'alfa_band': alfa_band,
        'gamma_band': gamma_band,
        'alfa_power': alfa_power,
        'gamma_power': gamma_power,
        'total_power': total_power,
        'alfa_peak_freq': alfa_peak_freq,
        'gamma_peak_freq': gamma_peak_freq,
        'alfa_peak_power': alfa_peak_power,
        'gamma_peak_power': gamma_peak_power,
        'pop_rate': pop_rate,
        'pop_rate_stable': pop_rate_stable,
        'stabilization_time': stabilization_time,
        'exc_spike_counts': exc_spike_counts,
        'inh_spike_counts': inh_spike_counts,
        'bin_size': bin_size,
        'time_bins': time_bins,
        'activity': activity
    }


def plot_results(sim_results, analysis_results):
    """
    Genera plots de los resultados de simulación y análisis
    """
    # Extraer datos
    spike_mon_exc = sim_results['spike_mon_exc']
    spike_mon_inh = sim_results['spike_mon_inh']
    params = sim_results['params']
    
    N_exc = params['N_exc']
    N_inh = params['N_inh'] 
    N_total = params['N_total']
    duration = params['duration']
    
    freqs = analysis_results['freqs']
    psd = analysis_results['psd']
    alfa_band = analysis_results['alfa_band']
    gamma_band = analysis_results['gamma_band']
    alfa_peak_freq = analysis_results['alfa_peak_freq']
    gamma_peak_freq = analysis_results['gamma_peak_freq']
    pop_rate = analysis_results['pop_rate']
    stabilization_time = analysis_results['stabilization_time']
    exc_spike_counts = analysis_results['exc_spike_counts']
    inh_spike_counts = analysis_results['inh_spike_counts']
    bin_size = analysis_results['bin_size']
    time_bins = analysis_results['time_bins']
    activity = analysis_results['activity']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Espectro de Actividad (top-left)
    axes[0, 0].plot(freqs, psd, 'k-', alpha=0.5)
    axes[0, 0].plot(freqs[alfa_band], psd[alfa_band], 'b-', linewidth=2, label='Alfa (8-12 Hz)')
    axes[0, 0].plot(freqs[gamma_band], psd[gamma_band], 'r-', linewidth=2, label='Gamma (30-50 Hz)')
    
    if alfa_peak_freq > 0:
        axes[0, 0].axvline(alfa_peak_freq, color='blue', linestyle='--', alpha=0.7, 
                          label=f'Pico α: {alfa_peak_freq:.1f}Hz')
    if gamma_peak_freq > 0:
        axes[0, 0].axvline(gamma_peak_freq, color='red', linestyle='--', alpha=0.7, 
                          label=f'Pico γ: {gamma_peak_freq:.1f}Hz')
    
    axes[0, 0].set_xlabel('Frecuencia (Hz)')
    axes[0, 0].set_ylabel('PSD')
    axes[0, 0].set_title('Espectro de Actividad')
    axes[0, 0].set_xlim(0, 100)
    axes[0, 0].legend()
    
    # 2. Actividad de Red (top-right)

    axes[0, 1].plot(time_bins, activity, 'k-', linewidth=1)
    axes[0, 1].set_xlabel('Tiempo (ms)')
    axes[0, 1].set_ylabel(f'Spikes/{bin_size/ms:.0f}ms')
    axes[0, 1].set_title('Actividad de Red')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribución de Actividad (bottom-left)
    duration_sec = float(duration/second)
    bins = np.arange(0, max(max(exc_spike_counts), max(inh_spike_counts)) + 2) - 0.5
    axes[1, 0].hist(exc_spike_counts, bins=bins, alpha=0.7, label='Excitatorias', 
                   color='steelblue', density=True)
    axes[1, 0].hist(inh_spike_counts, bins=bins, alpha=0.7, label='Inhibitorias', 
                   color='orange', density=True)
    axes[1, 0].set_xlabel(f'Spikes en {duration_sec:.0f}s')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].set_title('Distribución de Actividad')
    axes[1, 0].legend()
    
    # 4. Raster Plot (bottom-right)
    exc_mask = spike_mon_exc.i < N_exc
    axes[1, 1].plot(spike_mon_exc.t[exc_mask]/ms, spike_mon_exc.i[exc_mask], 
                   '.k', markersize=0.5, alpha=0.85)
    
    inh_mask = spike_mon_inh.i < N_inh
    axes[1, 1].plot(spike_mon_inh.t[inh_mask]/ms, 
                   spike_mon_inh.i[inh_mask] + N_exc, 
                   '.k', markersize=0.5, alpha=0.85)
    
    axes[1, 1].axhline(y=N_exc, color='r', linestyle='-', linewidth=1)
    axes[1, 1].set_xlabel('Tiempo (ms)')
    axes[1, 1].set_ylabel('Neurona')
    axes[1, 1].set_title('Raster Plot - Estilo Paper')
    axes[1, 1].set_xlim(0, float(duration/ms))
    axes[1, 1].set_ylim(0, N_total)
    
    plt.tight_layout()
    plt.show()
