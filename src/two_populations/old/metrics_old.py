import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit
import seaborn as sns
from brian2 import ms
from scipy import signal as sg

###### GENERAL ANALYZER FOR TWO POPULATIONS ######
class NeuralConnectivityAnalyzer:
    """Suite completa de análisis para conectividad entre poblaciones neuronales"""
    
    def __init__(self, analysis_dt=0.5*ms, warmup=500.0):
        self.analysis_dt = float(analysis_dt/ms)
        self.fs = 1000.0 / self.analysis_dt  # Hz
        self.warmup = warmup  # ms para descartar al inicio
        
    def preprocess(self, rate):
        # highpass 1 Hz + detrend
        sos = sg.butter(2, 1, btype='highpass', fs=self.fs, output='sos')
        return sg.detrend(sg.sosfilt(sos, rate))
        
        
    def spikes_to_population_rate(self, spike_monitor, N_neurons, smooth_window=5, analysis_dt=1.0):
        """Convierte spikes en tasa poblacional suavizada"""
        
        if analysis_dt is None:
            analysis_dt = self.analysis_dt
            
        spike_times = np.array(spike_monitor.t/ms)
        spike_neurons = np.array(spike_monitor.i)
        
        # Crear bins temporales
        max_time = np.max(spike_times) if len(spike_times) > 0 else 1000
        time_bins = np.arange(0, max_time + analysis_dt, analysis_dt)
        
        # Contar spikes por bin
        spike_counts, _ = np.histogram(spike_times, bins=time_bins)
        
        # Convertir a tasa (Hz) y normalizar por número de neuronas
        population_rate = spike_counts / (analysis_dt/1000) / N_neurons
        
        # Suavizar con ventana móvil
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            population_rate = np.convolve(population_rate, kernel, mode='same')
        
        # Asegurar que no hay NaNs o Infs
        population_rate = np.nan_to_num(population_rate, nan=0.0, posinf=0.0, neginf=0.0)
            
        return time_bins[:-1], population_rate
    
    def refine_peak(self, lags, corr):
        i = np.argmax(corr)
        if 0 < i < len(corr)-1:
            y1,y2,y3 = corr[i-1], corr[i], corr[i+1]
            denom = (y1 - 2*y2 + y3)
            delta = 0.5*(y1 - y3)/denom if denom!=0 else 0.0
            lag_ref = lags[i] + delta*(lags[1]-lags[0])
            val_ref = y2 - 0.25*(y1 - y3)*delta
            return lag_ref, val_ref
        return lags[i], corr[i]
    
    # Reemplazo para cross_correlation_analysis en metrics.py
    def cross_correlation_analysis(self, signal_A, signal_B, max_lag_ms=50):
        from scipy.signal import find_peaks
        
        # z-score con guardas
        A = signal_A.astype(float).copy()
        B = signal_B.astype(float).copy()
        stdA0, stdB0 = np.std(A), np.std(B)
        A = (A - A.mean()) / (stdA0 if stdA0 > 0 else 1.0)
        B = (B - B.mean()) / (stdB0 if stdB0 > 0 else 1.0)

        # xcorr "unbiased"
        xc   = scipy_signal.correlate(B, A, mode='full')
        lags = scipy_signal.correlation_lags(len(B), len(A), mode='full')
        den  = (len(A) - np.abs(lags)).clip(min=1)
        corr = xc / den

        # pasar lags a ms
        lags_ms = lags * self.analysis_dt

        # filtrar por ventana en ms (ojo unidades)
        mask = np.abs(lags_ms) <= max_lag_ms
        lags_ms = lags_ms[mask]
        corr    = corr[mask]

        # Peak quality analysis usando find_peaks
        abs_corr = np.abs(corr)
        prominence_threshold = np.max(abs_corr) * 0.1  # 10% del pico máximo
        
        peaks, properties = find_peaks(abs_corr, prominence=prominence_threshold, 
                                    height=np.max(abs_corr) * 0.05)
        
        # pico bruto
        i_peak = int(np.argmax(abs_corr))
        peak_lag  = lags_ms[i_peak]
        peak_value = corr[i_peak]
        
        # Peak quality assessment
        n_significant_peaks = len(peaks)
        peak_prominence = properties['prominences'][0] if len(peaks) > 0 and i_peak in peaks else 0
        
        if n_significant_peaks == 0:
            peak_quality = 'no_significant_peak'
        elif n_significant_peaks == 1:
            peak_quality = 'single_clear'
        else:
            peak_quality = 'multiple_peaks'
        
        # Refinamiento parabólico con validación de calidad
        refinement_applied = False
        if (0 < i_peak < len(corr)-1 and 
            peak_quality in ['single_clear', 'multiple_peaks'] and
            abs(peak_value) > 0.05):  # Minimum correlation threshold
            
            y1, y2, y3 = corr[i_peak-1], corr[i_peak], corr[i_peak+1]
            denom = (y1 - 2*y2 + y3)
            
            # Validar curvatura mínima para refinamiento válido
            min_curvature = 0.001
            if abs(denom) > min_curvature:
                delta = 0.5*(y1 - y3)/denom
                # Limitar delta para evitar refinamientos extremos
                delta = np.clip(delta, -2.0, 2.0)
                
                dt = lags_ms[1] - lags_ms[0]
                peak_lag_ref = peak_lag + delta*dt
                peak_value_ref = y2 - 0.25*(y1 - y3)*delta
                refinement_applied = True
            else:
                peak_lag_ref, peak_value_ref = peak_lag, peak_value
        else:
            peak_lag_ref, peak_value_ref = peak_lag, peak_value

        return {
            'lags': lags_ms,
            'correlation': corr,
            'peak_lag': float(peak_lag_ref),
            'peak_value': float(peak_value_ref),
            'peak_quality': peak_quality,
            'n_peaks': n_significant_peaks,
            'peak_prominence': float(peak_prominence),
            'refinement_applied': refinement_applied
        }

    
    def autocorrelation_analysis(self, signal, max_lag=100):
        """Autocorrelación de una población"""
        return self.cross_correlation_analysis(signal, signal, max_lag)
    
    def intrinsic_timescale(self, signal, max_lag_ms=100, threshold=np.exp(-1)):
        """Calcula intrinsic timescale como área bajo autocorrelación hasta threshold"""
        # Check for insufficient activity
        mean_activity = np.mean(signal)
        
        autocorr = self.autocorrelation_analysis(signal, max_lag_ms)
        lags = autocorr['lags']
        correlation = autocorr['correlation']
        
        correlation = correlation / np.max(correlation)
        
        # Solo lags positivos
        positive_mask = lags >= 0
        lags_pos = lags[positive_mask]
        corr_pos = correlation[positive_mask]
        
        if len(lags_pos) < 3:
            return {'tau': np.nan, 'area': 0, 'threshold_lag': 0, 'fit_quality': 'insufficient_data'}
        
        # Encontrar donde cruza el threshold
        above_threshold = corr_pos > threshold
        
        if not np.any(above_threshold):
            return {'tau': 0, 'area': 0, 'threshold_lag': 0, 'fit_quality': 'no_correlation'}
        
        # Encontrar último punto sobre threshold
        last_above_idx = np.where(above_threshold)[0][-1]
        
        if last_above_idx >= len(lags_pos) - 2:
            cutoff_lag = lags_pos[-1]
        else:
            # Interpolar para encontrar cruce exacto
            if last_above_idx < len(corr_pos) - 1:
                y1, y2 = corr_pos[last_above_idx], corr_pos[last_above_idx + 1]
                x1, x2 = lags_pos[last_above_idx], lags_pos[last_above_idx + 1]
                if y2 != y1:
                    cutoff_lag = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
                else:
                    cutoff_lag = x1
            else:
                cutoff_lag = lags_pos[last_above_idx]
        
        # Integrar área hasta el cutoff
        integration_mask = lags_pos <= cutoff_lag
        lags_integration = lags_pos[integration_mask]
        corr_integration = corr_pos[integration_mask]
        
        # Calcular área usando trapezoides
        area = np.trapz(corr_integration, lags_integration)
        
       # print(f"Intrinsic timescale debug logger: {lags_integration=} , {area=}, {mean_activity=}")
        
        if len(lags_integration) < 2 or area <= 0 or np.isnan(area):
            return {'tau': 0, 'area': 0, 'threshold_lag': cutoff_lag, 'fit_quality': 'insufficient_points'}

        # Quality assessment - more conservative for low activity
        if mean_activity < 0.2:  # Low activity
            if cutoff_lag > 30:
                quality = 'good'
            elif cutoff_lag > 10:
                quality = 'moderate'
            else:
                quality = 'poor'
        else:  # Normal activity
            if cutoff_lag > 50:
                quality = 'good'
            elif cutoff_lag > 20:
                quality = 'moderate'
            elif cutoff_lag > 5:
                quality = 'poor'
            else:
                quality = 'very_poor'
        
        return {
            'tau': area if area > 0 else 0,  # 0 en lugar de np.nan
            'area': area,
            'threshold_lag': cutoff_lag,
            'fit_quality': quality,
            'max_correlation': np.max(corr_pos) if len(corr_pos) > 0 else 0,
            'lags_used': lags_integration,
            'corr_used': corr_integration,
        }

        
    def phase_locking_value(self, signal_A, signal_B, freq_bands=None):
        """PLV/PLI robusto: recorta a longitud común antes de filtrar."""
        if freq_bands is None:
            freq_bands = {
                'alpha': (8, 12),
                'beta': (13, 30),
                'gamma': (30, 50),
                'broadband': (1, 100),
            }

        # Asegurar misma longitud
        L = int(min(len(signal_A), len(signal_B)))
        if L <= 3:
            return {k: {'plv': 0.0, 'pli': 0.0, 'phase_diff': np.array([]), 'phase_stability': 0.0}
                    for k in freq_bands.keys()}
        A = np.asarray(signal_A[:L], dtype=float)
        B = np.asarray(signal_B[:L], dtype=float)

        results = {}
        for band_name, (low_f, high_f) in freq_bands.items():
            sos = scipy_signal.butter(4, [low_f, high_f], btype='band', fs=self.fs, output='sos')
            fA = scipy_signal.sosfilt(sos, A)
            fB = scipy_signal.sosfilt(sos, B)

            hA = scipy_signal.hilbert(fA)
            hB = scipy_signal.hilbert(fB)

            phase_A = np.angle(hA)
            phase_B = np.angle(hB)
            phase_diff = phase_A - phase_B

            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))

            cross_spectrum = hA * np.conj(hB)
            imag_part = np.imag(cross_spectrum)
            pli = float(np.abs(np.mean(np.sign(imag_part)))) if imag_part.size else 0.0

            # Garantizar PLI ≤ PLV
            pli = float(min(pli, plv))
            phase_stability = float(max(0.0, 1 - np.var(np.unwrap(phase_diff)) / (2*np.pi)**2))

            results[band_name] = {
                'plv': plv,
                'pli': pli,
                'phase_diff': phase_diff,
                'phase_stability': phase_stability,
            }
        return results
        

    def spectral_coherence(self, signal_A, signal_B, nperseg=None):
        """Coherencia espectral entre poblaciones"""
        
        if nperseg is None:
            nperseg = int(min(len(signal_A)//8, self.fs*2))  # ~2 s; >=32 garantizado abajo
        nperseg = max(32, nperseg)
        noverlap = nperseg//4
        
        print(f"{nperseg=}, {noverlap=}, {len(signal_A)=}")
        
        # Add small amount of independent noise to avoid perfect coherence
        # noise_level = np.std(signal_A) * 0.01
        # signal_A_noisy = signal_A + np.random.normal(0, noise_level, len(signal_A))
        # signal_B_noisy = signal_B + np.random.normal(0, noise_level, len(signal_B))
        
        freqs, coherence = scipy_signal.coherence(
            signal_A, signal_B, fs=self.fs, window='hann', nperseg=nperseg,
            noverlap=noverlap, detrend='constant'
        )
        
        # Filter out high frequencies (>100Hz) for biological plausibility
        valid_freq_mask = freqs <= 100
        freqs_filtered = freqs[valid_freq_mask]
        coherence_filtered = coherence[valid_freq_mask]
        
        # Apply smoothing to filtered data
        if len(coherence_filtered) > 5:
            from scipy.ndimage import gaussian_filter1d
            coherence_filtered = gaussian_filter1d(coherence_filtered, sigma=1.0)
        
        # Limit maximum coherence to realistic value
        coherence_filtered = np.minimum(coherence_filtered, 0.95)
        
        # Find peak in filtered data
        if len(coherence_filtered) > 0:
            peak_idx = np.argmax(coherence_filtered)
            peak_freq = freqs_filtered[peak_idx]
            peak_coherence = coherence_filtered[peak_idx]
        else:
            peak_freq = 0
            peak_coherence = 0
        
        # Coherence in specific bands
        alpha_mask = (freqs_filtered >= 8) & (freqs_filtered <= 12)
        gamma_mask = (freqs_filtered >= 30) & (freqs_filtered <= 50)
        
        alpha_coherence = np.mean(coherence_filtered[alpha_mask]) if np.any(alpha_mask) else 0
        gamma_coherence = np.mean(coherence_filtered[gamma_mask]) if np.any(gamma_mask) else 0
        
        return {
            'freqs': freqs_filtered,
            'coherence': coherence_filtered,
            'peak_freq': peak_freq,
            'peak_coherence': peak_coherence,
            'alpha_coherence': alpha_coherence,
            'gamma_coherence': gamma_coherence
        }
    
    def power_spectrum(self, signal, target_freq_res=0.5, fs_analysis=None):
        """Espectro de potencias (Welch) + potencia en bandas."""
        if fs_analysis is None:
            fs_analysis = self.fs  # usa la fs del analizador

        # nperseg ~ fs / Δf, limitado por la señal
        est_nperseg = int(round(fs_analysis / max(target_freq_res, 1e-6)))
        nperseg = min(len(signal), max(64, est_nperseg))
        noverlap = nperseg // 2

        freqs, psd = scipy_signal.welch(
            signal, fs=fs_analysis, window='hann',
            nperseg=nperseg, noverlap=noverlap,
            detrend='constant', scaling='density'
        )

        # Máscaras de bandas
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        gamma_mask = (freqs >= 30) & (freqs <= 50)

        # Potencias (manejo de vacíos)
        alpha_power = float(np.trapz(psd[alpha_mask], freqs[alpha_mask])) if np.any(alpha_mask) else 0.0
        gamma_power = float(np.trapz(psd[gamma_mask], freqs[gamma_mask])) if np.any(gamma_mask) else 0.0
        total_power = float(np.trapz(psd, freqs)) if len(freqs) else 0.0

        peak_idx = int(np.argmax(psd)) if len(psd) else 0
        peak_freq = float(freqs[peak_idx]) if len(freqs) else 0.0
        peak_power = float(psd[peak_idx]) if len(psd) else 0.0

        return {
            'freqs': freqs,
            'psd': psd,
            'alpha_power': alpha_power,
            'gamma_power': gamma_power,
            'total_power': total_power,
            'peak_freq': peak_freq,
            'peak_power': peak_power
        }

    
    def complete_analysis(self, spike_mon_A, spike_mon_B, N_A, N_B,
                      t0_ms=500.0, view_ms=None):
        """
        Suite completa de análisis.
        t0_ms: tiempo (ms) a partir del cual se analizan/visualizan los datos.
        view_ms: si se da, limita lo que se DEVUELVE para plots a los primeros view_ms tras t0.
                (Las métricas se calculan sobre la señal post-corte completa igualmente.)
        """
        # 1) spikes -> tasas con el dt del analizador (misma malla temporal por dt fijo)
        tA, rA = self.spikes_to_population_rate(spike_mon_A, N_A, smooth_window=15, analysis_dt=None)
        tB, rB = self.spikes_to_population_rate(spike_mon_B, N_B, smooth_window=15, analysis_dt=None)

        # 2) sincronizar longitudes
        L = int(min(len(rA), len(rB)))
        tA, rA = tA[:L], rA[:L]
        tB, rB = tB[:L], rB[:L]

        # 3) preprocesado (hp 1 Hz + detrend)
        rA = self.preprocess(rA)
        rB = self.preprocess(rB)

        # 4) aplicar corte y re-referenciar tiempo a 0 ms
        cut_idx = int(round(t0_ms / self.analysis_dt))
        cut_idx = max(0, min(cut_idx, L - 1))

        time_post = tA[cut_idx:] - tA[cut_idx]
        rA_post   = rA[cut_idx:]
        rB_post   = rB[cut_idx:]

        # 5) series para PLOTS (opcional recorte visual)
        if view_ms is not None:
            Lview = int(round(view_ms / self.analysis_dt))
            time_plot = time_post[:Lview]
            rA_plot   = rA_post[:Lview]
            rB_plot   = rB_post[:Lview]
        else:
            time_plot = time_post
            rA_plot   = rA_post
            rB_plot   = rB_post

        # 6) métricas sobre señales post-corte (misma longitud garantizada)
        Lm = int(min(len(rA_post), len(rB_post)))
        rA_metrics = rA_post[:Lm]
        rB_metrics = rB_post[:Lm]

        results = {
            # series para gráficos (post-corte, quizás truncadas a view_ms)
            'time': time_plot,
            'rate_A': rA_plot,
            'rate_B': rB_plot,
            't0_ms': float(t0_ms),

            # info absoluta para raster (usamos t0_ms para llevarlo a 0–… ms en el plot)
            'spike_times_A': np.array(spike_mon_A.t/ms),
            'spike_neurons_A': np.array(spike_mon_A.i),
            'spike_times_B': np.array(spike_mon_B.t/ms),
            'spike_neurons_B': np.array(spike_mon_B.i),

            # Métricas (todas con las señales post-corte y longitudes iguales)
            'cross_correlation': self.cross_correlation_analysis(rA_metrics, rB_metrics),
            'autocorr_A': self.autocorrelation_analysis(rA_metrics),
            'autocorr_B': self.autocorrelation_analysis(rB_metrics),
            'int_A': self.intrinsic_timescale(rA_metrics),
            'int_B': self.intrinsic_timescale(rB_metrics),
            'plv_pli': self.phase_locking_value(rA_metrics, rB_metrics),
            'coherence': self.spectral_coherence(rA_metrics, rB_metrics),
            'psd_A': self.power_spectrum(rA_metrics, fs_analysis=self.fs),
            'psd_B': self.power_spectrum(rB_metrics, fs_analysis=self.fs),
        }
        return results
    

###### ANALSYS MAIN WRAPPER FUNCTION FOR TWO POPULATIONS ######

def analyze_simulation_results(spike_mon_exc_A, spike_mon_exc_B, N=1000, 
                             condition_name="test", warmup=500.0, state_monitors=True):
    """Función única para análisis de conectividad neuronal"""
    analyzer = NeuralConnectivityAnalyzer(analysis_dt=0.25*ms, warmup=warmup)
    
    # Verificar datos
    if len(spike_mon_exc_A.t) == 0 or len(spike_mon_exc_B.t) == 0:
        print(f"Warning: No spikes detected in {condition_name}")
        return None
    
    results = analyzer.complete_analysis(spike_mon_exc_A, spike_mon_exc_B, N, N, t0_ms=warmup)
    
    # Add state monitors if provided
    if state_monitors:
        results['state_monitor_A'] = state_monitors.get('A')
        results['state_monitor_B'] = state_monitors.get('B')
    
    # Imprimir resumen
    print(f"\n=== Análisis de Conectividad - {condition_name} ===")
    print(f"Cross-correlation peak: {results['cross_correlation']['peak_value']:.3f} at {results['cross_correlation']['peak_lag']:.1f}ms")
    print(f"PLV Alpha: {results['plv_pli']['alpha']['plv']:.3f}")
    print(f"PLI Alpha: {results['plv_pli']['alpha']['pli']:.3f}")
    print(f"PLV Gamma: {results['plv_pli']['gamma']['plv']:.3f}")
    print(f"PLI Gamma: {results['plv_pli']['gamma']['pli']:.3f}")
    print(f"Coherence peak: {results['coherence']['peak_coherence']:.3f} at {results['coherence']['peak_freq']:.1f}Hz")
    print(f"Intrinsic timescales: Pop A = {results['int_A']['tau']:.1f}ms ({results['int_A']['fit_quality']}), Pop B = {results['int_B']['tau']:.1f}ms ({results['int_B']['fit_quality']})")
    
    return results
    
    
###### CONECTIVITY DASHBOARD FOR TWO POPULATIONS ######

def plot_connectivity_dashboard(results_dict, figsize=(18, 12)):
    """Dashboard principal: Conectividad y Sincronización"""
    
    # Filtrar resultados válidos
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    if not valid_results:
        print("Error: No valid results to plot")
        return None
        
    conditions = list(valid_results.keys())
    n_conditions = len(conditions)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Cross-correlation curves
    ax1 = fig.add_subplot(gs[0, 0])
    for condition, results in valid_results.items():
        cc = results['cross_correlation']
        ax1.plot(cc['lags'], cc['correlation'], label=f"{condition}", linewidth=2)
    ax1.set_xlabel('Lag (ms)')
    ax1.set_ylabel('Cross-correlation')
    ax1.set_title('Cross-correlation Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. PLV by Frequency Bands
    ax2 = fig.add_subplot(gs[0, 1])
    bands = ['alpha', 'beta', 'gamma', 'broadband']
    x_pos = np.arange(len(bands))
    width = 0.35 / n_conditions
    
    for i, (condition, results) in enumerate(valid_results.items()):
        plv_values = [results['plv_pli'][band]['plv'] for band in bands]
        pli_values = [results['plv_pli'][band]['pli'] for band in bands]
        
        ax2.bar(x_pos + i*width*2, plv_values, width, label=f"{condition} PLV", alpha=0.8)
        ax2.bar(x_pos + i*width*2 + width, pli_values, width, label=f"{condition} PLI", alpha=0.6)
    
    ax2.set_xlabel('Frequency Bands')
    ax2.set_ylabel('Phase Locking')
    ax2.set_title('PLV & PLI by Frequency Band')
    ax2.set_xticks(x_pos + width*(n_conditions-0.5))
    ax2.set_xticklabels(bands)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Coherence Spectra
    ax3 = fig.add_subplot(gs[0, 2])
    for condition, results in valid_results.items():
        coh = results['coherence']
        # ax3.semilogy(coh['freqs'], coh['coherence'], label=condition, linewidth=2)
        ax3.plot(coh['freqs'], coh['coherence'], label=condition, linewidth=2)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Coherence')
    ax3.set_title('Spectral Coherence')
    ax3.set_xlim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Intrinsic Timescales
    ax4 = fig.add_subplot(gs[1, 0])
    pop_names = ['Pop A', 'Pop B']
    x_pos = np.arange(len(pop_names))
    width = 0.8 / n_conditions
    
    for i, (condition, results) in enumerate(valid_results.items()):
        tau_values = [results['int_A']['tau'], results['int_B']['tau']]
        bars = ax4.bar(x_pos + i*width, tau_values, width, label=condition, alpha=0.8)
        
        # Agregar indicadores de calidad de fit
        for j, bar in enumerate(bars):
            quality = results['int_A']['fit_quality'] if j == 0 else results['int_B']['fit_quality']
            if quality == 'poor':
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
    
    ax4.set_xlabel('Population')
    ax4.set_ylabel('Intrinsic Timescale (ms)')
    ax4.set_title('Intrinsic Timescales')
    ax4.set_xticks(x_pos + width*(n_conditions-1)/2)
    ax4.set_xticklabels(pop_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary Metrics
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ['Cross-corr', 'Alpha PLV', 'Gamma Coh', 'Avg INT']
    x_pos = np.arange(len(metrics))
    width = 0.8 / n_conditions
    
    for i, (condition, results) in enumerate(valid_results.items()):
        avg_tau = (results['int_A']['tau'] + results['int_B']['tau']) / 2
        # Normalizar tau: convertir a escala 0-1 asumiendo rango típico 0-100ms
        normalized_tau = min(avg_tau / 50.0, 1.0)  # 50ms = 1.0, valores mayores = 1.0
        
        values = [
            abs(results['cross_correlation']['peak_value']),
            results['plv_pli']['alpha']['plv'],
            results['coherence']['gamma_coherence'],
            normalized_tau
        ]
        ax5.bar(x_pos + i*width, values, width, label=condition, alpha=0.8)
    
    ax5.set_xlabel('Metrics')
    ax5.set_ylabel('Normalized Value')
    ax5.set_title('Summary Connectivity Metrics')
    ax5.set_xticks(x_pos + width*(n_conditions-1)/2)
    ax5.set_xticklabels(metrics, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. PLI vs PLV Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    for condition, results in valid_results.items():
        plv_alpha = results['plv_pli']['alpha']['plv']
        pli_alpha = results['plv_pli']['alpha']['pli']
        plv_gamma = results['plv_pli']['gamma']['plv']
        pli_gamma = results['plv_pli']['gamma']['pli']
        
        ax6.scatter(plv_alpha, pli_alpha, s=100, label=f"{condition} α", alpha=0.7)
        ax6.scatter(plv_gamma, pli_gamma, s=100, marker='s', label=f"{condition} γ", alpha=0.7)
    
    ax6.set_xlabel('PLV')
    ax6.set_ylabel('PLI')
    ax6.set_title('PLV vs PLI')
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

###### GENERAL DASHBOARD FOR TWO POPULATIONS COMPARISON ######

def plot_population_dashboard(results_dict, figsize=(18, 12)):
    """Dashboard secundario: Dinámicas Poblacionales Detalladas (post-corte)."""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        print("Error: No valid results to plot")
        return None

    conditions = list(valid_results.keys())
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # 1) Autocorrelaciones
    ax1 = fig.add_subplot(gs[0, 0])
    for condition, res in valid_results.items():
        ac_A, ac_B = res['autocorr_A'], res['autocorr_B']
        ax1.plot(ac_A['lags'], ac_A['correlation'],
                 label=f"{condition} Pop A", alpha=0.8, linewidth=2)
        ax1.plot(ac_B['lags'], ac_B['correlation'],
                 label=f"{condition} Pop B", alpha=0.8, linewidth=2, linestyle='--')

        if res['int_A'].get('lags_used') is not None and len(res['int_A']['lags_used']) > 0:
            ax1.plot(res['int_A']['lags_used'], res['int_A']['corr_used'],
                     'r:', alpha=0.7, linewidth=2, label=f"{condition} Pop A fit region")
    ax1.set_xlabel('Lag (ms)'); ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Autocorrelation Functions'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2) PSD
    ax2 = fig.add_subplot(gs[0, 1])
    for condition, res in valid_results.items():
        psd_A, psd_B = res['psd_A'], res['psd_B']
        # ax2.semilogy(psd_A['freqs'], psd_A['psd'],
        #              label=f"{condition} Pop A", alpha=0.8, linewidth=2)
        # ax2.semilogy(psd_B['freqs'], psd_B['psd'],
        #              label=f"{condition} Pop B", alpha=0.8, linewidth=2, linestyle='--')
        ax2.plot(psd_A['freqs'], psd_A['psd'],
                    label=f"{condition} Pop A", alpha=0.8, linewidth=2)
        ax2.plot(psd_B['freqs'], psd_B['psd'],
                    label=f"{condition} Pop B", alpha=0.8, linewidth=2, linestyle='--')
    ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Power Spectra'); ax2.set_xlim(0, 60)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # 3) Series de actividad 0–2000 ms post-corte
    ax3 = fig.add_subplot(gs[1, :])
    view_ms = [res['time'][-1] for condition, res in valid_results.items()][0]
    for condition, res in valid_results.items():
        t = res['time']  # ya empieza en 0 ms post-corte
        end_idx = np.searchsorted(t, view_ms, side='right')
        ax3.plot(t[:end_idx], res['rate_A'][:end_idx],
                 label=f"{condition} Pop A", alpha=0.7, linewidth=1)
        ax3.plot(t[:end_idx], res['rate_B'][:end_idx],
                 label=f"{condition} Pop B", alpha=0.7, linewidth=1, linestyle='--')
    ax3.set_xlabel('Time (ms)'); ax3.set_ylabel('Population Rate (Hz)')
    ax3.set_title(f'Population Activity Time Series (first {view_ms}ms post-cut)')
    ax3.legend(bbox_to_anchor=(1.05, 1))
    ax3.grid(True, alpha=0.3)

    # 4) Raster 0–2000 ms post-corte (primera condición para claridad)
    ax4 = fig.add_subplot(gs[2, 0])
    first_condition = conditions[0]
    res0 = valid_results[first_condition]
    t0 = float(res0.get('t0_ms', 0.0))
    raster_window = (0.0, view_ms)  # ms post-corte
    neuron_limit = 100

    # A
    tA_abs = res0['spike_times_A']; nA = res0['spike_neurons_A']
    mA = (tA_abs >= t0 + raster_window[0]) & (tA_abs < t0 + raster_window[1]) \
         & (nA < neuron_limit)
    ax4.scatter(tA_abs[mA] - t0, nA[mA], s=0.5, alpha=0.6, color='blue', label='Pop A')

    # B (offset vertical)
    tB_abs = res0['spike_times_B']; nB = res0['spike_neurons_B']
    mB = (tB_abs >= t0 + raster_window[0]) & (tB_abs < t0 + raster_window[1]) \
         & (nB < neuron_limit)
    ax4.scatter(tB_abs[mB] - t0, nB[mB] + neuron_limit + 10,
                s=0.5, alpha=0.6, color='red', label='Pop B')

    ax4.set_xlim(raster_window)
    ax4.set_xlabel('Time (ms)'); ax4.set_ylabel('Neuron ID')
    ax4.set_title(f'Raster Plot - {first_condition} (post-cut)')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    # 5) Potencia por banda/pop
    ax5 = fig.add_subplot(gs[2, 1])
    pop_names = ['Pop A', 'Pop B']; x = np.arange(len(pop_names)); w = 0.35
    for condition, res in valid_results.items():
        alpha_p = [res['psd_A']['alpha_power'], res['psd_B']['alpha_power']]
        gamma_p = [res['psd_A']['gamma_power'], res['psd_B']['gamma_power']]
        ax5.bar(x - w/2, alpha_p, w/2, label=f"{condition} Alpha", alpha=0.8)
        ax5.bar(x,       gamma_p, w/2, label=f"{condition} Gamma", alpha=0.6)
    ax5.set_xlabel('Population'); ax5.set_ylabel('Spectral Power')
    ax5.set_title('Alpha/Gamma Power by Population')
    ax5.set_xticks(x); ax5.set_xticklabels(pop_names)
    ax5.legend(); ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

###### NETWORK STATISTICS TWO POPULATIONS  ######

def print_network_statistics_table(results, network, N_exc=800, N_inh=200, T_total=3000, warmup=500):
    """Imprimir estadísticas en formato tabla comparativa"""
    
    def calculate_firing_rates(spike_monitor, N_exc, N_inh, T_sim):
        spike_times = spike_monitor.t/ms
        spike_times_filtered = spike_times[spike_times >= warmup]
        spike_indices_filtered = spike_monitor.i[spike_times >= warmup]
        T_analysis = (T_sim - warmup) / 1000
        
        exc_spikes = np.sum(spike_indices_filtered < N_exc)
        inh_spikes = np.sum(spike_indices_filtered >= N_exc)
        freq_exc_mean = exc_spikes / (N_exc * T_analysis)
        freq_inh_mean = inh_spikes / (N_inh * T_analysis)
        
        freq_exc_individual = [(spike_indices_filtered == i).sum() / T_analysis for i in range(N_exc)]
        freq_inh_individual = [(spike_indices_filtered == (i + N_exc)).sum() / T_analysis for i in range(N_inh)]
        
        return freq_exc_mean, freq_inh_mean, freq_exc_individual, freq_inh_individual
    
    # Alternative: More robust CV using individual neuron firing rates
    def calculate_synchrony_metrics(spike_monitor, N_exc, warmup=500, T_total=3000):
        """Alternative robust CV calculation using individual neuron rates"""
        spike_times = spike_monitor.t/ms
        spike_times_filtered = spike_times[spike_times >= warmup]
        spike_indices_filtered = spike_monitor.i[spike_times >= warmup]
        
        T_analysis = (T_total - warmup) / 1000  # Convert to seconds
        
        # Calculate individual neuron firing rates
        individual_rates = []
        for neuron_id in range(N_exc):
            neuron_spikes = np.sum(spike_indices_filtered == neuron_id)
            firing_rate = neuron_spikes / T_analysis
            individual_rates.append(firing_rate)
        
        individual_rates = np.array(individual_rates)
        
        # CV of individual firing rates
        mean_rate = np.mean(individual_rates)
        std_rate = np.std(individual_rates)
        
        if mean_rate > 1.0:
            cv_individual = std_rate / mean_rate
        else:
            cv_individual = 0
        
        # Population activity variability (original method but fixed)
        bin_size_ms = 10  # Larger bins for stability
        time_bins = np.arange(warmup, T_total, bin_size_ms)
        pop_activity = []
        
        for t in time_bins:
            mask = (spike_times_filtered >= t) & (spike_times_filtered < t + bin_size_ms)
            spikes_in_bin = np.sum(spike_indices_filtered[mask] < N_exc)
            # Normalize by number of neurons and bin size
            normalized_activity = spikes_in_bin / (N_exc * bin_size_ms / 1000)
            pop_activity.append(normalized_activity)
        
        pop_activity = np.array(pop_activity)
        mean_pop = np.mean(pop_activity)
        std_pop = np.std(pop_activity)
        
        if mean_pop > 0.01:
            cv_population = std_pop / mean_pop
            fano_factor = np.var(pop_activity) / mean_pop
        else:
            cv_population = 0
            fano_factor = 0
        
        print(f"    DEBUG: CV_individual={cv_individual:.3f}, CV_population={cv_population:.3f}")
        
        # Return the more stable measure
        return cv_individual, fano_factor

    def calculate_connection_stats(syn_intra, N_total):
        n_connections = len(syn_intra.i)
        max_possible = N_total * (N_total - 1)
        connection_prob = n_connections / max_possible
        sparsity = 1 - connection_prob
        return n_connections, connection_prob, sparsity
    
    # def calculate_burst_metrics(spike_monitor, N_exc, burst_threshold=50):
        
    #     burst_threshold = max(10, N_exc * 0.05) 
        
    #     spike_times = spike_monitor.t/ms
    #     spike_times_filtered = spike_times[spike_times >= warmup]
    #     spike_indices_filtered = spike_monitor.i[spike_times >= warmup]
        
    #     time_bins_1ms = np.arange(warmup, T_total, 1)
    #     exc_activity_1ms = []
        
    #     for t in time_bins_1ms:
    #         mask = (spike_times_filtered >= t) & (spike_times_filtered < t + 1)
    #         exc_spikes = np.sum(spike_indices_filtered[mask] < N_exc)
    #         exc_activity_1ms.append(exc_spikes)
        
    #     exc_activity_1ms = np.array(exc_activity_1ms)
    #     burst_mask = exc_activity_1ms > burst_threshold
    #     burst_periods = []
        
    #     in_burst = False
    #     burst_start = 0
        
    #     for i, is_burst in enumerate(burst_mask):
    #         if is_burst and not in_burst:
    #             burst_start = i
    #             in_burst = True
    #         elif not is_burst and in_burst:
    #             burst_periods.append((burst_start, i-1))
    #             in_burst = False
        
    #     if burst_periods:
    #         burst_durations = [end - start + 1 for start, end in burst_periods]
    #         burst_rate = len(burst_periods) / ((T_total - warmup) / 1000)
    #         mean_burst_duration = np.mean(burst_durations)
    #         burst_coverage = np.sum(burst_mask) / len(burst_mask)
    #     else:
    #         burst_rate = 0
    #         mean_burst_duration = 0
    #         burst_coverage = 0
        
    #     return burst_rate, mean_burst_duration, burst_coverage
    
    # Recopilar datos para ambas poblaciones
    stats = {}
    N_total = N_exc + N_inh
    
    for pop_name in ['A', 'B']:
        spike_mon = results[pop_name]['spike_monitor']
        group = network.populations[pop_name]['group']
        syn_intra = network.populations[pop_name]['syn_intra']
        
        freq_exc_mean, freq_inh_mean, freq_exc_ind, freq_inh_ind = calculate_firing_rates(
            spike_mon, N_exc, N_inh, T_total)
        cv_exc, fano_factor = calculate_synchrony_metrics(spike_mon, N_exc)
        n_conn, conn_prob, sparsity = calculate_connection_stats(syn_intra, N_total)
       # burst_rate, burst_duration, burst_coverage = calculate_burst_metrics(spike_mon, N_exc)
        
        # Neuronas activas
        active_exc = np.sum(np.array(freq_exc_ind) > 0.1)
        active_inh = np.sum(np.array(freq_inh_ind) > 0.1)
        
        # Pesos sinápticos
        weights = syn_intra.w
        exc_weights = weights[syn_intra.i < N_exc]
        inh_weights = weights[syn_intra.i >= N_exc]
        
        # Nivel de sincronía
        if cv_exc < 0.5:
            sync_level = "Alta"
        elif cv_exc < 1.0:
            sync_level = "Moderada"
        else:
            sync_level = "Asíncrona"
        
        stats[pop_name] = {
            'connections': n_conn,
            'conn_prob': conn_prob,
            'sparsity': sparsity,
            'freq_exc': freq_exc_mean,
            'freq_inh': freq_inh_mean,
            'ratio_ei': freq_exc_mean/max(freq_inh_mean, 0.01) if freq_inh_mean > 0 else 0,
            'active_exc_pct': 100*active_exc/N_exc,
            'active_inh_pct': 100*active_inh/N_inh,
            'cv': cv_exc,
            'fano': fano_factor,
            'sync_level': sync_level,
            # 'burst_rate': burst_rate,
            # 'burst_duration': burst_duration,
            # 'burst_coverage': burst_coverage*100,
            'exc_weight_mean': np.mean(exc_weights),
            'exc_weight_std': np.std(exc_weights),
            'inh_weight_mean': np.mean(inh_weights),
            'inh_weight_std': np.std(inh_weights)
        }
    
    # Imprimir tabla comparativa
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN ESTADÍSTICAS DE RED (Post-warmup {warmup}ms)")
    print(f"{'='*80}")
    
    print(f"{'Métrica':<25} {'Población A':<25} {'Población B':<25}")
    print(f"{'-'*75}")
    
    print(f"{'ARQUITECTURA':<25}")
    print(f"{'  Conexiones':<25} {stats['A']['connections']:<25} {stats['B']['connections']}")
    print(f"{'  Prob. conexión':<25} {stats['A']['conn_prob']:.4f}{'':<21} {stats['B']['conn_prob']:.4f}")
    print(f"{'  Sparsity':<25} {stats['A']['sparsity']:.3f}{'':<22} {stats['B']['sparsity']:.3f}")
    
    print(f"\n{'ACTIVIDAD':<25}")
    print(f"{'  Freq exc (Hz)':<25} {stats['A']['freq_exc']:.2f}{'':<23} {stats['B']['freq_exc']:.2f}")
    print(f"{'  Freq inh (Hz)':<25} {stats['A']['freq_inh']:.2f}{'':<23} {stats['B']['freq_inh']:.2f}")
    print(f"{'  Ratio E/I':<25} {stats['A']['ratio_ei']:.2f}{'':<23} {stats['B']['ratio_ei']:.2f}")
    print(f"{'  Activas exc (%)':<25} {stats['A']['active_exc_pct']:.1f}{'':<24} {stats['B']['active_exc_pct']:.1f}")
    print(f"{'  Activas inh (%)':<25} {stats['A']['active_inh_pct']:.1f}{'':<24} {stats['B']['active_inh_pct']:.1f}")
    
    print(f"\n{'SINCRONÍA':<25}")
    print(f"{'  CV':<25} {stats['A']['cv']:.3f}{'':<22} {stats['B']['cv']:.3f}")
    print(f"{'  Fano Factor':<25} {stats['A']['fano']:.3f}{'':<22} {stats['B']['fano']:.3f}")
    print(f"{'  Nivel':<25} {stats['A']['sync_level']:<25} {stats['B']['sync_level']}")
    
    # print(f"\n{'BURSTING':<25}")
    # print(f"{'  Rate (bursts/s)':<25} {stats['A']['burst_rate']:.2f}{'':<23} {stats['B']['burst_rate']:.2f}")
    # print(f"{'  Duración (ms)':<25} {stats['A']['burst_duration']:.1f}{'':<24} {stats['B']['burst_duration']:.1f}")
    # print(f"{'  Cobertura (%)':<25} {stats['A']['burst_coverage']:.1f}{'':<24} {stats['B']['burst_coverage']:.1f}")
    
    print(f"\n{'PESOS SINÁPTICOS':<25}")
    exc_a = f"{stats['A']['exc_weight_mean']:.3f}±{stats['A']['exc_weight_std']:.3f}"
    exc_b = f"{stats['B']['exc_weight_mean']:.3f}±{stats['B']['exc_weight_std']:.3f}"
    inh_a = f"{stats['A']['inh_weight_mean']:.3f}±{stats['A']['inh_weight_std']:.3f}"
    inh_b = f"{stats['B']['inh_weight_mean']:.3f}±{stats['B']['inh_weight_std']:.3f}"
    print(f"{'  Exc mean±std':<25} {exc_a:<25} {exc_b}")
    print(f"{'  Inh mean±std':<25} {inh_a:<25} {inh_b}")
    
    print(f"\n{'='*80}")
    
    return stats

