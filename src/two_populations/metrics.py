import numpy as np
from scipy import signal as scipy_signal
from brian2 import ms
from scipy import signal as sg
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def validate_signals(signal_A, signal_B):
    """Validate and synchronize signal lengths"""
    A = np.asarray(signal_A, dtype=float).copy()
    B = np.asarray(signal_B, dtype=float).copy()
    
    L = min(len(A), len(B))
    if L <= 3:
        return None, None, L
    
    return A[:L], B[:L], L

def preprocess_signal(signal, fs=None):
    """Detrend + highpass 1Hz (4th order, zero-phase)"""
    
    import numpy as np
    
    signal = np.asarray(signal, dtype=float)
    
    # Detrend primero (más robusto contra drift pre-filtrado)
    # print("signal shape:", signal.shape)
    # print("has NaN:", np.isnan(signal).any())
    # print("has inf:", np.isinf(signal).any())

    signal = sg.detrend(signal)
    
    # HPF solo si fs lo permite
    fc = 1.0
    if fs <= 4 * fc:  # Margen más conservador
        return signal
    
    # Orden 4 (estándar en neurociencia)
    sos = sg.butter(4, fc, btype='highpass', fs=fs, output='sos')
    return sg.sosfiltfilt(sos, signal)

def spikes_to_population_rate(spike_monitor, N_neurons, smooth_window=5, analysis_dt=0.5, T_total=1000):
    spike_times = np.array(spike_monitor.t/ms)
    
    if len(spike_times) == 0:
        max_time = T_total if T_total else 1000  # Default si no hay spikes
    else:
        max_time = T_total if T_total else np.max(spike_times)
    
    time_bins = np.arange(0, max_time + analysis_dt, analysis_dt)
    
    spike_counts, _ = np.histogram(spike_times, bins=time_bins)
    population_rate = spike_counts / (analysis_dt/1000) / N_neurons
    
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        population_rate = np.convolve(population_rate, kernel, mode='same')
    
    return time_bins[:-1], np.nan_to_num(population_rate, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# CORE ANALYSIS FUNCTIONS (STANDALONE)
# =============================================================================

def cross_correlation_analysis(signal_A, signal_B, max_lag_ms=500, dt=0.5):
    """Standalone cross-correlation with peak refinement"""
    A, B, L = validate_signals(signal_A, signal_B)
    if A is None:
        return {'peak_lag': 0, 'peak_value': 0, 'lags': np.array([]), 'correlation': np.array([])}
    
    # Z-score normalization
    std_A, std_B = np.std(A), np.std(B)
    A = (A - A.mean()) / (std_A if std_A > 0 else 1.0)
    B = (B - B.mean()) / (std_B if std_B > 0 else 1.0)
    
    # Cross-correlation
    xc = scipy_signal.correlate(B, A, mode='full')
    lags = scipy_signal.correlation_lags(len(B), len(A), mode='full')
    den = (len(A) - np.abs(lags)).clip(min=1)
    corr = xc / den
    
    # Convert to time units
    lags_ms = lags * dt
    mask = np.abs(lags_ms) <= max_lag_ms
    lags_ms, corr = lags_ms[mask], corr[mask]
    
    # Peak detection and refinement
    i_peak = np.argmax((corr)) # np.abs ??
    peak_lag, peak_value = lags_ms[i_peak], corr[i_peak]
    
    # Parabolic refinement
    if 0 < i_peak < len(corr)-1 and abs(peak_value) > 0.05:
        y1, y2, y3 = corr[i_peak-1], corr[i_peak], corr[i_peak+1]
        denom = (y1 - 2*y2 + y3)
        if abs(denom) > 0.001:
            delta = np.clip(0.5*(y1 - y3)/denom, -2.0, 2.0)
            peak_lag = peak_lag + delta * dt
            peak_value = y2 - 0.25*(y1 - y3)*delta
    
    return {
        'lags': lags_ms,
        'correlation': corr,
        'peak_lag': float(peak_lag),
        'peak_value': float(peak_value),
        # Convención explícita para interpretar el signo del retardo
        'lag_sign_convention': 'corr(B, A): lag>0 => A precede a B'}
    
def phase_locking_analysis(signal_A, signal_B, freq_bands=None, fs=None):
    """PLV/PLI analysis for frequency bands"""
    
    # Añadimos 'broadband' como en la versión antigua
    if freq_bands is None:

        # en phase_locking_analysis default
        freq_bands = {'theta': (4,7), 'alpha': (8,12), 'beta': (13,30), 'gamma': (30,50)}
        
        # Solo añadir broadband si fs lo permite
        if fs >= 250:  # Nyquist > 100Hz
            freq_bands['broadband'] = (1, 100)

    A, B, L = validate_signals(signal_A, signal_B)
    if A is None:
        return {k: {'plv': 0.0, 'pli': 0.0, 'phase_stability': 0.0, 'env_corr': 0.0} for k in freq_bands.keys()}
    
    results = {}
    
    for band_name, (low_f, high_f) in freq_bands.items():
        
        # Asegurar que la banda es válida respecto a Nyquist
        nyq = 0.5 * fs
        low_f = float(low_f)
        high_f = float(min(high_f, nyq * 0.98))
        if high_f <= low_f or high_f <= 0:
            # Banda inválida con esta fs -> valores nulos
            results[band_name] = {'plv': 0.0, 'pli': 0.0, 'env_corr': 0.0}
            continue
        
        # Band-pass filter
        sos = scipy_signal.butter(4, [low_f, high_f], btype='band', fs=fs, output='sos')
        fA = scipy_signal.sosfiltfilt(sos, A)
        fB = scipy_signal.sosfiltfilt(sos, B)
        
        # Hilbert transform and phase
        hA = scipy_signal.hilbert(fA)
        hB = scipy_signal.hilbert(fB)
        phase_A = np.angle(hA)
        phase_B = np.angle(hB)
        phase_diff = phase_A - phase_B
        
        # PLV
        plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
        
        # PLI
        cross_spectrum = hA * np.conj(hB)
        pli = float(np.abs(np.mean(np.sign(np.imag(cross_spectrum))))) if cross_spectrum.size else 0.0
        pli = min(pli, plv)  # PLI ≤ PLV constraint
        
        # métrica adicional de estabilidad de fase (compat)
        phase_stability = float(max(0.0, 1 - np.var(np.unwrap(phase_diff)) / (2*np.pi)**2))
        
        # NEW: Envelope correlation (amplitude locking)
        env_A = np.abs(hA)
        env_B = np.abs(hB)
        
        # Pearson correlation of envelopes
        if np.std(env_A) > 0 and np.std(env_B) > 0:
            env_corr = float(np.corrcoef(env_A, env_B)[0, 1])
        else:
            env_corr = 0.0
        
        results[band_name] = {
            'plv': plv, 
            'pli': pli, 
            'phase_stability': phase_stability,  # ← FALTABA (compatibilidad)
            'env_corr': np.abs(env_corr)
        }
    
    return results

def spectral_coherence_analysis(signal_A, signal_B, fs=None, nperseg=None):
    """Coherence analysis between signals"""
    A, B, L = validate_signals(signal_A, signal_B)
    if A is None:
        return {'freqs': np.array([]), 'coherence': np.array([]), 'peak_freq': 0, 'peak_coherence': 0}
    
    if nperseg is None:
        # Δf ~ 1.0 Hz
        target_df = 1.0
        nperseg = int(max(256, min(L // 3, fs / target_df)))
    
    freqs, coherence = scipy_signal.coherence(
        A, B, fs=fs, window='hann', nperseg=nperseg,
        noverlap=nperseg//2, detrend='constant'
    )
    
    # Filter biological frequencies
    valid_mask = freqs <= 100
    freqs_filt = freqs[valid_mask]
    coh_filt = coherence[valid_mask]
    
    # Suavizado y cap como en la versión antigua
    if len(coh_filt) > 5:
        coh_filt = gaussian_filter1d(coh_filt, sigma=1.0)
    
    if len(coh_filt) > 0:
        peak_idx = np.argmax(coh_filt)
        peak_freq, peak_coherence = freqs_filt[peak_idx], coh_filt[peak_idx]
    else:
        peak_freq, peak_coherence = 0, 0
        
    # Coherencias por banda (para el dashboard antiguo)
    theta_mask = (freqs_filt < 8)
    alpha_mask = (freqs_filt >= 8) & (freqs_filt <= 12)
    beta_mask = (freqs_filt > 12) & (freqs_filt < 30)
    gamma_mask = (freqs_filt >= 30) & (freqs_filt <= 70)
    broad_mask = (freqs_filt > 70)
    
    theta_coherence = float(np.mean(coh_filt[theta_mask])) if np.any(theta_mask) else 0.0
    alpha_coherence = float(np.mean(coh_filt[beta_mask])) if np.any(beta_mask) else 0.0
    beta_coherence = float(np.mean(coh_filt[alpha_mask])) if np.any(alpha_mask) else 0.0
    gamma_coherence = float(np.mean(coh_filt[gamma_mask])) if np.any(gamma_mask) else 0.0
    broad_coherence = float(np.mean(coh_filt[broad_mask])) if np.any(broad_mask) else 0.0
    
    return {
        'freqs': freqs_filt,
        'coherence': coh_filt,
        'peak_freq': float(peak_freq),
        'peak_coherence': float(peak_coherence),
        'theta_coherence': theta_coherence,
        'alpha_coherence': alpha_coherence,
        'beta_coherence': beta_coherence,
        'gamma_coherence': gamma_coherence,
        'broad_coherence': broad_coherence,
    }

def power_spectrum_analysis(signal, fs=None, freq_bands=None):
    """Power spectrum with band-specific power"""
    if freq_bands is None:
        freq_bands = {'alpha': (8, 12), 'beta': (13, 30), 'gamma': (30, 50)}
    
    if len(signal) < 64:
        # devolver estructura completa aunque vacía para compat
        base = {'freqs': np.array([]), 'psd': np.array([]), 'total_power': 0.0,
                'peak_freq': 0.0, 'peak_power': 0.0}
        
        for band in freq_bands.keys():
            base[f'{band}_power'] = 0.0
            
        return base
    
    # Δf objetivo ~ 0.5 Hz
    nperseg = min(len(signal), max(64, int(fs / 0.5)))
    freqs, psd = scipy_signal.welch(
        signal, fs=fs, window='hann', nperseg=nperseg,
        noverlap=nperseg//2, detrend='constant'
    )
    
    results = {'freqs': freqs, 'psd': psd}
    nyq = 0.5 * fs
    for band_name, (low_f, high_f) in freq_bands.items():
        high_eff = min(high_f, nyq * 0.98)
        if high_eff <= low_f:
            results[f'{band_name}_power'] = 0.0
            continue
        mask = (freqs >= low_f) & (freqs <= high_eff)
        power = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        results[f'{band_name}_power'] = power
        
    # totales/peaks como la antigua
    results['total_power'] = float(np.trapz(psd, freqs)) if len(freqs) else 0.0
    if len(psd):
        pidx = int(np.argmax(psd))
        results['peak_freq'] = float(freqs[pidx])
        results['peak_power'] = float(psd[pidx])
    else:
        results['peak_freq'] = 0.0
        results['peak_power'] = 0.0
    
    return results

def intrinsic_timescale_analysis(signal, max_lag_ms=500, dt=0.5):
    """Timescale: exponential fit + integrated AC"""
    autocorr = cross_correlation_analysis(signal, signal, max_lag_ms, dt)
    
    lags = autocorr['lags']
    corr = autocorr['correlation']
    
    if len(lags) == 0:
        return {'tau_exp': 0, 'tau_int': 0, 'quality': 'no_data'}
    
    # Normalizar
    corr_norm = corr / np.max(np.abs(corr)) if np.max(np.abs(corr)) > 0 else corr
    pos_mask = lags >= 0
    lags_pos, corr_pos = lags[pos_mask], corr_norm[pos_mask]
    
    # === 1. Exponential fit (MEJORADO) ===
    # Estrategia: fit solo hasta primer mínimo local (antes de oscilaciones)
    from scipy.signal import find_peaks
    
    # Encontrar primer mínimo (valley)
    valleys, _ = find_peaks(-corr_pos)
    
    if len(valleys) > 0:
        # Fit solo hasta primer valley
        fit_end = min(valleys[0], int(50/dt))  # Max 50ms
    else:
        # Sin valleys claros: usar ventana fija
        fit_end = min(len(corr_pos), int(50/dt))
    
    # Fit exponencial en esta ventana
    lags_fit = lags_pos[:fit_end]
    corr_fit = corr_pos[:fit_end]
    
    if len(lags_fit) < 3:
        tau_exp = 0
        quality = 'insufficient_data'
    else:
        # Linearizar: log(AC) = -t/tau + log(A)
        corr_fit_safe = np.clip(corr_fit, 1e-10, None)
        log_corr = np.log(corr_fit_safe)
        
        # Fit lineal (robusto a outliers)
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(lags_fit, log_corr, deg=1)
        slope = p.coef[1]
        
        if slope < -1e-6:  # Decay negativo
            tau_exp = float(-1.0 / slope)
            tau_exp = np.clip(tau_exp, 0, max_lag_ms)  # Clip razonable
            quality = 'good' if tau_exp > 5 else 'moderate'
        else:
            tau_exp = 0
            quality = 'no_decay'
    
    # === 2. Integrated AC (sin cambios) ===
    zero_cross = np.where(corr_pos <= 0.1)[0]
    end_idx = zero_cross[0] if len(zero_cross) > 0 else len(corr_pos)
    tau_int = float(np.trapz(corr_pos[:end_idx], lags_pos[:end_idx]))
    
    return {
        'tau_exp': float(tau_exp),
        'tau_int': float(tau_int),
        'quality': quality
    }

# def intrinsic_timescale_analysis(signal, max_lag_ms=500, dt=0.5):
#     """Timescale: exponential fit + integrated AC"""
#     autocorr = cross_correlation_analysis(signal, signal, max_lag_ms, dt)
    
#     lags = autocorr['lags']
#     corr = autocorr['correlation']
    
#     if len(lags) == 0:
#         return {'tau_exp': 0, 'tau_int': 0, 'quality': 'no_data'}
    
#     # Normalizar y positivos
#     corr_norm = corr / np.max(np.abs(corr)) if np.max(np.abs(corr)) > 0 else corr
#     pos_mask = lags >= 0
#     lags_pos, corr_pos = lags[pos_mask], corr_norm[pos_mask]
    
#     # === 1. Exponential fit (método actual) ===
#     threshold = np.exp(-1)
#     above = corr_pos > threshold
    
#     if not np.any(above) or len(lags_pos) < 3:
#         tau_exp = 0
#         quality = 'insufficient_decay'
#     else:
#         last_above = np.where(above)[0][-1]
#         if last_above < len(lags_pos) - 1:
#             x1, x2 = lags_pos[last_above], lags_pos[last_above + 1]
#             y1, y2 = corr_pos[last_above], corr_pos[last_above + 1]
#             if y2 != y1:
#                 cutoff = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
#             else:
#                 cutoff = x1
#             area_base = np.trapz(corr_pos[:last_above+1], lags_pos[:last_above+1])
#             area_partial = 0.5 * (cutoff - x1) * (y1 + threshold)
#             tau_exp = float(max(0.0, area_base + area_partial))
#             quality = 'good' if cutoff > 20 else 'moderate'
#         else:
#             tau_exp = 0
#             quality = 'no_decay'
    
#     # === 2. Integrated AC (más robusto) ===
#     # Integrar hasta primer cruce por 0 o max_lag
#     zero_cross = np.where(corr_pos <= 1/np.e)[0]
#     if len(zero_cross) > 0:
#         end_idx = zero_cross[0]
#     else:
#         end_idx = len(corr_pos)
    
#     tau_int = float(np.trapz(corr_pos[:end_idx], lags_pos[:end_idx]))
    
#     return {
#         'tau_exp': float(tau_exp) if not np.isnan(tau_exp) else np.nan,
#         'tau_int': float(tau_int) if not np.isnan(tau_int) else np.nan,
#         'quality': quality
#     }
# =============================================================================
# MAIN CONNECTIVITY ANALYZER (SIMPLIFIED)
# =============================================================================

class ConnectivityAnalyzer:
    """Simplified connectivity analyzer - orchestrates core functions"""
    
    def __init__(self, analysis_dt=0.5, fs=None, signal_mode='firing_rate', T_total = 1000):
        """
        Args:
            signal_mode: 'firing_rate' (spikes) o 'lfp' (voltage)
        """
        self.analysis_dt = float(analysis_dt)
        self.fs = float(fs) if fs is not None else 1000.0 / float(analysis_dt)
        self.signal_mode = signal_mode
        self.T_total = T_total
    
    def analyze_pair(self, spike_mon_A, spike_mon_B, N_A, N_B, warmup_ms=500,
                    voltage_mon_A=None, voltage_mon_B=None, N_exc_sampled_A=None, N_exc_sampled_B=None):
        """Análisis con switch firing_rate / LFP. Si spike_mon_B es None, analiza solo A."""
        
        # Detectar modo single population
        single_pop = spike_mon_B is None
        
        # ====== EXTRACCIÓN DE SEÑAL A ======
        if self.signal_mode == 'lfp':
            time_A, signal_A_raw, fs_A = voltage_to_lfp(voltage_mon_A, N_exc_sampled_A)
            self.fs = fs_A
            self.analysis_dt = 1000.0 / self.fs
            time_fr_A, fr_A_raw = spikes_to_population_rate(spike_mon_A, N_A, analysis_dt=0.5, T_total=self.T_total)
        else:
            time_A, signal_A_raw = spikes_to_population_rate(spike_mon_A, N_A, analysis_dt=self.analysis_dt, T_total=self.T_total)
            time_fr_A, fr_A_raw = time_A, signal_A_raw
        
        # ====== EXTRACCIÓN DE SEÑAL B (solo si existe) ======
        if not single_pop:
            if self.signal_mode == 'lfp':
                time_B, signal_B_raw, fs_B = voltage_to_lfp(voltage_mon_B, N_exc_sampled_B)
                time_fr_B, fr_B_raw = spikes_to_population_rate(spike_mon_B, N_B, analysis_dt=0.5, T_total=self.T_total)
            else:
                time_B, signal_B_raw = spikes_to_population_rate(spike_mon_B, N_B, analysis_dt=self.analysis_dt, T_total=self.T_total)
                time_fr_B, fr_B_raw = time_B, signal_B_raw
        
            ## Validate pair
            signal_A, signal_B, L = validate_signals(signal_A_raw, signal_B_raw)
            if signal_A is None:
                return self._empty_results()
            
        else:
            signal_A = signal_A_raw
            signal_B = None
            
            
        # Warmup cutoff
        if warmup_ms > 0:
            cut_idx = int(warmup_ms / self.analysis_dt)
            cut_idx = max(0, min(cut_idx, len(signal_A_raw)-1))
            time_cut = time_A[cut_idx:] - time_A[cut_idx]
            signal_A_raw_cut = signal_A_raw[cut_idx:]
            signal_B_raw_cut = signal_B_raw[cut_idx:] if not single_pop else None
            
            # AÑADIR: cortar también FR cuando signal_mode='lfp'
            if self.signal_mode == 'lfp':
                cut_idx_fr = int(warmup_ms / 0.5)  # FR tiene dt=0.5ms
                time_fr_cut = time_fr_A[cut_idx_fr:] - time_fr_A[cut_idx_fr]
                fr_A_raw_cut = fr_A_raw[cut_idx_fr:]
                fr_B_raw_cut = fr_B_raw[cut_idx_fr:] if not single_pop else None
            else:
                time_fr_cut = time_cut
                fr_A_raw_cut = signal_A_raw_cut
                fr_B_raw_cut = signal_B_raw_cut
        else:
            time_cut = time_A
            signal_A_raw_cut = signal_A_raw
            signal_B_raw_cut = signal_B_raw if not single_pop else None
            time_fr_cut = time_fr_A if self.signal_mode == 'lfp' else time_cut
            fr_A_raw_cut = fr_A_raw if self.signal_mode == 'lfp' else signal_A_raw_cut
            fr_B_raw_cut = fr_B_raw if (self.signal_mode == 'lfp' and not single_pop) else signal_B_raw_cut
            
        # Preprocess
        signal_A_proc = preprocess_signal(signal_A_raw_cut, self.fs)
        signal_B_proc = preprocess_signal(signal_B_raw_cut, self.fs) if not single_pop else None
        
        # ====== MÉTRICAS INDIVIDUALES (siempre) ======
        ac_A = cross_correlation_analysis(signal_A_proc, signal_A_proc, dt=self.analysis_dt)
        ts_A = intrinsic_timescale_analysis(signal_A_proc, dt=self.analysis_dt)
        psA = power_spectrum_analysis(signal_A_proc, fs=self.fs)
        
        # ====== MÉTRICAS PAIRWISE (solo si hay B) ======
        if not single_pop:
            cc = cross_correlation_analysis(signal_A_proc, signal_B_proc, dt=self.analysis_dt)
            ac_B = cross_correlation_analysis(signal_B_proc, signal_B_proc, dt=self.analysis_dt)
            ts_B = intrinsic_timescale_analysis(signal_B_proc, dt=self.analysis_dt)
            plv = phase_locking_analysis(signal_A_proc, signal_B_proc, fs=self.fs)
            coh = spectral_coherence_analysis(signal_A_proc, signal_B_proc, fs=self.fs)
            psB = power_spectrum_analysis(signal_B_proc, fs=self.fs)
        else:
            cc = None
            ac_B = None
            ts_B = None
            plv = None
            coh = None
            psB = None
        
        # Resultados
        results = {
            'signal_mode': self.signal_mode,
            'single_population': single_pop,
            'time_series': {
                'time': time_cut,
                'signal_A': signal_A_raw_cut,
                'signal_B': signal_B_raw_cut,
                'time_fr': time_fr_cut,
                'fr_A': fr_A_raw_cut,
                'fr_B': fr_B_raw_cut if not single_pop else None
            },
            'cross_correlation': cc,
            'phase_locking': plv,
            'coherence': coh,
            'power_A': psA,
            'power_B': psB,
            'timescale_A': ts_A,
            'timescale_B': ts_B,
            't0_ms': float(warmup_ms),
            'autocorr_A': ac_A,
            'autocorr_B': ac_B,
            'int_A': {**ts_A, 'fit_quality': ts_A.get('quality', 'unknown')},
            'int_B': {**ts_B, 'fit_quality': ts_B.get('quality', 'unknown')} if ts_B else None,
            'plv_pli': plv,
            'psd_A': psA,
            'psd_B': psB,
            'analysis_dt': self.analysis_dt,
            'spike_times_A': np.array(spike_mon_A.t/ms),
            'spike_neurons_A': np.array(spike_mon_A.i),
            'spike_times_B': np.array(spike_mon_B.t/ms) if not single_pop else None,
            'spike_neurons_B': np.array(spike_mon_B.i) if not single_pop else None,
        }
        
        return results
    
    def _empty_results(self):
        """Return empty results structure"""
        return {
            'time_series': {'time': np.array([]), 'rate_A': np.array([]), 'rate_B': np.array([])},
            'cross_correlation': {'peak_lag': 0, 'peak_value': 0},
            'phase_locking': {'alpha': {'plv': 0, 'pli': 0}},
            'coherence': {'peak_freq': 0, 'peak_coherence': 0},
            'power_A': {'alpha_power': 0}, 'power_B': {'alpha_power': 0},
            'timescale_A': {'tau': 0}, 'timescale_B': {'tau': 0}
        }

# =============================================================================
# WRAPPER FUNCTIONS (BACKWARDS COMPATIBILITY)
# =============================================================================

def analyze_simulation_results(spike_mon_A, spike_mon_B, N=1000,
                            condition_name="test", warmup=500.0,
                            state_monitors=None, delays=None, 
                            signal_mode='firing_rate', T_total=None, **kwargs):
    """Main analysis wrapper - soporta single population"""
    
    if T_total is None:
        T_total = float(max(spike_mon_A.t/ms)) if len(spike_mon_A.t) > 0 else 1000
    
    analyzer = ConnectivityAnalyzer(analysis_dt=0.5, fs=None, signal_mode=signal_mode,  T_total=T_total)
    
    # Extraer voltage monitors si existen
    voltage_mon_A = state_monitors.get('A', {}).get('voltage') if state_monitors else None
    voltage_mon_B = state_monitors.get('B', {}).get('voltage') if state_monitors else None
    N_exc_A = state_monitors.get('A', {}).get('v_n_exc_sampled') if state_monitors else None
    N_exc_B = state_monitors.get('B', {}).get('v_n_exc_sampled') if state_monitors else None
    
    results = analyzer.analyze_pair(
        spike_mon_A, spike_mon_B, N, N, warmup_ms=warmup,
        voltage_mon_A=voltage_mon_A, voltage_mon_B=voltage_mon_B,
        N_exc_sampled_A=N_exc_A, N_exc_sampled_B=N_exc_B
    )
    
    single_pop = results.get('single_population', False)
    
    # Legacy keys
    results['condition'] = condition_name
    
    # Métricas pairwise (solo si existe población B)
    if not single_pop:
        results['cross_corr_peak'] = results['cross_correlation']['peak_value']
        results['cross_corr_lag'] = results['cross_correlation']['peak_lag']
        results['plv_alpha'] = results['phase_locking']['alpha']['plv']
        results['pli_alpha'] = results['phase_locking']['alpha']['pli']
        results['coherence_peak'] = results['coherence']['peak_coherence']
        results['tau_B'] = results['timescale_B'].get('tau_exp', results['timescale_B'].get('tau', 0))
    else:
        results['cross_corr_peak'] = None
        results['cross_corr_lag'] = None
        results['plv_alpha'] = None
        results['pli_alpha'] = None
        results['coherence_peak'] = None
        results['tau_B'] = None
    
    # Métricas siempre disponibles
    results['tau_A'] = results['timescale_A'].get('tau_exp', results['timescale_A'].get('tau', 0))
    
    # State monitors
    if state_monitors:
        results['state_monitor_A'] = state_monitors.get('A', {}).get('currents')
        results['state_monitor_B'] = state_monitors.get('B', {}).get('currents') if not single_pop else None
        results['voltage_monitor_A'] = state_monitors.get('A', {}).get('voltage')
        results['voltage_monitor_B'] = state_monitors.get('B', {}).get('voltage') if not single_pop else None
        
    if delays:
        results['delays_AB'] = delays.get('AB')
        results['delays_BA'] = delays.get('BA')
    
    # Logger
    logger.info(f"\n=== {condition_name} ({signal_mode}) ===")
    if not single_pop:
        logger.info(f"Cross-corr: {results['cross_corr_peak']:.3f} @ {results['cross_corr_lag']:.1f}ms")
        logger.info(f"Alpha PLV: {results['plv_alpha']:.3f}, PLI: {results['pli_alpha']:.3f}")
        logger.info(f"Coherence: {results['coherence_peak']:.3f} @ {results['coherence']['peak_freq']:.1f}Hz")
        logger.info(f"Timescales: A={results['tau_A']:.1f}ms, B={results['tau_B']:.1f}ms")
    else:
        logger.info(f"Single population analysis")
        logger.info(f"Timescale A: {results['tau_A']:.1f}ms")
    
    return results
    
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
        
        logger.info(f"    DEBUG: CV_individual={cv_individual:.3f}, CV_population={cv_population:.3f}")
        
        # Return the more stable measure
        return cv_individual, fano_factor

    def calculate_connection_stats(syn_intra, N_total):
        n_connections = len(syn_intra.i)
        max_possible = N_total * (N_total - 1)
        connection_prob = n_connections / max_possible
        sparsity = 1 - connection_prob
        return n_connections, connection_prob, sparsity
    
    stats = {}
    N_total = N_exc + N_inh
    
    available_pops = [name for name in ['A', 'B'] if name in results]
    
    for pop_name in available_pops:
        spike_mon = results[pop_name]['spike_monitor']
        group = network.populations[pop_name]['group']
        syn_intra = network.populations[pop_name]['syn_intra']
        
        freq_exc_mean, freq_inh_mean, freq_exc_ind, freq_inh_ind = calculate_firing_rates(
            spike_mon, N_exc, N_inh, T_total)
        cv_exc, fano_factor = calculate_synchrony_metrics(spike_mon, N_exc)
        n_conn, conn_prob, sparsity = calculate_connection_stats(syn_intra, N_total)

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
            'exc_weight_mean': np.mean(exc_weights),
            'exc_weight_std': np.std(exc_weights),
            'inh_weight_mean': np.mean(inh_weights),
            'inh_weight_std': np.std(inh_weights)
        }
    
   # Imprimir tabla
    logger.info(f"\n{'='*80}")
    title = f"ESTADÍSTICAS DE RED - Población {available_pops[0]}" if len(available_pops) == 1 else "COMPARACIÓN ESTADÍSTICAS DE RED"
    logger.info(f"{title} (Post-warmup {warmup}ms)")
    logger.info(f"{'='*80}")

    logger.info(f"{'Métrica':<25} {'Población A':<25} {'Población B':<25}")
    logger.info(f"{'-'*75}")

    # Helper: obtener valor con formato o vacío si no existe
    def fmt(pop, key, spec):
        if pop not in stats:
            return ""
        val = stats[pop][key]
        return f"{val:{spec}}" if not isinstance(val, str) else val

    logger.info(f"{'ARQUITECTURA':<25}")
    logger.info(f"{'  Conexiones':<25} {fmt('A','connections','d'):<25} {fmt('B','connections','d')}")
    logger.info(f"{'  Prob. conexión':<25} {fmt('A','conn_prob','.4f'):<25} {fmt('B','conn_prob','.4f')}")
    logger.info(f"{'  Sparsity':<25} {fmt('A','sparsity','.3f'):<25} {fmt('B','sparsity','.3f')}")

    logger.info(f"\n{'ACTIVIDAD':<25}")
    logger.info(f"{'  Freq exc (Hz)':<25} {fmt('A','freq_exc','.2f'):<25} {fmt('B','freq_exc','.2f')}")
    logger.info(f"{'  Freq inh (Hz)':<25} {fmt('A','freq_inh','.2f'):<25} {fmt('B','freq_inh','.2f')}")
    logger.info(f"{'  Ratio E/I':<25} {fmt('A','ratio_ei','.2f'):<25} {fmt('B','ratio_ei','.2f')}")
    logger.info(f"{'  Activas exc (%)':<25} {fmt('A','active_exc_pct','.1f'):<25} {fmt('B','active_exc_pct','.1f')}")
    logger.info(f"{'  Activas inh (%)':<25} {fmt('A','active_inh_pct','.1f'):<25} {fmt('B','active_inh_pct','.1f')}")

    logger.info(f"\n{'SINCRONÍA':<25}")
    logger.info(f"{'  CV':<25} {fmt('A','cv','.3f'):<25} {fmt('B','cv','.3f')}")
    logger.info(f"{'  Fano Factor':<25} {fmt('A','fano','.3f'):<25} {fmt('B','fano','.3f')}")
    logger.info(f"{'  Nivel':<25} {stats.get('A',{}).get('sync_level',''):<25} {stats.get('B',{}).get('sync_level','')}")

    logger.info(f"\n{'PESOS SINÁPTICOS':<25}")
    exc_a = f"{stats['A']['exc_weight_mean']:.3f}±{stats['A']['exc_weight_std']:.3f}" if 'A' in stats else ""
    exc_b = f"{stats['B']['exc_weight_mean']:.3f}±{stats['B']['exc_weight_std']:.3f}" if 'B' in stats else ""
    inh_a = f"{stats['A']['inh_weight_mean']:.3f}±{stats['A']['inh_weight_std']:.3f}" if 'A' in stats else ""
    inh_b = f"{stats['B']['inh_weight_mean']:.3f}±{stats['B']['inh_weight_std']:.3f}" if 'B' in stats else ""
    logger.info(f"{'  Exc mean±std':<25} {exc_a:<25} {exc_b}")
    logger.info(f"{'  Inh mean±std':<25} {inh_a:<25} {inh_b}")

    logger.info(f"\n{'='*80}")
    return stats


def population_lfp(v_matrix, N_exc, weight_inh=0.3):
    """Agregación espacial: LFP-like signal"""
    v_exc = v_matrix[:N_exc].mean(axis=0)
    v_inh = v_matrix[N_exc:].mean(axis=0)
    return v_exc #+ weight_inh * v_inh


def voltage_to_lfp(voltage_monitor, N_exc_sampled, weight_inh=0.3):
    """Extract LFP con fs automático"""
    v_matrix = voltage_monitor.v[:]
    times = np.array(voltage_monitor.t / ms)
    
    # Detectar fs real del monitor
    if len(times) > 1:
        dt_actual = np.mean(np.diff(times))  # en ms
        fs_hz = 1000.0 / dt_actual
    else:
        fs_hz = 2000.0  # default
    
    v_exc = v_matrix[:N_exc_sampled].mean(axis=0)
    v_inh = v_matrix[N_exc_sampled:].mean(axis=0)
    lfp = v_exc #+ weight_inh * v_inh
    
    return times, lfp, fs_hz


import numpy as np
from scipy.signal import hilbert, correlate, spectrogram
from scipy.stats import entropy

# ============= LFP & PHASE =============

def extract_lfp(V_monitor, start_ms, dt):
    """Extract LFP as mean membrane potential"""
    start_idx = int(start_ms / dt)
    V_matrix = V_monitor.v[:, start_idx:]  # mV ya debe estar
    lfp = np.mean(V_matrix, axis=0)
    return lfp

def compute_instantaneous_phase(signal, fs, freq_band=(40, 70)):
    """Hilbert transform for phase"""
    from scipy.signal import butter, filtfilt
    
    # Bandpass filter
    nyq = 0.5 * fs
    low, high = freq_band[0]/nyq, freq_band[1]/nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    
    # Phase via Hilbert
    analytic = hilbert(filtered)
    phase = np.angle(analytic)  # [-π, π]
    return phase

# ============= ROUTING STATES =============

def routing_state_analysis(lfp_A, lfp_B, fs, freq_band=(40, 70)):
    """
    Compute routing states from phase relationships.
    Returns fractions of time in Top/Bottom states and switch rate.
    """
    # Instantaneous phases
    phase_A = compute_instantaneous_phase(lfp_A, fs, freq_band)
    phase_B = compute_instantaneous_phase(lfp_B, fs, freq_band)
    
    # Relative phase [0, 1]
    delta_phi = ((phase_B - phase_A) % (2*np.pi)) / (2*np.pi)
    
    # States: Top (Y leads, 0 < ΔΦ < 0.5), Bottom (X leads, 0.5 < ΔΦ < 1)
    is_top_state = delta_phi < 0.5
    top_fraction = np.mean(is_top_state)
    
    # Switch rate (transitions per second)
    state_binary = is_top_state.astype(int)
    transitions = np.abs(np.diff(state_binary))
    dt = 1.0 / fs
    switch_rate_hz = np.sum(transitions) / (len(transitions) * dt)
    
    # Bimodality (peaks at ΔΦ↑* and ΔΦ↓*)
    hist, bins = np.histogram(delta_phi, bins=50, range=(0, 1))
    bimodality = np.std(hist) / np.mean(hist)  # Simple index
    
    return {
        'delta_phi': delta_phi,
        'top_state_fraction': top_fraction,
        'bottom_state_fraction': 1 - top_fraction,
        'switch_rate_hz': switch_rate_hz,
        'phase_bimodality': bimodality
    }

# ============= TRANSFER ENTROPY =============

def transfer_entropy_simple(source, target, lag=1, bins=50):
    """
    Simplified TE: H[Y_t | Y_{t-lag}] - H[Y_t | X_{t-lag}, Y_{t-lag}]
    Uses histogram-based entropy estimation.
    """
    # Prepare time series
    y_t = target[lag:]
    y_past = target[:-lag]
    x_past = source[:-lag]
    
    # Discretize
    def digitize(x):
        return np.digitize(x, bins=np.linspace(x.min(), x.max(), bins))
    
    y_t_d = digitize(y_t)
    y_past_d = digitize(y_past)
    x_past_d = digitize(x_past)
    
    # Joint distributions
    def joint_entropy(x, y):
        xy = np.column_stack([x, y])
        return entropy(np.histogram2d(x, y, bins=bins)[0].flatten())
    
    # H[Y_t, Y_{t-lag}]
    h_y_joint = joint_entropy(y_t_d, y_past_d)
    # H[Y_{t-lag}]
    h_y_past = entropy(np.histogram(y_past_d, bins=bins)[0])
    # H[Y_t, Y_{t-lag}, X_{t-lag}]
    h_full = entropy(np.histogramdd(np.column_stack([y_t_d, y_past_d, x_past_d]), bins=bins)[0].flatten())
    # H[Y_{t-lag}, X_{t-lag}]
    h_cond = joint_entropy(y_past_d, x_past_d)
    
    # TE = H[Y_t | Y_{t-lag}] - H[Y_t | X_{t-lag}, Y_{t-lag}]
    te = (h_y_joint - h_y_past) - (h_full - h_cond)
    return max(0, te)  # TE >= 0

def compute_bidirectional_te(lfp_A, lfp_B, max_lag_ms=500, dt=0.1):
    """TE in both directions over lag range"""
    max_lag_samples = int(max_lag_ms / dt)
    lags = np.arange(1, max_lag_samples)
    
    te_AB = [transfer_entropy_simple(lfp_A, lfp_B, lag) for lag in lags]
    te_BA = [transfer_entropy_simple(lfp_B, lfp_A, lag) for lag in lags]
    
    # Optimal lags
    tau_opt_AB = lags[np.argmax(te_AB)] * dt
    tau_opt_BA = lags[np.argmax(te_BA)] * dt
    
    # Anisotropy
    max_te_AB, max_te_BA = np.max(te_AB), np.max(te_BA)
    delta_te = (max_te_BA - max_te_AB) / max(abs(max_te_AB), abs(max_te_BA))
    
    return {
        'te_AB': te_AB,
        'te_BA': te_BA,
        'tau_opt_AB_ms': tau_opt_AB,
        'tau_opt_BA_ms': tau_opt_BA,
        'delta_te': delta_te,
        'lags_ms': lags * dt
    }

# ============= BURST DETECTION =============

def detect_gamma_bursts(lfp, fs, freq_range=(40, 70), percentile=95):
    """Detect transient gamma bursts via spectrogram thresholding"""
    signal_duration = len(lfp) / fs
    
    # Adaptar ventanas a longitud de señal
    if signal_duration < 1.0:
        # Señal corta: ventanas más pequeñas
        nperseg = max(int(0.1 * fs), 128)  # 100ms o mínimo
        noverlap = int(0.75 * nperseg)  # 75% overlap
    else:
        # Señal larga: ventanas del paper
        nperseg = int(0.3 * fs)  # 300ms
        noverlap = int(0.25 * fs)  # 250ms overlap
    
    f, t, Sxx = spectrogram(lfp, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Filter gamma band
    mask = (f >= freq_range[0]) & (f <= freq_range[1])
    Sxx_gamma = Sxx[mask, :]
    f_gamma = f[mask]
    
    # Skip si muy pocos time bins
    if Sxx_gamma.shape[1] < 3:
        return {
            'freqs': f_gamma,
            'times': t,
            'bursts': np.zeros_like(Sxx_gamma, dtype=bool),
            'power': Sxx_gamma
        }
    
    # Threshold per frequency
    burst_binary = np.zeros_like(Sxx_gamma, dtype=bool)
    for i in range(len(f_gamma)):
        thresh = np.percentile(Sxx_gamma[i, :], percentile)
        burst_binary[i, :] = Sxx_gamma[i, :] > thresh
    
    return {
        'freqs': f_gamma,
        'times': t,
        'bursts': burst_binary,
        'power': Sxx_gamma
    }

def compute_burst_overlap(bursts_A, bursts_B):
    """Jaccard-like overlap con protección"""
    n_A = np.sum(bursts_A, axis=1)
    n_B = np.sum(bursts_B, axis=1)
    n_both = np.sum(bursts_A & bursts_B, axis=1)
    
    # Evitar divisiones por 0
    denominator = np.sqrt(n_A * n_B)
    overlap = np.divide(n_both, denominator, 
                    out=np.zeros_like(n_both, dtype=float),
                    where=denominator > 1)  # Solo si >1 burst
    
    return overlap

# XC normalizado por ventana
def compute_xc_timeresolved(lfp_A, lfp_B, fs, window_ms=50, step_ms=10):
    window = int(window_ms * fs / 1000)
    step = int(step_ms * fs / 1000)
    max_lag_samples = int(20 * fs / 1000)
    
    n_windows = (len(lfp_A) - window) // step
    xc_matrix = np.zeros((2*max_lag_samples+1, n_windows))
    
    for i in range(n_windows):
        start = i * step
        chunk_A = lfp_A[start:start+window]
        chunk_B = lfp_B[start:start+window]
        
        # Normalizar chunks
        chunk_A = (chunk_A - chunk_A.mean()) / (chunk_A.std() + 1e-10)
        chunk_B = (chunk_B - chunk_B.mean()) / (chunk_B.std() + 1e-10)
        
        xc_full = correlate(chunk_A, chunk_B, mode='full')
        
        # Normalizar XC por longitud ventana
        xc_full = xc_full / window
        
        center = len(xc_full) // 2
        xc_matrix[:, i] = xc_full[center-max_lag_samples:center+max_lag_samples+1]
    
    return xc_matrix, np.arange(n_windows)*step_ms/1000

# ============= WRAPPER =============

def palmigiano_analysis(results, start_ms=500):
    """
    Complete analysis inspired by Palmigiano et al.
    Add to existing analyze_simulation_results.
    """
    dt = results['dt']
    fs = 1000.0 / dt
    
    # Extract LFPs
    lfp_A = extract_lfp(results['A']['voltage_monitor'], start_ms, dt)
    lfp_B = extract_lfp(results['B']['voltage_monitor'], start_ms, dt)
    
    metrics = {}
    
    # Routing states
    routing = routing_state_analysis(lfp_A, lfp_B, fs)
    metrics['routing'] = routing
    
    # Transfer Entropy
    te = compute_bidirectional_te(lfp_A, lfp_B, max_lag_ms=500, dt=dt)
    metrics['transfer_entropy'] = te
    
    # Burst coordination
    bursts_A = detect_gamma_bursts(lfp_A, fs)
    bursts_B = detect_gamma_bursts(lfp_B, fs)
    overlap = compute_burst_overlap(bursts_A['bursts'], bursts_B['bursts'])
    
    metrics['bursts'] = {
        'overlap': overlap,
        'peak_overlap': np.max(overlap),
        'peak_frequency': bursts_A['freqs'][np.argmax(overlap)],
        'A': bursts_A,
        'B': bursts_B
    }
    
    # XC time-resolved
    xc_matrix, xc_times = compute_xc_timeresolved(lfp_A, lfp_B, fs)
    xc_star = np.max(xc_matrix, axis=0)  # Max por ventana
    
    # Conditional phase

    from scipy.interpolate import interp1d
    time_lfp = np.arange(len(routing['delta_phi'])) / fs
    interp_func = interp1d(xc_times, xc_star, kind='linear', 
                        fill_value='extrapolate')
    xc_star_interp = interp_func(time_lfp)

    # Usar interpolado para máscaras
    thresh_low = np.percentile(xc_star_interp, 50)
    thresh_high = np.percentile(xc_star_interp, 50)

    metrics['xc_resolved'] = {
        'matrix': xc_matrix,
        'times': xc_times,
        'xc_star': xc_star,
        'delta_phi_low': routing['delta_phi'][xc_star_interp < thresh_low],
        'delta_phi_high': routing['delta_phi'][xc_star_interp > thresh_high]
    }
    
    return metrics, lfp_A , lfp_B, fs


# ISI AND REFRACTORY ANALYSIS   

def analyze_ISI(spike_mon, Ne, Ni, t_start=100*ms, t_end=None):
    """
    Calcula y visualiza ISI (Inter-Spike Intervals)
    
    Parameters:
    -----------
    spike_mon : SpikeMonitor
    Ne, Ni : int - número de neuronas exc/inh
    t_start : tiempo para empezar análisis (evitar transitorio)
    t_end : tiempo final (None = hasta el final)
    """
    if t_end is None:
        t_end = spike_mon.t[-1]
    
    # Filtrar spikes en ventana temporal
    mask = (spike_mon.t >= t_start) & (spike_mon.t <= t_end)
    spike_times = spike_mon.t[mask]
    spike_indices = spike_mon.i[mask]
    
    # Calcular ISI por neurona
    isi_exc = []
    isi_inh = []
    
    for i in range(Ne + Ni):
        neuron_spikes = spike_times[spike_indices == i]
        if len(neuron_spikes) > 1:
            isis = np.diff(neuron_spikes / ms)  # en ms
            if i < Ne:
                isi_exc.extend(isis)
            else:
                isi_inh.extend(isis)
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Histogramas separados
    axes[0, 0].hist(isi_exc, bins=50, alpha=0.7, label='Exc', color='red', density=True)
    axes[0, 0].set_xlabel('ISI (ms)')
    axes[0, 0].set_ylabel('Densidad')
    axes[0, 0].set_title('ISI Excitatorias')
    axes[0, 0].axvline(4, color='k', ls='--', label='Refractario explícito')
    axes[0, 0].legend()
    
    axes[0, 1].hist(isi_inh, bins=50, alpha=0.7, label='Inh', color='blue', density=True)
    axes[0, 1].set_xlabel('ISI (ms)')
    axes[0, 1].set_title('ISI Inhibitorias')
    axes[0, 1].axvline(4, color='k', ls='--')
    axes[0, 1].legend()
    
    # 2. Comparación directa
    axes[1, 0].hist([isi_exc, isi_inh], bins=50, alpha=0.6, 
                    label=['Exc', 'Inh'], color=['red', 'blue'], density=True)
    axes[1, 0].set_xlabel('ISI (ms)')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].set_title('Comparación ISI')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 50)  # zoom a primeros 50 ms
    
    # 3. CDF (función acumulativa)
    isi_exc_sorted = np.sort(isi_exc)
    isi_inh_sorted = np.sort(isi_inh)
    axes[1, 1].plot(isi_exc_sorted, np.linspace(0, 1, len(isi_exc_sorted)), 
                    label='Exc', color='red')
    axes[1, 1].plot(isi_inh_sorted, np.linspace(0, 1, len(isi_inh_sorted)), 
                    label='Inh', color='blue')
    axes[1, 1].set_xlabel('ISI (ms)')
    axes[1, 1].set_ylabel('CDF')
    axes[1, 1].set_title('Distribución Acumulativa')
    axes[1, 1].axvline(4, color='k', ls='--')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 50)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Estadísticas
    print(f"ISI Excitatorias: media={np.mean(isi_exc):.2f} ms, std={np.std(isi_exc):.2f} ms")
    print(f"ISI Inhibitorias: media={np.mean(isi_inh):.2f} ms, std={np.std(isi_inh):.2f} ms")
    print(f"ISI < 4ms (Exc): {100*np.sum(np.array(isi_exc)<4)/len(isi_exc):.1f}%")
    print(f"ISI <4ms (Inh): {100*np.sum(np.array(isi_inh)<4)/len(isi_inh):.1f}%")
    
    return isi_exc, isi_inh



def estimate_intrinsic_refractoriness(spike_mon, Ne, Ni, t_start=100*ms):
    """
    Estima período refractario intrínseco desde ISI observados
    """
    mask = spike_mon.t >= t_start
    spike_times = spike_mon.t[mask]
    spike_indices = spike_mon.i[mask]
    
    isi_exc = []
    isi_inh = []
    
    for i in range(Ne + Ni):
        neuron_spikes = spike_times[spike_indices == i]
        if len(neuron_spikes) > 1:
            isis = np.diff(neuron_spikes / ms)
            if i < Ne:
                isi_exc.extend(isis)
            else:
                isi_inh.extend(isis)
    
    # Estadísticas clave
    results = {
        'exc': {
            'min': np.min(isi_exc) if isi_exc else np.nan,
            'p1': np.percentile(isi_exc, 1) if isi_exc else np.nan,
            'p5': np.percentile(isi_exc, 5) if isi_exc else np.nan,
            'median': np.median(isi_exc) if isi_exc else np.nan,
            'mean': np.mean(isi_exc) if isi_exc else np.nan,
        },
        'inh': {
            'min': np.min(isi_inh) if isi_inh else np.nan,
            'p1': np.percentile(isi_inh, 1) if isi_inh else np.nan,
            'p5': np.percentile(isi_inh, 5) if isi_inh else np.nan,
            'median': np.median(isi_inh) if isi_inh else np.nan,
            'mean': np.mean(isi_inh) if isi_inh else np.nan,
        }
    }
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for ax, (name, isis, color) in zip(axes, [
        ('Excitatorias', isi_exc, 'red'),
        ('Inhibitorias', isi_inh, 'blue')
    ]):
        ax.hist(isis, bins=100, alpha=0.7, color=color, density=True)
        ax.axvline(results[name.lower()[:3]]['min'], 
                   color='k', ls='--', label=f'Mín: {results[name.lower()[:3]]["min"]:.2f}ms')
        ax.axvline(results[name.lower()[:3]]['p5'], 
                   color='orange', ls='--', label=f'P5: {results[name.lower()[:3]]["p5"]:.2f}ms')
        ax.set_xlabel('ISI (ms)')
        ax.set_ylabel('Densidad')
        ax.set_title(f'ISI {name}')
        ax.set_xlim(0, 50)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Resumen
    print("="*60)
    print("PERÍODO REFRACTARIO INTRÍNSECO (sin refractory explícito)")
    print("="*60)
    for cell_type in ['exc', 'inh']:
        print(f"\n{cell_type.upper()}:")
        print(f"  ISI mínimo:    {results[cell_type]['min']:.2f} ms")
        print(f"  ISI P1:        {results[cell_type]['p1']:.2f} ms")
        print(f"  ISI P5:        {results[cell_type]['p5']:.2f} ms")
        print(f"  ISI mediana:   {results[cell_type]['median']:.2f} ms")
    
    return results, (isi_exc, isi_inh)

# Uso: