import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal as scipy_signal
from src.two_populations.helpers.logger import setup_logger
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.signal import spectrogram
from scipy.stats import gaussian_kde

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

# =============================================================================
# PLOTTING FUNCTIONS (SEPARATED FROM ANALYSIS)
# =============================================================================

# def _band_order(phase_locking_dict):
#     pref = ['theta','alpha','beta','gamma','broadband']
#     bands = list(phase_locking_dict.keys())
#     # mantiene el orden preferido y añade el resto al final
#     ordered = [b for b in pref if b in bands] + [b for b in bands if b not in pref]
#     return ordered

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
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(ncol=2)
    ax1.grid(True, alpha=0.3)


    # 2) PLV & PLI de TODAS las bandas (barras agrupadas por condición)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(plv_bands))
    # ancho pequeño para poder meter PLV y PLI por condición sin solapar
    width = 0.35 / max(1, n_conditions * 2)  # *2 porque hay PLV+PLI
    offset = -width * n_conditions  # centrar grupo

    for i, (condition, res) in enumerate(valid_results.items()):
        src = res.get('phase_locking', res.get('plv_pli', {}))
        plv_vals = [float(src.get(b, {}).get('plv', 0)) for b in plv_bands]
        pli_vals = [float(src.get(b, {}).get('pli', 0)) for b in plv_bands]
        env_vals = [float(src.get(b, {}).get('env_corr', 0)) for b in plv_bands]  # NUEVO
        
        width_adj = width * 2/3  # Reducir ancho para caber 3 barras
        ax2.bar(x + (i*3)*width_adj - width_adj*(n_conditions-1), plv_vals, width_adj,
                label=f'{condition} PLV', alpha=0.85)
        ax2.bar(x + ((i*3)+1)*width_adj - width_adj*(n_conditions-1), pli_vals, width_adj,
                label=f'{condition} PLI', alpha=0.6)
        # ax2.bar(x + ((i*3)+2)*width_adj - width_adj*(n_conditions-1), env_vals, width_adj,
        #         label=f'{condition} Env', alpha=0.4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(plv_bands)
    ax2.set_xlabel('Frequency Bands')
    ax2.set_ylabel('Phase Locking')
    ax2.set_title('PLV / PLI')
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
    xAB = np.arange(2)
    width_ts = 0.35 / max(1, n_conditions)

    for i, (condition, res) in enumerate(valid_results.items()):
        ts_A = res.get('timescale_A', {})
        ts_B = res.get('timescale_B', {})
        
        # Exponential
        tau_exp_A = float(ts_A.get('tau_exp', ts_A.get('tau', 0)))
        tau_exp_B = float(ts_B.get('tau_exp', ts_B.get('tau', 0)))
        
        # Integrated
        tau_int_A = float(ts_A.get('tau_int', 0))
        tau_int_B = float(ts_B.get('tau_int', 0))
        
        ax4.bar(xAB + (i*2)*width_ts, [tau_exp_A, tau_exp_B], width_ts,
                label=f'{condition} Exp', alpha=0.85)
        ax4.bar(xAB + (i*2+1)*width_ts, [tau_int_A, tau_int_B], width_ts,
                label=f'{condition} Int', alpha=0.6)

    ax4.set_xticks(xAB + width_ts*n_conditions)
    ax4.set_xticklabels(['A', 'B'])
    ax4.set_xlabel('Population')
    ax4.set_ylabel('Timescale (ms)')
    ax4.set_title('Intrinsic Timescales (Exp vs Int)')
    ax4.legend(fontsize=7)
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

    try:
        plt.tight_layout()
    except:
        pass  # Skip if axes incompatible
    return fig

def plot_population_dashboard(results_dict, figsize=(20, 11)):
    """Dashboard optimizado para 1 o 2 poblaciones"""
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if not valid_results:
        return None

    # Detectar modo
    first_res = next(iter(valid_results.values()))
    single_pop = first_res.get('single_population', False)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 5, hspace=0.25, wspace=0.4, height_ratios=[0.9, 1.3],
                        width_ratios=[0.7, 0.7, 1.2, 1.2, 0.1])
    
    # 1) Autocorr
    
    # ax1 = fig.add_subplot(gs[0, 0])
    # for condition, res in valid_results.items():
    #     ac_A = res.get('autocorr_A', {})
    #     ac_B = res.get('autocorr_B', {})
    #     if len(ac_A.get('lags', [])) > 0:
    #         ax1.plot(ac_A['lags'], ac_A['correlation'], label=f"{condition} A", alpha=0.9, lw=2.5)
    #     if len(ac_B.get('lags', [])) > 0:
    #         ax1.plot(ac_B['lags'], ac_B['correlation'], label=f"{condition} B", alpha=0.9, lw=2.5, ls='--')
    # ax1.set_xlabel('Lag (ms)', fontsize=11)
    # ax1.set_ylabel('Autocorrelation', fontsize=11)
    # ax1.set_title('Autocorrelation', fontsize=12, weight='bold')
    # ax1.legend(fontsize=9)
    # ax1.grid(True, alpha=0.3)
    # ax1.tick_params(labelsize=10)

   # 1) Autocorrelación
    colors = {'A': 'steelblue', 'B': 'darkorange'}
    pops = ['A'] if single_pop else ['A', 'B']
    
    for idx, pop in enumerate(pops):
        ax = fig.add_subplot(gs[0, idx], sharey=None if idx == 0 else ax1a)
        if idx == 0:
            ax1a = ax
            
        for condition, res in valid_results.items():
            ac = res.get(f'autocorr_{pop}', {})
            ts = res.get(f'timescale_{pop}', {})
            lags = ac.get('lags', [])
            corr = ac.get('correlation', [])
            if len(lags) == 0:
                continue
            
            mask = (lags >= -5) & (lags <= 45)
            ax.plot(lags[mask], corr[mask], label=condition, lw=2.5, alpha=0.9, color=colors[pop])
            
            # Área integrada
            pos = (lags >= 0) & (lags <= 45)
            zero_idx = np.where(corr[pos] <= 0)[0]
            if len(zero_idx) > 0:
                ax.fill_between(lags[pos][:zero_idx[0]], 0, corr[pos][:zero_idx[0]], 
                               alpha=0.2, color=colors[pop])
            
            # Exp fit
            tau = ts.get('tau_exp', 0)
            if tau > 0:
                fit_x = lags[pos][lags[pos] <= 4*tau]
                fit_y = np.exp(-fit_x / tau)
                ax.plot(fit_x, fit_y, 'k--', lw=1.5, alpha=0.8)
        
        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.4)
        ax.axvline(0, color='k', ls='-', lw=0.8, alpha=0.3)
        ax.set_xlabel('Lag (ms)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('AC', fontsize=11)
        ax.set_title(f'AC Pop {pop}', fontsize=12, weight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 45)
        ax.tick_params(labelsize=10)
        if idx > 0:
            ax.tick_params(labelleft=False)
    
    # 2) PSD
    ax2 = fig.add_subplot(gs[0, 2])
    ax2_twin = ax2.twinx() if not single_pop else None
    
    for condition, res in valid_results.items():
        psd_A = res.get('psd_A', res.get('power_A', {}))
        if len(psd_A.get('freqs', [])) > 0:
            ax2.plot(psd_A['freqs'], psd_A['psd'], label=f"{condition} A", alpha=0.9, lw=1.5, color='steelblue')
            ax2.fill_between(psd_A['freqs'], 0, psd_A['psd'], alpha=0.5, facecolor='steelblue')
        
        if not single_pop:
            psd_B = res.get('psd_B', res.get('power_B', {}))
            if len(psd_B.get('freqs', [])) > 0:
                ax2_twin.plot(psd_B['freqs'], psd_B['psd'], label=f"{condition} B", alpha=0.9, lw=1.5, color='darkorange')
                ax2_twin.fill_between(psd_B['freqs'], 0, psd_B['psd'], alpha=0.3, facecolor='darkorange')
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('PSD Pop A', color='steelblue', fontsize=11, weight='bold')
    if ax2_twin:
        ax2_twin.set_ylabel('PSD Pop B', color='darkorange', fontsize=11, weight='bold')
        ax2_twin.tick_params(axis='y', labelcolor='darkorange', labelsize=10)
        ax2_twin.set_ylim(bottom=0)
    ax2.set_title('Power Spectra', fontsize=12, weight='bold')
    ax2.set_xlim(0, 60)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=10)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3) Delays (skip si single_pop)
    if not single_pop:
        ax3 = fig.add_subplot(gs[0, 3])
        _plot_delay_distribution(ax3, valid_results, legend_mode='bottom')
    
    # 4) Series temporales
    signal_mode = first_res.get('signal_mode', 'firing_rate')
    is_lfp = (signal_mode == 'lfp')
    view_ms = min((float(r.get('time_series', {}).get('time', [0])[-1] if len(r.get('time_series', {}).get('time', [])) else 0) 
                   for r in valid_results.values()), default=0.0)
    
    if is_lfp:
        from scipy.ndimage import gaussian_filter1d
        gs_lfp = gs[1, :].subgridspec(2, 1, hspace=0.2, height_ratios=[1, 1])
        ax4_lfp = fig.add_subplot(gs_lfp[0])
        ax4_fr = fig.add_subplot(gs_lfp[1], sharex=ax4_lfp)
        
        for condition, res in valid_results.items():
            ts = res.get('time_series', {})
            time_fr = ts.get('time_fr', ts.get('time', np.array([])))
            if len(time_fr) == 0:
                continue
            end_idx = np.searchsorted(time_fr, view_ms, side='right')
            
            # LFP
            lfp_A = gaussian_filter1d(ts.get('signal_A', np.array([])), 2)
            ax4_lfp.plot(time_fr[:end_idx], lfp_A[:end_idx], label=f'{condition} A', lw=1.5, alpha=0.95, color='steelblue')
            if not single_pop:
                lfp_B = gaussian_filter1d(ts.get('signal_B', np.array([])), 2)
                ax4_lfp.plot(time_fr[:end_idx], lfp_B[:end_idx], label=f'{condition} B', lw=1.5, alpha=0.85, color='darkorange')
            
            # FR
            fr_A = gaussian_filter1d(ts.get('fr_A', np.array([])), 4)
            ax4_fr.plot(time_fr[:end_idx], fr_A[:end_idx], label=f'{condition} A', lw=1.5, alpha=0.95, color='steelblue')
            if not single_pop:
                fr_B = gaussian_filter1d(ts.get('fr_B', np.array([])), 4)
                ax4_fr.plot(time_fr[:end_idx], fr_B[:end_idx], label=f'{condition} B', lw=1.5, alpha=0.85, color='darkorange')
        
        ax4_lfp.set_ylabel('LFP (mV)', fontsize=12, weight='bold')
        ax4_lfp.set_title('Mean Field Potential', fontsize=13, weight='bold', pad=10)
        ax4_lfp.legend(fontsize=10, loc='upper right')
        ax4_lfp.grid(True, alpha=0.3)
        ax4_lfp.tick_params(labelbottom=False, labelsize=10)
        
        ax4_fr.set_xlabel('Time (ms)', fontsize=12, weight='bold')
        ax4_fr.set_ylabel('Rate (Hz)', fontsize=12, weight='bold')
        ax4_fr.set_title('Population Firing Rate', fontsize=13, weight='bold', pad=10)
        ax4_fr.legend(fontsize=10, loc='upper right')
        ax4_fr.grid(True, alpha=0.3)
        ax4_fr.set_ylim(bottom=0)
        ax4_fr.tick_params(labelsize=10)
    else:
        from scipy.ndimage import gaussian_filter1d
        ax4 = fig.add_subplot(gs[1, :])
        for condition, res in valid_results.items():
            ts = res.get('time_series', {})
            time_fr = ts.get('time_fr', ts.get('time', np.array([])))
            if len(time_fr) == 0:
                continue
            end_idx = np.searchsorted(time_fr, view_ms, side='right')
            
            fr_A = gaussian_filter1d(ts.get('fr_A', np.array([])), 4)
            ax4.plot(time_fr[:end_idx], fr_A[:end_idx], label=f'{condition} A', lw=1.5, alpha=0.9, color='steelblue')
            if not single_pop:
                fr_B = gaussian_filter1d(ts.get('fr_B', np.array([])), 4)
                ax4.plot(time_fr[:end_idx], fr_B[:end_idx], label=f'{condition} B', lw=1.5, alpha=0.9, color='darkorange')
        
        ax4.set_xlabel('Time (ms)', fontsize=12)
        ax4.set_ylabel('Rate (Hz)', fontsize=12)
        ax4.set_title('Population Activity', fontsize=13, weight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=10)
    
    return fig


def compute_synchronization_metrics(res, t_max_ms, dt_bin=50, filter_band=(1, 50)):
    """
    Métricas genéricas de sincronización temporal:
    - VS_A, VS_B: vector strength intra-poblacional
    - PLV_AB: phase locking inter-poblacional
    """
    
    spike_times_A = res.get('spike_times_A', np.array([]))
    spike_times_B = res.get('spike_times_B', np.array([]))
    
    ts = res.get('time_series', {})
    t_rate = ts.get('time', np.array([]))
    rate_A = ts.get('rate_A', np.array([]))
    rate_B = ts.get('rate_B', np.array([]))
    
    if len(spike_times_A) == 0 or len(rate_A) == 0:
        return None, None, None, None
    
    # Preparar señales
    min_len = min(len(t_rate), len(rate_A), len(rate_B))
    t_rate, rate_A, rate_B = t_rate[:min_len], rate_A[:min_len], rate_B[:min_len]
    
    dt_rate = np.median(np.diff(t_rate))
    fs = 1000.0 / dt_rate
    low_f, high_f = filter_band[0], min(filter_band[1], 0.45*fs)
    
    if high_f <= low_f or min_len < 100:
        return None, None, None, None
    
    # Filtrar y extraer fases poblacionales
    sos = scipy_signal.butter(3, [low_f, high_f], btype='band', fs=fs, output='sos')
    filt_A = scipy_signal.sosfiltfilt(sos, rate_A)
    filt_B = scipy_signal.sosfiltfilt(sos, rate_B)
    
    phase_A_full = np.angle(scipy_signal.hilbert(filt_A))
    phase_B_full = np.angle(scipy_signal.hilbert(filt_B))
    
    # Bins temporales
    t_bins = np.arange(0, min(t_max_ms, t_rate[-1]), dt_bin)
    VS_A = np.zeros(len(t_bins))
    VS_B = np.zeros(len(t_bins))
    PLV_AB = np.zeros(len(t_bins))
    
    for i, t_center in enumerate(t_bins):
        t_start, t_end = t_center, t_center + dt_bin
        
        # === INTRA: Vector Strength ===
        # Spikes de A en esta ventana
        mask_A = (spike_times_A >= t_start) & (spike_times_A < t_end)
        spikes_window_A = spike_times_A[mask_A]
        
        if len(spikes_window_A) > 1:
            # Fase poblacional en cada spike
            phases_at_spikes = np.interp(spikes_window_A, t_rate, phase_A_full)
            VS_A[i] = np.abs(np.mean(np.exp(1j * phases_at_spikes)))
        
        # Spikes de B en esta ventana
        mask_B = (spike_times_B >= t_start) & (spike_times_B < t_end)
        spikes_window_B = spike_times_B[mask_B]
        
        if len(spikes_window_B) > 1:
            phases_at_spikes = np.interp(spikes_window_B, t_rate, phase_B_full)
            VS_B[i] = np.abs(np.mean(np.exp(1j * phases_at_spikes)))
        
        # === INTER: PLV entre poblaciones ===
        idx_start = np.searchsorted(t_rate, t_start)
        idx_end = np.searchsorted(t_rate, t_end)
        
        if idx_end > idx_start and idx_end <= len(phase_A_full):
            phase_diff = phase_A_full[idx_start:idx_end] - phase_B_full[idx_start:idx_end]
            PLV_AB[i] = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return VS_A, VS_B, PLV_AB, t_bins


def _plot_delay_distribution(ax, valid_results, legend_mode):
    """Plot delay distribution histograms with comprehensive statistics"""
    
    delay_data = {}
    for condition, res in valid_results.items():
        delays_AB = res.get('delays_AB', res.get('delay_samples', None))
        delays_BA = res.get('delays_BA', None)
        delay_config = res.get('delay_config', {})
        
        # Fallback a estadísticas pre-calculadas
        if delays_AB is None:
            delay_stats = res.get('delay_statistics', {})
            if delay_stats and 'mean' in delay_stats:
                delay_data[condition] = {
                    'stats_only': True, 
                    'stats': delay_stats,
                    'config': delay_config
                }
                continue
        
        if delays_AB is not None:
            delay_data[condition] = {
                'AB': np.asarray(delays_AB).flatten(),
                'BA': np.asarray(delays_BA).flatten() if delays_BA is not None else None,
                'stats_only': False,
                'config': delay_config
            }
    
    if not delay_data:
        ax.text(0.5, 0.5, 'No delay data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Delay Distribution')
        return
    
    # Plot histograms
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    has_histogram = False
    
    for idx, (condition, data) in enumerate(delay_data.items()):
        if data.get('stats_only'):
            continue
            
        delays_AB = data['AB']
        delays_BA = data.get('BA')
        color = colors[idx]
        
        if len(delays_AB) > 0:
            ax.hist(delays_AB, bins=30, alpha=0.6, color=color, 
                   label=f'{condition} A→B', edgecolor='black', linewidth=0.5)
            has_histogram = True
            
        if delays_BA is not None and len(delays_BA) > 0:
            ax.hist(delays_BA, bins=30, alpha=0.4, color=color, 
                   label=f'{condition} B→A', edgecolor='gray', linewidth=0.5, 
                   linestyle='--', histtype='step')
    
    # Estadísticas completas
    stats_text = []
    for condition, data in delay_data.items():
        config = data.get('config', {})
        
        if data.get('stats_only'):
            st = data['stats']
            stats_text.append(f"{condition}:")
            stats_text.append(f"  μ={st.get('mean', 0):.2f}, Med={st.get('median', 0):.2f}")
            stats_text.append(f"  σ={st.get('std', 0):.2f}")
            # Añadir params nativos si están disponibles
            _append_native_params(stats_text, config)
            continue
        
        delays_AB = data['AB']
        if len(delays_AB) < 2:
            continue
        
        # Métricas descriptivas
        mean_val = float(np.mean(delays_AB))
        median_val = float(np.median(delays_AB))
        std_val = float(np.std(delays_AB))
        min_val = float(np.min(delays_AB))
        max_val = float(np.max(delays_AB))
        
        # Moda robusta (bin más frecuente del histograma)
        counts, bins = np.histogram(delays_AB, bins=30)
        mode_val = float(bins[np.argmax(counts)] + (bins[1] - bins[0])/2)
        
        # Forma de distribución
        skew_val = float(stats.skew(delays_AB))
        kurt_val = float(stats.kurtosis(delays_AB))
        cv = std_val / mean_val if mean_val > 0 else 0
        
        # Textbox compacto
        stats_text.append(f"{condition} (A→B):")
        stats_text.append(f"  μ={mean_val:.1f}, Med={median_val:.1f}, Mo={mode_val:.1f}")
        stats_text.append(f"  σ={std_val:.1f}, [{min_val:.1f},{max_val:.1f}]")
        stats_text.append(f"  Sk={skew_val:.2f}, Ku={kurt_val:.2f}, CV={cv:.2f}")
        
        # Parámetros nativos de la distribución
        _append_native_params(stats_text, config)
        
        # Bidireccionalidad
        delays_BA = data.get('BA')
        if delays_BA is not None and len(delays_BA) > 1:
            mean_ba = float(np.mean(delays_BA))
            stats_text.append(f"  B→A: μ={mean_ba:.1f} (Δ={mean_ba-mean_val:.1f})")
    
    # Textbox
    if stats_text:
        text_str = '\n'.join(stats_text)
        
        if legend_mode == 'bottom':
            # Formato horizontal compacto
            compact_text = []
            for condition, data in delay_data.items():
                if data.get('stats_only'):
                    st = data['stats']
                    compact_text.append(f"{condition}: μ={st.get('mean',0):.1f}, σ={st.get('std',0):.1f}")
                else:
                    delays = data['AB']
                    if len(delays) > 1:
                        compact_text.append(f"{condition}: μ={np.mean(delays):.1f}, σ={np.std(delays):.1f}, Mo={bins[np.argmax(counts)]:.1f}")
            
            text_str = ' | '.join(compact_text)
            ax.text(0.5, 0.5, text_str, transform=ax.transAxes,
                fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        else:
            # Original (lado derecho)
            ax.text(0.97, 0.97, text_str, transform=ax.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                family='monospace')
    
    ax.set_xlabel('Delay (ms)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Delay Distribution', fontweight='bold', fontsize=12)

def _append_native_params(stats_text, delay_config):
    """Append native distribution parameters to stats text"""
    if not delay_config or 'type' not in delay_config:
        return
    
    dist_type = delay_config['type']
    params = delay_config.get('params', {})
    
    if dist_type == 'lognormal':
        alpha = params.get('alpha', delay_config.get('value', 0))
        beta = params.get('beta', 0)
        stats_text.append(f"  LogN: α={alpha:.2f}, β={beta:.2f}")
        
    elif dist_type == 'gamma':
        shape = params.get('shape', 0)
        scale = params.get('scale', 0)
        stats_text.append(f"  Γ: k={shape:.2f}, θ={scale:.2f}")
        
    elif dist_type == 'beta':
        alpha = params.get('alpha', 0)
        beta = params.get('beta', 0)
        scale = params.get('scale', 0)
        stats_text.append(f"  β: α={alpha:.1f}, β={beta:.1f}, s={scale:.1f}")
        
    elif dist_type == 'uniform':
        low = params.get('low', 0)
        high = params.get('high', 0)
        stats_text.append(f"  U: [{low:.1f}, {high:.1f}]")
        
    elif dist_type == 'constant':
        value = delay_config.get('value', params.get('value', 0))
        stats_text.append(f"  δ: {value:.1f}ms")

        
def plot_spectrogram(results_dict, figsize=(16, 6)):
    """Espectrograma optimizado"""
    from scipy import signal as sg
    from scipy.ndimage import gaussian_filter

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    for idx, pop in enumerate(['A', 'B']):
        condition = list(results_dict.keys())[0]
        res = results_dict[condition]
        
        ts = res.get('time_series', {})
        lfp = ts.get(f'signal_{pop}', np.array([]))
        
        if len(lfp) == 0:
            continue
        
        fs = 1000.0  # tu fs real
        nperseg = 512
        noverlap = int(nperseg * 0.8)
        
        # Z-score + detrend
        lfp_proc = (lfp - np.mean(lfp)) / np.std(lfp)
        
        f, t, Sxx = sg.spectrogram(lfp_proc, fs=fs, 
                                nperseg=nperseg, 
                                noverlap=noverlap,
                                window='hann',
                                scaling='density')
        
        
        mask = f <= 60
        f_masked = f[mask]  # FIX: enmascarar f también
        
        # Suavizado + escala log directa (sin normalización por freq)
        Sxx_smooth = gaussian_filter(Sxx[mask, :], sigma=(0.5, 0.5))
        Sxx_db = Sxx_smooth#10 * np.log10(Sxx_smooth + 1e-12)
        
        # Percentiles para escala dinámica
        vmin = np.percentile(Sxx_db, 12)
        vmax = np.percentile(Sxx_db, 88)
        
        ax = axes[idx]
        im = ax.pcolormesh(t, f_masked, Sxx_db, 
                        shading='gouraud',
                        cmap='inferno', # jet
                        vmin=vmin, vmax=vmax,
                        rasterized=True)
        
        ax.set_ylabel('Frequency (Hz)' if idx == 0 else '', fontsize=13)
        ax.set_xlabel('Time (s)', fontsize=13)
        ax.set_ylim(2, 60)  # enfoque <80 Hz
        ax.set_title(f'Population {pop}', fontsize=14, weight='bold', pad=10)
        
        cbar = plt.colorbar(im, ax=ax, aspect=25)
        cbar.set_label('Power (dB)', rotation=270, labelpad=18, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        ax.tick_params(labelsize=11)

    fig.suptitle('LFP Spectrograms', fontsize=16, y=0.99, weight='bold')
    plt.tight_layout()
    return fig



def plot_palmigiano_dashboard(palmi_metrics, lfp_A, lfp_B, fs, save_path=None):
    """Palmigiano dashboard - optimized layout"""
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.5,
                  top=0.95, bottom=0.05, left=0.06, right=0.98)
    
    # # ========== Row 1: Spectrograms ==========
    # for idx, (lfp, label) in enumerate([(lfp_A, 'A'), (lfp_B, 'B')]):
    #     ax = fig.add_subplot(gs[0, idx*3:(idx+1)*3])
        
    #     nperseg = 512
    #     noverlap = int(nperseg * 0.8)
    #     lfp_proc = (lfp - np.mean(lfp)) / (np.std(lfp) + 1e-10)
        
    #     f, t, Sxx = spectrogram(lfp_proc, fs=1000, nperseg=nperseg, 
    #                             noverlap=noverlap, window='hann', scaling='density')
        
    #     mask = f <= 80
    #     f_masked = f[mask]
    #     Sxx_smooth = gaussian_filter(Sxx[mask, :], sigma=(0.5, 0.5))
        
    #     vmin, vmax = np.percentile(Sxx_smooth, [10, 90])
        
    #     im = ax.pcolormesh(t, f_masked, Sxx_smooth, shading='gouraud',
    #                       cmap='inferno', vmin=vmin, vmax=vmax, rasterized=True)
    #     ax.set_ylabel('Frequency (Hz)', fontsize=11)
    #     ax.set_xlabel('Time (s)', fontsize=11)
    #     ax.set_title(f'Population {label}', fontsize=12, weight='bold')
    #     plt.colorbar(im, ax=ax, label='Power', aspect=18)
    
    # ========== Row 2: Phase Distribution + XC Heatmap ==========
    delta_phi = palmi_metrics['routing']['delta_phi']
    xc_data = palmi_metrics['xc_resolved']
    
    # # Polar 
    # ax_polar = fig.add_subplot(gs[0, 0], projection='polar')
    # ax_polar.hist(delta_phi * 2*np.pi, bins=60, density=False,
    #               color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
    # ax_polar.set_title('ΔΦ Polar', pad=12, fontsize=10)
    
    # Linear histogram (2 columns)
    ax_phase = fig.add_subplot(gs[0, :])
    counts, bins = np.histogram(delta_phi, bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax_phase.bar(bin_centers, counts, width=np.diff(bins)[0]*0.95,
                 color='steelblue', alpha=0.8, edgecolor='none')
    
    if len(delta_phi) > 10:
        kde = gaussian_kde(delta_phi, bw_method=0.02)
        x_smooth = np.linspace(0, 1, 500)
        kde_scaled = kde(x_smooth) * len(delta_phi) / 80
        ax_phase.plot(x_smooth, kde_scaled, 'darkblue', linewidth=2.5, label='KDE')
    
    ax_phase.axvline(0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax_phase.set_xlim([0, 1])
    ax_phase.set_xlabel('ΔΦ', fontsize=11)
    ax_phase.set_ylabel('Count', fontsize=11)
    ax_phase.set_title('Phase Difference Distribution', fontsize=12, weight='bold')
    ax_phase.legend(fontsize=9)
    ax_phase.grid(alpha=0.3)
    
    # # XC heatmap (3 columns)
    # ax_xc = fig.add_subplot(gs[1, 3:])
    # lags_ms = np.linspace(-20, 20, xc_data['matrix'].shape[0])
    # im = ax_xc.imshow(xc_data['matrix'], aspect='auto', cmap='RdBu_r',
    #                   extent=[0, xc_data['times'][-1], lags_ms[0], lags_ms[-1]],
    #                   interpolation='bilinear')
    # ax_xc.axhline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    # ax_xc.set_xlabel('Time (s)', fontsize=11)
    # ax_xc.set_ylabel('Lag (ms)', fontsize=11)
    # ax_xc.set_title('Time-resolved XC', fontsize=12, weight='bold')
    # plt.colorbar(im, ax=ax_xc, label='XC', aspect=18)
    
    # ========== Row 3: Transfer Entropy (4 columns, más espacio) ==========
    te_data = palmi_metrics['transfer_entropy']
    ax_te = fig.add_subplot(gs[1, :])
    
    ax_te.plot(te_data['lags_ms'], te_data['te_AB'], 'o-', 
               label='A→B', color='steelblue', markersize=4, linewidth=2.5)
    ax_te.plot(te_data['lags_ms'], te_data['te_BA'], 's-', 
               label='B→A', color='coral', markersize=4, linewidth=2.5)
    ax_te.axvline(te_data['tau_opt_AB_ms'], color='steelblue', 
                  linestyle='--', alpha=0.5, linewidth=2)
    ax_te.axvline(te_data['tau_opt_BA_ms'], color='coral', 
                  linestyle='--', alpha=0.5, linewidth=2)
    ax_te.set_xlabel('Lag (ms)', fontsize=11)
    ax_te.set_ylabel('Transfer Entropy (bits)', fontsize=11)
    ax_te.set_title('Bidirectional Transfer Entropy', fontsize=12, weight='bold')
    ax_te.legend(fontsize=10, frameon=True, loc='upper left')
    ax_te.grid(alpha=0.3)
    
#     # ========== Row 3: Summary (pegado a la derecha) ==========
#     routing = palmi_metrics['routing']
#     ax_summary = fig.add_subplot(gs[2, 5:])
#     ax_summary.axis('off')
    
#     summary_text = f"""ROUTING METRICS
# ───────────────
# Top: {routing['top_state_fraction']:.1%}
# Bottom: {routing['bottom_state_fraction']:.1%}
# Switch: {routing['switch_rate_hz']:.1f} Hz
# Bimodality: {routing['phase_bimodality']:.2f}

# TRANSFER ENTROPY
# ───────────────
# τ_opt A→B: {te_data['tau_opt_AB_ms']:.1f} ms
# τ_opt B→A: {te_data['tau_opt_BA_ms']:.1f} ms
# ΔTE: {te_data['delta_te']:.2f}"""
    
#     ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
#                     fontsize=10, va='top', family='monospace',
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
#     # ========== Row 4: Routing Pie (centrado, pegado con summary) ==========
#     ax_routing = fig.add_subplot(gs[2, 4:5])
#     fractions = [routing['top_state_fraction'], routing['bottom_state_fraction']]
#     labels = [f'Top\n{fractions[0]:.1%}', f'Bottom\n{fractions[1]:.1%}']
#     colors_pie = ['steelblue', 'coral']
#     ax_routing.pie(fractions, labels=labels, colors=colors_pie, 
#                    startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
#     ax_routing.set_title(f'Routing States (Switch: {routing["switch_rate_hz"]:.1f} Hz)', 
#                         fontsize=12, weight='bold', pad=10)
    
    plt.suptitle('Palmigiano-style Analysis Dashboard', 
                 fontsize=15, fontweight='bold', y=0.97)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    plt.tight_layout()
    
    return fig

# --

# """
# Versión expandida del plot de comparación con TODAS las métricas del dashboard.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Dict, Optional

# def plot_delay_comparison_with_distributions(results_db: Dict, save_path: Optional[str] = None):
#     """
#     Grid 3x4 con TODAS las métricas de conectividad + distribuciones de delay.
    
#     Incluye:
#     - Phase locking: PLV/PLI alpha y gamma
#     - Cross-correlation: peak y lag
#     - Coherence: peak, alpha, gamma
#     - Timescales: tau_A, tau_B
#     - Delay distributions
#     """
#     if not results_db:
#         print("[WARNING] results_db vacío")
#         return None
    
#     # Sort configs by parameters
#     configs = list(results_db.keys())
#     configs_with_params = [(config, extract_parameters_for_sorting(config)) for config in configs]
#     configs_with_params.sort(key=lambda x: (x[1][0], x[1][1]))
#     configs = [item[0] for item in configs_with_params]
    
#     # TODAS las métricas (11 + 1 plot de distribuciones)
#     key_metrics = [
#         # Row 1: Phase locking
#         'plv_alpha', 'pli_alpha', 'plv_gamma', 'pli_gamma',
#         # Row 2: Cross-correlation & Coherence
#         'cc_peak', 'cc_lag', 'coherence_peak', 'alpha_coherence',
#         # Row 3: More coherence & timescales
#         'gamma_coherence', 'tau_A', 'tau_B'
#     ]
    
#     # Map internal names to display names
#     metric_labels = {
#         'plv_alpha': 'PLV Alpha',
#         'pli_alpha': 'PLI Alpha',
#         'plv_gamma': 'PLV Gamma',
#         'pli_gamma': 'PLI Gamma',
#         'cc_peak': 'Cross Corr Peak',
#         'cc_lag': 'Cross Corr Lag',
#         'coherence_peak': 'Coherence Peak',
#         'alpha_coherence': 'Alpha Coherence',
#         'gamma_coherence': 'Gamma Coherence',
#         'tau_A': 'Tau A',
#         'tau_B': 'Tau B'
#     }
    
#     fig, axes = plt.subplots(3, 4, figsize=(20, 14))
#     axes = axes.flatten()
    
#     # Simplificar nombres para eje x
#     def simplify_config_name(config_name):
#         parts = config_name.split('_')
#         for i, part in enumerate(parts):
#             if part in ['delta', 'lognormal', 'gamma', 'uniform', 'beta', 'exponential']:
#                 return '_'.join(parts[i:]).replace('_input_', '_')
#         return '_'.join(parts[-3:])
    
#     simplified_names = [simplify_config_name(config) for config in configs]
#     x = np.arange(len(configs))
    
#     # Plot primeras 11 métricas
#     for i, metric in enumerate(key_metrics):
#         ax = axes[i]
        
#         means = []
#         stds = []
        
#         for c in configs:
#             agg = results_db[c].get('aggregated', {})
#             metric_data = agg.get(metric, {})
            
#             mean_val = metric_data.get('mean', 0)
#             std_val = metric_data.get('std', 0)
            
#             # Handle NaN values
#             if mean_val is None or np.isnan(mean_val):
#                 mean_val = 0
#             if std_val is None or np.isnan(std_val):
#                 std_val = 0
            
#             # Use absolute value for lag
#             if metric == 'cc_lag':
#                 mean_val = abs(mean_val)
                
#             means.append(mean_val)
#             stds.append(std_val)
        
#         if len(means) > 0:
#             means = np.array(means)
#             stds = np.array(stds)
            
#             # Color coding by metric type
#             if 'plv' in metric or 'pli' in metric:
#                 color = 'darkgreen'
#                 fillcolor = 'lightgreen'
#             elif 'cc' in metric:
#                 color = 'darkblue'
#                 fillcolor = 'lightblue'
#             elif 'coherence' in metric:
#                 color = 'darkred'
#                 fillcolor = 'lightcoral'
#             else:  # tau
#                 color = 'darkorange'
#                 fillcolor = 'moccasin'
            
#             ax.plot(x, means, 'o-', linewidth=2.5, markersize=7, 
#                    color=color, markerfacecolor=color, 
#                    markeredgecolor='white', markeredgewidth=1.5)
            
#             ax.fill_between(x, means - stds, means + stds, 
#                            alpha=0.25, color=fillcolor)
            
#             ax.set_xticks(x)
#             ax.set_xticklabels(simplified_names, rotation=45, ha='right', fontsize=8)
#             ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10, fontweight='bold')
#             ax.set_title(metric_labels.get(metric, metric), fontweight='bold', fontsize=11)
#             ax.grid(True, alpha=0.3)
            
#             # Adaptive y-axis: show meaningful range
#             if np.std(means) > 0:
#                 mean_val = np.mean(means)
#                 std_val = np.std(means)
#                 y_range = max(3*std_val, mean_val*0.3)  # At least 3 std or 30% of mean
#                 ax.set_ylim(max(0, mean_val - y_range), mean_val + y_range)
    
#     # Último plot: distribuciones de delay
#     ax_dist = axes[11]
#     plot_delay_distributions(results_db, ax=ax_dist, show_legend=True)
    
#     plt.suptitle('Connectivity Metrics & Delay Distributions (Complete Dashboard)', 
#                 fontsize=16, fontweight='bold', y=0.995)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.97)
    
#     if save_path:
#         fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         print(f"[PLOT] Saved expanded comparison plot: {save_path}")
    
#     return fig


# def extract_parameters_for_sorting(config_name: str):
#     """Extract numeric parameters from config name for sorting."""
#     params = []
#     parts = config_name.split('_')
    
#     for i, part in enumerate(parts):
#         if part in ['delta', 'lognormal', 'gamma', 'uniform', 'beta', 'exponential']:
#             # Extract next 2 numeric values
#             for j in range(i+1, min(i+3, len(parts))):
#                 try:
#                     if parts[j].replace('.', '').replace('p', '').replace('-', '').isdigit():
#                         value = float(parts[j].replace('p', '.').split('-')[0])
#                         params.append(value)
#                 except:
#                     continue
#             break
    
#     # Ensure at least 2 parameters
#     while len(params) < 2:
#         params.append(0)
    
#     return params[:2]


# def plot_delay_distributions(results_db: Dict, ax=None, show_legend: bool = True):
#     """
#     Plot delay distributions as overlaid KDE curves.
#     Simplified version for the comparison grid.
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6))
    
#     from scipy.stats import gaussian_kde
    
#     # Collect delay data
#     delay_samples = {}
#     for config, data in results_db.items():
#         delay_stats = data.get('delay_statistics', {})
#         if delay_stats and 'mean' in delay_stats:
#             # Reconstruct approximate distribution from stats
#             mean = delay_stats.get('mean', 5)
#             std = delay_stats.get('std', 1)
            
#             # Simple label from config
#             label = config.split('_')[-2] if '_' in config else config[:10]
#             delay_samples[label] = (mean, std)
    
#     if not delay_samples:
#         ax.text(0.5, 0.5, 'No delay data', ha='center', va='center',
#                transform=ax.transAxes, fontsize=12)
#         return
    
#     # Plot distributions
#     x_range = np.linspace(0, 25, 500)
#     colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(delay_samples))))
    
#     for idx, (label, (mean, std)) in enumerate(delay_samples.items()):
#         # Approximate Gaussian for visualization
#         from scipy.stats import norm
#         y = norm.pdf(x_range, mean, std)
        
#         color = colors[idx % len(colors)]
#         ax.plot(x_range, y, linewidth=2, alpha=0.7, label=label, color=color)
#         ax.axvline(mean, color=color, linestyle='--', alpha=0.5, linewidth=1)
    
#     ax.set_xlabel('Delay (ms)', fontweight='bold', fontsize=10)
#     ax.set_ylabel('Probability Density', fontweight='bold', fontsize=10)
#     ax.set_title('Delay Distributions', fontweight='bold', fontsize=11)
#     ax.grid(True, alpha=0.3)
    
#     if show_legend and len(delay_samples) <= 10:
#         ax.legend(fontsize=7, loc='upper right', ncol=2)
    
#     return ax