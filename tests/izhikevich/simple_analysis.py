import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

def izhikevich_simple(a=0.02, b=0.2, c=-65, d=8, I=10):
    """
    Análisis súper simple del modelo de Izhikevich
    Una función todo-en-uno para exploración rápida
    """
    
    def derivatives(t, state):
        v, u = state
        dvdt = 0.04*v**2 + 5*v + 140 - u + I
        dudt = a*(b*v - u)
        return [dvdt, dudt]
    
    def spike_event(t, state):
        return state[0] - 29.5
    spike_event.terminal = True
    spike_event.direction = 1
    
    # Crear figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. RETRATO DE FASE
    print("📊 Generando retrato de fase...")
    
    # Campo vectorial
    v = np.linspace(-80, 40, 12)
    u = np.linspace(-20, 20, 12)
    V, U = np.meshgrid(v, u)
    
    DV = 0.04*V**2 + 5*V + 140 - U + I
    DU = a*(b*V - U)
    M = np.sqrt(DV**2 + DU**2)
    M[M == 0] = 1
    
    ax1.quiver(V, U, DV/M, DU/M, alpha=0.6, scale=20)
    
    # Nullclines
    v_null = np.linspace(-80, 40, 300)
    u_null_v = 0.04*v_null**2 + 5*v_null + 140 + I
    u_null_u = b * v_null
    
    ax1.plot(v_null, u_null_v, 'r-', linewidth=2, label='v-nullcline')
    ax1.plot(v_null, u_null_u, 'b-', linewidth=2, label='u-nullcline')
    
    # Puntos fijos
    discriminant = (5-b)**2 - 4*0.04*(140+I)
    if discriminant >= 0:
        v1 = (-(5-b) + np.sqrt(discriminant)) / 0.08
        v2 = (-(5-b) - np.sqrt(discriminant)) / 0.08
        u1, u2 = b*v1, b*v2
        
        for v_fp, u_fp in [(v1, u1), (v2, u2)]:
            if -80 <= v_fp <= 40 and -20 <= u_fp <= 20:
                ax1.plot(v_fp, u_fp, 'yo', markersize=10, markeredgecolor='black')
    
    # Trayectorias
    colors = ['red', 'blue', 'green']
    for i, (v0, u0) in enumerate([(-70, 0), (-60, 5), (-50, -5)]):
        try:
            sol = solve_ivp(derivatives, [0, 50], [v0, u0], events=spike_event,
                          method='DOP853', rtol=1e-4, max_step=0.5)
            if sol.success and len(sol.t) > 1:
                t_plot = np.linspace(0, sol.t[-1], 200)
                traj = sol.sol(t_plot).T
                ax1.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2)
                ax1.plot(v0, u0, 'o', color=colors[i], markersize=8)
        except:
            pass
    
    ax1.axvline(x=30, color='k', linestyle='--', alpha=0.7)
    ax1.set_xlabel('v (mV)', fontweight='bold')
    ax1.set_ylabel('u', fontweight='bold')
    ax1.set_title(f'Retrato de Fase\nI={I}, a={a}, b={b}', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-80, 40)
    ax1.set_ylim(-20, 20)
    
    # 2. SERIE TEMPORAL
    print("⏱️  Generando serie temporal...")
    
    def simulate_euler(t_max=200):
        dt = 0.5  # 0.5ms estándar
        steps = int(t_max / dt)
        v, u = -70, 0
        v_trace, t_trace = [], []
        
        for step in range(steps):
            dv = 0.04*v**2 + 5*v + 140 - u + I
            du = a*(b*v - u)
            
            v += dv * dt
            u += du * dt
            
            if v >= 30:
                v_trace.append(30)
                t_trace.append(step * dt)
                v = c
                u += d
                
            v_trace.append(v)
            t_trace.append(step * dt)
            
        return np.array(t_trace), np.array(v_trace)
    
    try:
        t_plot, v_trace = simulate_euler()
        ax2.plot(t_plot, v_trace, 'b-', linewidth=1.5)
        ax2.axhline(y=30, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Tiempo (ms)', fontweight='bold')
        ax2.set_ylabel('v (mV)', fontweight='bold')
        ax2.set_title(f'Serie Temporal\nI={I}', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-80, 40)
    except:
        ax2.text(0.5, 0.5, 'Error en simulación', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
    
    # 3. BIFURCACIÓN vs I
    print("🔀 Calculando bifurcaciones...")
    
    I_range = np.linspace(-5, 20, 100)
    stable_v, stable_I = [], []
    unstable_v, unstable_I = [], []
    
    for I_test in I_range:
        discriminant = (5-b)**2 - 4*0.04*(140+I_test)
        if discriminant >= 0:
            v1 = (-(5-b) + np.sqrt(discriminant)) / 0.08
            v2 = (-(5-b) - np.sqrt(discriminant)) / 0.08
            
            for v_fp in [v1, v2]:
                if -100 <= v_fp <= 50:
                    # Análisis de estabilidad simple
                    trace = 0.08*v_fp + 5 - a
                    det = (0.08*v_fp + 5)*(-a) - (-1)*(a*b)
                    
                    if trace < 0 and det > 0:
                        stable_v.append(v_fp)
                        stable_I.append(I_test)
                    else:
                        unstable_v.append(v_fp)
                        unstable_I.append(I_test)
    
    if stable_v:
        ax3.plot(stable_I, stable_v, 'b-', linewidth=2, label='Estable')
    if unstable_v:
        ax3.plot(unstable_I, unstable_v, 'r--', linewidth=2, label='Inestable')
    
    ax3.axvline(x=I, color='green', linewidth=3, alpha=0.7, label=f'I actual = {I}')
    ax3.set_xlabel('Corriente I', fontweight='bold')
    ax3.set_ylabel('v* (mV)', fontweight='bold')
    ax3.set_title('Bifurcaciones vs I', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. CURVA F-I
    print("📈 Calculando curva F-I...")
    
    I_freq = np.linspace(0, 30, 15)
    frequencies = []
    
    for I_test in I_freq:
        def simulate_freq(t_max=1000):
            dt = 0.5  # 0.5ms estándar
            steps = int(t_max / dt)
            v, u = -70, 0
            spike_count = 0
            
            for step in range(steps):
                dv = 0.04*v**2 + 5*v + 140 - u + I_test
                du = a*(b*v - u)
                
                v += dv * dt
                u += du * dt
                
                if v >= 30:
                    spike_count += 1
                    v = c
                    u += d
                    
            return spike_count  # spikes en 1 segundo
        
        try:
            freq = simulate_freq()
        except:
            freq = 0
        
        frequencies.append(freq)
    
    ax4.plot(I_freq, frequencies, 'go-', linewidth=3, markersize=6)
    ax4.axvline(x=I, color='red', linewidth=3, alpha=0.7, label=f'I actual = {I}')
    ax4.set_xlabel('Corriente I', fontweight='bold')
    ax4.set_ylabel('Frecuencia (Hz)', fontweight='bold')
    ax4.set_title('Curva F-I', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    plt.suptitle(f'🧠 Análisis Modelo Izhikevich - Tipo Neuronal\n' + 
                f'a={a}, b={b}, c={c}, d={d}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✅ ¡Análisis completado!")
    
    return fig

def compare_neuron_types_simple():
    """Comparación rápida de tipos neuronales"""
    print("🧠 Comparando tipos neuronales...")
    
    types = {
        'RS': (0.02, 0.2, -65, 8, 'blue'),
        'IB': (0.02, 0.2, -55, 4, 'red'),
        'CH': (0.02, 0.2, -50, 2, 'green'),
        'FS': (0.1, 0.2, -65, 2, 'orange'),
        'LTS': (0.02, 0.25, -65, 2, 'purple')
    }
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, (name, (a, b, c, d, color)) in enumerate(types.items()):
        ax = axes[i]
        I = 15  # Corriente más alta para asegurar spikes
        
        # Simulación con reset manual en threshold
        def simulate_with_reset(t_max=200):
            dt = 0.5  # 0.5ms para estabilidad numérica (estándar Izhikevich)
            t, steps = 0, int(t_max / dt)
            v, u = -70, 0
            v_trace, t_trace = [], []
            
            for step in range(steps):
                # Derivadas
                dv = 0.04*v**2 + 5*v + 140 - u + I
                du = a*(b*v - u)
                
                # Integración Euler con dt=0.5ms
                v += dv * dt
                u += du * dt
                
                # Reset si spike
                if v >= 30:
                    v_trace.append(30)  # Mostrar el spike
                    t_trace.append(t)
                    v = c  # Reset voltaje
                    u += d  # Reset recovery
                    
                v_trace.append(v)
                t_trace.append(t)
                t += dt
                
            return np.array(t_trace), np.array(v_trace)
        
        try:
            t_plot, v_trace = simulate_with_reset()
            ax.plot(t_plot, v_trace, color=color, linewidth=2)
            ax.axhline(y=30, color='k', linestyle='--', alpha=0.5)
                
        except:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f'{name}\na={a}, b={b}', fontweight='bold', color=color)
        ax.set_xlabel('Tiempo (ms)')
        if i == 0:
            ax.set_ylabel('v (mV)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-80, 40)
    
    plt.suptitle('🧠 Comparación de Tipos Neuronales', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig

# Ejemplos de uso rápido
if __name__ == "__main__":
    print("🚀 Análisis Simple del Modelo de Izhikevich")
    print("=" * 50)
    
    # Análisis completo de neurona RS
    print("\n1️⃣ Neurona Regular Spiking (RS):")
    izhikevich_simple(a=0.02, b=0.2, c=-65, d=8, I=10)
    
    # Comparación rápida
    print("\n2️⃣ Comparación de tipos:")
    compare_neuron_types_simple()
    
    print("\n✨ Para análisis personalizado, llama:")
    print("izhikevich_simple(a=0.02, b=0.25, c=-65, d=2, I=15)")