import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

class FixedIzhikevichAnalysis:
    def __init__(self, a=0.02, b=0.2, c=-65, d=8):
        """
        Análisis completamente corregido del modelo de Izhikevich
        Sin warnings numéricos y con integración robusta
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    def derivatives(self, t, state, I):
        """Derivadas del sistema para solve_ivp"""
        v, u = state
        dvdt = 0.04*v**2 + 5*v + 140 - u + I
        dudt = self.a*(self.b*v - u)
        return [dvdt, dudt]
    
    def spike_event(self, t, state, I):
        """Detecta spikes cuando v cruza 29.5 mV (antes del reset)"""
        v, u = state
        return v - 29.5
    
    spike_event.terminal = True
    spike_event.direction = 1
    
    def simulate_robust(self, initial_state, I, t_max=100):
        """Simulación con dt=0.5ms para estabilidad numérica"""
        dt = 0.5  # 0.5ms estándar Izhikevich
        steps = int(t_max / dt)
        v, u = initial_state
        
        v_trace, u_trace = [], []
        
        for step in range(steps):
            # Derivadas
            dvdt = 0.04*v**2 + 5*v + 140 - u + I
            dudt = self.a*(self.b*v - u)
            
            # Integración Euler
            v += dvdt * dt
            u += dudt * dt
            
            # Reset si spike
            if v >= 30:
                v = self.c
                u += self.d
                
            v_trace.append(v)
            u_trace.append(u)
            
        trajectory = np.column_stack([v_trace, u_trace])
        return trajectory, True
    
    def find_fixed_points(self, I):
        """Encuentra puntos fijos analíticamente con validación"""
        # Ecuación cuadrática: 0.04*v² + (5-b)*v + (140+I) = 0
        a_coef = 0.04
        b_coef = 5 - self.b
        c_coef = 140 + I
        
        discriminant = b_coef**2 - 4*a_coef*c_coef
        
        if discriminant < 0:
            return []
        
        v1 = (-b_coef + np.sqrt(discriminant)) / (2*a_coef)
        v2 = (-b_coef - np.sqrt(discriminant)) / (2*a_coef)
        
        u1 = self.b * v1
        u2 = self.b * v2
        
        # Filtrar puntos fijos físicamente razonables
        fixed_points = []
        for v_fp, u_fp in [(v1, u1), (v2, u2)]:
            if (not np.isnan(v_fp) and not np.isnan(u_fp) and 
                -100 <= v_fp <= 50 and -50 <= u_fp <= 50):
                fixed_points.append((v_fp, u_fp))
        
        return fixed_points
    
    def analyze_stability(self, v_fp, u_fp):
        """Análisis de estabilidad del punto fijo"""
        # Jacobiano en el punto fijo
        J11 = 0.08*v_fp + 5
        J12 = -1
        J21 = self.a * self.b
        J22 = -self.a
        
        trace = J11 + J22
        det = J11*J22 - J12*J21
        
        # Clasificar estabilidad
        if det < 0:
            return "saddle"
        elif det > 0 and trace < 0:
            return "stable"
        elif det > 0 and trace > 0:
            return "unstable"
        else:
            return "marginal"
    
    def phase_portrait(self, I=10, v_range=(-80, 40), u_range=(-20, 20)):
        """Diagrama de fases limpio y robusto"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Campo vectorial con grilla más espaciosa
        v = np.linspace(v_range[0], v_range[1], 16)
        u = np.linspace(u_range[0], u_range[1], 16)
        V, U = np.meshgrid(v, u)
        
        # Calcular campo vectorial
        DV = 0.04*V**2 + 5*V + 140 - U + I
        DU = self.a*(self.b*V - U)
        
        # Normalizar y filtrar vectores extremos
        M = np.sqrt(DV**2 + DU**2)
        M[M == 0] = 1
        
        # Filtrar vectores muy grandes
        mask = M < 200
        ax.quiver(V[mask], U[mask], DV[mask]/M[mask], DU[mask]/M[mask], 
                 M[mask], scale=25, alpha=0.6, cmap='viridis')
        
        # Nullclines suavizadas
        v_null = np.linspace(v_range[0], v_range[1], 500)
        u_null_v = 0.04*v_null**2 + 5*v_null + 140 + I
        u_null_u = self.b * v_null
        
        # Filtrar nullclines para evitar valores extremos
        mask_v = np.abs(u_null_v) < 100
        mask_u = np.abs(u_null_u) < 100
        
        ax.plot(v_null[mask_v], u_null_v[mask_v], 'r-', linewidth=3, 
               label='v-nullcline', alpha=0.8)
        ax.plot(v_null[mask_u], u_null_u[mask_u], 'b-', linewidth=3, 
               label='u-nullcline', alpha=0.8)
        
        # Puntos fijos con análisis de estabilidad
        fixed_points = self.find_fixed_points(I)
        for i, (v_fp, u_fp) in enumerate(fixed_points):
            if v_range[0] <= v_fp <= v_range[1] and u_range[0] <= u_fp <= u_range[1]:
                stability = self.analyze_stability(v_fp, u_fp)
                
                color_map = {'stable': 'blue', 'unstable': 'red', 
                           'saddle': 'purple', 'marginal': 'orange'}
                marker_map = {'stable': 'o', 'unstable': 's', 
                            'saddle': '^', 'marginal': 'D'}
                
                ax.plot(v_fp, u_fp, marker_map.get(stability, 'o'), 
                       markersize=12, color=color_map.get(stability, 'gray'),
                       markeredgecolor='black', markeredgewidth=2,
                       label=f'Punto fijo ({stability})')
        
        # Umbral de disparo
        ax.axvline(x=30, color='black', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Umbral (v=30)')
        
        # Trayectorias robustas
        colors = ['crimson', 'navy', 'forestgreen', 'darkorange', 'purple']
        initial_conditions = [(-70, 0), (-60, 5), (-50, -8), (-65, 12), (-55, -3)]
        
        for i, (v0, u0) in enumerate(initial_conditions):
            if (v_range[0] <= v0 <= v_range[1] and 
                u_range[0] <= u0 <= u_range[1]):
                
                trajectory, success = self.simulate_robust([v0, u0], I, t_max=50)
                
                if success and len(trajectory) > 1:
                    ax.plot(trajectory[:, 0], trajectory[:, 1], 
                           color=colors[i % len(colors)], linewidth=2.5, 
                           alpha=0.8, zorder=5)
                    ax.plot(v0, u0, 'o', color=colors[i % len(colors)], 
                           markersize=10, markeredgecolor='white', 
                           markeredgewidth=2, zorder=6)
        
        ax.set_xlabel('v (mV)', fontsize=14, fontweight='bold')
        ax.set_ylabel('u', fontsize=14, fontweight='bold')
        ax.set_title(f'Diagrama de Fases Robusto - Modelo Izhikevich\n' + 
                    f'I = {I}, a={self.a}, b={self.b}, c={self.c}, d={self.d}', 
                    fontsize=14, fontweight='bold')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(v_range)
        ax.set_ylim(u_range)
        
        plt.tight_layout()
        return fig, ax
    
    def bifurcation_diagram(self, I_range=(-5, 25), n_points=300):
        """Diagrama de bifurcaciones sin warnings"""
        print(f"🔍 Calculando bifurcaciones para I ∈ [{I_range[0]}, {I_range[1]}]...")
        
        I_values = np.linspace(I_range[0], I_range[1], n_points)
        
        stable_points = []
        unstable_points = []
        saddle_points = []
        
        # Barra de progreso simple
        for i, I in enumerate(I_values):
            if i % 50 == 0:
                progress = int(50 * i / n_points)
                bar = '█' * progress + '░' * (50 - progress)
                print(f'\r[{bar}] {100*i/n_points:.1f}%', end='', flush=True)
            
            fixed_points = self.find_fixed_points(I)
            
            for v_fp, u_fp in fixed_points:
                stability = self.analyze_stability(v_fp, u_fp)
                
                if stability == "stable":
                    stable_points.append((I, v_fp))
                elif stability == "unstable":
                    unstable_points.append((I, v_fp))
                elif stability == "saddle":
                    saddle_points.append((I, v_fp))
        
        print('\r[██████████████████████████████████████████████████] 100.0%')
        
        # Análisis de frecuencia con dt=0.5ms
        print("📊 Calculando curva F-I...")
        I_freq = np.linspace(max(0, I_range[0]), I_range[1], 25)
        frequencies = []
        
        for I in I_freq:
            # Simulación con dt=0.5ms
            dt = 0.5
            t_max = 1000  # 1 segundo
            steps = int(t_max / dt)
            v, u = -70, 0
            spike_count = 0
            
            for step in range(steps):
                dvdt = 0.04*v**2 + 5*v + 140 - u + I
                dudt = self.a*(self.b*v - u)
                
                v += dvdt * dt
                u += dudt * dt
                
                if v >= 30:
                    spike_count += 1
                    v = self.c
                    u += self.d
                    
            frequencies.append(spike_count)  # Hz
        
        # Plots mejorados
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Diagrama de bifurcaciones
        if stable_points:
            stable_points = np.array(stable_points)
            ax1.plot(stable_points[:, 0], stable_points[:, 1], 'b-', 
                    linewidth=3, label='Puntos fijos estables', alpha=0.8)
        
        if unstable_points:
            unstable_points = np.array(unstable_points)
            ax1.plot(unstable_points[:, 0], unstable_points[:, 1], 'r--', 
                    linewidth=3, label='Puntos fijos inestables', alpha=0.8)
            
        if saddle_points:
            saddle_points = np.array(saddle_points)
            ax1.plot(saddle_points[:, 0], saddle_points[:, 1], 'purple', 
                    linestyle=':', linewidth=3, label='Puntos saddle', alpha=0.8)
        
        ax1.set_xlabel('Corriente I', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Voltaje equilibrio v* (mV)', fontsize=14, fontweight='bold')
        ax1.set_title('Diagrama de Bifurcaciones', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Curva F-I mejorada
        ax2.plot(I_freq, frequencies, 'g-', linewidth=4, marker='o', 
                markersize=6, markerfacecolor='lightgreen', 
                markeredgecolor='darkgreen', markeredgewidth=2)
        ax2.fill_between(I_freq, frequencies, alpha=0.3, color='green')
        
        ax2.set_xlabel('Corriente I', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frecuencia (Hz)', fontsize=14, fontweight='bold')
        ax2.set_title('Curva F-I (Frecuencia vs Corriente)', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def compare_neuron_types(self):
        """Comparación robusta de tipos neuronales"""
        print("🧠 Comparando tipos neuronales...")
        
        neuron_types = {
            'RS': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8, 'color': 'blue', 'I': 10},
            'IB': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4, 'color': 'red', 'I': 12},
            'CH': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2, 'color': 'green', 'I': 15},
            'FS': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2, 'color': 'orange', 'I': 18},
            'LTS': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2, 'color': 'purple', 'I': 8}
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        I = 10
        
        for i, (name, params) in enumerate(neuron_types.items()):
            if i >= 5:
                break
            
            # Crear instancia temporal
            color = params.pop('color')
            I_neuron = params.pop('I')  # Corriente específica por tipo
            temp_neuron = FixedIzhikevichAnalysis(**{k: v for k, v in params.items() if k not in ['color', 'I']})
            ax = axes[i]
            
            # Campo vectorial simplificado
            v = np.linspace(-80, 35, 10)
            u = np.linspace(-15, 15, 10)
            V, U = np.meshgrid(v, u)
            
            DV = 0.04*V**2 + 5*V + 140 - U + I_neuron
            DU = temp_neuron.a*(temp_neuron.b*V - U)
            
            M = np.sqrt(DV**2 + DU**2)
            M[M == 0] = 1
            
            # Filtrar y normalizar
            mask = M < 300
            ax.quiver(V[mask], U[mask], DV[mask]/M[mask], DU[mask]/M[mask], 
                     alpha=0.5, scale=15, color='gray')
            
            # Nullclines
            v_null = np.linspace(-80, 35, 300)
            u_null_v = 0.04*v_null**2 + 5*v_null + 140 + I_neuron
            u_null_u = temp_neuron.b * v_null
            
            # Filtrar nullclines
            mask_v = np.abs(u_null_v) < 30
            mask_u = np.abs(u_null_u) < 30
            
            ax.plot(v_null[mask_v], u_null_v[mask_v], 'r-', linewidth=2, alpha=0.7)
            ax.plot(v_null[mask_u], u_null_u[mask_u], 'b-', linewidth=2, alpha=0.7)
            
            # Trayectoria con dt=0.5ms
            dt = 0.5
            t_max = 100
            steps = int(t_max / dt)
            v, u = -70, 0
            v_trace, u_trace = [], []
            
            for step in range(steps):
                dvdt = 0.04*v**2 + 5*v + 140 - u + I_neuron
                dudt = temp_neuron.a*(temp_neuron.b*v - u)
                
                v += dvdt * dt
                u += dudt * dt
                
                if v >= 30:
                    v = temp_neuron.c
                    u += temp_neuron.d
                    
                v_trace.append(v)
                u_trace.append(u)
            
            if len(v_trace) > 1:
                ax.plot(v_trace, u_trace, color=color, linewidth=3, alpha=0.9)
            
            ax.plot(-70, 0, 'o', color=color, markersize=10, 
                   markeredgecolor='black', markeredgewidth=2)
            ax.axvline(x=30, color='k', linestyle='--', alpha=0.5)
            
            # Restaurar parámetros para el título
            params['color'] = color
            params['I'] = I_neuron
            
            ax.set_title(f'{name} - {color.title()}\n' + 
                        f'a={params["a"]}, b={params["b"]}, c={params["c"]}, d={params["d"]}\nI={I_neuron}',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('v (mV)', fontweight='bold')
            ax.set_ylabel('u', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-80, 35)
            ax.set_ylim(-15, 15)
        
        # Remover subplot extra
        if len(neuron_types) < 6:
            fig.delaxes(axes[5])
        
        plt.suptitle('🧠 Comparación de Tipos Neuronales\nDiagramas de Fase Robustos', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig, axes

def main():
    """Función principal con manejo robusto"""
    print("🚀 Iniciando Análisis Robusto del Modelo de Izhikevich")
    print("=" * 60)
    
    try:
        # Crear analizador
        analyzer = FixedIzhikevichAnalysis()
        
        # 1. Diagrama de fases
        print("\n📊 Generando diagrama de fases...")
        fig1, ax1 = analyzer.phase_portrait(I=10)
        plt.show()
        print("✅ Diagrama de fases completado")
        
        # 2. Diagrama de bifurcaciones
        print("\n📈 Generando diagrama de bifurcaciones...")
        fig2, (ax2, ax3) = analyzer.bifurcation_diagram(I_range=(-3, 20))
        plt.show()
        print("✅ Diagrama de bifurcaciones completado")
        
        # 3. Comparación de tipos neuronales
        print("\n🔬 Comparando tipos neuronales...")
        fig3, axes3 = analyzer.compare_neuron_types()
        plt.show()
        print("✅ Comparación completada")
        
        print("\n🎉 ¡Análisis completado sin warnings!")
        return analyzer
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        return None

# Ejemplo de uso directo
if __name__ == "__main__":
    analyzer = main()