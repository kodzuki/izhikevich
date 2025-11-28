import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class BifurcationComparator:
    def __init__(self):
        self.dt = 0.5  # Paso temporal estándar
        
    def simulate_izhikevich(self, a, b, c, d, I, t_max=200):
        """Simulación Izhikevich con dt=0.5ms"""
        steps = int(t_max / self.dt)
        v, u = -70, 0
        v_trace, u_trace = [], []
        
        for step in range(steps):
            dvdt = 0.04*v**2 + 5*v + 140 - u + I
            dudt = a*(b*v - u)
            
            v += dvdt * self.dt
            u += dudt * self.dt
            
            if v >= 30:
                v = c
                u += d
                
            v_trace.append(v)
            u_trace.append(u)
            
        return np.array(v_trace), np.array(u_trace)
    
    def find_fixed_point(self, b, I):
        """Encuentra punto fijo para parámetros dados"""
        discriminant = (5-b)**2 - 4*0.04*(140+I)
        if discriminant < 0:
            return None
        
        v_fp = (-(5-b) + np.sqrt(discriminant)) / 0.08
        u_fp = b * v_fp
        return v_fp, u_fp
    
    def analyze_stability(self, v_fp, u_fp, a, b):
        """Análisis de estabilidad"""
        J11 = 0.08*v_fp + 5
        J12 = -1
        J21 = a*b
        J22 = -a
        
        trace = J11 + J22
        det = J11*J22 - J12*J21
        discriminant = trace**2 - 4*det
        
        return trace, det, discriminant
    
    def saddle_node_bifurcation(self):
        """Bifurcación Saddle-Node variando I"""
        print("🔍 Analizando Bifurcación Saddle-Node...")
        
        # Parámetros fijos para saddle-node
        a, b, c, d = 0.02, 0.2, -65, 8
        
        # Corriente crítica teórica
        I_critical = ((5-b)**2)/0.16 - 140
        
        # Rango alrededor de la bifurcación
        I_range = np.linspace(I_critical-2, I_critical+5, 200)
        
        stable_points = []
        unstable_points = []
        
        for I in I_range:
            fp = self.find_fixed_point(b, I)
            if fp:
                v_fp, u_fp = fp
                trace, det, _ = self.analyze_stability(v_fp, u_fp, a, b)
                
                if trace < 0 and det > 0:
                    stable_points.append((I, v_fp))
                else:
                    unstable_points.append((I, v_fp))
        
        # Plot bifurcación
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Diagrama de bifurcación
        if stable_points:
            stable_points = np.array(stable_points)
            ax1.plot(stable_points[:, 0], stable_points[:, 1], 'b-', 
                    linewidth=3, label='Estable')
        if unstable_points:
            unstable_points = np.array(unstable_points)
            ax1.plot(unstable_points[:, 0], unstable_points[:, 1], 'r--', 
                    linewidth=3, label='Inestable')
        
        ax1.axvline(I_critical, color='green', linestyle=':', linewidth=2, 
                   label=f'I crítico = {I_critical:.2f}')
        ax1.set_xlabel('Corriente I')
        ax1.set_ylabel('v* (mV)')
        ax1.set_title('Bifurcación Saddle-Node')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Retratos de fase antes/después
        I_values = [I_critical-1, I_critical+1]
        titles = ['Antes (sin punto fijo)', 'Después (con punto fijo)']
        axes_phase = [ax2, ax3]
        
        for i, (I_test, title) in enumerate(zip(I_values, titles)):
            ax = axes_phase[i]
            
            # Campo vectorial
            v = np.linspace(-80, 40, 12)
            u = np.linspace(-15, 15, 12)
            V, U = np.meshgrid(v, u)
            
            DV = 0.04*V**2 + 5*V + 140 - U + I_test
            DU = a*(b*V - U)
            M = np.sqrt(DV**2 + DU**2)
            M[M == 0] = 1
            
            ax.quiver(V, U, DV/M, DU/M, alpha=0.6, scale=20)
            
            # Nullclines
            v_null = np.linspace(-80, 40, 300)
            u_null_v = 0.04*v_null**2 + 5*v_null + 140 + I_test
            u_null_u = b * v_null
            
            ax.plot(v_null, u_null_v, 'r-', linewidth=2, label='v-nullcline')
            ax.plot(v_null, u_null_u, 'b-', linewidth=2, label='u-nullcline')
            
            # Punto fijo si existe
            fp = self.find_fixed_point(b, I_test)
            if fp:
                v_fp, u_fp = fp
                ax.plot(v_fp, u_fp, 'ko', markersize=10, markerfacecolor='yellow')
            
            # Trayectoria
            v_traj, u_traj = self.simulate_izhikevich(a, b, c, d, I_test, 100)
            ax.plot(v_traj, u_traj, 'purple', linewidth=2)
            ax.plot(-70, 0, 'go', markersize=8)
            
            ax.set_title(f'{title}\nI = {I_test:.1f}')
            ax.set_xlabel('v (mV)')
            ax.set_ylabel('u')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-80, 40)
            ax.set_ylim(-15, 15)
        
        # Serie temporal
        I_demo = I_critical + 2
        v_demo, _ = self.simulate_izhikevich(a, b, c, d, I_demo, 200)
        t_demo = np.arange(len(v_demo)) * self.dt
        
        ax4.plot(t_demo, v_demo, 'b-', linewidth=1.5)
        ax4.axhline(30, color='r', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Tiempo (ms)')
        ax4.set_ylabel('v (mV)')
        ax4.set_title(f'Serie Temporal (I = {I_demo:.1f})')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Bifurcación Saddle-Node en Modelo Izhikevich', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def hopf_bifurcation(self):
        """Bifurcación de Hopf variando b"""
        print("🔍 Analizando Bifurcación de Hopf...")
        
        # Parámetros fijos
        a, c, d, I = 0.02, -65, 8, 10
        
        # Rango de b para Hopf
        b_range = np.linspace(0.15, 0.35, 200)
        
        stable_points = []
        unstable_points = []
        hopf_points = []
        
        for b in b_range:
            fp = self.find_fixed_point(b, I)
            if fp:
                v_fp, u_fp = fp
                trace, det, _ = self.analyze_stability(v_fp, u_fp, a, b)
                
                # Detectar Hopf (trace ≈ 0, det > 0)
                if abs(trace) < 0.01 and det > 0:
                    hopf_points.append((b, v_fp))
                elif trace < 0 and det > 0:
                    stable_points.append((b, v_fp))
                else:
                    unstable_points.append((b, v_fp))
        
        # Plot bifurcación
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Diagrama de bifurcación
        if stable_points:
            stable_points = np.array(stable_points)
            ax1.plot(stable_points[:, 0], stable_points[:, 1], 'b-', 
                    linewidth=3, label='Estable')
        if unstable_points:
            unstable_points = np.array(unstable_points)
            ax1.plot(unstable_points[:, 0], unstable_points[:, 1], 'r--', 
                    linewidth=3, label='Inestable')
        if hopf_points:
            hopf_points = np.array(hopf_points)
            ax1.plot(hopf_points[:, 0], hopf_points[:, 1], 'go', 
                    markersize=8, label='Bifurcación Hopf')
            b_hopf = hopf_points[0, 0]
        else:
            b_hopf = 0.25  # Aproximación
        
        ax1.axvline(b_hopf, color='green', linestyle=':', linewidth=2, 
                   label=f'b crítico ≈ {b_hopf:.3f}')
        ax1.set_xlabel('Parámetro b')
        ax1.set_ylabel('v* (mV)')
        ax1.set_title('Bifurcación de Hopf')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Retratos de fase antes/después de Hopf
        b_values = [b_hopf-0.02, b_hopf+0.02]
        titles = ['Antes (espiral estable)', 'Después (espiral inestable)']
        axes_phase = [ax2, ax3]
        
        for i, (b_test, title) in enumerate(zip(b_values, titles)):
            ax = axes_phase[i]
            
            # Campo vectorial
            v = np.linspace(-80, 40, 12)
            u = np.linspace(-15, 15, 12)
            V, U = np.meshgrid(v, u)
            
            DV = 0.04*V**2 + 5*V + 140 - U + I
            DU = a*(b_test*V - U)
            M = np.sqrt(DV**2 + DU**2)
            M[M == 0] = 1
            
            ax.quiver(V, U, DV/M, DU/M, alpha=0.6, scale=20)
            
            # Nullclines
            v_null = np.linspace(-80, 40, 300)
            u_null_v = 0.04*v_null**2 + 5*v_null + 140 + I
            u_null_u = b_test * v_null
            
            ax.plot(v_null, u_null_v, 'r-', linewidth=2)
            ax.plot(v_null, u_null_u, 'b-', linewidth=2)
            
            # Punto fijo
            fp = self.find_fixed_point(b_test, I)
            if fp:
                v_fp, u_fp = fp
                ax.plot(v_fp, u_fp, 'ko', markersize=10, markerfacecolor='yellow')
            
            # Múltiples trayectorias para mostrar espiral
            colors = ['purple', 'orange', 'green']
            initial_conditions = [(-65, 0), (-60, 5), (-70, -3)]
            
            for j, (v0, u0) in enumerate(initial_conditions):
                v_traj, u_traj = self.simulate_izhikevich(a, b_test, c, d, I, 150)
                # Empezar desde condición inicial específica
                v_traj[0], u_traj[0] = v0, u0
                ax.plot(v_traj, u_traj, color=colors[j], linewidth=2, alpha=0.8)
                ax.plot(v0, u0, 'o', color=colors[j], markersize=6)
            
            ax.set_title(f'{title}\\nb = {b_test:.3f}')
            ax.set_xlabel('v (mV)')
            ax.set_ylabel('u')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-80, 40)
            ax.set_ylim(-15, 15)
        
        # Serie temporal mostrando oscilaciones
        b_demo = b_hopf + 0.03
        v_demo, _ = self.simulate_izhikevich(a, b_demo, c, d, I, 300)
        t_demo = np.arange(len(v_demo)) * self.dt
        
        ax4.plot(t_demo, v_demo, 'b-', linewidth=1.5)
        ax4.axhline(30, color='r', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Tiempo (ms)')
        ax4.set_ylabel('v (mV)')
        ax4.set_title(f'Oscilaciones (b = {b_demo:.3f})')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Bifurcación de Hopf en Modelo Izhikevich', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def compare_bifurcations(self):
        """Comparación directa de ambas bifurcaciones"""
        print("🔄 Comparando Bifurcaciones...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Saddle-Node
        a, b, c, d = 0.02, 0.2, -65, 8
        I_critical = ((5-b)**2)/0.16 - 140
        I_range = np.linspace(I_critical-1, I_critical+3, 100)
        
        stable_sn, unstable_sn = [], []
        for I in I_range:
            fp = self.find_fixed_point(b, I)
            if fp:
                v_fp, u_fp = fp
                trace, det, _ = self.analyze_stability(v_fp, u_fp, a, b)
                if trace < 0 and det > 0:
                    stable_sn.append((I, v_fp))
                else:
                    unstable_sn.append((I, v_fp))
        
        if stable_sn:
            stable_sn = np.array(stable_sn)
            ax1.plot(stable_sn[:, 0], stable_sn[:, 1], 'b-', linewidth=3, label='Estable')
        if unstable_sn:
            unstable_sn = np.array(unstable_sn)
            ax1.plot(unstable_sn[:, 0], unstable_sn[:, 1], 'r--', linewidth=3, label='Inestable')
        
        ax1.axvline(I_critical, color='green', linestyle=':', linewidth=2)
        ax1.set_xlabel('Corriente I')
        ax1.set_ylabel('v* (mV)')
        ax1.set_title('Saddle-Node: Aparición de Punto Fijo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Hopf
        I = 10
        b_range = np.linspace(0.18, 0.28, 100)
        
        stable_h, unstable_h, hopf_h = [], [], []
        for b_test in b_range:
            fp = self.find_fixed_point(b_test, I)
            if fp:
                v_fp, u_fp = fp
                trace, det, _ = self.analyze_stability(v_fp, u_fp, a, b_test)
                if abs(trace) < 0.005 and det > 0:
                    hopf_h.append((b_test, v_fp))
                elif trace < 0 and det > 0:
                    stable_h.append((b_test, v_fp))
                else:
                    unstable_h.append((b_test, v_fp))
        
        if stable_h:
            stable_h = np.array(stable_h)
            ax2.plot(stable_h[:, 0], stable_h[:, 1], 'b-', linewidth=3, label='Estable')
        if unstable_h:
            unstable_h = np.array(unstable_h)
            ax2.plot(unstable_h[:, 0], unstable_h[:, 1], 'r--', linewidth=3, label='Inestable')
        if hopf_h:
            hopf_h = np.array(hopf_h)
            ax2.plot(hopf_h[:, 0], hopf_h[:, 1], 'go', markersize=8, label='Hopf')
            b_hopf = hopf_h[0, 0]
            ax2.axvline(b_hopf, color='green', linestyle=':', linewidth=2)
        
        ax2.set_xlabel('Parámetro b')
        ax2.set_ylabel('v* (mV)')
        ax2.set_title('Hopf: Cambio de Estabilidad')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Series temporales comparativas
        # Saddle-Node: antes y después
        I_before = I_critical - 0.5
        I_after = I_critical + 1
        
        v_before, _ = self.simulate_izhikevich(a, b, c, d, I_before, 200)
        v_after, _ = self.simulate_izhikevich(a, b, c, d, I_after, 200)
        t = np.arange(len(v_before)) * self.dt
        
        ax3.plot(t, v_before, 'r-', linewidth=2, label=f'I={I_before:.1f} (no equilibrio)')
        ax3.plot(t, v_after, 'b-', linewidth=2, label=f'I={I_after:.1f} (con equilibrio)')
        ax3.axhline(30, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Tiempo (ms)')
        ax3.set_ylabel('v (mV)')
        ax3.set_title('Saddle-Node: Comportamiento Temporal')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Hopf: antes y después
        if hopf_h:
            b_before = b_hopf - 0.01
            b_after = b_hopf + 0.01
        else:
            b_before, b_after = 0.22, 0.26
        
        v_stable, _ = self.simulate_izhikevich(a, b_before, c, d, I, 200)
        v_unstable, _ = self.simulate_izhikevich(a, b_after, c, d, I, 200)
        
        ax4.plot(t, v_stable, 'b-', linewidth=2, label=f'b={b_before:.3f} (estable)')
        ax4.plot(t, v_unstable, 'r-', linewidth=2, label=f'b={b_after:.3f} (oscilatorio)')
        ax4.axhline(30, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Tiempo (ms)')
        ax4.set_ylabel('v (mV)')
        ax4.set_title('Hopf: Comportamiento Temporal')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Comparación: Saddle-Node vs Hopf', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

def run_bifurcation_analysis():
    """Ejecuta análisis completo de bifurcaciones"""
    comparator = BifurcationComparator()
    
    print("🚀 Análisis de Bifurcaciones en Modelo Izhikevich")
    print("=" * 50)
    
    # 1. Saddle-Node
    fig1 = comparator.saddle_node_bifurcation()
    plt.show()
    
    # 2. Hopf
    fig2 = comparator.hopf_bifurcation()
    plt.show()
    
    # 3. Comparación
    fig3 = comparator.compare_bifurcations()
    plt.show()
    
    print("✅ Análisis completado!")

if __name__ == "__main__":
    run_bifurcation_analysis()