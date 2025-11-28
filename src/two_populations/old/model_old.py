from brian2 import *
import matplotlib.pyplot as plt

import datetime  # Faltante para timestamp
import shutil    # Para mover archivos
import numpy as np  # Algunos usos no están importados

class IzhikevichNetwork:
    def __init__(self, dt_val=0.1, T_total=3000, fixed_seed=42, variable_seed=42, warmup=200, device_mode='runtime'):
        """Parámetros globales de la simulación"""
        self.dt_val = dt_val
        self.T_total = T_total
        
        self.fixed_seed_A = fixed_seed - 1
        self.variable_seed_A = variable_seed - 1
        
        self.fixed_seed_B = fixed_seed + 1
        self.variable_seed_B = variable_seed + 1
        
        self.fixed_seed_common = fixed_seed
        self.variable_seed_common = variable_seed
        
        self.device_mode = device_mode
        
        # Configurar Brian2
        defaultclock.dt = dt_val * ms
        
        # Escalado para dt
        self.noise_scale = np.sqrt(dt_val)**(-1)
        
        #self.warmup_time = warmup
        
        # Contenedores
        self.populations = {}
        self.synapses = []
        self.monitors = {}
        
        # Parámetros heterogeneidad
        # self.re = np.random.rand(800)
        # self.ri = np.random.rand(200)
        
    def create_population(self, name, Ne=800, Ni=200, k_exc=0.5, k_inh=1.0, 
                         noise_exc=5.0, noise_inh=2.0, p_intra=1.0, delay = 0,  noise_type='gaussian', step=False):
        """
        Crear una población Izhikevich
        
        Parameters:
        - name: identificador único
        - Ne, Ni: número de neuronas exc/inh
        - k_exc, k_inh: fuerza sináptica exc/inh
        - noise_exc, noise_inh: amplitud ruido talámico
        - p_intra: probabilidad conexión intra-población
        """
        
        fixed_rng_common = np.random.RandomState(self.fixed_seed_common)
        variable_rng_common = np.random.RandomState(self.variable_seed_common)
        
        fixed_rng_pop = np.random.RandomState(self.fixed_seed_A if name == 'A' else self.fixed_seed_B)
        variable_rng_pop = np.random.RandomState(self.variable_seed_A if name == 'A' else self.variable_seed_B)
        
        seed(self.fixed_seed_common)
        
        print(f"Creating population {name} using common seeds:")
        
        print(f"{self.fixed_seed_common=}, {self.variable_seed_common=}")
        
        print(f"Fixed seed for {name}: {self.fixed_seed_A if name == 'A' else self.fixed_seed_B}, \n Variable seed for {name}: {self.variable_seed_A if name == 'A' else self.variable_seed_B}")
        
        re = fixed_rng_common.rand(Ne)
        ri = fixed_rng_common.rand(Ni)
        
        # re = self.re
        # ri = self.ri
        
        a_vals = np.concatenate([0.02*np.ones(Ne), 0.02+0.08*ri])
        b_vals = np.concatenate([0.2*np.ones(Ne), 0.25-0.05*ri])
        c_vals = np.concatenate([-65+15*re**2, -65*np.ones(Ni)])
        d_vals = np.concatenate([8-6*re**2, 2*np.ones(Ni)])
        
        # Ruido talámico
        time_steps = int(self.T_total / self.dt_val)
    
        def step_profile(n, start=0.25, end=0.75, base=0.0, elev=1.0):
            prof = np.full(n, base, dtype=float); s,e = int(n*start), int(n*end)
            prof[s:e] = elev; return prof
            
            
        if noise_type == 'gaussian':
            noise_exc_vals = fixed_rng_pop.randn(time_steps, Ne) * noise_exc * self.noise_scale
            noise_inh_vals = fixed_rng_pop.randn(time_steps, Ni) * noise_inh * self.noise_scale
            
        elif noise_type == 'poisson':
            dt_ms = self.dt_val  # dt en ms
            lambda_exc = noise_exc**2 / dt_ms  # std target = 5.0
            lambda_inh = noise_inh**2 / dt_ms   # std target = 2.0
            
            # Poisson con rate proporcional a noise_exc/inh
            noise_exc_vals = fixed_rng_pop.poisson(lambda_exc, (time_steps, Ne)) - lambda_exc # /np.sqrt(lambda_exc)
            noise_inh_vals = fixed_rng_pop.poisson(lambda_inh, (time_steps, Ni)) - lambda_inh # /np.sqrt(lambda_inh)
            
            # Tren excitatorio (solo positivos)
            # noise_exc_vals = np.abs(fixed_rng.poisson(lambda_exc, (time_steps, Ne)) - lambda_exc) # /np.sqrt(lambda_exc)
            # noise_inh_vals = np.abs(fixed_rng.poisson(lambda_inh, (time_steps, Ni)) - lambda_inh) # /np.sqrt(lambda_inh)
            
        elif noise_type == 'step':
            segs = [time_steps//3, time_steps//3, time_steps - 2*(time_steps//3)]
            
            exc_parts = [np.zeros((segs[0], Ne)),
                        np.ones((segs[1], Ne))*noise_exc,
                        np.zeros((segs[2], Ne))]
            
            inh_parts = [np.zeros((segs[0], Ni)),
                        np.ones((segs[1], Ni))*noise_inh,
                        np.zeros((segs[2], Ni))]
            
            noise_exc_vals = np.concatenate(exc_parts, axis=0)
            noise_inh_vals = np.concatenate(inh_parts, axis=0)

        elif noise_type == 'none':
            
            noise_exc_vals = np.zeros((time_steps, Ne)) 
            noise_inh_vals = np.zeros((time_steps, Ni))
            
        if step and noise_type != 'none':
                
            mu_exc    = step_profile(time_steps, 0.25, 0.75, base=0.0,  elev=0.0)
            sigma_exc = step_profile(time_steps, 0.25, 0.75, base=0.8,   elev=1.1)

            mu_inh    = step_profile(time_steps, 0.25, 0.75, base=0.0, elev=0.0)
            sigma_inh = step_profile(time_steps, 0.25, 0.75, base=0.8,  elev=1.1)
            
            noise_exc_vals = sigma_exc[:,None]*noise_exc_vals # mu_exc[:,None]
            noise_inh_vals = sigma_inh[:,None]*noise_inh_vals # + mu_inh[:,None]
            
        noise_combined = np.concatenate([noise_exc_vals, noise_inh_vals], axis=1)
        stimulus = TimedArray(noise_combined, dt=self.dt_val*ms)

        # Ecuaciones neurales
        eqs = f'''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_thalamic) / ms : 1
        du/dt = a*(b*v - u) / ms : 1
        dI_syn/dt = -I_syn/tau_syn : 1  # Nuevo: decaimiento exponencial
        I_thalamic = stimulus_{name}(t, i) : 1
        tau_syn : second (constant)  # Constante de tiempo sináptica
        a : 1 (constant)
        b : 1 (constant) 
        c : 1 (constant)
        d : 1 (constant)
        '''
        
        # eqs = f'''
        # dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_thalamic) / ms : 1
        # du/dt = a*(b*v - u) / ms : 1
        # I_syn : 1
        # I_accum : 1
        # I_thalamic = stimulus_{name}(t, i) : 1
        # a : 1 (constant)
        # b : 1 (constant) 
        # c : 1 (constant)
        # d : 1 (constant)
        # '''
            
        # Crear grupo neuronal
        G = NeuronGroup(Ne + Ni, eqs, threshold='v >= 30', 
                    reset='v = c; u += d', method='heun', #heun
                    namespace={f'stimulus_{name}': stimulus})
        
        # Asignar parámetros
        G.a = a_vals
        G.b = b_vals
        G.c = c_vals
        G.d = d_vals
        G.v = -65 + 5*variable_rng_pop.randn(Ne + Ni) 
        G.u = G.b * G.v
        G.I_syn = 0
        G.tau_syn = 1.5*ms#3*ms G.tau_syn = 2*ms 
        # G.I_accum = 0
        
        # Sinapsis intra-población
        syn_intra = Synapses(G, G, 'w : 1', on_pre='I_syn_post += w') # on_pre='I_syn_post += w')

        # np.random.seed(self.variable_seed + hash("inter_connection"))
        syn_intra.connect(p=p_intra)  # Ahora usa variable_seed temporalmente
        # np.random.seed(self.fixed_seed)  # Restaurar para otras operaciones
        syn_intra.delay = delay*ms
        
        # Generar pesos solo para conexiones existentes
        n_connections = len(syn_intra.i)
        n_exc_connections = np.sum(syn_intra.i < Ne)  # Conexiones desde excitatorias
        n_inh_connections = n_connections - n_exc_connections
        
        # Asignar pesos según tipo de conexión
        weights = np.zeros(n_connections)
        weights[:n_exc_connections] = k_exc * fixed_rng_common.rand(n_exc_connections)  # Exc weights
        weights[n_exc_connections:] = -k_inh * fixed_rng_common.rand(n_inh_connections)  # Inh weights

        syn_intra.w = weights
    
        # Guardar población
        self.populations[name] = {
            'group': G,
            'Ne': Ne,
            'Ni': Ni,
            'syn_intra': syn_intra,
            'stimulus': stimulus
        }
        
        self.synapses.append(syn_intra)
        
        return G
    
    def connect_populations(self, source_name, target_name, p_inter=0.01, 
                       weight_scale=4.0, weight_dist='constant', weight_params=None,
                        delay_value=0.0, delay_dist='constant', delay_params=None):
        """
        Conectar poblaciones con distribuciones configurables
        
        Parameters:
        - weight_dist: 'constant', 'uniform', 'gaussian', 'beta'
        - weight_params: dict con parámetros de distribución
        - delay_dist: 'constant', 'uniform', 'gaussian', 'beta'  
        - delay_params: dict con parámetros de distribución
        """
        
        source = self.populations[source_name]['group']
        target = self.populations[target_name]['group']
        source_Ne = self.populations[source_name]['Ne']
        target_Ne = self.populations[target_name]['Ne']
        
        fixed_rng_common = np.random.RandomState(self.fixed_seed_common)
        variable_rng_common = np.random.RandomState(self.variable_seed_common)
        
        seed(self.fixed_seed_common)
        
        print(f"Connecting two populations using seeds: {self.fixed_seed_common=}, {self.variable_seed_common=}")
        
        # Solo conexiones excitatorias inter-población
        syn_inter = Synapses(source[:source_Ne], target, 'w : 1', 
                            on_pre='I_syn_post += w') # 'I_syn_post += w')
        
        syn_inter.connect(p=p_inter*1.25) # factor 1.25 para compensar que las conexiones solo salen de excitatorias y queremos p ~ 10 : 1 (intra/inter)
        
        # Pesos y delays
        # Generar pesos según distribución
        n_conn = len(syn_inter)
        k_base = 0.5
         
        if weight_dist == 'constant':
            weights = np.ones(n_conn)
        elif weight_dist == 'uniform':
            low, high = weight_params.get('low', 0), weight_params.get('high', 1)
            weights = fixed_rng_common.uniform(low, high, n_conn)
        
        syn_inter.w = weight_scale * k_base * weights
        
        # Generar delays según distribución
        if delay_dist == 'constant':
            delays = np.full(n_conn, delay_value)
        elif delay_dist == 'uniform':
            low, high = delay_params.get('low', 0.2), delay_params.get('high', 6.0)
            delays = variable_rng_common.uniform(low, high, n_conn)
        elif delay_dist == 'gaussian':
            mu, sigma = delay_params.get('mu', 3.0), delay_params.get('sigma', 1.0)
            delays = variable_rng_common.normal(mu, sigma, n_conn)
        elif delay_dist == 'beta':
            alpha, beta = delay_params.get('alpha', 2), delay_params.get('beta', 2)
            scale = delay_params.get('scale', 10)
            delays = scale * variable_rng_common.beta(alpha, beta, n_conn)
        
        syn_inter.delay = np.clip(delays, float(defaultclock.dt/ms), None) * ms
        
        self.synapses.append(syn_inter)
        
        return syn_inter
    
    def setup_monitors(self, population_names):
        """Configurar monitores para poblaciones especificadas"""
        
        for name in population_names:
            G = self.populations[name]['group']
            Ne = self.populations[name]['Ne']
            
            # Monitores
            spike_mon = SpikeMonitor(G)
            state_mon = StateMonitor(G, ['v', 'I_syn', 'I_thalamic'], 
                                record=range(0, min(100, Ne)))
            
            self.monitors[name] = {
                'spikes': spike_mon,
                'states': state_mon
            }
    
    def run_simulation(self):
        """Ejecutar simulación completa"""
        
        # Operación de red para aplicación sináptica
        # @network_operation(dt=1.0*ms, when='before_groups')
        # def apply_synapses():
        #     for pop_data in self.populations.values():
        #         G = pop_data['group']
        #         G.I_syn = G.I_accum
        #         G.I_accum = 0
        
        # Crear red explícita
        net_objects = []
        
        # Añadir poblaciones
        for pop_data in self.populations.values():
            net_objects.append(pop_data['group'])
        
        # Añadir sinapsis
        net_objects.extend(self.synapses)
        
        # Añadir monitores
        for mon_data in self.monitors.values():
            net_objects.extend(mon_data.values())
        
        # Añadir operación de red
        # net_objects.append(apply_synapses)
        
        # Ejecutar
        net = Network(*net_objects)
        net.run(self.T_total * ms)
        
        # En standalone, devolver self para acceder a monitores después
        if self.device_mode in ['cpp_standalone', 'cuda_standalone']:
            return self
        else:
            return self.get_results() #warmup_time=self.warmup_time
    
    def get_results(self, warmup_time=None):
        """Extraer resultados - llamar SOLO después de run() en standalone"""
        if warmup_time is None:
            warmup_time = 0 #self.warmup_time
            
        results = {}
        
        for name, monitors in self.monitors.items():
            
            # En standalone, acceder directamente a los arrays
            if self.device_mode in ['cpp_standalone', 'cuda_standalone']:
                # Acceso directo sin filtrado temporal durante simulación
                spike_times = np.array(monitors['spikes'].t / ms)
                spike_indices = np.array(monitors['spikes'].i)
                
                # Filtrar después de obtener datos
                spike_mask = spike_times >= warmup_time
                spike_times_filtered = spike_times[spike_mask]
                spike_indices_filtered = spike_indices[spike_mask]
                
                # Estados
                state_times = np.array(monitors['states'].t / ms)
                state_mask = state_times >= warmup_time
                times_filtered = state_times[state_mask]
                
                results[name] = {
                    'spike_monitor': monitors['spikes'],
                    'state_monitor': monitors['states'], 
                    # 'spike_times': spike_times_filtered,
                    # 'spike_indices': spike_indices_filtered,
                    # 'potentials': np.array(monitors['states'].v)[:, state_mask],
                    # 'I_syn': np.array(monitors['states'].I_syn)[:, state_mask],
                    # 'I_thalamic': np.array(monitors['states'].I_thalamic)[:, state_mask],
                    # 'times': times_filtered
                }
            else:
                # Runtime mode - método original
                spike_mask = monitors['spikes'].t >= warmup_time*ms
                spike_times_filtered = monitors['spikes'].t[spike_mask] / ms
                spike_indices_filtered = monitors['spikes'].i[spike_mask]
                
                state_mask = monitors['states'].t >= warmup_time*ms
                times_filtered = monitors['states'].t[state_mask] / ms
                
                results[name] = {
                    'spike_monitor': monitors['spikes'],
                    'state_monitor': monitors['states'], 
                    # 'spike_times': np.array(spike_times_filtered),
                    # 'spike_indices': np.array(spike_indices_filtered),
                    # 'potentials': np.array(monitors['states'].v)[:, state_mask],
                    # 'I_syn': np.array(monitors['states'].I_syn)[:, state_mask],
                    # 'I_thalamic': np.array(monitors['states'].I_thalamic)[:, state_mask],
                    # 'times': np.array(times_filtered)
                }
                
        if 'I_syn' in monitors['states'].record_variables:
            results[name]['I_syn'] = monitors['states'].I_syn[:, state_mask]
            results[name]['I_thalamic'] = monitors['states'].I_thalamic[:, state_mask]
                
        results['dt'] = self.dt_val
        results['T_total'] = self.T_total #- warmup_time
        
        return results

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
    
    
    
# # if __name__ == '__main__':
    
# #     from metrics import plot_connectivity_dashboard, analyze_simulation_results, plot_population_dashboard

# #     start_scope()

# #     k_factor = 5

# #     # Crear simulador
# #     sim = IzhikevichNetwork(dt_val=1.0, T_total=1200, seed_val=42)

# #     # Crear poblaciones A y B
# #     pop_A = sim.create_population('A', Ne=800, Ni=200, k_exc=k_factor*0.5, k_inh=k_factor*1.0,
# #                                     noise_exc=5, noise_inh=2, p_intra=0.1, delay = 1.0,  noise_type='poisson')

# #     pop_B = sim.create_population('B', Ne=800, Ni=200, k_exc=k_factor*0.5, k_inh=k_factor*1.0,
# #                                     noise_exc=5, noise_inh=2, p_intra=0.1, delay = 1.0, noise_type='poisson')

# #     inter_k_factor = 0.5

# #     # Conexiones bidireccionales
# #     syn_AB = sim.connect_populations('A', 'B', p_inter=0.01, 
# #                                     weight_scale=k_factor*inter_k_factor, delay_value=15) # p_inter: 0.1% hasta un 3%
# #     syn_BA = sim.connect_populations('B', 'A', p_inter=0.01, 
# #                                     weight_scale=k_factor*inter_k_factor, delay_value=15) # p_inter: 0.1% hasta un 3%

# #     # Configurar monitores
# #     sim.setup_monitors(['A', 'B'])
# #     results = sim.run_simulation()

# #     results_dict = {}

# #     # Una línea para analizar
# #     results_dict = {
# #         'baseline': analyze_simulation_results(results['A']['spike_monitor'], results['B']['spike_monitor'], 1000, "Baseline")
# #     }

# #     # Dashboard
# #     fig1 = plot_connectivity_dashboard(results_dict)  # Principal
# #     fig2 = plot_population_dashboard(results_dict)    # Detallado
        
    
# #     from metrics import print_network_statistics_table

# #     print_network_statistics_table(results, sim, 800, 200, 1200, 200)

# #     plot_raster_results(results)
            
    


# #########



# # from brian2 import *
# # import numpy as np
# # import matplotlib.pyplot as plt

# # class IzhikevichNetwork:
# #     def __init__(self, dt_val=1.0, T_total=1000, seed_val=42):
# #         """Parámetros globales de la simulación"""
# #         self.dt_val = dt_val
# #         self.T_total = T_total
# #         self.seed_val = seed_val
        
# #         # Configurar Brian2
# #         np.random.seed(seed_val)
# #         seed(seed_val)
# #         defaultclock.dt = dt_val * ms
        
# #         # Escalado para dt
# #         self.noise_scale = np.sqrt(dt_val)**(-1)
        
# #         # Contenedores
# #         self.populations = {}
# #         self.synapses = []
# #         self.monitors = {}
        
# #         # Parámetros heterogeneidad
# #         self.re = np.random.rand(800)
# #         self.ri = np.random.rand(200)
        
# #     def create_population(self, name, Ne=800, Ni=200, k_exc=0.5, k_inh=1.0, 
# #                          noise_exc=5.0, noise_inh=2.0, p_intra=1.0):
# #         """
# #         Crear una población Izhikevich
        
# #         Parameters:
# #         - name: identificador único
# #         - Ne, Ni: número de neuronas exc/inh
# #         - k_exc, k_inh: fuerza sináptica exc/inh
# #         - noise_exc, noise_inh: amplitud ruido talámico
# #         - p_intra: probabilidad conexión intra-población
# #         """
        
# #         re = self.re
# #         ri = self.ri
        
# #         a_vals = np.concatenate([0.02*np.ones(Ne), 0.02+0.08*ri])
# #         b_vals = np.concatenate([0.2*np.ones(Ne), 0.25-0.05*ri])
# #         c_vals = np.concatenate([-65+15*re**2, -65*np.ones(Ni)])
# #         d_vals = np.concatenate([8-6*re**2, 2*np.ones(Ni)])
        
# #         # Ruido talámico
# #         time_steps = int(self.T_total / self.dt_val)
# #         noise_exc_vals = np.random.randn(time_steps, Ne) * noise_exc * self.noise_scale
# #         noise_inh_vals = np.random.randn(time_steps, Ni) * noise_inh * self.noise_scale
# #         noise_combined = np.concatenate([noise_exc_vals, noise_inh_vals], axis=1)
# #         stimulus = TimedArray(noise_combined, dt=self.dt_val*ms)
        
# #         # Ecuaciones neurales
# #         eqs = f'''
# #         dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_thalamic) / ms : 1
# #         du/dt = a*(b*v - u) / ms : 1
# #         I_syn : 1
# #         I_accum : 1
# #         I_thalamic = stimulus_{name}(t, i) : 1
# #         a : 1 (constant)
# #         b : 1 (constant) 
# #         c : 1 (constant)
# #         d : 1 (constant)
# #         '''
        
# #         # Crear grupo neuronal
# #         G = NeuronGroup(Ne + Ni, eqs, threshold='v >= 30', 
# #                     reset='v = c; u += d', method='rk4',
# #                     namespace={f'stimulus_{name}': stimulus})
        
# #         # Asignar parámetros
# #         G.a = a_vals
# #         G.b = b_vals
# #         G.c = c_vals
# #         G.d = d_vals
# #         G.v = -65 + 5*np.random.randn(Ne + Ni) 
# #         G.u = G.b * G.v
# #         G.I_syn = 0
# #         G.I_accum = 0
        
# #         # Matriz de conectividad intra-población
# #         S_intra = np.concatenate([
# #             k_exc * np.random.rand(Ne + Ni, Ne),
# #             -k_inh * np.random.rand(Ne + Ni, Ni)
# #         ], axis=1)
        
# #         # Sinapsis intra-población
# #         syn_intra = Synapses(G, G, 'w : 1', on_pre='I_accum_post += w')
# #         syn_intra.connect(p=p_intra)
# #         syn_intra.w = (S_intra.T).flatten()
        
# #         # Guardar población
# #         self.populations[name] = {
# #             'group': G,
# #             'Ne': Ne,
# #             'Ni': Ni,
# #             'syn_intra': syn_intra,
# #             'stimulus': stimulus
# #         }
        
# #         self.synapses.append(syn_intra)
        
# #         return G
    
# #     def connect_populations(self, source_name, target_name, p_inter=0.01, 
# #                            weight_scale=4.0, delay_value = 0.0):
# #         """
# #         Conectar dos poblaciones con delays
        
# #         Parameters:
# #         - source_name, target_name: nombres de poblaciones
# #         - p_inter: probabilidad conexión inter-población
# #         - weight_scale: escalado de pesos inter-población
# #         - delay_range: rango de delays (min, max) en ms
# #         """
        
# #         source = self.populations[source_name]['group']
# #         target = self.populations[target_name]['group']
# #         source_Ne = self.populations[source_name]['Ne']
# #         target_Ne = self.populations[target_name]['Ne']
        
# #         # Solo conexiones excitatorias inter-población
# #         syn_inter = Synapses(source[:source_Ne], target, 'w : 1', 
# #                             on_pre='I_accum_post += w')
# #         syn_inter.connect(p=p_inter)
        
# #         # Pesos y delays
# #         k_base = 0.5  # valor base del paper
# #         syn_inter.w = weight_scale * k_base * np.random.rand(len(syn_inter.w))
        
# #         # Delays distribuidos uniformemente
# #         delays = delay_value * np.ones(len(syn_inter.w))
# #         syn_inter.delay = delays * ms
        
# #         self.synapses.append(syn_inter)
        
# #         return syn_inter
    
# #     def setup_monitors(self, population_names):
# #         """Configurar monitores para poblaciones especificadas"""
        
# #         for name in population_names:
# #             G = self.populations[name]['group']
# #             Ne = self.populations[name]['Ne']
            
# #             # Monitores
# #             spike_mon = SpikeMonitor(G)
# #             state_mon = StateMonitor(G, ['v', 'I_syn', 'I_thalamic'], 
# #                                 record=range(0, min(100, Ne)))
            
# #             self.monitors[name] = {
# #                 'spikes': spike_mon,
# #                 'states': state_mon
# #             }
    
# #     def run_simulation(self):
# #         """Ejecutar simulación completa"""
        
# #         # Operación de red para aplicación sináptica
# #         @network_operation(dt=1.0*ms, when='before_groups')
# #         def apply_synapses():
# #             for pop_data in self.populations.values():
# #                 G = pop_data['group']
# #                 G.I_syn = G.I_accum
# #                 G.I_accum = 0
        
# #         # Crear red explícita
# #         net_objects = []
        
# #         # Añadir poblaciones
# #         for pop_data in self.populations.values():
# #             net_objects.append(pop_data['group'])
        
# #         # Añadir sinapsis
# #         net_objects.extend(self.synapses)
        
# #         # Añadir monitores
# #         for mon_data in self.monitors.values():
# #             net_objects.extend(mon_data.values())
        
# #         # Añadir operación de red
# #         net_objects.append(apply_synapses)
        
# #         # Ejecutar
# #         net = Network(*net_objects)
# #         net.run(self.T_total * ms)
        
# #         return self.get_results()
    
# #     def get_results(self):
# #         """Extraer resultados de monitores"""
# #         results = {}
        
# #         for name, monitors in self.monitors.items():
# #             results[name] = {
# #                 'spike_monitor': monitors['spikes'],  # Objeto completo
# #                 'state_monitor': monitors['states'],  # Objeto completo
# #                 'spike_times': np.array(monitors['spikes'].t / ms),
# #                 'spike_indices': np.array(monitors['spikes'].i),
# #                 'potentials': np.array(monitors['states'].v),
# #                 'I_syn': np.array(monitors['states'].I_syn),
# #                 'I_thalamic': np.array(monitors['states'].I_thalamic),
# #                 'times': np.array(monitors['states'].t / ms)
# #             }
        
# #         results['dt'] = self.dt_val
# #         results['T_total'] = self.T_total
        
# #         return results
