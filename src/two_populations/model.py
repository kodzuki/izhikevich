from brian2 import *
import numpy as np
    
from src.two_populations.helpers.seed_manager import SeedManager, SeedConfig, SeedMapping
from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

class IzhikevichNetwork:
    def __init__(self, dt_val=0.1, T_total=3000, seed_config=None, trial=0, **kwargs):
        """Parámetros globales de la simulación con SeedManager integrado. Inicializamos la red 
            con los parámetros base (dt , T_total, fixed_seed, variable_seed)."""
    
        start_scope()
        # Configuración básica
        self.dt_val = dt_val
        self.T_total = T_total
        
        # SeedManager
        if seed_config is None:

            fixed_seed = kwargs.get('fixed_seed', 100)
            variable_seed = kwargs.get('variable_seed', 200) 
            seed_config = SeedConfig(fixed_seed=fixed_seed, variable_seed=variable_seed)
        
        self.seed_manager = SeedManager(seed_config, trial)
        
        # Configurar Brian2
        defaultclock.dt = dt_val * ms
        
        # Escalado para dt
        self.noise_scale = np.sqrt(dt_val)**(-1)
        
        # Contenedores
        self.populations = {}
        self.synapses = []
        self.monitors = {}
        self.inter_synapses = {}  # Guardar sinapsis inter-población
        
        logger.success(f"Network initialized with seeds: {self.seed_manager.get_seed_summary()}")
    
    def update_trial(self, trial: int):
        """Actualizar seed del trial para experimentos multi-trial"""
        
        self.seed_manager.update_trial(trial)
        
        logger.info(f"Updated to trial {trial}")
    
    def create_population(self, name, Ne=800, Ni=200, k_exc=0.5, k_inh=1.0, 
                    noise_exc=5.0, noise_inh=2.0, p_intra=0.1, delay=0.0, 
                    noise_type='gaussian', step=False,
                    stim_start_ms=None, stim_duration_ms=None, 
                    stim_base=1.0, stim_elevated=None):
        """Crear población Izhikevich usando las seeds propias y compartidas:
            - Heterogeneidad neuronal (a,b,c,d,v,u): variable_seed específica de la población + común entre trials
            - Conectividad intra-población (sinapsis (p) , pesos (k)): fixed_seed específica de la población + común entre trials
            - Estímulo externo (ruido talámico): variable_seed específica de la población + variable entre trials
            - Condiciones iniciales (v,u): variable_seed específica de la población + variable entre trials
            
            name: Identificador de la población ('A' o 'B')
            Ne, Ni: Número de neuronas excitatorias e inhibitorias
            k_exc, k_inh: Escalado de pesos excitatorios e inhibitorios
            noise_exc, noise_inh: Escala de ruido talámico excitatorio e inhibitor
            p_intra: Probabilidad de conexión intra-población
            delay: Retardo sináptico intra-población (ms)
            noise_type: Tipo de ruido talámico ('gaussian', 'poisson')
            step: Si True, modula el ruido con un perfil escalón (solo si noise_type != 'none')
        """
        
        fixed_common_seed, var_common_seed = self.seed_manager.fixed_seed_common, self.seed_manager._var_base_common
        fixed_pop_seed, var_pop_seed = (self.seed_manager.fixed_seed_A, self.seed_manager._var_base_A) if 'A' in name else (self.seed_manager.fixed_seed_B , self.seed_manager._var_base_B)
        
        # RNGs simplificados via SeedManager
        fixed_rng_common = np.random.RandomState(fixed_common_seed)# Fijo entre trials + común A-B (FC)
        variable_rng_common = np.random.RandomState(var_common_seed)# Variable entre trials + común en A-B (VC) 
        variable_rng_pop = np.random.RandomState(var_pop_seed) # Variable entre trials + distinto en A-B (VD)
        fixed_rng_pop = np.random.RandomState(fixed_pop_seed) # Fijo entre trials + distinto entre A-B (FD)
        
        # Configurar Brian2 seed para conectividad - FD
        seed(fixed_pop_seed)
    
        # === PARÁMETROS NEURONALES (estructura distinta A/B - distintas ROIs) ===
        re = fixed_rng_pop.rand(Ne)
        ri = fixed_rng_pop.rand(Ni)

        # Utilizamos heterogeneidad basada en Izhikevich 2003:
            # Para Inh - fijo en c y d , uniform en a y b
            # Para Exc - fijo en a y b , uniform^2 en c y d
        a_vals = np.concatenate([0.02*np.ones(Ne), 0.02+0.08*ri])
        b_vals = np.concatenate([0.2*np.ones(Ne), 0.25-0.05*ri])
        c_vals = np.concatenate([-65+15*re**2, -65*np.ones(Ni)])
        d_vals = np.concatenate([8-6*re**2, 2*np.ones(Ni)]) # 2.5
        
        logger.debug(f"Creating population {name} with seeds: fixed_common={fixed_common_seed} / fixed_for_{name}={fixed_pop_seed} / variable_common={var_common_seed} / variable_for_{name}={var_pop_seed}")
        
            
        # === RUIDO TALÁMICO ===
            # Actividad baseline - Estímulo basal - Ruido blanco gaussiano o Poisson - Cada neurona recibe una señal de ruido independiente - Con step se introduce un escalón de ruido 
        stimulus_values = self._create_stimulus(name=name, Ne=Ne, Ni=Ni, noise_exc=noise_exc, noise_inh=noise_inh, noise_type=noise_type, step=step, 
                                                    rng=variable_rng_pop,  stim_start_ms=stim_start_ms, stim_duration_ms=stim_duration_ms,  stim_base=stim_base, stim_elevated=stim_elevated)
        
        stimulus = TimedArray(stimulus_values, dt=self.dt_val*ms)
        
        # === ECUACIONES iguales para Exc , Inh ===
            # El estímulo externo se introduce en forma de corriente I_thalamic para cada neurona i en cada paso de tiempo t
            # La corriente sináptica se modela con una variable I_syn que decae exponencialmente (tau_syn)
            # La dinámica de v y u sigue las ecuaciones estándar de Izhikevich
            # El parámetro tau_syn es constante para todas las neuronas (1.5 ms)
        eqs = f'''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_thalamic) / ms : 1 (unless refractory)
        du/dt = a*(b*v - u) / ms : 1
        dI_syn/dt = -I_syn/tau_syn : 1
        I_thalamic = stimulus_{name}(t, i) : 1
        tau_syn : second
        a : 1 (constant)
        b : 1 (constant) 
        c : 1 (constant)
        d : 1 (constant)
        '''
        
        # === DEFINIMOS NEURONGROUP - Inh y Exc juntas - Usamos método heun ===
        G = NeuronGroup(Ne + Ni, eqs, threshold='v >= 30', reset='v = c; u += d', method='heun', namespace={f'stimulus_{name}': stimulus}, refractory='4*ms') # TODO AÑADIR PERIODO REFRACTARIO , refractory='1*ms'
        
        # Asignación de parámetros y condiciones iniciales
        # Se inicializan todas las variables de la red neuronal
        # Inicializamosparámetros a,b,c,d - VC
        G.a = a_vals
        G.b = b_vals
        G.c = c_vals
        G.d = d_vals
        # Inicializamos parámetros CI (u,v) - VD
        G.v = -65 + 5*variable_rng_pop.randn(Ne + Ni)
        G.u = G.b * G.v
        # Inicializamosparámetros sinápsis
        G.I_syn = 0
        
        # === ASIGNACIÓN DIFERENCIADA DE TAU_SYN ===
        # Define constantes específicas para cada tipo
        tau_syn_exc = 1.5 * ms  # Decay para excitatorias
        tau_syn_inh = 1.5 * ms  # Decay para inhibitorias (ejemplo: más lento)

        # Asigna según índice
        G.tau_syn[:Ne] = tau_syn_exc  # Primeras Ne neuronas (excitatorias)
        G.tau_syn[Ne:] = tau_syn_inh  # Últimas Ni neuronas (inhibitorias)

        
        # === CONECTIVIDAD INTRA POP===
            # Sinapsis recurrentes (Exc->Exc, Exc->Inh, Inh->Exc, Inh->Inh) - p_intra - FD - Se actualiza la neurona postsináptica con un peso
            # Probabilidad p_intra con seed fija (FD)
            # Añadimos delays uniformes a las sinapsis intra-población
        syn_intra = Synapses(G, G, 'w : 1', on_pre='I_syn_post += w')
        syn_intra.connect(condition='i!=j', p=p_intra)
        syn_intra.delay = delay * ms
        
        # Definimos los pesos sinápticos de la red - Calculamos conexiones Exc e Inh
        n_connections = len(syn_intra.i)
        n_exc_connections = np.sum(syn_intra.i < Ne)
        n_inh_connections = n_connections - n_exc_connections
        
        # Distribuimos uniformemente los pesos Exc e Inh - FD
            # Exc - Valores positivos , peso más débil
            # Inh - Valores negativos , peso más fuerte  
        weights = np.zeros(n_connections)
        weights[:n_exc_connections] = k_exc * fixed_rng_pop.rand(n_exc_connections)
        weights[n_exc_connections:] = -k_inh * fixed_rng_pop.rand(n_inh_connections)
        syn_intra.w = weights
        
        logger.info(f"Intra-connectivity de  {name}: {n_exc_connections} Exc, {n_inh_connections} Inh synapses")
        
        # Guardar población
        self.populations[name] = {
            'group': G,
            'Ne': Ne,
            'Ni': Ni,
            'syn_intra': syn_intra,
            'stimulus': stimulus,
            'params': {'a': a_vals, 'b': b_vals, 'c': c_vals, 'd': d_vals},
            'weights': weights,
            'initial_conditions': {'v': G.v[:], 'u': G.u[:]}
        }
        
        # Guardar sinapsis
        self.synapses.append(syn_intra)
        return G
    
    def create_population2(self, name, Ne=800, Ni=200, k_exc=0.5, k_inh=1.0,
                    noise_exc=1.0, noise_inh=0.3, p_intra=0.1, delay=0.0, 
                    rate_hz=2.0, stim_start_ms=None, stim_duration_ms=None, 
                    stim_base=1.0, stim_elevated=None):
        """
        Población con PoissonInput → conductancias sinápticas (Palmigiano-style).
        Solo neuronas excitatorias reciben input talámico.
        """
        # Seeds
        fixed_common_seed, var_common_seed = self.seed_manager.fixed_seed_common, self.seed_manager._var_base_common
        fixed_pop_seed, var_pop_seed = (self.seed_manager.fixed_seed_A, self.seed_manager._var_base_A) if 'A' in name else (self.seed_manager.fixed_seed_B, self.seed_manager._var_base_B)
        
        fixed_rng_pop = np.random.RandomState(fixed_pop_seed)
        variable_rng_pop = np.random.RandomState(var_pop_seed)
        seed(fixed_pop_seed)
        
        # Parámetros neuronales (igual)
        re = fixed_rng_pop.rand(Ne)
        ri = fixed_rng_pop.rand(Ni)
        
        a_vals = np.concatenate([0.02*np.ones(Ne), 0.02+0.08*ri])
        b_vals = np.concatenate([0.2*np.ones(Ne), 0.25-0.05*ri])
        c_vals = np.concatenate([-65+15*re**2, -65*np.ones(Ni)])
        d_vals = np.concatenate([8-6*re**2, 2*np.ones(Ni)])
        
        ## Ecuaciones con conversión de unidades
        eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_thalamic) / ms : 1 (unless refractory)
        du/dt = a*(b*v - u) / ms : 1
        dI_syn/dt = -I_syn/tau_syn : 1
        dg_exc/dt = -g_exc/tau_exc : 1
        I_thalamic = g_exc : 1
        tau_syn : second
        tau_exc : second
        a : 1 (constant)
        b : 1 (constant)
        c : 1 (constant)
        d : 1 (constant)
        '''
        
        G = NeuronGroup(Ne + Ni, eqs, threshold='v >= 30', 
                        reset='v = c; u += d', method='heun', refractory='4*ms') # , refractory='4*ms'
    
        # Inicialización (igual)
        G.a = a_vals
        G.b = b_vals
        G.c = c_vals
        G.d = d_vals
        G.v = -65 + 5*variable_rng_pop.randn(Ne + Ni)
        G.u = G.b * G.v
        G.I_syn = 0
        
        G.g_exc = 0.0  # adimensional ahora
        G.tau_syn = 1.5 * ms
        G.tau_exc = 3.0 * ms  # Decay input (Palmigiano)
        
        # Conectividad (igual)
        syn_intra = Synapses(G, G, 'w : 1', on_pre='I_syn_post += w')
        syn_intra.connect(condition='i!=j', p=p_intra) 
        # syn_intra.connect(condition='i!=j', p=p_intra) #TODO Evitar conexiones recurrentes? Evitar conexiones recíprocas?
        syn_intra.delay = delay * ms
        
        n_connections = len(syn_intra.i)
        n_exc_connections = np.sum(syn_intra.i < Ne)
        n_inh_connections = n_connections - n_exc_connections
        
        weights = np.zeros(n_connections)
        weights[:n_exc_connections] = k_exc * fixed_rng_pop.rand(n_exc_connections)
        weights[n_exc_connections:] = -k_inh * fixed_rng_pop.rand(n_inh_connections)
        syn_intra.w = weights
        
        seed(var_pop_seed)
        # PoissonInput talámico
        thalamic = self._create_stimulus2(name, G, Ne, Ni, noise_exc, noise_inh, rate_hz, stim_start_ms, stim_duration_ms, stim_base, stim_elevated) 
        seed(fixed_pop_seed)
        
        logger.info(f"Population {name} (PoissonInput): {Ne}E/{Ni}I, "
                    f"{n_exc_connections}+{n_inh_connections} syn")
        
        self.populations[name] = {
            'group': G,
            'Ne': Ne,
            'Ni': Ni,
            'syn_intra': syn_intra,
            'thalamic': thalamic
        }
        
        self.synapses.append(syn_intra)
        return G
    
      
    def _create_stimulus2(self, name, neurons, Ne, Ni, noise_exc, noise_inh, rate_hz,
                    stim_start_ms=None, stim_duration_ms=None, 
                    stim_base=1.0, stim_elevated=None):
        from brian2 import PoissonGroup, Synapses, TimedArray
        
        N_sources = 100
        
        # En _create_stimulus2, ANTES de crear PoissonGroup:
        logger.info(f"Creating Poisson for {name} with seed state: {np.random.get_state()[1][0]}")
        
        time_steps = int(self.T_total / self.dt_val)
        
        # Rate profile
        if stim_elevated is None:
            stim_elevated = stim_base
        
        start_idx = int(stim_start_ms / self.dt_val) if stim_start_ms else int(time_steps * 0.25)
        end_idx = start_idx + int(stim_duration_ms / self.dt_val) if stim_duration_ms else int(time_steps * 0.75)
        
        rate_profile = np.full(time_steps, stim_base)
        rate_profile[start_idx:end_idx] = stim_elevated
        
        # fixed_pop_seed, var_pop_seed = (self.seed_manager.fixed_seed_A, self.seed_manager._var_base_A) if 'A' in name else (self.seed_manager.fixed_seed_B, self.seed_manager._var_base_B)
        # variable_rng_pop = np.random.RandomState(var_pop_seed)
        
        #JITTER para decorrelación (usa seed actual de np.random)
        rate_profile *= (1 + 0.2 * np.random.randn(time_steps))  # 3%
        rate_profile = np.clip(rate_profile, 0, None)
        
        # print(f"rate profile {rate_profile[:10]}")
        
        rate_array_exc = TimedArray(rate_profile * rate_hz * Hz, dt=self.dt_val*ms, 
                                name=f'rate_{name}_exc')
        rate_array_inh = TimedArray(rate_profile * rate_hz * Hz, dt=self.dt_val*ms,
                                    name=f'rate_{name}_inh')
        
        # N_sources × N_neurons fuentes independientes
        poisson_exc = PoissonGroup(N_sources * Ne, rates='rate_array(t)', 
                            namespace={'rate_array': rate_array_exc})
        poisson_inh = PoissonGroup(N_sources * Ni, rates='rate_array(t)', 
                                namespace={'rate_array': rate_array_inh})
        
                # N_sources × N_neurons fuentes independientes
        # poisson_exc = PoissonGroup(N_sources * Ne, rates='rate_array(t)', 
        #                         namespace={f'rate_array_{name}_exc': rate_array})
        # poisson_inh = PoissonGroup(N_sources * Ni, rates='rate_array(t)', 
        #                         namespace={f'rate_array_{name}_inh': rate_array})
        
        # Cada neurona recibe de SUS 1000 fuentes
        syn_exc = Synapses(poisson_exc, neurons[:Ne], 'w : 1', on_pre='g_exc_post += w')
        syn_exc.connect(j='i // N_sources')
        syn_exc.w = noise_exc
        
        syn_inh = Synapses(poisson_inh, neurons[Ne:], 'w : 1', on_pre='g_exc_post += w')
        syn_inh.connect(j='i // N_sources')
        syn_inh.w = noise_inh
        
        # # PoissonGroup (una sola para compartir rate)
        # poisson = PoissonGroup(N_sources, rates='rate_array(t)', 
        #                     namespace={'rate_array': rate_array})
        
        # # Conectar TODAS las fuentes a TODAS las neuronas (replicar PoissonInput)
        # syn_exc = Synapses(poisson, neurons[:Ne], 'w : 1', on_pre='g_exc_post += w')
        # syn_exc.connect()  # Totalmente conectado: N_sources × Ne conexiones
        # syn_exc.w = noise_exc
        
        # syn_inh = Synapses(poisson, neurons[Ne:], 'w : 1', on_pre='g_exc_post += w')
        # syn_inh.connect()
        # syn_inh.w = noise_inh
        
        logger.info(f"PoissonGroup {name}: {N_sources} sources @ {rate_hz}Hz → "
                    f"{N_sources*Ne} exc + {N_sources*Ni} inh synapses")
        
        return [poisson_exc, poisson_inh, syn_exc, syn_inh]
    
    
    # def _create_stimulus2(self, name, neurons, Ne, noise_exc, noise_inh, rate_hz):
    #     """PoissonInput → conductancias sinápticas"""
    #     from brian2 import PoissonInput
        
    #     N_sources = 1000 
    #     rate_per_source = rate_hz*Hz #2.005 * Hz
    #     weight = noise_exc
            
    #     thalamic_exc = PoissonInput(
    #         target=neurons[:Ne],
    #         target_var='g_exc',
    #         N=N_sources,
    #         rate=rate_per_source,
    #         weight=weight
    #     )
        
    #     weight = noise_inh #weight*0.4
            
    #     thalamic_inh = PoissonInput(
    #         target=neurons[Ne:],
    #         target_var='g_exc',
    #         N=N_sources,
    #         rate=rate_per_source,
    #         weight=weight
    #     )
        
    #     logger.info(f"PoissonInput {name}: {N_sources} sources @ {rate_per_source}")
        
    #     return [thalamic_exc, thalamic_inh]   # Se añade automáticamente a la red mágica
    
    def _create_stimulus(self, name, Ne, Ni, noise_exc, noise_inh, noise_type='none', 
                    step=False, rng=None, stim_start_ms=None, stim_duration_ms=None, 
                    stim_base=1.0, stim_elevated=None):
        """
        Estímulo talámico con modulación temporal flexible.
        
        Args:
            stim_base: Factor multiplicativo baseline (default 1.0 = ruido normal)
            stim_elevated: Factor durante ventana temporal (si None, usa stim_base)
            stim_start_ms, stim_duration_ms: Ventana de modulación
        
        Ejemplos:
            base=0, elevated=3.0 → silencio + pulso de 3x
            base=1.0, elevated=3.0 → ruido constante + pulso elevado
            base=0.5, elevated=2.0 → ruido reducido + pulso fuerte
        """
        
        time_steps = int(self.T_total / self.dt_val)
        
        # Crear ruido base
        if noise_type == 'gaussian':
            noise_exc_vals = rng.randn(time_steps, Ne) * noise_exc * self.noise_scale
            noise_inh_vals = rng.randn(time_steps, Ni) * noise_inh * self.noise_scale
        elif noise_type == 'poisson':
            lambda_exc = noise_exc**2 / self.dt_val
            lambda_inh = noise_inh**2 / self.dt_val
            noise_exc_vals = rng.poisson(lambda_exc, (time_steps, Ne)) - lambda_exc
            noise_inh_vals = rng.poisson(lambda_inh, (time_steps, Ni)) - lambda_inh
        else:
            noise_exc_vals = np.zeros((time_steps, Ne))
            noise_inh_vals = np.zeros((time_steps, Ni))
            
        logger.info(f"[DEBUG] step={step}, noise_type={noise_type}")
        
        # Aplicar modulación temporal
        if step and noise_type != 'none':
            
            logger.info(f"[DEBUG] ENTRANDO EN MODULACIÓN")
            logger.info(f"[DEBUG] {stim_base=}")
            logger.info(f"[DEBUG] {stim_elevated=}")
            logger.info(f"[DEBUG] {stim_start_ms=}")
            logger.info(f"[DEBUG] {stim_duration_ms=}")
            
            if stim_elevated is None:
                stim_elevated = stim_base
            
            # Calcular ventana
            if stim_start_ms is None:
                start_idx = int(time_steps * 0.25)
            else:
                start_idx = int(stim_start_ms / self.dt_val)
            
            if stim_duration_ms is None:
                end_idx = int(time_steps * 0.75)
            else:
                end_idx = start_idx + int(stim_duration_ms / self.dt_val)
            
            logger.info(f"[DEBUG] {stim_elevated=}")
            
            # Perfil: base fuera, elevated dentro
            profile = np.full(time_steps, stim_base)
            profile[start_idx:end_idx] = stim_elevated
            
            logger.info(f"[DEBUG] {max(profile)}")
            
            noise_exc_vals = profile[:, None] * noise_exc_vals
            noise_inh_vals = profile[:, None] * noise_inh_vals
            
            logger.info(f"Stim {name}: base={stim_base:.1f}x, elevated={stim_elevated:.1f}x "
                        f"at {stim_start_ms}ms for {stim_duration_ms}ms")
        
        return np.concatenate([noise_exc_vals, noise_inh_vals], axis=1)
    
    def connect_populations(self, source_name, target_name, p_inter=0.0, weight_scale=0.0, weight_dist='constant', delay_value=0.0, delay_dist='constant', delay_params=None):
        """Conectar poblaciones source -> target con sinapsis excitatorias (E->E, E->I).
            Usamos las seeds propias y compartidas:
            - Conectividad inter-población (p): fixed_seed común entre poblaciones + común entre trials (FC)
            - Pesos sinápticos (k): fixed_seed común entre poblaciones + común entre trials (FC)
            - Retardos sinápticos (delay): variable_seed común entre poblaciones + variable entre trials (VC)"""
            
        fixed_common_seed, var_common_seed = self.seed_manager.fixed_seed_common, self.seed_manager._var_base_common
        fixed_pop_seed, var_pop_seed = (self.seed_manager.fixed_seed_A, self.seed_manager._var_base_A) if 'A' in source_name else (self.seed_manager.fixed_seed_B , self.seed_manager._var_base_B)
        
        fixed_rng_common = np.random.RandomState(fixed_common_seed)# Fijo entre trials + común A-B (FC)
        variable_rng_common = np.random.RandomState(var_common_seed)# Variable entre trials + común en A-B (VC) 

        source = self.populations[source_name]['group']
        target = self.populations[target_name]['group']
        source_Ne = self.populations[source_name]['Ne'] 
        
        seed(fixed_common_seed)
        
        logger.info(f"Connecting {source_name}->{target_name} using seeds: " f"fixed={fixed_common_seed}, variable(current)={var_common_seed}")

        # Solo conexiones excitatorias inter-población: E -> E , E -> I
        # La neurona postsináptica se actualiza con un peso
        # Compensamos la baja probabilidad de conexión con un factor 1.25
        syn_inter = Synapses(source[:source_Ne], target, 'w : 1', on_pre='I_syn_post += w')
        syn_inter.connect(condition='i!=j', p=p_inter * 1.25)
        
        # Comprobamos las conexiones creadas
        n_conn = len(syn_inter)
        if n_conn == 0:
            logger.error(f"WARNING: No connections created between {source_name} and {target_name}")
            return None
        
        # Definimos el peso base para las conexiones excitatorias
        k_base = 0.5
        
        # Comprobamos los parámetros de pesos y delays
        if delay_params is None:
            delay_params = {}
        
        logger.info(f"Distribución de pesos y conectividad inter: {weight_dist=} , {k_base=} , {weight_scale=},  {p_inter=}, {n_conn=}")
        
        # === DISTRIBUCIÓN DE PESOS INTER-POBLACION ===
        if weight_dist == 'constant':
            weights = np.ones(n_conn)
        elif weight_dist == 'uniform':
            weights = fixed_rng_common.rand(n_conn)
        else:
            weights = np.ones(n_conn)
        
        # Multipilicamos: factor base * factor de escala (inter/intra) * distribución
        syn_inter.w = weight_scale * k_base * weights
        
        logger.info(f"Distribución de delays inter: {delay_dist=}")
        
        # === DELAYS (distribución configurable no negativa) ===
        # Podemos elegir delays distribuidos de forma constante, uniforme, beta, gamma, lognormal o weibull
        if delay_dist == 'constant':
            delays = np.full(n_conn, delay_value)
            
        elif delay_dist == 'uniform':
            low, high = delay_params.get('low', 0.2), delay_params.get('high', 6.0)
            delays = variable_rng_common.uniform(low, high, n_conn)

        elif delay_dist == 'gamma':
            shape, scale = delay_params.get('shape', 3.5), delay_params.get('scale', 1.0)
            delays = variable_rng_common.gamma(shape, scale, n_conn)
            
        elif delay_dist == 'lognormal':
            mean, sigma = delay_params.get('alpha', 1.0), delay_params.get('beta', 0.6)
            delays = variable_rng_common.lognormal(mean, sigma, n_conn)
        
        elif delay_dist == 'beta':
            alpha, beta = delay_params.get('alpha', 2), delay_params.get('beta', 2)
            scale = delay_params.get('scale', 10)
            delays = scale * variable_rng_common.beta(alpha, beta, n_conn)
            
        # elif delay_dist == 'weibull':
        #     a, scale = delay_params.get('a', 1.8), delay_params.get('scale', 3.0)
        #     delays = scale*variable_rng_common.weibull(a, n_conn)
            
        else:
            delays = np.full(n_conn, delay_value)
        
        # CLIPPING crítico para estabilidad # TODO REVISAR
        syn_inter.delay = np.clip(delays, float(defaultclock.dt/ms), None) * ms
        
        # Guardamos las sinápsis inter
        self.synapses.append(syn_inter)
        self.inter_synapses[f"{source_name}{target_name}"] = syn_inter
        
        return syn_inter
    
    def setup_monitors(self, population_names, record_v_dt=0.5, sample_fraction=0.5, monitor_conductance=False):
        for name in population_names:
            if name in self.populations:
                G = self.populations[name]['group']
                Ne = self.populations[name]['Ne']
                Ni = self.populations[name]['Ni']
                
                fixed_pop_seed = self.seed_manager.fixed_seed_A if 'A' in name else self.seed_manager.fixed_seed_B
                rng_sample = np.random.RandomState(fixed_pop_seed)
                
                spike_mon = SpikeMonitor(G)
                current_mon = StateMonitor(G, ['I_syn', 'I_thalamic'], record=range(0, min(100, Ne)))
                
                # QUITAR ESTAS DOS LÍNEAS:
                # g_mon = StateMonitor(G, 'g_exc', record=[0, 1, 2], dt=0.1*ms)
                # self.monitors[name]['g_exc_debug'] = g_mon
                
                n_exc = int(Ne * sample_fraction)
                n_inh = int(Ni * sample_fraction)
                sample_exc = rng_sample.choice(Ne, n_exc, replace=False)
                sample_inh = Ne + rng_sample.choice(Ni, n_inh, replace=False)
                sample_indices = np.concatenate([sample_exc, sample_inh])
                
                v_mon = StateMonitor(G, 'v', record=sample_indices, dt=record_v_dt*ms)
                
                # Crear dict
                self.monitors[name] = {
                    'spikes': spike_mon,
                    'currents': current_mon,
                    'voltage': v_mon,
                    'v_sample_indices': sample_indices,
                    'v_n_exc_sampled': n_exc
                }
                
                # Añadir g_exc
                if monitor_conductance:
                    g_mon = StateMonitor(G, 'g_exc', record=[0,1,2], dt=0.1*ms)
                    self.monitors[name]['g_exc_debug'] = g_mon
        
    def run_simulation(self):
        net_objects = []
        
        # Poblaciones
        for pop_data in self.populations.values():
            net_objects.append(pop_data['group'])
            
            # AÑADIR POISSONINPUT
            if 'thalamic' in pop_data:
                if isinstance(pop_data['thalamic'], list):
                    net_objects.extend(pop_data['thalamic'])
                else:
                    net_objects.append(pop_data['thalamic'])
        
        # Sinapsis (tanto intra como inter)
        net_objects.extend(self.synapses)
        
        # Monitores
        for mon_data in self.monitors.values():
            for key, obj in mon_data.items():
                if not key.startswith('v_'):
                    net_objects.append(obj)
        
        net = Network(*net_objects)
        net.run(self.T_total * ms)
        return self.get_results()
    
    def get_results(self):
        results = {}

        for name, monitors in self.monitors.items():
            results[name] = {
                'spike_monitor': monitors['spikes'],
                'state_monitor': monitors['currents'],
                'voltage_monitor': monitors['voltage'],
                'v_sample_indices': monitors['v_sample_indices'],
                'v_n_exc_sampled': monitors['v_n_exc_sampled'],
                'spike_times': np.array(monitors['spikes'].t / ms),
                'spike_indices': np.array(monitors['spikes'].i),
                'times': np.array(monitors['currents'].t / ms),
                'I_syn': np.array(monitors['currents'].I_syn),
                'I_thalamic': np.array(monitors['currents'].I_thalamic),
                'potentials': np.array(monitors['voltage'].v),
                'potentials_times': np.array(monitors['voltage'].t / ms)
            }
            
            # Añadir g_exc si existe
            if 'g_exc_debug' in monitors:
                results[name]['g_exc'] = np.array(monitors['g_exc_debug'].g_exc)
                results[name]['g_exc_times'] = np.array(monitors['g_exc_debug'].t / ms)
                
        results['dt'] = self.dt_val
        results['T_total'] = self.T_total
        results['seed_summary'] = self.seed_manager.get_seed_summary()
        
        for conn_name, syn in self.inter_synapses.items():
            results[f'delays_{conn_name}'] = np.array(syn.delay/ms)
        
        return results
    
    def get_network_info(self):
        """Información del estado de la red"""
        info = {
            'populations': list(self.populations.keys()),
            'total_synapses': len(self.synapses),
            'seed_config': self.seed_manager.get_seed_summary(),
            'validation': self.seed_manager.validate_configuration()
        }
        return info