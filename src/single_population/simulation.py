from brian2 import *
import numpy as np

def run_izhikevich_simulation(params, duration=3000*ms):
    """
    Ejecuta simulación del modelo Izhikevich con parámetros dados
    
    params: dict con parámetros a optimizar
    duration: duración de la simulación  
    
    Returns: dict con resultados del espectro
    """
    start_scope()
    np.random.seed(42)  # Para reproducibilidad
    defaultclock.dt = 0.01*ms  # Timestep más grande para mejor eficiencia
    
    N_exc = 800
    N_inh = 200
    N_total = N_exc + N_inh
    
    # Ecuaciones del modelo
    equations = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_thalamic)/ms : 1
    du/dt = a*(b*v - u)/ms : 1 
    I_syn : 1
    I_thalamic: 1
    a : 1
    b : 1  
    c : 1
    d : 1
    '''
    
    # Crear grupos neuronales
    exc_neurons = NeuronGroup(N_exc, equations, 
                              threshold='v >= 30',
                              reset='v = c; u += d',
                              method='euler')
    
    inh_neurons = NeuronGroup(N_inh, equations,
                              threshold='v >= 30', 
                              reset='v = c; u += d',
                              method='euler')
    
    # Input talámico parametrizado
    input_strength_exc = params.get('input_strength_exc', 2.0)
    input_strength_inh = params.get('input_strength_inh', 1.0)
    input_rate_exc = params.get('input_rate_exc', 1.5)
    input_rate_inh = params.get('input_rate_inh', 1.0)
    input_update_freq = params.get('input_update_freq', 1.0)  # en ms
    
    @network_operation(dt=input_update_freq*ms)
    def update_input():
        exc_neurons.I_thalamic = input_strength_exc * np.random.poisson(input_rate_exc, N_exc)
        inh_neurons.I_thalamic = input_strength_inh * np.random.poisson(input_rate_inh, N_inh)
    
    # Heterogeneidad neuronal
    r_exc = np.random.rand(N_exc)
    r_inh = np.random.rand(N_inh)
    
    # Parámetros neuronales
    exc_neurons.a = 0.02
    exc_neurons.b = 0.2
    exc_neurons.c = -65 + 15 * r_exc**2
    exc_neurons.d = 8 - 6 * r_exc**2
    
    inh_neurons.a = 0.02 + 0.08 * r_inh
    inh_neurons.b = 0.25 - 0.05 * r_inh
    inh_neurons.c = -65
    inh_neurons.d = 2
    
    # Condiciones iniciales
    exc_neurons.v = -65 + 10*np.random.randn(N_exc)
    exc_neurons.u = exc_neurons.b * exc_neurons.v
    inh_neurons.v = -65 + 10*np.random.randn(N_inh)
    inh_neurons.u = inh_neurons.b * inh_neurons.v
    
    # Reset sináptico
    exc_neurons.run_regularly('I_syn = 0', when='before_synapses', dt=1.0*ms)
    inh_neurons.run_regularly('I_syn = 0', when='before_synapses', dt=1.0*ms)
    
    # Conectividad sináptica parametrizada
    k_exc = params.get('k_exc', 5.0)
    k_inh = params.get('k_inh', 5.0)
    connectivity = params.get('connectivity', 0.1)
    
    # Delays parametrizados
    delay_exc = params.get('delay_exc', 1.0) * ms
    delay_inh = params.get('delay_inh', 1.0) * ms
    
    # Sinapsis excitatorias
    syn_ee = Synapses(exc_neurons, exc_neurons, 'w : 1', on_pre='I_syn += w')
    syn_ee.connect(p=connectivity)
    syn_ee.w = f'{k_exc} * rand()'
    syn_ee.delay = delay_exc
    
    syn_ei = Synapses(exc_neurons, inh_neurons, 'w : 1', on_pre='I_syn += w')
    syn_ei.connect(p=connectivity * params.get('ei_connectivity_factor', 1.0))
    syn_ei.w = f'{k_exc} * rand()'
    syn_ei.delay = delay_exc
    
    # Sinapsis inhibitorias
    syn_ie = Synapses(inh_neurons, exc_neurons, 'w : 1', on_pre='I_syn += w')
    syn_ie.connect(p=connectivity * params.get('ie_connectivity_factor', 1.0))
    syn_ie.w = f'-{k_inh} * rand()'
    syn_ie.delay = delay_inh
    
    syn_ii = Synapses(inh_neurons, inh_neurons, 'w : 1', on_pre='I_syn += w')
    syn_ii.connect(p=connectivity * params.get('ii_connectivity_factor', 1.0))
    syn_ii.w = f'-{k_inh} * rand()'
    syn_ii.delay = delay_inh
    
    # Monitores
    spike_mon_exc = SpikeMonitor(exc_neurons)
    spike_mon_inh = SpikeMonitor(inh_neurons)
    pop_mon_exc = PopulationRateMonitor(exc_neurons)
    
    # Ejecutar simulación
    run(duration)
    
    return {
        'spike_mon_exc': spike_mon_exc,
        'spike_mon_inh': spike_mon_inh,
        'pop_mon_exc': pop_mon_exc,
        'synapses': {
            'syn_ee': syn_ee,
            'syn_ei': syn_ei, 
            'syn_ie': syn_ie,
            'syn_ii': syn_ii
        },
        'params': {
            'N_exc': N_exc,
            'N_inh': N_inh,
            'N_total': N_total,
            'connectivity': connectivity,
            'duration': duration
        }
    }