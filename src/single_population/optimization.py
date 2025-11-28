from brian2 import *
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from izhikevich_model_analysis.simulation import run_izhikevich_simulation


def objective_function(param_values, param_names, optimization_target='combined'):
    """
    Función objetivo para optimización
    
    param_values: array con valores de parámetros
    param_names: lista con nombres de parámetros
    optimization_target: 'combined', 'alfa_only', 'gamma_only'
    """
    # Convertir array a dict
    params = dict(zip(param_names, param_values))
    
    try:
        # Ejecutar simulación
        results = run_izhikevich_simulation(params, duration=3000*ms, plot=False)
        
        if results is None:  # Error en la simulación
            return 1e6
        
        # Verificar que la simulación no colapse - CRITERIOS MEJORADOS
        spike_rate_exc = results['spike_count_exc'] / (800 * 3.0)  # Hz por neurona
        spike_rate_inh = results['spike_count_inh'] / (200 * 3.0)
        
        # Rangos fisiológicos más estrictos
        if spike_rate_exc < 1.0 or spike_rate_exc > 50.0:  # Muy poca o mucha actividad exc
            return 1e6
        if spike_rate_inh < 5.0 or spike_rate_inh > 100.0:  # Muy poca o mucha actividad inh
            return 1e6
        if np.mean(results['pop_rate']) > 200:  # Tasa poblacional demasiado alta
            return 1e6
            
        alfa_proportion = results['alfa_power'] / results['total_power']
        gamma_proportion = results['gamma_power'] / results['total_power']
        
        # Penalizar si no hay picos detectables
        if results['alfa_peak_power'] < 1e-6 and optimization_target in ['combined', 'alfa_only']:
            return 1e6
        if results['gamma_peak_power'] < 1e-6 and optimization_target in ['combined', 'gamma_only']:
            return 1e6
        
        if optimization_target == 'combined':
            # Maximizar alfa + gamma con peso balanceado
            combined_score = alfa_proportion + 0.5 * gamma_proportion  # Alfa más importante
            objective = 1.0 - combined_score
        elif optimization_target == 'alfa_only':
            objective = 1.0 - alfa_proportion
        elif optimization_target == 'gamma_only':
            objective = 1.0 - gamma_proportion
        elif optimization_target == 'ratio':
            # Mantener ratio específico alfa/gamma
            target_ratio = 2.5  # Alfa debería ser ~2.5x más fuerte que gamma
            actual_ratio = alfa_proportion / (gamma_proportion + 1e-6)
            objective = abs(actual_ratio - target_ratio) + (1.0 - alfa_proportion - gamma_proportion)
            
        return objective
        
    except Exception as e:
        print(f"Error en simulación: {e}")
        return 1e6
    

def optimize_alfa_gamma(method='differential_evolution', optimization_target='combined', 
                       max_iterations=50, n_trials=1):
    """
    Optimiza parámetros para maximizar picos alfa-gamma
    
    method: 'differential_evolution', 'grid_search', 'random_search'
    optimization_target: 'combined', 'alfa_only', 'gamma_only', 'ratio'
    max_iterations: número máximo de evaluaciones
    n_trials: número de pruebas independientes
    """
    
    # Definir parámetros a optimizar y sus rangos
    param_config = {
        'input_strength_exc': (1.0, 4.0),     # Fuerza input excitatorio
        'input_strength_inh': (0.5, 2.5),     # Fuerza input inhibitorio  
        'input_rate_exc': (0.5, 3.0),         # Frecuencia input excitatorio
        'input_rate_inh': (0.5, 2.5),         # Frecuencia input inhibitorio
        'k_exc': (1.5, 4.0),                  # Constante acoplamiento excitatorio
        'k_inh': (3.0, 8.0),                  # Constante acoplamiento inhibitorio
        'delay_exc': (0.5, 3.0),              # Delay sinapsis excitatorias (ms)
        'delay_inh': (0.2, 2.0),              # Delay sinapsis inhibitorias (ms)
        'connectivity': (0.05, 0.2),          # Probabilidad conectividad
        'ei_connectivity_factor': (0.8, 2.0), # Factor conectividad E->I
        'ie_connectivity_factor': (0.8, 2.5), # Factor conectividad I->E
    }
    
    param_names = list(param_config.keys())
    bounds = list(param_config.values())
    
    print(f"=== OPTIMIZACIÓN ALFA-GAMMA ===")
    print(f"Método: {method}")
    print(f"Objetivo: {optimization_target}")
    print(f"Parámetros a optimizar: {len(param_names)}")
    print(f"Rango de evaluaciones: {max_iterations}")
    print()
    
    best_results = []
    
    for trial in range(n_trials):
        print(f"--- Prueba {trial + 1}/{n_trials} ---")
        start_time = time.time()
        
        if method == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds,
                args=(param_names, optimization_target),
                seed=42 + trial,
                maxiter=max_iterations//10,  # DE usa generaciones
                popsize=10,
                atol=1e-6,
                disp=True
            )
            best_params = dict(zip(param_names, result.x))
            best_score = result.fun
            
        elif method == 'random_search':
            best_score = float('inf')
            best_params = {}
            
            for i in range(max_iterations):
                # Generar parámetros aleatorios
                random_params = {}
                for param, (min_val, max_val) in param_config.items():
                    random_params[param] = np.random.uniform(min_val, max_val)
                
                score = objective_function(
                    list(random_params.values()), 
                    param_names, 
                    optimization_target
                )
                
                if score < best_score:
                    best_score = score
                    best_params = random_params.copy()
                    print(f"Iteración {i+1}: Nuevo mejor = {1-best_score:.4f}")
        
        elif method == 'grid_search':
            # Grid search simplificado (solo algunos parámetros clave)
            key_params = ['k_exc', 'k_inh', 'input_strength_exc', 'delay_inh']
            n_points = int(max_iterations**(1/len(key_params)))  # n^4 = max_iterations
            
            best_score = float('inf')
            best_params = {}
            
            # Usar valores por defecto para parámetros no optimizados
            default_params = {k: (v[0] + v[1])/2 for k, v in param_config.items()}
            
            iteration = 0
            for k_exc in np.linspace(*param_config['k_exc'], n_points):
                for k_inh in np.linspace(*param_config['k_inh'], n_points):
                    for input_exc in np.linspace(*param_config['input_strength_exc'], n_points):
                        for delay_inh in np.linspace(*param_config['delay_inh'], n_points):
                            test_params = default_params.copy()
                            test_params.update({
                                'k_exc': k_exc,
                                'k_inh': k_inh, 
                                'input_strength_exc': input_exc,
                                'delay_inh': delay_inh
                            })
                            
                            score = objective_function(
                                list(test_params.values()),
                                param_names,
                                optimization_target
                            )
                            
                            if score < best_score:
                                best_score = score
                                best_params = test_params.copy()
                                print(f"Grid {iteration+1}: Nuevo mejor = {1-best_score:.4f}")
                            
                            iteration += 1
                            if iteration >= max_iterations:
                                break
                        if iteration >= max_iterations:
                            break
                    if iteration >= max_iterations:
                        break
                if iteration >= max_iterations:
                    break
        
        elapsed = time.time() - start_time
        print(f"Tiempo: {elapsed:.1f}s")
        print(f"Mejor score: {1-best_score:.4f}")
        print()
        
        # Evaluar mejor resultado
        final_results = run_izhikevich_simulation(best_params, duration=3000*ms, plot=True)
        
        best_results.append({
            'params': best_params,
            'score': best_score,
            'results': final_results,
            'trial': trial
        })
    
    # Mostrar mejor resultado global
    global_best = min(best_results, key=lambda x: x['score'])
    print("=== MEJOR RESULTADO GLOBAL ===")
    print(f"Score: {1-global_best['score']:.4f}")
    print("Parámetros óptimos:")
    for param, value in global_best['params'].items():
        print(f"  {param}: {value:.3f}")
    
    results = global_best['results']
    print(f"\nMétricas espectrales:")
    print(f"  Proporción Alfa: {results['alfa_power']/results['total_power']:.3f}")
    print(f"  Proporción Gamma: {results['gamma_power']/results['total_power']:.3f}")
    print(f"  Pico Alfa: {results['alfa_peak_freq']:.1f} Hz")
    print(f"  Pico Gamma: {results['gamma_peak_freq']:.1f} Hz")
    
    return global_best