from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from izhikevich_model_analysis.simulation import run_izhikevich_simulation
from izhikevich_model_analysis.analysis import analysis

def parameter_sweep_2d(param1_name, param1_range, param2_name, param2_range, 
                       base_params, n_points=10):
    """
    Barrido 2D de parámetros con seguimiento de progreso
    """
    results = np.zeros((n_points, n_points, 4))  # alfa_freq, gamma_freq, alfa_power, gamma_power
    
    param1_values = np.linspace(*param1_range, n_points)
    param2_values = np.linspace(*param2_range, n_points)
    
    print(f"Barriendo {param1_name} vs {param2_name}")
    print(f"Simulaciones totales: {n_points * n_points}")
    
    for i, p1 in enumerate(param1_values):
        for j, p2 in enumerate(param2_values):
            params = base_params.copy()
            params[param1_name] = p1
            params[param2_name] = p2
            
            try:
                sim = run_izhikevich_simulation(params, duration=1000*ms)
                analysis_data = analysis(sim)
                
                if analysis_data and analysis_data['total_power'] > 0:
                    results[i,j,0] = analysis_data['alfa_peak_freq']
                    results[i,j,1] = analysis_data['gamma_peak_freq']  
                    results[i,j,2] = analysis_data['alfa_power']/analysis_data['total_power']
                    results[i,j,3] = analysis_data['gamma_power']/analysis_data['total_power']
                else:
                    results[i,j,:] = np.nan  # Simulación fallida
                    
            except Exception as e:
                print(f"Error en ({p1:.2f}, {p2:.2f}): {e}")
                results[i,j,:] = np.nan
                
        print(f"Completado: {(i+1)*n_points}/{n_points*n_points} ({100*(i+1)/n_points:.0f}%)")
    
    return results, param1_values, param2_values

def plot_sweep_results(results, param1_values, param2_values, param1_name, param2_name):
    """
    Crea heatmaps de los resultados del barrido
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Títulos y etiquetas
    titles = ['Frecuencia Pico Alfa (Hz)', 'Frecuencia Pico Gamma (Hz)', 
              'Proporción Potencia Alfa', 'Proporción Potencia Gamma']
    
    # Crear meshgrid para plots
    X, Y = np.meshgrid(param2_values, param1_values)
    
    for idx, (ax, title) in enumerate(zip(axes.flat, titles)):
        data = results[:,:,idx]
        
        # Manejar NaN values
        if np.all(np.isnan(data)):
            ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            continue
            
        # Crear heatmap
        if idx < 2:  # Frecuencias
            vmin, vmax = np.nanmin(data), np.nanmax(data)
            cmap = 'viridis'
        else:  # Proporciones 
            vmin, vmax = 0, np.nanmax(data)
            cmap = 'plasma'
            
        im = ax.imshow(data, extent=[param2_values[0], param2_values[-1], 
                                    param1_values[0], param1_values[-1]], 
                      aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Añadir contornos para mejor visualización
        if not np.all(np.isnan(data)):
            contours = ax.contour(X, Y, data, levels=5, colors='white', alpha=0.3, linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        ax.set_xlabel(param2_name.replace('_', ' ').title())
        ax.set_ylabel(param1_name.replace('_', ' ').title()) 
        ax.set_title(title)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if idx < 2:
            cbar.set_label('Hz')
        else:
            cbar.set_label('Proporción')
    
    plt.tight_layout()
    plt.show()

def plot_regime_map(results, param1_values, param2_values, param1_name, param2_name):
    """
    Mapa de regímenes dominantes (alfa, gamma, mixto, ninguno)
    """
    alfa_prop = results[:,:,2]
    gamma_prop = results[:,:,3]
    
    # Clasificar regímenes
    regime_map = np.zeros_like(alfa_prop, dtype=int)
    
    # 0: Sin actividad, 1: Alfa dominante, 2: Gamma dominante, 3: Mixto
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            if np.isnan(alfa_prop[i,j]) or np.isnan(gamma_prop[i,j]):
                regime_map[i,j] = 0  # Sin datos
            elif alfa_prop[i,j] > 0.1 and gamma_prop[i,j] > 0.1:
                if abs(alfa_prop[i,j] - gamma_prop[i,j]) < 0.05:
                    regime_map[i,j] = 3  # Mixto balanceado
                elif alfa_prop[i,j] > gamma_prop[i,j]:
                    regime_map[i,j] = 1  # Alfa dominante
                else:
                    regime_map[i,j] = 2  # Gamma dominante
            elif alfa_prop[i,j] > 0.1:
                regime_map[i,j] = 1  # Solo alfa
            elif gamma_prop[i,j] > 0.1:
                regime_map[i,j] = 2  # Solo gamma
            else:
                regime_map[i,j] = 0  # Sin oscilaciones
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = ['white', 'blue', 'red', 'purple']
    labels = ['Sin oscilaciones', 'Alfa dominante', 'Gamma dominante', 'Mixto']
    
    im = ax.imshow(regime_map, extent=[param2_values[0], param2_values[-1], 
                                     param1_values[0], param1_values[-1]], 
                  aspect='auto', origin='lower', cmap=plt.matplotlib.colors.ListedColormap(colors))
    
    ax.set_xlabel(param2_name.replace('_', ' ').title())
    ax.set_ylabel(param1_name.replace('_', ' ').title())
    ax.set_title('Mapa de Regímenes Oscilatorios')
    
    # Colorbar personalizado
    cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(labels)
    
    plt.tight_layout()
    plt.show()

