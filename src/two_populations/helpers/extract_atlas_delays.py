import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_ftract_matrices(base_path, parcellation="Lausanne2008-60"):
    """
    Extrae las matrices de retardos y constantes sinápticas del atlas F-TRACT.
    """
    parcel_dir = os.path.join(base_path, parcellation)
    
    # Archivos objetivo (usamos las medianas poblacionales)
    files_to_extract = {
        'delays': 'dcm_axonal_delay__median.txt.gz',
        'tau_exc': 'dcm_excitatory_tc__median.txt.gz',
        'tau_inh': 'dcm_inhibitory_tc__median.txt.gz',
        'probability': 'probability.txt.gz' # Útil para filtrar conexiones espurias
    }
    
    matrices = {}
    
    for key, filename in files_to_extract.items():
        filepath = os.path.join(parcel_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Archivo no encontrado: {filepath}")
            continue
            
        print(f"Cargando {key} desde {filename}...")
        # Cargar el txt.gz como un dataframe de pandas (suele estar delimitado por espacios/tabulaciones)
        # Reemplazamos NaNs por ceros temporalmente, aunque para delays sin conexión usaremos np.inf o 0 dependiendo de Brian2
        df = pd.read_csv(filepath, sep='\s+', header=None)
        matrices[key] = df.values
        
    return matrices

def clean_and_export_delays(matrices, export_dir, prob_threshold=0.1):
    """
    Limpia la matriz de delays y la guarda para su uso en model.py.
    Filtra las conexiones con baja probabilidad de existencia.
    """
    os.makedirs(export_dir, exist_ok=True)
    
    delays = matrices['delays']
    probs = matrices['probability']
    
    # 1. Aplicar máscara de probabilidad: si la probabilidad de conexión es muy baja, 
    # asumimos que no hay vía directa (marcamos el delay como NaN)
    mask_valid = probs >= prob_threshold
    clean_delays = np.where(mask_valid, delays, np.nan)
    
    # 2. Análisis estadístico rápido (Ground Truth vs Nuestro Modelo)
    valid_delays = clean_delays[~np.isnan(clean_delays)]
    print("\n--- Estadísticas de Delays Axonales (Humanos Sanos) ---")
    print(f"Mediana global: {np.nanmedian(clean_delays):.2f} ms")
    print(f"Media global:   {np.nanmean(clean_delays):.2f} ms")
    print(f"Máximo delay:   {np.nanmax(clean_delays):.2f} ms")
    
    # 3. Extraer las medianas de las constantes sinápticas
    tau_e = np.nanmedian(matrices['tau_exc'])
    tau_i = np.nanmedian(matrices['tau_inh'])
    print(f"\n--- Constantes Sinápticas ---")
    print(f"Tau Excitatorio: {tau_e:.2f} ms")
    print(f"Tau Inhibitorio: {tau_i:.2f} ms")
    
    # 4. Guardar para Brian2
    np.save(os.path.join(export_dir, 'human_delays_matrix.npy'), clean_delays)
    
    # Guardar un diccionario con los escalares
    config = {
        'tau_exc_ms': float(tau_e),
        'tau_inh_ms': float(tau_i),
        'median_delay_ms': float(np.nanmedian(clean_delays))
    }
    pd.Series(config).to_json(os.path.join(export_dir, 'human_bio_params.json'))
    print(f"\n✅ Datos exportados a {export_dir}")

if __name__ == "__main__":
    # Ajusta estas rutas a tu estructura local
    RAW_PATH = "data/raw/f_tract_00_v2210/"
    EXPORT_PATH = "data/exports/human_connectome/"
    
    # Extraer y procesar
    mats = extract_ftract_matrices(RAW_PATH, parcellation="Lausanne2008-60")
    if mats:
        clean_and_export_delays(mats, EXPORT_PATH, prob_threshold=0.1)