#!/bin/bash

source ~/.bashrc

conda init
conda activate neurophysics

# --- FIX PARA QUE ENCUENTRE 'src' ---
# Añadimos la raíz del proyecto al path de Python
# Asumiendo que estás en notebooks/two_populations y la raíz está 2 arriba:
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/../..

echo "=== INICIO DEL JOB ==="
echo "Nodo: $(hostname)"
echo "Ruta: $(pwd)"
echo "Cores asignados: $SLURM_CPUS_PER_TASK"

# Crear carpeta de los resultados si no existe
python -u sweep_3d_autocorrelation2.py

