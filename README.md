---

# 🧠 Neurodelays

Proyecto de investigación en **neurofísica computacional**:
**Influencia de los retrasos temporales en la dinámica de acoplamiento neuronal.**

---

## 🎯 Objetivo

Estudiar cómo distintas **distribuciones de retrasos temporales** (delta, gaussiana, beta y distribuciones realistas derivadas de DTI) afectan la **sincronización y conectividad funcional** entre regiones cerebrales, usando:

* Modelo neuronal de **Izhikevich** (implementado en Brian2).
* Simulaciones de redes poblacionales (inicialmente 2 ROIs).
* Métricas de conectividad funcional (cross-correlation, PLV, PLI, coherencia espectral).

📈 A largo plazo: escalar a subredes y finalmente al **conectoma completo de ratón**.

---

## 📂 Estructura del repositorio

```
PROJECT_CONTEXT.md   # descripción detallada del proyecto y roadmap
README.md            # este archivo

notebooks/           # exploración y experimentos iniciales
├─ data_analysis/    # análisis preliminar y casos específicos
├─ tutorials/        # notebooks didácticos (Brian2, Izhikevich simple)
└─ two_populations/  # simulaciones y análisis de 2 ROIs

src/                 # código estable
├─ single_population # simulaciones, análisis y optimización de 1 población
├─ two_populations   # modelo, métricas y barridos de 2 ROIs
└─ theoretical       # análisis teóricos (bifurcaciones, comparaciones)

data/                # conectomas y distribuciones de retrasos
├─ raw/              # datos originales (DTI, ROI)
└─ processed/        # distribuciones derivadas

results/             # salidas de simulación
├─ experiments/      # resultados organizados por experimento
└─ figures/          # figuras destacadas

configs/             # parámetros de simulación y barridos
reports/             # reportes de sesión y notas teóricas
archive/             # material antiguo, copias, pruebas
```

---

## 🚀 Estado actual

* Notebooks principales:

  * `initial_analysis.ipynb` → análisis preliminar.
  * `lisette_analysis.ipynb` → caso específico.
  * `two_izhikevich_populations*.ipynb` → núcleo de simulaciones con 2 ROIs.
* Módulos estables:

  * `src/single_population/` → base de una población con barridos y optimización.
  * `src/two_populations/` → simulaciones de dos poblaciones acopladas.

Próximos pasos inmediatos:

1. Consolidar simulaciones con retraso delta fijo.
2. Ampliar a distribuciones gaussiana y beta.
3. Integrar distribuciones realistas derivadas de DTI.

---

## ⚙️ Dependencias principales

* [Brian2](https://brian2.readthedocs.io)
* NumPy, SciPy, pandas
* matplotlib, seaborn
* PyYAML (para configs)
