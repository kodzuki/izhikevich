---

# Proyecto: Neurodelays – Influencia de los retrasos temporales en la dinámica neuronal

## 🎯 Objetivo general

Estudiar cómo diferentes **distribuciones de retrasos temporales** (delta, gaussiana, beta y realistas a partir de DTI) influyen en la **dinámica de acoplamiento** entre dos regiones cerebrales (ROIs), utilizando el modelo neuronal de **Izhikevich** y métricas de conectividad funcional.
A largo plazo, escalar a múltiples ROIs y eventualmente al **conectoma completo de ratón**.

---

## 🤝 Rol de GPT en el proyecto

GPT actúa como **colaborador de investigación** en neurofísica computacional y ayuda en:

1. **Diseño y planificación**

   * Definir experimentos y barridos de parámetros.
   * Proponer configuraciones de simulación y métricas adecuadas.

2. **Codificación y simulación**

   * Asistir en el desarrollo de simulaciones en **Python/Brian2**.
   * Modularizar notebooks en código estable (`src/`).
   * Optimizar para ejecución local o en clúster.

3. **Análisis de resultados**

   * Calcular métricas: cross-correlation, PLV, PLI, coherencia espectral, etc.
   * Sugerir visualizaciones y ayudar a interpretar resultados.

4. **Bibliografía y teoría**

   * Resumir papers relevantes.
   * Extraer ecuaciones y supuestos clave para fundamentar los experimentos.

5. **Documentación**

   * Mantener reportes claros en `reports/`.
   * Ayudar a escribir resúmenes y notas de progreso.

---

## 📂 Organización del repositorio

* `notebooks/` → exploración inicial y experimentos rápidos.
* `src/` → código estable (modelos, simulación, análisis, optimización).
* `data/` → conectomas y distribuciones de retrasos (raw + processed).
* `results/` → métricas, figuras y salidas de simulaciones.
* `configs/` → parámetros de simulación y barridos.
* `reports/` → reportes de sesión y notas teóricas.
* `archive/` → material antiguo, duplicados y pruebas.

---

## 🚀 Estado actual

* **Repositorio ordenado**: notebooks seleccionados, código modular en `src/`, resultados estructurados.
* **Módulos disponibles**:

  * `src/single_population/` → simulaciones y optimización de una población.
  * `src/two_populations/` → simulación, métricas y barridos en 2 ROIs.
  * `src/theoretical/` → análisis bifurcacional y comparaciones teóricas.
* **Notebooks activos**:

  * `initial_analysis.ipynb` (base de análisis).
  * `lisette_analysis.ipynb` (caso específico).
  * `two_izhikevich_populations*.ipynb` (core, inputs, avanzado).
* **Próximo paso**: establecer una simulación mínima (2 ROIs, retraso delta fijo) y documentar métricas básicas.

---

## 🗺️ Roadmap de fases

1. **Base mínima**

   * 2 ROIs, delays fijos (delta).
   * Calcular métricas básicas y guardar resultados.

2. **Distribuciones sintéticas**

   * Introducir gaussianas, betas y comparar con delta.
   * Ejecutar barridos de parámetros (µ, CV, fuerza de acoplamiento).

3. **Distribuciones realistas (DTI)**

   * Procesar tractografías → distribuciones de retrasos.
   * Comparar dinámica realista vs. sintética.

4. **Escalado**

   * De 2 ROIs → subredes pequeñas → conectoma completo.
   * Añadir métricas avanzadas (dPLI, Granger, Transfer Entropy).

5. **Producto final**

   * Reportes con figuras clave.
   * Notebook limpio para publicación.
   * Manuscrito con resultados.

---

## 📌 Cómo trabajar con GPT

Al inicio de cada sesión, recordar este contexto y dar una tarea concreta, por ejemplo:

* *“Hoy quiero planear los barridos de µ y CV en retrasos”*
* *“Necesito ayuda para organizar los notebooks en la carpeta `src/`”*
* *“Resúmeme un paper sobre efectos de delays en sincronía neuronal”*
* *“Analiza estas métricas y sugiere visualizaciones claras”*

GPT debe:

* Proponer opciones claras y explicar trade-offs.
* Escribir código modular y comentado (cuando se pida).
* Recordar mantener orden entre `notebooks/`, `src/`, `configs/`, `results/` y `reports/`.
* Ayudar a avanzar **por fases**, empezando simple y escalando gradualmente.

---

## 🧾 Backlog inicial

* [ ] Definir primer experimento con 2 ROIs y retraso delta fijo.
* [ ] Guardar resultados (métricas + figuras) en `results/experiments/two_populations/`.
* [ ] Preparar reporte de sesión con interpretación de métricas.
* [ ] Revisar papers base sobre delays en sincronía neuronal y resumir en `reports/`.

---

## 📚 Bibliografía inicial

* Izhikevich, E. M. (2003). *Simple model of spiking neurons*. IEEE Transactions on Neural Networks.
* Deco, G., Jirsa, V., & McIntosh, A. R. (2011). *Emerging concepts for the dynamical organization of resting-state activity in the brain*. Nature Reviews Neuroscience.
* Petkoski, S., & Jirsa, V. (2019). *Transmission time delays organize the brain network synchronization*. Philosophical Transactions of the Royal Society A.
