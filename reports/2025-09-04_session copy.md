# 🧠 Sesión de Análisis de Conectividad

**Fecha:** 2025-09-04
**Condición:** `AtoB_delta4ms`

---

## ✅ Cambios principales implementados

* Corrección en `complete_analysis` para cortar y alinear señales (`t0_ms` y `view_ms`) evitando offsets en time series y raster plots.
* Normalización y refinamiento de `cross_correlation_analysis` con ajuste parabólico sub-bin.
* Ajuste de `power_spectrum` (Welch) con resolución configurable y cálculo robusto de potencia en bandas.
* Refactorización de `phase_locking_value` para garantizar la restricción matemática `PLI ≤ PLV`.
* Limpieza y validación de `spectral_coherence` (suavizado gaussiano, recorte a <100 Hz y limitación a 0.95).
* Ajustes en `intrinsic_timescale`: cálculo robusto de τ mediante integración hasta cruce con `exp(-1)` y clasificación de calidad (`good`, `moderate`, `poor`, `very_poor`).
* Dashboards actualizados:

  * **Connectivity Dashboard**: métricas principales (cross-corr, PLV/PLI, coherencia, INT).
  * **Population Dashboard**: autocorrelaciones, PSD, series temporales (2000 ms post-corte), raster (1000 ms post-corte), y potencias Alpha/Gamma por población.

---

## 📊 Resultados principales (condición `AtoB_delta4ms`)

* **Cross-correlation peak:** `0.915` at `6.1 ms`
* **PLV / PLI**:

  * Alpha → PLV = `0.929`, PLI = `0.809`
  * Gamma → PLV = `0.628`, PLI = `0.628`
* **Spectral coherence:** peak `0.950` at `4.1 Hz`
* **Intrinsic timescales:**

  * Pop A: `7.7 ms` (**moderate**)
  * Pop B: `7.3 ms` (**poor**)

---

## 📈 Observaciones a destacar

* **Alta sincronía** entre poblaciones: cross-corr >0.9 con desfase consistente (\~6 ms).
* **Coherencia robusta** en baja frecuencia (<10 Hz), con pico claro en \~4 Hz.
* **PLV vs PLI**: Alpha muestra fuerte acoplamiento de fase (PLV \~0.93) pero con PLI reducido (\~0.81), sugiriendo contribuciones de volumen común.
* **Timescales bajos (\~7 ms)** → actividad poblacional rápida, sin integraciones largas; Pop B más inestable (`poor`).
* **Raster y tasas poblacionales** confirman descargas periódicas y oscilaciones gamma/alpha.

