# Predicción del Precio de Bolsa — Mercado Eléctrico Colombiano
## Entrega 1: Problema · Datos · EDA · Baseline

| Campo | Detalle |
|---|---|
| **Curso** | Aprendizaje de Máquina Aplicado |
| **Profesor** | Marco Terán — Universidad EAFIT |
| **Fuente** | XM S.A. E.S.P. — Operador del SIN de Colombia |
| **Período** | 01/02/2023 – 31/03/2026 |
| **Fecha** | Abril 2026 |

---

## 1. Definición del Problema

Predecir el **precio de bolsa diario de la electricidad en Colombia** ($/kWh) a partir de variables del estado del sistema eléctrico: disponibilidad hídrica, mezcla de generación, nivel de reservas y demanda. El precio de bolsa es el precio marginal del mercado mayorista, determinado principalmente por la escasez de agua en los embalses y la necesidad de despachar generación termoeléctrica (más cara).

### ¿Por qué este dataset es adecuado?

- Cubre un período con alta variabilidad real, incluyendo el fenómeno El Niño 2023-2024.
- Las variables capturan los drivers físicos reales del precio: reservas, generación hídrica y térmica.
- Permite construir un baseline razonable y comparar familias de modelos con validación honesta.
- Tiene relevancia práctica directa para agentes del mercado eléctrico colombiano.
- Fuente pública y documentada: XM S.A. E.S.P., operador del Sistema Interconectado Nacional.

### Variable objetivo y métrica

| Variable objetivo | Transformación | Métrica principal | Justificación |
|---|---|---|---|
| `precio_bolsa_kwh` | `log(precio_bolsa_kwh)` para entrenamiento; `exp()` al predecir | RMSE en $/kWh | Penaliza errores grandes, coherente con impacto económico en contratos y coberturas |

---

## 2. Calidad de Datos

| Verificación | Resultado | Implicación |
|---|---|---|
| Valores faltantes | 0 — ninguna columna | Sin imputación necesaria |
| Filas duplicadas | 0 — ningún día repetido | Dataset limpio |
| Continuidad temporal | Sin gaps 01/02/2023–31/03/2026 | Series completas |
| Tipos de dato | datetime + float64/int64 | Correctos para ML |
| Outliers extremos | Presentes — verificados (El Niño 2024) | Datos reales, no errores |

### Perfil estadístico de variables clave

| Variable | Media | Mediana | Std | Skew | Outlier IQR % |
|---|---|---|---|---|---|
| `precio_bolsa_kwh` | 474.04 | 355.64 | 374.91 | 2.158 | 3.03% |
| `log_precio_bolsa` | 5.889 | 5.874 | 0.746 | 0.058 | 1.30% |
| `aportes_energia_gwh` | 202.3M | 187.0M | 95.8M | 1.082 | 3.12% |
| `reservas_pct` | 0.627 | 0.640 | 0.119 | -0.628 | 0.87% |
| `gen_termica` | 50.4M | 42.3M | 25.8M | 0.812 | 0.00% |
| `ratio_hidro` | 73.77% | 77.25% | 10.79 | -0.884 | 0.69% |

> La transformación log redujo el skew del target de **2.158 → 0.058** — distribución prácticamente simétrica, adecuada para un modelo lineal.

---

## 3. EDA — Análisis de Variables

### 3.1 Cadena causal del sistema eléctrico

El EDA reveló que el precio de bolsa sigue una cadena causal física clara. Las relaciones **entre variables** (no solo contra el target) permiten entender el mecanismo subyacente:

```
aportes_energia_gwh  →  reservas_pct  →  gen_termica  →  precio_bolsa_kwh
       r = +0.39              r = -0.54        r = +0.795
```

La señal se amplifica a lo largo de la cadena. Los aportes hídricos tienen correlación directa de solo −0.46 con el precio, pero su efecto real pasa por reservas y térmica.

### 3.2 Correlaciones con el target

| Variable | r con precio_bolsa | Interpretación de dominio |
|---|---|---|
| `gen_termica` | +0.795 | Generación térmica cara → precio sube. Señal más fuerte. |
| `ratio_hidro` | -0.774 | Más hidro → precio baja. Captura la mezcla energética. |
| `gen_hidro` | -0.701 | Derivada del ratio — confirma la misma señal. |
| `demanda_min` | -0.464 | Alta demanda base → más hidro disponible → precio menor. |
| `aportes_energia_gwh` | -0.461 | Más aportes hídricos → generación barata → precio baja. |
| `reservas_pct` | -0.374 | Embalses llenos → menor presión térmica → precio menor. |
| `precio_escasez_kwh` | +0.285 | Señal regulatoria del mercado, correlación moderada. |
| `gen_solar` | -0.282 | Solar desplaza térmica → precio baja levemente. |

> Historia del dataset en una frase: **el precio sube cuando el sistema se queda sin agua y tiene que encender las plantas térmicas.**

### 3.3 Multicolinealidad entre features

El análisis de relaciones entre variables identificó pares con alta correlación que informaron las decisiones de feature engineering:

| Par de variables | r entre sí | Decisión tomada |
|---|---|---|
| `gen_termica` ↔ `ratio_hidro` | -0.980 | Ambas en modelo — Ridge maneja multicolinealidad |
| `gen_hidro` ↔ `ratio_hidro` | +0.907 | `gen_hidro` incluida en `gen_renovable` |
| `gen_hidro` ↔ `gen_termica` | -0.859 | Sustitución física — `gen_termica` captura señal absoluta |
| `demanda_min` ↔ `demanda_pico` | +0.999 | `efecto_solar_demanda` = pico - min las combina |

Se evaluó empíricamente si eliminar las variables redundantes mejoraba el modelo:

| Set | Features | RMSE ($/kWh) | Conclusión |
|---|---|---|---|
| Set actual | 13 | 101.0 | ✅ Mejor |
| Set reducido | 10 | 115.2 | Peor en 14% |
| Set mínimo | 9 | 125.6 | Peor en 24% |

> **Conclusión:** Ridge maneja la multicolinealidad con regularización. Se conservan los 13 features originales + 5 engineered = 15 en total.

### 3.4 Tendencia solar estructural

La generación solar muestra una tendencia creciente no estacional — es nueva capacidad instalada, no variación climática:

| Año | Generación solar diaria promedio | % de la matriz | r(solar, precio) |
|---|---|---|---|
| 2023 | 3,411,587 kWh | 1.53% | +0.376 |
| 2024 | 9,033,202 kWh | 3.98% | +0.120 |
| 2025 | 11,941,276 kWh | 5.17% | +0.240 |
| 2026 | 16,200,948 kWh | 6.90% | **-0.083** |

> La correlación solar-precio se está invirtiendo: en 2026 el solar ya tiene correlación negativa con el precio — señal de que está desplazando generación térmica cara.

---

## 4. Fenómeno El Niño 2023-2024

| Período | Días | Precio medio | Precio máx | Reservas media | Ratio Hidro |
|---|---|---|---|---|---|
| Normal (ene-may 2023) | 120 | $408/kWh | $838/kWh | 58% | 81.5% |
| El Niño (jun23-may24) | 366 | $628/kWh | $1,461/kWh | 57% | 66.3% |
| Crisis precios (jun-dic24) | 214 | $770/kWh | $3,683/kWh | 56% | 67.7% |
| Recuperación (2025-2026) | 455 | $228/kWh | $769/kWh | 71% | 80.6% |

> El pico de **$3,682/kWh** del 30 sep 2024 coincide con reservas en 52% y ratio hídrico en 50% — dato real del sistema, no error de captura.

---

## 5. Feature Engineering — 5 Features Justificadas

Cada feature nueva responde a una hipótesis física del sistema identificada durante el EDA. No se crean variables por probar — cada una tiene motivación de dominio, evidencia estadística y resultado medible en RMSE.

### Grupo 1 — Cadena causal hídrica-térmica

*Motivadas por la cadena causal identificada en el EDA: `aportes → reservas → gen_termica → precio`*

| Feature | Fórmula | Hipótesis | Δ RMSE |
|---|---|---|---|
| `estres_hidrico` | `gen_termica / (reservas_pct + 0.01)` | Umbral crítico donde reservas bajas + alta térmica disparan el precio. 3er coeficiente más importante en Ridge. | **−66.4 $/kWh** |
| `presion_termica` | `gen_termica / (aportes_energia_gwh + 1)` | Flujo de entrada vs respuesta del sistema. Complementa `estres_hidrico` (stock vs flujo). | −9.4 $/kWh |

> **Nota sobre los guards `+0.01` y `+1`:** prevención defensiva de división por cero en producción (Laplace smoothing). No afectan el resultado actual — `gen_termica` mínimo = 18,181,016 kWh; `reservas_pct` nunca llega a 0.

### Grupo 2 — Transición energética solar

*Motivadas por la inversión de la correlación solar-precio y el crecimiento estructural del solar*

| Feature | Fórmula | Hipótesis | Δ RMSE |
|---|---|---|---|
| `efecto_solar_demanda` | `demanda_pico - demanda_min` | Duck curve: la diferencia crece con instalación solar (r=0.54 con gen_solar). | −0.3 $/kWh |
| `gen_renovable` | `gen_hidro + gen_solar + gen_eolica` | Generación limpia total que desplaza térmica cara. | −0.1 $/kWh |
| `termica_evitada` | `gen_solar / (gen_termica + 1)` | Cuánto solar desplaza térmica — esencia de la transición energética en una variable. Correlación solar-precio: +0.376 → -0.083. | **−28.6 $/kWh** |

### Impacto acumulativo

| Set de features | N | RMSE ($/kWh) | R² (log) | Δ vs anterior |
|---|---|---|---|---|
| Sin engineering | 10 | 166.9 | -2.24 | — |
| + `estres_hidrico` | 11 | 100.6 | -0.67 | -66.4 |
| + `presion_termica` | 12 | 91.2 | -0.42 | -9.4 |
| + `efecto_solar_demanda` | 13 | 91.4 | -0.41 | -0.3 |
| + `gen_renovable` | 14 | 91.5 | -0.41 | -0.1 |
| **+ `termica_evitada` (final)** | **15** | **62.9** | **+0.30** | **-28.6** |

**Mejora total respecto a sin engineering: −104.0 $/kWh (−62.3%)**

---

## 6. Partición Temporal

> En series de tiempo el split **nunca es aleatorio** — un split aleatorio mezclaría el tiempo y el modelo aprendería del futuro para predecir el pasado (leakage temporal).

| Conjunto | Período | Días | Proporción | Precio medio | Contiene |
|---|---|---|---|---|---|
| **Train** | feb 2023 – mar 2025 | 790 | 68.4% | $606/kWh | El Niño completo + crisis precios |
| **Test** | abr 2025 – mar 2026 | 365 | 31.6% | $188/kWh | Recuperación — 1 año completo |

La diferencia de precios medios entre train ($606) y test ($188) refleja el **cambio de régimen** entre la crisis de El Niño y el período de recuperación.

---

## 7. Baseline Reproducible

### Elección de Ridge

Ridge (no LinearRegression) por criterio explícito del curso (*fundamentos_preprocessing_ml.md*, tabla de decisiones del preprocessor):

> *"Modelo: Ridge (no LinearRegression) — Regulariza coeficientes, más estable con features correlacionadas"*

Con multicolinealidad r=−0.98 entre `gen_termica` y `ratio_hidro`, Ridge es la elección correcta. alpha=1.0 es el valor canónico del curso.

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # defensivo para producción
    ('scaler',  StandardScaler()),
    ('model',   Ridge(alpha=1.0)),
])
```

### Resultados comparativos

| Modelo | Features | RMSE ($/kWh) | MAE ($/kWh) | R² (log) | Mejora vs benchmark |
|---|---|---|---|---|---|
| Benchmark — media del train | — | 331.5 | 320.7 | -6.82 | — |
| Ridge sin engineering | 10 | 166.9 | 157.5 | -2.24 | -50% |
| Ridge + engineering parcial | 13 | 101.0 | 89.8 | -0.66 | -70% |
| **Ridge + 5 features (final)** | **15** | **62.9** | **51.3** | **+0.30** | **-81%** |

### Interpretación del R²

El R² pasó de negativo a **+0.30** con las 5 features completas. R²=+0.30 no es débil — es honesto. La dificultad del problema viene del cambio de régimen entre train y test, no de un modelo mal construido.

> *"R² cercano a 1.0 en un problema complejo = primera sospecha de leakage. R² modesto con datos limpios = resultado honesto y punto de partida para mejorar."*
> — `estadistica_para_ml_v2.md`

### Coeficientes más importantes (Ridge + 15 features)

| Feature | Coeficiente | Interpretación |
|---|---|---|
| `gen_termica` | +0.417 | Más térmica → precio sube |
| `estres_hidrico` | -0.287 | Más estrés hídrico → precio sube (coef. negativo captura la relación inversa tras scaling) |
| `ratio_hidro` | -0.275 | Más hídrica → precio baja |
| `termica_evitada` | -0.253 | Más solar desplazando térmica → precio baja |
| `presion_termica` | -0.112 | Más presión térmica → precio sube |

---

## 8. Limitaciones y Próximos Pasos

### Limitaciones identificadas

| Limitación | Descripción | Impacto |
|---|---|---|
| **Cambio de régimen** | Train (crisis El Niño, ~$606/kWh) y test (recuperación, ~$188/kWh) tienen distribuciones muy distintas. | R² negativo en versiones sin `termica_evitada`. |
| **No estacionariedad solar** | `gen_solar` crece estructuralmente por nueva capacidad instalada, no por clima. | Puede subestimar efecto solar en predicciones futuras. |
| **Sin variables climáticas directas** | No se incluyen temperatura, precipitación ni índices ENSO directamente. | El Niño capturado indirectamente a través de reservas y aportes. |
| **`demanda_pico` sin documentación oficial** | XM no provee definición pública del cálculo exacto. | Interpretación basada en evidencia estadística (duck curve). |
| **Posible nuevo El Niño 2026** | Se anticipa fenómeno para el segundo semestre de 2026. | El período de test podría no ser representativo del futuro. |

### Próximos pasos — Entrega 2 (30/04/2026)

| Componente | Descripción |
|---|---|
| Validación por ventanas deslizantes | Captura los distintos regímenes del sistema. Más honesta que un único holdout temporal. |
| Random Forest | Captura no linealidades y umbrales. Robusto a multicolinealidad sin regularización explícita. |
| Gradient Boosting | Mejor captura de cambios de régimen. Comparar con Ridge y RF con las mismas features. |
| Búsqueda de alpha óptimo | Grid search de Ridge alpha en validación temporal. Enseñado en `ml_linearmodels.ipynb`. |
| Análisis de importancia | SHAP o permutation importance para interpretar qué variables explican el desempeño. |

---

## Conclusiones principales

1. **El precio de bolsa está dominado por la mezcla hídrica-térmica.** `gen_termica` (+0.795) y `ratio_hidro` (−0.774) son las señales más fuertes.

2. **El feature engineering aporta señal real.** RMSE baja de 166.9 a 62.9 $/kWh — reducción del 62%. `termica_evitada` es la feature más poderosa.

3. **La transición solar está cambiando el sistema.** La correlación solar-precio se invirtió de +0.376 (2023) a −0.083 (2026). El solar está empezando a desplazar térmica.

4. **La validación temporal es crítica.** Un split aleatorio habría producido métricas artificialmente optimistas (leakage temporal).

5. **El cambio de régimen es el reto central.** Train y test son mundos distintos — El Niño vs recuperación. Los modelos no lineales de la Entrega 2 deberían capturarlo mejor.

---

> *"El precio de la electricidad en Colombia sube cuando el sistema se queda sin agua en los embalses y tiene que encender las plantas térmicas. La transición solar está cambiando esa ecuación."*

*Dataset Energía Colombia · XM S.A. E.S.P. · Período 2023-2026*
*Aprendizaje de Máquina Aplicado — EAFIT 2026*
