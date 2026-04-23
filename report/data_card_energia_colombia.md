# Data Card — Dataset Energía Eléctrica Colombia
### Aprendizaje de Máquina Aplicado · Marco Terán 2026

---

## 1. Identificación del dataset

| Campo | Descripción |
|---|---|
| **Nombre** | Dataset Energía Eléctrica Colombia 2023-2026 |
| **Versión** | 1.0 — integración de 7 fuentes XM |
| **Fecha de construcción** | Abril 2026 |
| **Período cubierto** | 01/02/2023 – 31/03/2026 |
| **Granularidad** | Diaria — una fila por día |
| **Filas totales** | 1,155 |
| **Columnas totales** | 16 (tras integración) |
| **Tipo de tarea** | Regresión — predicción del precio de bolsa |

---

## 2. Fuente de los datos

| Campo | Descripción |
|---|---|
| **Fuente original** | XM S.A. E.S.P. — Operador del Sistema Interconectado Nacional (SIN) de Colombia |
| **URL de referencia** | https://www.xm.com.co |
| **Forma de obtención** | Descarga de reportes históricos del mercado eléctrico mayorista |
| **Licencia** | Datos públicos del mercado eléctrico colombiano |
| **Archivos originales** | 7 archivos CSV individuales, integrados en un único dataset |

---

## 3. Descripción del problema

### 3.1 Pregunta de negocio

> ¿Es posible predecir el precio de bolsa de la electricidad en Colombia a partir de variables del estado diario del sistema eléctrico?

### 3.2 Motivación

El precio de bolsa es el precio marginal de la electricidad en el mercado mayorista colombiano. Su correcta estimación tiene implicaciones directas para:

- Agentes del mercado (generadores, comercializadores, grandes consumidores)
- Planeación del despacho energético
- Gestión del riesgo financiero en el sector eléctrico
- Política energética ante fenómenos climáticos como El Niño

### 3.3 Variable objetivo

`precio_bolsa_kwh` — precio de bolsa ponderado diario en $/kWh

**Transformación aplicada:** `log(precio_bolsa_kwh)` para entrenamiento, dado que la distribución original tiene skew=2.158 y presencia de valores extremos reales (máx $3,682/kWh durante El Niño 2024). Al predecir se aplica `exp()` para volver a la escala original.

---

## 4. Variables del dataset

### 4.1 Variable objetivo

| Variable | Tipo | Descripción | Rango |
|---|---|---|---|
| `precio_bolsa_kwh` | float | Precio de bolsa ponderado diario ($/kWh) | 101.90 – 3,682.63 |

### 4.2 Features originales conservadas

| Variable | Tipo | Descripción | Rango aprox. |
|---|---|---|---|
| `aportes_energia_gwh` | float | Aportes hídricos diarios al sistema en GWh | 55M – 643M |
| `precio_escasez_kwh` | float | Precio de escasez ponderado ($/kWh) — señal regulatoria | 790 – 1,192 |
| `reservas_pct` | float | Nivel de embalses como porcentaje de capacidad útil | 0.26 – 0.84 |
| `gen_hidro` | int | Generación hidroeléctrica diaria (kWh) | 78M – 214M |
| `gen_termica` | int | Generación termoeléctrica diaria (kWh) | 18M – 122M |
| `gen_solar` | int | Generación solar diaria (kWh) | 1.1M – 21.6M |
| `gen_eolica` | int | Generación eólica diaria (kWh) | 0 – 916K |
| `ratio_hidro` | float | Porcentaje de generación hídrica sobre total (%) | 39.1 – 89.7 |
| `demanda_min` | int | Demanda mínima diaria del sistema (kWh) | 6.3M – 50M |
| `demanda_pico` | int | Demanda en horas pico diaria (kWh) | 8.2M – 66M |

### 4.3 Features eliminadas por redundancia

| Variable eliminada | Razón |
|---|---|
| `gen_total_kwh` | Suma exacta de gen_hidro + gen_termica + gen_solar + gen_eolica. r=0.051 con target. |
| `demanda_kwh` | Exactamente igual a `demanda_promedio` (verificado celda a celda). |
| `demanda_max` | Correlación 1.000 con `demanda_kwh`. |
| `demanda_promedio` | Exactamente igual a `demanda_kwh`. |
| `demanda_pico` | Correlación 0.9999 con `demanda_max` pero con diferencia sistemática relacionada al efecto solar — se conserva `demanda_pico`. |

### 4.4 Features engineered

| Variable creada | Fórmula | Justificación |
|---|---|---|
| `estres_hidrico` | `gen_termica / (reservas_pct + 0.01)` | Captura el umbral crítico donde reservas bajas + alta térmica disparan el precio. Tercer coeficiente más importante en Ridge. |
| `efecto_solar_demanda` | `demanda_pico - demanda_min` | Captura el duck curve — impacto de la solar en la demanda neta. La diferencia crece con la instalación solar (r=0.54 con gen_solar). |
| `gen_renovable` | `gen_hidro + gen_solar + gen_eolica` | Generación limpia total que desplaza térmica y presiona el precio hacia abajo. |

---

## 5. Calidad de los datos

| Verificación | Resultado |
|---|---|
| Valores faltantes | **0** — ninguna columna tiene NaN en el dataset integrado |
| Filas duplicadas | **0** — ningún día repetido |
| Rango de fechas | Continuo del 01/02/2023 al 31/03/2026, sin gaps |
| Tipos de dato | Correctos: datetime para fecha, float64/int64 para numéricas |
| Outliers extremos | Presentes y verificados como datos reales (fenómeno El Niño 2024) |

---

## 6. Partición del dataset

| Conjunto | Período | Días | Proporción |
|---|---|---|---|
| **Train** | 01/02/2023 – 31/03/2025 | 790 | 68.4% |
| **Test** | 01/04/2025 – 31/03/2026 | 365 | 31.6% |

**Criterio de división:** temporal — no aleatoria. En series de tiempo, la división aleatoria introduce leakage temporal (el modelo aprendería del futuro para predecir el pasado).

**Justificación del corte en marzo 2025:**
- El train contiene el fenómeno El Niño completo (366 días) y la crisis de precios (91 días sep-nov 2024)
- El test cubre exactamente un año completo — captura estacionalidad completa
- El test representa un régimen de recuperación genuinamente distinto al train

---

## 7. Contexto de dominio relevante

### 7.1 El fenómeno El Niño 2023-2024

El período de entrenamiento incluye el fenómeno El Niño más severo de los últimos años en Colombia:

| Indicador | Valor normal | Valor durante El Niño |
|---|---|---|
| Reservas hidráulicas | ~65% | **32%** (mínimo, abril 2024) |
| Ratio generación hídrica | ~78% | **48%** (mínimo, abril 2024) |
| Precio de bolsa promedio | ~$350/kWh | **$1,544/kWh** (octubre 2024) |
| Precio máximo registrado | — | **$3,682/kWh** (30 sep 2024) |

### 7.2 Expansión solar en curso

La generación solar muestra una tendencia creciente estructural — no estacional:

| Año | Generación solar diaria promedio | Participación en matriz |
|---|---|---|
| 2023 | 3,411,587 kWh | 1.53% |
| 2024 | 9,033,202 kWh | 3.98% |
| 2025 | 11,941,276 kWh | 5.17% |
| 2026 | 16,200,948 kWh | 6.90% |

Esta tendencia implica que `gen_solar` tiene una componente no estacionaria. Se conserva como feature con la limitación documentada.

---

## 8. Limitaciones y riesgos

| Limitación | Descripción | Impacto |
|---|---|---|
| **Cambio de régimen** | Train (crisis El Niño) y test (recuperación) tienen distribuciones del target muy diferentes: media $606 vs $188/kWh | R² negativo en baseline — el modelo lineal no generaliza entre regímenes |
| **No estacionariedad de solar** | `gen_solar` crece estructuralmente por nueva capacidad instalada, no solo por clima | El modelo puede subestimar el efecto solar en predicciones futuras |
| **Sin variables climáticas** | No se incluyen temperatura, precipitación ni índices ENSO directamente | El Niño se captura indirectamente a través de reservas y aportes hídricos |
| **`demanda_pico` sin documentación** | XM no provee definición pública del cálculo exacto de esta variable | Interpretación basada en evidencia estadística (duck curve), no en documentación oficial |
| **Posible nuevo El Niño 2026** | Se anticipa un fenómeno El Niño para el segundo semestre de 2026 | El período de test podría no ser representativo de las condiciones futuras |
| **Granularidad diaria** | Los precios horarios tienen mayor variabilidad que el agregado diario | El modelo predice precios promedio diarios, no picos intradiarios |

---

## 9. Métricas de evaluación

**Métrica principal:** RMSE en escala original ($/kWh) — penaliza errores grandes, coherente con el impacto económico de una mala predicción de precio.

**Métrica secundaria:** MAE en escala original — más interpretable, menos sensible a los picos extremos del Niño.

**Métrica de ajuste:** R² en escala log — mide qué tanto explica el modelo respecto a predecir siempre la media.

**Justificación:** En mercados eléctricos, un error grande en la predicción del precio tiene consecuencias económicas desproporcionadas (contratos, coberturas, despacho). El RMSE captura esa asimetría mejor que el MAE.

---

## 10. Resultados del baseline

| Modelo | RMSE ($/kWh) | MAE ($/kWh) | R² (log) |
|---|---|---|---|
| Media del train (benchmark mínimo) | 331.5 | 320.7 | −6.82 |
| Ridge sin feature engineering | 166.9 | 157.5 | −2.24 |
| **Ridge + feature engineering** | **101.0** | **89.8** | **−0.66** |

**Interpretación:** El R² negativo refleja el cambio de régimen entre train (El Niño, precios altos) y test (recuperación, precios bajos). El modelo aprende correctamente las relaciones del sistema pero no generaliza entre regímenes tan distintos. El feature engineering reduce el RMSE en un 70% respecto al benchmark mínimo y un 40% respecto al Ridge sin engineering, confirmando que las variables derivadas (`estres_hidrico`, `efecto_solar_demanda`) capturan señal real del dominio.

**Conclusión del baseline:** el problema es difícil para un modelo lineal con un único período de entrenamiento. La Entrega 2 explorará validación por ventanas temporales y modelos no lineales (Random Forest, Gradient Boosting) que puedan capturar mejor los cambios de régimen.

---

*Dataset construido para el proyecto aplicado del curso Aprendizaje de Máquina Aplicado — EAFIT 2026*
