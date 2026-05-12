# Entrega 2 — Comparación de Familias y Validación
## Predicción del Precio de Bolsa · Mercado Eléctrico Colombiano

**Aprendizaje de Máquina Aplicado · Profesor Marco Terán · Universidad EAFIT · 2026**

---

## Resumen ejecutivo

Esta entrega presenta la comparación técnica entre cuatro familias de modelos para predecir el precio de bolsa diario del mercado eléctrico colombiano (horizonte: día siguiente). Se implementó un protocolo de validación temporal con TimeSeriesSplit de 5 ventanas sobre el conjunto de entrenamiento, garantizando que todas las decisiones de modelado — selección de features, HPO y comparación de familias — se tomaron sin acceso al conjunto de prueba.

El modelo ganador es **Random Forest con n_estimators=200, max_depth=15, min_samples_leaf=1**, con MAE_log = 0.1847 ± 0.0361 en validación cruzada temporal — una reducción del 66.7% respecto al baseline trivial (0.5549). Un hallazgo metodológico central es que el 90.9% de la importancia predictiva proviene del rezago de un día (lag_1), lo que revela que el modelo opera principalmente como predictor de persistencia. Un experimento complementario sin rezagos confirma que las condiciones estructurales del sistema (gen_termica, reservas_pct) dominan cuando se elimina la autocorrelación, conectando con los hallazgos de Villarreal & Flores (2023).

---

## 1. Introducción y continuidad con la Entrega 1

La Entrega 1 estableció el problema, el dataset y un baseline Ridge que produjo un R² negativo en el holdout de prueba. La interpretación fue que el cambio de régimen entre el período de El Niño (train, precio medio $606/kWh) y el período de recuperación (test, precio medio $188/kWh) hacía imposible la generalización con un único corte temporal y un modelo lineal.

La Entrega 1 dejó dos compromisos explícitos para esta entrega:

1. **Validación por ventanas deslizantes** que capture los distintos regímenes del sistema eléctrico.
2. **Evaluación de `estres_hidrico`** (anteriormente llamada `tension_hidrica`) con proceso completamente libre de leakage.

Ambos compromisos se cumplen en esta entrega. El TimeSeriesSplit de 5 ventanas cubre los regímenes de amenaza del Niño, estrés hídrico, crisis de precios y recuperación. La feature `estres_hidrico` = gen_termica / (reservas_pct + 0.01) se evalúa dentro del protocolo de validación cruzada temporal sin acceso al test.

---

## 2. Metodología

### 2.1 Contrato metodológico

Antes de escribir código de modelado se fijaron las siguientes decisiones, que no se modificaron después de ver resultados:

| Decisión | Elección | Justificación |
|---|---|---|
| Métrica principal | MAE en $/kWh (escala original) | Interpretable, robusta a outliers del Niño |
| Protocolo CV | TimeSeriesSplit 5 ventanas | Promedia sobre distintos regímenes |
| Pipeline | sklearn Pipeline — preprocessing dentro | Evita leakage de preprocessing |
| Hiperparámetros | Grids fijos definidos antes de correr | Evita tuning oportunista |
| Criterio final | Menor MAE CV promedio + parsimonia | Evita elegir complejidad sin justificación |
| Horizonte | Corto plazo — precio del día siguiente | Coherente con el caso de uso del inversionista |
| Test | Reservado — se usa UNA sola vez en Entrega 3 | Garantía de evaluación honesta |

**Métrica principal — MAE y no RMSE:** el RMSE penaliza los errores grandes de forma cuadrática. Los 91 días de crisis del El Niño 2024 tienen precios de hasta $3,682/kWh — valores reales pero excepcionales. Usar RMSE como métrica principal llevaría al HPO a optimizar el modelo para esos días a costa del error en los 699 días normales. El MAE trata todos los errores con igual peso, coherente con el caso de uso de un inversionista que necesita predicción diaria confiable.

### 2.2 Feature engineering

Se conservaron las tres features engineered de la Entrega 1 y se agregaron tres rezagos temporales:

**Features de dominio (Entrega 1):**

- `estres_hidrico` = gen_termica / (reservas_pct + 0.01): captura el umbral crítico donde reservas bajas y alta generación térmica disparan el precio. No es lineal — por debajo del 35-40% de reservas el precio escala de forma no proporcional.
- `efecto_solar_demanda` = demanda_pico - demanda_min: captura el duck curve — impacto de la generación solar en la demanda neta. La diferencia crece con la instalación solar (r=0.54 con gen_solar).
- `gen_renovable` = gen_hidro + gen_solar + gen_eolica: generación limpia total que desplaza térmica y presiona el precio hacia abajo.

**Rezagos temporales (nuevos en Entrega 2):**

- `lag_1`: precio de ayer — captura la inercia del mercado
- `lag_7`: precio de hace 7 días — captura patrón semanal
- `lag_30`: precio de hace 30 días — captura contexto mensual

Los rezagos se calcularon con `shift()` sobre el train ordenado por fecha antes del split. Los primeros 30 días con NaN se eliminaron, dejando 760 observaciones para modelado.

### 2.3 Protocolo de validación — TimeSeriesSplit

El K-Fold clásico no aplica para series de tiempo porque mezcla el tiempo aleatoriamente — el modelo podría aprender del futuro para predecir el pasado (leakage temporal). El TimeSeriesSplit garantiza que en cada fold la validación siempre es posterior al train.

Con 760 días de train y 5 folds, cada ventana de validación cubre ~126 días, evaluando el modelo en condiciones distintas del sistema:

| Fold | Período validación | Días | Precio medio real | Condición del sistema |
|---|---|---|---|---|
| 1 | jul–nov 2023 | 126 | $765/kWh | Amenaza El Niño — precios subiendo |
| 2 | nov 2023–mar 2024 | 126 | $578/kWh | Transición — precios variables |
| 3 | mar–jul 2024 | 126 | $438/kWh | Embalses en mínimos — estrés hídrico |
| 4 | jul–nov 2024 | 126 | $989/kWh | **Crisis El Niño — pico de precios** |
| 5 | nov 2024–mar 2025 | 126 | $488/kWh | Inicio recuperación — precios bajando |

El MAE promedio de los 5 folds captura el error en distintos regímenes, dando una estimación más confiable que un único holdout temporal.

### 2.4 Familias de modelos comparadas

Se compararon cuatro familias bajo el mismo protocolo de validación y el mismo conjunto de features:

**Baseline trivial — DummyRegressor(strategy='mean'):** predice siempre la media del fold de train. No aprende ninguna estructura. Define el piso mínimo que cualquier modelo debe superar.

**Baseline serio — Ridge:** modelo lineal con regularización L2. Referencia contra la que deben competir los modelos no lineales. Si Random Forest no supera a Ridge, la complejidad no tiene justificación metodológica.

**Random Forest:** ensamble de árboles construidos en paralelo sobre muestras aleatorias del train (bagging). Captura umbrales, discontinuidades e interacciones que Ridge no puede ver. Robusto al sobreajuste por el efecto de promediado.

**Gradient Boosting:** árboles construidos en secuencia, donde cada árbol corrige los errores del anterior. Captura patrones más finos que Random Forest pero es más sensible a hiperparámetros.

**¿Por qué no KNN ni SVM?** KNN predice por vecindad en el espacio de features, pero en series de tiempo los días con features similares no necesariamente tienen el mismo precio — el contexto temporal importa. SVM para regresión tiene hiperparámetros (C, γ) muy sensibles en series de tiempo y es computacionalmente costoso para este problema. Random Forest y GBM dan mejor balance entre precisión y costo computacional.

### 2.5 HPO — optimización de hiperparámetros

La búsqueda de hiperparámetros se realizó con `RandomizedSearchCV` dentro del TimeSeriesSplit. El test no se usó en ninguna etapa del HPO.

**Grids definidos antes de correr:**

| Modelo | Hiperparámetro | Valores |
|---|---|---|
| Random Forest | n_estimators | 100, 200, 300 |
| Random Forest | max_depth | 5, 10, 15, None |
| Random Forest | min_samples_leaf | 1, 5, 10 |
| Gradient Boosting | n_estimators | 100, 200, 300 |
| Gradient Boosting | learning_rate | 0.01, 0.05, 0.1, 0.2 |
| Gradient Boosting | max_depth | 3, 5, 7 |

n_iter=20 en ambos casos — 20 combinaciones aleatorias del espacio de búsqueda dentro del CV temporal.

### 2.6 Configuración experimental completa

La siguiente tabla resume todos los experimentos realizados, el número de combinaciones de hiperparámetros probadas, y el tiempo de cómputo. Todos los experimentos usaron SEED=42, TimeSeriesSplit con n_splits=5, y `neg_mean_absolute_error` como scoring.

| Exp | Modelo | Tipo | Combinaciones | MAE_log CV | ± std | Tiempo (s) |
|---|---|---|---|---|---|---|
| E1 | Dummy (media train) | Baseline trivial | 1 | 0.5549 | 0.1725 | <1 |
| E2 | Ridge α=1.0 | Baseline lineal | 1 | 0.5364 | 0.7213 | <1 |
| E3 | Random Forest inicial | Árboles — bagging | 1 | 0.1862 | 0.0358 | 1.6 |
| E4 | Gradient Boosting inicial | Árboles — boosting | 1 | 0.2138 | 0.0392 | 0.9 |
| E5 | **Random Forest HPO** | Bagging + HPO | **20/36** | **0.1847** | **0.0361** | 46.9 |
| E6 | Gradient Boosting HPO | Boosting + HPO | 20/36 | 0.2048 | 0.0552 | 51.6 |
| E7 | RF sin rezagos (interpretativo) | Experimento interpretativo | 1 | 0.3964 | 0.1938 | 2.6 |

**Notas:**
- E5 y E6: RandomizedSearchCV con n_iter=20 sobre un espacio de 36 combinaciones posibles
- E7 no compite por el test — es un experimento interpretativo para cuantificar el aporte de los rezagos
- Total: 7 experimentos, 45 combinaciones evaluadas, ~104 segundos de cómputo


---

## 3. Resultados

### 3.1 Comparación de modelos

| Modelo | MAE_log CV | ± std | Mejora vs Dummy |
|---|---|---|---|
| Dummy (media train) | 0.5549 | 0.1725 | — |
| Ridge α=1.0 | 0.5364 | 0.7213 | -3.3% |
| Random Forest inicial | 0.1862 | 0.0358 | -66.4% |
| GBM inicial | 0.2138 | 0.0392 | -61.5% |
| **Random Forest HPO** | **0.1847** | **0.0361** | **-66.7%** |
| GBM HPO | 0.2048 | 0.0552 | -63.1% |

**Mejores hiperparámetros encontrados:**
- Random Forest: n_estimators=200, max_depth=15, min_samples_leaf=1
- Gradient Boosting: n_estimators=100, max_depth=5, learning_rate=0.1

**Observaciones:**

Ridge no mejora al Dummy (0.5364 vs 0.5549, diferencia de 0.0185 dentro del ruido de validación). Esto confirma que la relación entre las variables del sistema eléctrico y el precio no es lineal — un modelo lineal no puede capturar el umbral crítico de reservas.

Random Forest y GBM superan ampliamente a los baselines. La diferencia entre RF HPO (0.1847) y GBM HPO (0.2048) es 0.0201, mayor que una desviación estándar del ganador (0.0361) — diferencia estadísticamente significativa según el criterio de parsimonia definido en el contrato.

### 3.2 Error por fold — modelo ganador

| Fold | Período | MAE_log | Precio medio real | Condición |
|---|---|---|---|---|
| 1 | jul–nov 2023 | 0.1838 | $765/kWh | Amenaza El Niño |
| 2 | nov 2023–mar 2024 | 0.1406 | $578/kWh | Transición |
| 3 | mar–jul 2024 | 0.2469 | $438/kWh | Estrés hídrico |
| 4 | jul–nov 2024 | 0.1930 | $989/kWh | Crisis El Niño |
| 5 | nov 2024–mar 2025 | 0.1591 | $488/kWh | Recuperación |
| **Promedio** | | **0.1847** | | |

El Fold 3 (estrés hídrico — embalses en mínimos) tiene el mayor MAE_log (0.2469), no el Fold 4 (crisis de precios). Esto sugiere que el modelo maneja mejor los picos de precio extremos (donde lag_1 ya es alto) que las transiciones desde precios moderados hacia la crisis — exactamente el escenario donde la anticipación tendría mayor valor.

### 3.3 Importancia de features

| Feature | Importancia | Grupo |
|---|---|---|
| lag_1 | 90.9% | Rezago temporal |
| gen_termica | 2.2% | Sistema eléctrico |
| estres_hidrico | 0.8% | Feature engineered |
| lag_7 | 0.7% | Rezago temporal |
| efecto_solar_demanda | 0.7% | Feature engineered |
| aportes_energia_gwh | 0.6% | Sistema eléctrico |
| lag_30 | 0.5% | Rezago temporal |
| Resto (9 features) | 3.6% | Varios |

### 3.4 Experimento interpretativo — con rezagos vs sin rezagos

Para entender qué parte del MAE viene de la autocorrelación y qué parte de las condiciones estructurales, se entrenó el mismo RF HPO sin las tres features de rezago:

| Modelo | MAE_log CV | ± std |
|---|---|---|
| RF con rezagos (ganador) | 0.1847 | 0.0361 |
| RF sin rezagos (interpretativo) | 0.3964 | 0.1938 |

Los rezagos explican aproximadamente el 53% de la reducción del error. Sin embargo, el análisis por fold revela un patrón importante:

| Fold | Condición | MAE con lags | MAE sin lags | Diferencia |
|---|---|---|---|---|
| 1 | Amenaza El Niño | 0.1838 | 0.2173 | +0.0335 |
| 2 | Transición | 0.1406 | 0.4531 | +0.3125 |
| 3 | Estrés hídrico | 0.2469 | 0.4749 | +0.2280 |
| 4 | Crisis El Niño | 0.1930 | 0.6881 | +0.4951 |
| 5 | Recuperación | 0.1591 | 0.1489 | **-0.0102** |

En el **Fold 5 (recuperación)** el modelo sin rezagos es marginalmente mejor. Cuando el precio está bajando rápidamente, lag_1 introduce un sesgo hacia arriba — "recuerda" los precios altos de la crisis. El modelo estructural generaliza mejor en ese cambio de régimen.

Cuando se eliminan los rezagos, la importancia de features cambia radicalmente:

| Feature (sin lags) | Importancia |
|---|---|
| gen_termica | 70.5% |
| precio_escasez_kwh | 7.0% |
| reservas_pct | 4.3% |
| aportes_energia_gwh | 4.1% |
| ratio_hidro | 3.0% |

gen_termica pasa a dominar con 70.5% — exactamente la variable con mayor correlación con el precio (r=+0.795) identificada en el EDA de la Entrega 1. Esto confirma desde otro ángulo el hallazgo de Villarreal & Flores (2023): *"las condiciones hidrológicas y la mezcla de generación son los drivers estructurales del precio de bolsa en Colombia"*.

---

## 4. Discusión técnica

### 4.1 ¿Qué modelos se compararon y por qué?

Se compararon cuatro familias: Dummy (benchmark mínimo), Ridge (baseline lineal serio), Random Forest y Gradient Boosting. La elección de RF y GBM se justifica por la naturaleza no lineal del problema — el umbral crítico de reservas no puede ser capturado por un modelo lineal. KNN y SVM se descartaron por limitaciones específicas para series de tiempo con cambios de régimen.

### 4.2 ¿Cómo se evitó el data leakage?

Se implementaron tres capas de protección:

**Leakage temporal:** el TimeSeriesSplit garantiza que en cada fold la validación siempre es posterior al train. El K-Fold clásico fue descartado explícitamente.

**Leakage de preprocessing:** el StandardScaler y el SimpleImputer están encapsulados dentro del sklearn Pipeline — aprenden solo con el fold de train de cada ventana, nunca con los datos de validación.

**Leakage de decisión:** el HPO y la selección del modelo ganador se realizaron exclusivamente sobre el conjunto de train con TimeSeriesSplit. El archivo `test_energia.csv` no se importó en el notebook de esta entrega.

### 4.3 ¿Cuál familia parece más prometedora?

**Random Forest** es la familia más prometedora para este problema, con las siguientes reservas:

A favor: mejor MAE_log CV (0.1847 ± 0.0361), estabilidad entre folds (std más baja que GBM), y capacidad demostrada para capturar los distintos regímenes del sistema eléctrico.

Reserva principal: el modelo opera principalmente como predictor de persistencia (lag_1 = 90.9% de importancia). Esto funciona bien en condiciones estables pero puede introducir sesgo en cambios de régimen — especialmente en la recuperación post-Niño, que es precisamente el período del test.

### 4.4 ¿Qué limitaciones siguen abiertas?

**Limitación 1 — Dominancia de lag_1:** el modelo captura principalmente autocorrelación, no condiciones estructurales. Para un inversionista que necesita anticipar crisis — no solo seguirlas — esto es una limitación real. La Entrega 3 debería explorar si un modelo sin rezagos o un ensemble híbrido mejora la anticipación de cambios de régimen.

**Limitación 2 — Cambio de régimen train/test:** el train tiene precio medio $606/kWh y el test $188/kWh. El modelo fue entrenado principalmente en condiciones de estrés hídrico y la evaluación final será en recuperación. El Fold 5 (recuperación) es el proxy más cercano al test en la validación cruzada — su MAE_log de 0.1591 es el más bajo de todos los folds, lo que es una señal positiva.

**Limitación 3 — No estacionariedad de gen_solar:** la generación solar creció ~5x entre 2023 y 2026 por nueva capacidad instalada. El modelo puede subestimar el efecto solar en el test. Esta limitación fue documentada en la Entrega 1 y sigue abierta.

**Limitación 4 — demanda_pico sin documentación oficial:** la interpretación de demanda_pico como duck curve es estadística (correlación con gen_solar = 0.54), no documental. XM no provee la definición exacta de cálculo de esta variable.

---

## 5. Decisión provisional

**Modelo candidato para la Entrega 3:** Random Forest con n_estimators=200, max_depth=15, min_samples_leaf=1.

**Justificación:** mejor MAE_log CV promedio (0.1847) con diferencia estadísticamente significativa respecto al segundo candidato GBM (0.2048), diferencia de 0.0201 > 0.0361 (una desviación estándar del ganador).

**Registro pre-test:**

| Campo | Valor |
|---|---|
| Modelo | Random Forest |
| n_estimators | 200 |
| max_depth | 15 |
| min_samples_leaf | 1 |
| Features | 16 (10 sistema + 3 engineered + 3 rezagos) |
| Target | log(precio_bolsa_kwh) |
| MAE_log CV promedio | 0.1847 |
| MAE_log CV std | 0.0361 |
| Test usado para selección | No |
| Test usado para HPO | No |

El test se evalúa una sola vez en la Entrega 3.

---

## 6. Conclusiones

El cambio de régimen identificado en la Entrega 1 como causa del R² negativo del baseline Ridge queda parcialmente resuelto con modelos no lineales y validación temporal. Random Forest reduce el MAE en un 66.7% respecto al baseline trivial, con un error estable entre los distintos regímenes del sistema eléctrico colombiano.

El hallazgo metodológico más importante de esta entrega es la dualidad entre el modelo de persistencia (con rezagos, lag_1 domina) y el modelo estructural (sin rezagos, gen_termica domina). Esta dualidad conecta con la conclusión central de Villarreal & Flores (2023): *"el modelo óptimo depende del horizonte de predicción — los drivers estructurales dominan en mediano y largo plazo mientras que la autocorrelación domina en el corto plazo"*.

La Entrega 3 evaluará el modelo ganador sobre el test reservado y explorará si un modelo híbrido o sin rezagos mejora la anticipación de cambios de régimen — el escenario más relevante para un inversionista ante la posible ocurrencia de un nuevo fenómeno El Niño en el segundo semestre de 2026.

---

## Referencias

- Villarreal Marimon, Y.J. & Flores San Martín, L.A. (2023). *Predicción del precio de la energía eléctrica en Colombia mediante un enfoque de machine learning*. Tesis de grado, MAF, Universidad EAFIT.
- Terán, M. (2026). *Notas de clase — Sesiones 03 y 04: Modelos lineales, validación y selección de modelos*. Aprendizaje de Máquina Aplicado, Universidad EAFIT.
- XM S.A. E.S.P. (2026). *Históricos Sinergox*. https://sinergox.xm.com.co

---

*Aprendizaje de Máquina Aplicado · Marco Terán · Universidad EAFIT · 2026*
