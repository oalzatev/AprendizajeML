# Predicción del Precio de Bolsa — Mercado Eléctrico Colombiano

**Curso:** Aprendizaje de Máquina Aplicado  
**Profesor:** Marco Terán — Universidad EAFIT  
**Año:** 2026

---

## Descripción del proyecto

Predicción del **precio de bolsa diario de la electricidad en Colombia** ($/kWh)
a partir de variables del estado del sistema eléctrico: disponibilidad hídrica,
mezcla de generación, nivel de reservas y demanda.

El precio de bolsa es el precio marginal del mercado mayorista colombiano,
determinado principalmente por la escasez de agua en los embalses y la necesidad
de despachar generación termoeléctrica (más cara).

---

## Estructura del repositorio

```
ml-proj/
├── README.md                         ← Este archivo
├── requirements.txt                  ← Dependencias Python
├── .gitignore
├── data/
│   ├── raw/                          ← 7 archivos CSV originales de XM
│   └── processed/
│       ├── dataset_energia_colombia.csv   ← Dataset integrado (1,155 filas × 16 cols)
│       ├── train_energia.csv              ← Train (790 días) + 5 features engineered
│       └── test_energia.csv              ← Test  (365 días) + 5 features engineered
├── notebooks/
│   └── entrega1_energia_colombia.ipynb   ← Notebook Entrega 1 (EDA + baseline)
├── figures/                          ← Todas las figuras generadas
├── src/
│   └── preprocessing.py             ← Módulo reutilizable (features + split + métricas)
└── report/
    ├── data_card_energia_colombia.md ← Data card del dataset
    └── reporte_entrega1.pdf          ← Reporte técnico Entrega 1
```

---

## Dataset

| Campo | Detalle |
|---|---|
| **Fuente** | XM S.A. E.S.P. — Operador del SIN de Colombia |
| **Período** | 01/02/2023 – 31/03/2026 |
| **Granularidad** | Diaria |
| **Filas** | 1,155 |
| **Variables originales** | 16 (tras integración de 7 archivos) |
| **Features engineered** | 5 adicionales |
| **Target** | `precio_bolsa_kwh` → entrenado como `log(precio_bolsa_kwh)` |
| **Faltantes** | 0 |

### Variables principales

| Variable | Descripción |
|---|---|
| `precio_bolsa_kwh` | **TARGET** — Precio de bolsa ponderado diario ($/kWh) |
| `aportes_energia_gwh` | Aportes hídricos diarios al sistema |
| `reservas_pct` | Nivel de embalses (% capacidad útil) |
| `gen_hidro` | Generación hidroeléctrica diaria (kWh) |
| `gen_termica` | Generación termoeléctrica diaria (kWh) |
| `gen_solar` | Generación solar diaria (kWh) |
| `gen_eolica` | Generación eólica diaria (kWh) |
| `ratio_hidro` | % de generación hídrica sobre total |
| `demanda_min` | Demanda mínima diaria (kWh) |
| `demanda_pico` | Demanda en horas pico (kWh) |
| `precio_escasez_kwh` | Precio de escasez ponderado — señal regulatoria |

### Features engineered

| Feature | Fórmula | Grupo | Justificación |
|---|---|---|---|
| `estres_hidrico` | `gen_termica / (reservas_pct + 0.01)` | Hídrica-térmica | Umbral crítico de crisis hídrica |
| `presion_termica` | `gen_termica / (aportes_energia_gwh + 1)` | Hídrica-térmica | Flujo de aportes vs respuesta del sistema |
| `efecto_solar_demanda` | `demanda_pico - demanda_min` | Solar | Duck curve — impacto solar en demanda neta |
| `gen_renovable` | `gen_hidro + gen_solar + gen_eolica` | Solar | Generación limpia total |
| `termica_evitada` | `gen_solar / (gen_termica + 1)` | Solar | Cuánto solar desplaza térmica cara |

---

## Cadena causal del sistema

```
aportes_energia_gwh → reservas_pct → gen_termica → precio_bolsa_kwh
       r=+0.39              r=-0.54       r=+0.795
```

El precio sube cuando el sistema se queda sin agua y enciende plantas térmicas.
`termica_evitada` captura la transición energética: el solar desplazando térmica.

---

## Partición temporal

> ⚠️ El split **nunca es aleatorio** — un split aleatorio introduce leakage temporal.

| Conjunto | Período | Días | Contiene |
|---|---|---|---|
| Train | feb 2023 – mar 2025 | 790 | El Niño completo + crisis de precios |
| Test | abr 2025 – mar 2026 | 365 | Período de recuperación — 1 año completo |

---

## Resultados del baseline (Entrega 1)

| Modelo | Features | RMSE ($/kWh) | R² (log) |
|---|---|---|---|
| Benchmark — Media del train | — | 331.5 | −6.82 |
| Ridge sin engineering | 10 | 166.9 | −2.24 |
| Ridge + engineering parcial | 13 | 101.0 | −0.66 |
| **Ridge + 5 features completas** | **15** | **62.9** | **+0.30** |

El feature engineering reduce el RMSE en un **62%** respecto al baseline sin engineering.
`termica_evitada` es la feature más poderosa: captura el desplazamiento solar de la
térmica — la esencia de la transición energética en una sola variable.

---

## Cómo reproducir

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd ml-proj

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el notebook principal
jupyter notebook notebooks/entrega1_energia_colombia.ipynb
```

> **Nota:** el notebook carga `dataset_energia_colombia.csv` desde `data/processed/`.
> Los CSVs originales de XM están disponibles en `data/raw/` para reproducir
> la integración completa desde cero.

---

## Entregas

| Entrega | Fecha | Peso | Estado |
|---|---|---|---|
| Entrega 1: Problema, datos, EDA y baseline | 09/04/2026 | 5% | ✅ Completada |
| Entrega 2: Comparación de familias y validación | 30/04/2026 | 10% | 🔄 En progreso |
| Entrega 3: Modelo final, interpretación y comunicación | 14/05/2026 | 20% | ⏳ Pendiente |

---

## Limitaciones conocidas

- **Cambio de régimen:** train (El Niño, ~$606/kWh) y test (recuperación, ~$188/kWh) tienen distribuciones muy distintas
- **No estacionariedad solar:** `gen_solar` crece estructuralmente — la correlación solar-precio se está invirtiendo (2023: r=+0.376 → 2026: r=-0.083)
- **Sin variables climáticas directas:** El Niño se captura indirectamente a través de reservas y aportes
- **Posible nuevo El Niño en segundo semestre 2026**

---

## Módulo reutilizable

```python
from src.preprocessing import (
    add_engineered_features,  # 5 features engineered
    split_temporal,            # split por fecha, nunca aleatorio
    regression_report,         # métricas en escala log y original
    FEATURES,                  # lista de 15 features
    TARGET,                    # 'precio_bolsa_kwh'
    FECHA_CORTE,               # '2025-03-31'
)
```

---

*"Primero auditamos. Luego decidimos. Después transformamos. Y solo entonces automatizamos."*  
— Marco Terán, Aprendizaje de Máquina Aplicado 2026
