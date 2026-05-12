# Predicción del Precio de Bolsa — Mercado Eléctrico Colombiano

**Curso:** Aprendizaje de Máquina Aplicado · Profesor Marco Terán · EAFIT · 2026

---

## Descripción

Predicción del **precio de bolsa diario** de la electricidad en Colombia ($/kWh).
Horizonte: corto plazo — precio del día siguiente.

---

## Estructura del repositorio

```
ml-proj/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                              ← 7 CSVs originales de XM
│   └── processed/
│       ├── dataset_energia_colombia.csv  ← Dataset integrado (1,155 × 16)
│       ├── train_energia.csv             ← Train (790 días)
│       └── test_energia.csv             ← Test (365 días) ← NO tocar hasta Entrega 3
├── notebooks/
│   ├── entrega1_energia_colombia.ipynb  ← EDA y baseline
│   └── entrega2_energia_colombia.ipynb  ← Comparación familias y validación
├── figures/
├── report/
│   ├── data_card_energia_colombia.md
│   ├── reporte_entrega1.pdf
│   ├── reporte_entrega2.pdf
│   └── reporte_entrega2.md
└── src/
    └── preprocessing.py
```

---

## Partición temporal

| Conjunto | Período | Días | Nota |
|---|---|---|---|
| Train | feb 2023 – mar 2025 | 790 | El Niño + crisis precios |
| Test | abr 2025 – mar 2026 | 365 | **Reservado — solo Entrega 3** |

---

## Resultados Entrega 2 — TimeSeriesSplit 5 folds

| Modelo | MAE_log CV | ± std |
|---|---|---|
| Dummy (media train) | 0.5549 | 0.1725 |
| Ridge α=1.0 | 0.5364 | 0.7213 |
| Random Forest HPO | **0.1847** | **0.0361** |
| GBM HPO | 0.2048 | 0.0552 |

**Modelo ganador:** Random Forest (n_estimators=200, max_depth=15, min_samples_leaf=1)

---

## Entregas

| Entrega | Fecha | Peso | Estado |
|---|---|---|---|
| Entrega 1: EDA y baseline | 09/04/2026 | 5% | ✅ Completada |
| Entrega 2: Familias y validación | 30/04/2026 | 10% | ✅ Completada |
| Entrega 3: Modelo final | 14/05/2026 | 20% | ⏳ Pendiente |

---

## Reproducir

```bash
pip install -r requirements.txt
jupyter notebook notebooks/entrega2_energia_colombia.ipynb
```

> Nota: ajustar ruta con `os.chdir('/ruta/a/ml-proj')` si es necesario.
