"""
preprocessing.py
Funciones reutilizables de preprocesamiento para el proyecto de
predicción del precio de bolsa — Mercado Eléctrico Colombiano.

Aprendizaje de Máquina Aplicado · Marco Terán · EAFIT 2026
"""

import numpy as np
import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────────────
SEED         = 42
TARGET       = 'precio_bolsa_kwh'
TARGET_LOG   = 'log_precio_bolsa'
FECHA_CORTE  = '2025-03-31'

FEATURES = [
    # Variables originales
    'aportes_energia_gwh', 'precio_escasez_kwh', 'reservas_pct',
    'gen_hidro', 'gen_termica', 'gen_solar', 'gen_eolica', 'ratio_hidro',
    'demanda_min', 'demanda_pico',
    # Features engineered — Grupo 1: cadena causal hídrica-térmica
    'estres_hidrico',
    'presion_termica',
    # Features engineered — Grupo 2: transición energética solar
    'efecto_solar_demanda',
    'gen_renovable',
    'termica_evitada',
]


# ── Feature engineering ───────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega 5 features derivadas con justificación de dominio.

    Debe aplicarse DESPUÉS del split temporal, por separado en train y test.
    Es determinística — no aprende estadísticas del dataset.

    Grupo 1 — Cadena causal hídrica-térmica:
      - estres_hidrico:  reservas bajas + alta térmica → precio sube
      - presion_termica: flujo de aportes vs respuesta del sistema

    Grupo 2 — Transición energética solar:
      - efecto_solar_demanda: duck curve (impacto solar en demanda neta)
      - gen_renovable:        generación limpia total
      - termica_evitada:      cuánto solar desplaza térmica cara

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas originales del dataset de energía.

    Returns
    -------
    pd.DataFrame
        Copia del DataFrame con 5 columnas adicionales.
    """
    out = df.copy()

    # ── Grupo 1: Cadena causal hídrica-térmica ────────────────────────────────
    # Motivación: sección 6a identificó la cadena
    # aportes → reservas → gen_termica → precio (r progresivo hasta +0.795)

    # 1. Estrés hídrico — umbral crítico donde reservas bajas + térmica alta
    #    disparan el precio. Coeficiente más importante en Ridge.
    #    +0.01: guard contra división por cero si reservas → 0
    out['estres_hidrico'] = out['gen_termica'] / (out['reservas_pct'] + 0.01)

    # 2. Presión térmica — cuánta térmica se necesita por unidad de agua entrante
    #    Captura el flujo upstream: aportes bajos → presión térmica alta
    #    +1: guard defensivo para producción
    out['presion_termica'] = out['gen_termica'] / (out['aportes_energia_gwh'] + 1)

    # ── Grupo 2: Transición energética solar ──────────────────────────────────
    # Motivación: correlación solar-precio se está invirtiendo año a año
    # 2023: r=+0.376 → 2026: r=-0.083  (solar empieza a desplazar térmica)

    # 3. Efecto solar en demanda (duck curve)
    #    La diferencia pico-mínimo crece con la instalación solar (r=0.54)
    out['efecto_solar_demanda'] = out['demanda_pico'] - out['demanda_min']

    # 4. Generación renovable total — captura cuánto del sistema es limpio
    out['gen_renovable'] = out['gen_hidro'] + out['gen_solar'] + out['gen_eolica']

    # 5. Térmica evitada — cuánto solar desplaza térmica cara
    #    Alto ratio = solar funcionando como sustituto → precio baja
    #    Bajo ratio = solar no evita encender plantas → precio alto
    #    +1: guard defensivo
    out['termica_evitada'] = out['gen_solar'] / (out['gen_termica'] + 1)

    return out.replace([np.inf, -np.inf], np.nan)


# ── Transformación del target ─────────────────────────────────────────────────
def transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna log_precio_bolsa al DataFrame."""
    out = df.copy()
    out[TARGET_LOG] = np.log(out[TARGET])
    return out


# ── Split temporal ────────────────────────────────────────────────────────────
def split_temporal(df: pd.DataFrame, fecha_corte: str = FECHA_CORTE):
    """
    Divide el dataset por fecha de corte — nunca aleatoriamente.

    En series de tiempo el split aleatorio introduce leakage temporal:
    el modelo aprendería del futuro para predecir el pasado.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo con columna 'date'.
    fecha_corte : str
        Fecha de corte 'YYYY-MM-DD'. Todo lo anterior va a train.

    Returns
    -------
    tuple : (train, test)
    """
    train = df[df['date'] <= fecha_corte].copy()
    test  = df[df['date'] >  fecha_corte].copy()

    print(f'Train: {train["date"].min().date()} → {train["date"].max().date()} | {len(train):,} filas')
    print(f'Test : {test["date"].min().date()} → {test["date"].max().date()} | {len(test):,} filas')

    overlap = len(set(train['date']).intersection(set(test['date'])))
    assert overlap == 0, f'⚠️ Solapamiento detectado: {overlap} días'

    return train, test


# ── Reporte de métricas ───────────────────────────────────────────────────────
def regression_report(name: str, y_true_log, y_pred_log, y_true_orig) -> dict:
    """
    Calcula y muestra métricas en escala log y en escala original.

    Parameters
    ----------
    name : str         Nombre del modelo / experimento.
    y_true_log         Valores reales en escala log.
    y_pred_log         Predicciones en escala log.
    y_true_orig        Valores reales en escala original ($/kWh).

    Returns
    -------
    dict con métricas calculadas.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse_log  = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log   = mean_absolute_error(y_true_log, y_pred_log)
    r2_log    = r2_score(y_true_log, y_pred_log)
    y_pred_orig = np.exp(y_pred_log)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig  = mean_absolute_error(y_true_orig, y_pred_orig)
    r2_orig   = r2_score(y_true_orig, y_pred_orig)

    print(f'\n{"─"*60}')
    print(f'  {name}')
    print(f'{"─"*60}')
    print(f'  Escala log:      RMSE={rmse_log:.4f}  MAE={mae_log:.4f}  R²={r2_log:.4f}')
    print(f'  Escala original: RMSE={rmse_orig:,.1f} $/kWh  MAE={mae_orig:,.1f} $/kWh  R²={r2_orig:.4f}')

    return {
        'modelo': name,
        'rmse_log': rmse_log, 'mae_log': mae_log, 'r2_log': r2_log,
        'rmse_orig': rmse_orig, 'mae_orig': mae_orig, 'r2_orig': r2_orig,
    }
