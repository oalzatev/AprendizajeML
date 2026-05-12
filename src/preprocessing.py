"""
preprocessing.py
Funciones reutilizables — Predicción precio de bolsa Colombia
Aprendizaje de Máquina Aplicado · Marco Terán · EAFIT 2026
"""
import numpy as np
import pandas as pd

SEED        = 42
TARGET      = 'precio_bolsa_kwh'
TARGET_LOG  = 'log_precio_bolsa'
FECHA_CORTE = '2025-03-31'

FEATURES = [
    'aportes_energia_gwh','precio_escasez_kwh','reservas_pct',
    'gen_hidro','gen_termica','gen_solar','gen_eolica','ratio_hidro',
    'demanda_min','demanda_pico',
    'estres_hidrico','efecto_solar_demanda','gen_renovable',
    'lag_1','lag_7','lag_30',
]
FEATURES_SIN_LAGS = [f for f in FEATURES if 'lag' not in f]


def add_engineered_features(df, include_lags=True):
    """
    Agrega features de dominio y rezagos temporales.
    Aplicar DESPUÉS del split, sobre datos ordenados por fecha.
    """
    out = df.copy().sort_values('date').reset_index(drop=True)
    out['estres_hidrico']       = out['gen_termica'] / (out['reservas_pct'] + 0.01)
    out['efecto_solar_demanda'] = out['demanda_pico'] - out['demanda_min']
    out['gen_renovable']        = out['gen_hidro'] + out['gen_solar'] + out['gen_eolica']
    if include_lags:
        out['lag_1']  = out[TARGET_LOG].shift(1)
        out['lag_7']  = out[TARGET_LOG].shift(7)
        out['lag_30'] = out[TARGET_LOG].shift(30)
    return out.replace([np.inf, -np.inf], np.nan)


def split_temporal(df, fecha_corte=FECHA_CORTE):
    """Split temporal — nunca aleatorio en series de tiempo."""
    train = df[df['date'] <= fecha_corte].copy()
    test  = df[df['date'] >  fecha_corte].copy()
    print(f"Train: {train['date'].min().date()} → {train['date'].max().date()} | {len(train):,} filas")
    print(f"Test : {test['date'].min().date()} → {test['date'].max().date()} | {len(test):,} filas")
    assert len(set(train['date']).intersection(set(test['date']))) == 0
    return train, test


def regression_report(name, y_true_log, y_pred_log, y_true_orig=None):
    """Métricas en escala log y original."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse  = mean_squared_error(y_true_log, y_pred_log)
    mae  = mean_absolute_error(y_true_log, y_pred_log)
    r2   = r2_score(y_true_log, y_pred_log)
    print(f"\n  {name}: MSE={mse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    res = {'modelo': name, 'mse_log': mse, 'mae_log': mae, 'r2_log': r2}
    if y_true_orig is not None:
        pred_orig = np.exp(y_pred_log)
        rmse_o = np.sqrt(mean_squared_error(y_true_orig, pred_orig))
        mae_o  = mean_absolute_error(y_true_orig, pred_orig)
        print(f"  Escala orig: RMSE={rmse_o:,.1f} $/kWh  MAE={mae_o:,.1f} $/kWh")
        res.update({'rmse_orig': rmse_o, 'mae_orig': mae_o})
    return res
