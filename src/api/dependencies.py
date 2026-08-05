"""
FastAPI dependencies — shared resources loaded once at startup.

FastAPI's dependency injection system lets you declare what a route
needs (a loaded model, a database connection, etc.) and FastAPI
provides it automatically. This avoids reloading models on every request.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from functools import lru_cache
from src.api.config import settings

# ── Data loading functions ─────────────────────────────────────────
# lru_cache means the function only runs once — subsequent calls
# return the cached result. This prevents reloading data on every request.

@lru_cache(maxsize=1)
def get_risk_table() -> pd.DataFrame:
    return pd.read_csv(settings.CUSTOMER_RISK_TABLE)

@lru_cache(maxsize=1)
def get_shap_values() -> pd.DataFrame:
    return pd.read_csv(settings.SHAP_VALUES)

@lru_cache(maxsize=1)
def get_feature_importance() -> pd.DataFrame:
    return pd.read_csv(settings.FEATURE_IMPORTANCE)

@lru_cache(maxsize=1)
def get_forecasts() -> pd.DataFrame:
    df = pd.read_csv(settings.ALL_FORECASTS)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

@lru_cache(maxsize=1)
def get_alerts() -> pd.DataFrame:
    return pd.read_csv(settings.INVENTORY_ALERTS)

@lru_cache(maxsize=1)
def get_folds() -> pd.DataFrame:
    return pd.read_csv(settings.WALKFORWARD_FOLDS)

@lru_cache(maxsize=1)
def get_walkforward_summary() -> pd.DataFrame:
    return pd.read_csv(settings.WALKFORWARD_SUMMARY)

# ── Model loading ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_churn_model():
    return joblib.load(settings.CHURN_MODEL)

@lru_cache(maxsize=1)
def get_churn_scaler():
    return joblib.load(settings.CHURN_SCALER)

@lru_cache(maxsize=1)
def get_shap_explainer():
    return joblib.load(settings.SHAP_EXPLAINER)

@lru_cache(maxsize=1)
def get_optimal_threshold() -> float:
    return float(np.load(settings.OPTIMAL_THRESHOLD))

# ── LLM components ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm_client():
    from src.module3_llm.llm_client import RetailLLMClient
    return RetailLLMClient()


@lru_cache(maxsize=1)
def get_churn_narrator():
    from src.module3_llm.churn_narrator import ChurnNarrator
    return ChurnNarrator(get_llm_client())


@lru_cache(maxsize=1)
def get_demand_narrator():
    from src.module3_llm.demand_narrator import DemandNarrator
    return DemandNarrator(get_llm_client())


@lru_cache(maxsize=1)
def get_winback_generator():
    from src.module3_llm.winback_generator import WinBackGenerator
    return WinBackGenerator(get_llm_client())


@lru_cache(maxsize=1)
def get_query_engine():
    from src.module3_llm.query_engine import QueryEngine
    return QueryEngine(
        risk_table_path=str(settings.CUSTOMER_RISK_TABLE),
        alerts_path=str(settings.INVENTORY_ALERTS),
        folds_path=str(settings.WALKFORWARD_FOLDS),
        llm_client=get_llm_client()
    )


@lru_cache(maxsize=1)
def get_retail_data():
    """Full cleaned transaction log — used for win-back favorite-product lookup.
    ~42MB, loaded once and cached for the life of the server process."""
    return pd.read_csv(settings.CLEAN_RETAIL)


@lru_cache(maxsize=1)
def get_seasonality_analysis() -> pd.DataFrame:
    return pd.read_csv(settings.SEASONALITY_ANALYSIS)