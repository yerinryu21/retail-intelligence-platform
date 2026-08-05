"""
API configuration settings.
Using a config class keeps all settings in one place —
easy to change paths, model names, or settings without
hunting through multiple files.
"""

import os
from pathlib import Path

# ── Project root directory ─────────────────────────────────────────
# This file is at src/api/config.py
# Project root is 2 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings:
    """Central configuration for the Retail Intelligence API"""
    
    # ── API Settings ───────────────────────────────────────────────
    API_TITLE = "Retail Intelligence Platform API"
    API_DESCRIPTION = """
    AI-powered retail analytics API providing:
    - Customer churn prediction with SHAP explainability
    - Demand forecasting with uncertainty quantification  
    - LLM-generated business insights via Ollama
    """
    API_VERSION = "1.0.0"
    
    # ── Data paths ─────────────────────────────────────────────────
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    
    CUSTOMER_RISK_TABLE = DATA_DIR / "customer_risk_table.csv"
    CUSTOMER_FEATURES = DATA_DIR / "customer_features.csv"
    SHAP_VALUES = DATA_DIR / "shap_values.csv"
    FEATURE_IMPORTANCE = DATA_DIR / "feature_importance.csv"
    SEGMENT_SUMMARIES = DATA_DIR / "segment_summaries.csv"
    
    ALL_FORECASTS = DATA_DIR / "all_forecasts.csv"
    SEASONALITY_ANALYSIS = DATA_DIR / "seasonality_analysis.csv"
    CLEAN_RETAIL = DATA_DIR / "clean_retail.csv"
    INVENTORY_ALERTS = DATA_DIR / "inventory_alerts.csv"
    WALKFORWARD_FOLDS = DATA_DIR / "walkforward_folds.csv"
    WALKFORWARD_SUMMARY = DATA_DIR / "walkforward_summary.csv"
    
    # ── Model paths ────────────────────────────────────────────────
    MODELS_DIR = PROJECT_ROOT / "models"
    
    CHURN_MODEL = MODELS_DIR / "churn_model_tuned.pkl"
    CHURN_SCALER = MODELS_DIR / "churn_scaler.pkl"
    SHAP_EXPLAINER = MODELS_DIR / "shap_explainer.pkl"
    OPTIMAL_THRESHOLD = MODELS_DIR / "optimal_threshold.npy"
    
    # ── LLM Settings ───────────────────────────────────────────────
    OLLAMA_MODEL = "llama3"
    OLLAMA_TEMPERATURE = 0.1
    
    # ── Feature columns ────────────────────────────────────────────
    FEATURE_COLS = [
        'Frequency', 'Monetary', 'AvgOrderValue',
        'UniqueProducts', 'AvgQuantity', 'DaysActive', 'OrdersPerDay'
    ]
    
    # ── API limits ─────────────────────────────────────────────────
    MAX_CUSTOMERS_PER_REQUEST = 100
    MAX_PRODUCTS_PER_REQUEST = 20
    DEFAULT_PAGE_SIZE = 20

settings = Settings()