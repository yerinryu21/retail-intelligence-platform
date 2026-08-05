"""
Churn prediction service layer.
Contains business logic for churn endpoints.
Separates logic from route handlers — makes testing easier.

IMPORTANT: customer_risk_table.csv and shap_values.csv are NOT row-aligned.
Always join via the ShapRowIndex column, never by direct positional index.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.api.config import settings

RISK_TIER_COUNTS_MAP = {
    "extreme_risk_count": "⚫ Extreme Risk",
    "high_risk_count": "🔴 High Risk",
    "medium_risk_count": "🟡 Medium Risk",
    "low_risk_count": "🟢 Low Risk",
}


def _tier_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {
        key: int((df['RiskTier'] == label).sum())
        for key, label in RISK_TIER_COUNTS_MAP.items()
    }


def get_churn_summary(risk_table: pd.DataFrame) -> Dict:
    """Calculate summary statistics for churn dashboard"""

    summary = {
        "total_customers": len(risk_table),
        "total_revenue_at_risk": float(risk_table['RevenueAtRisk'].sum()),
        "avg_churn_probability": float(risk_table['ChurnProbability'].mean()),
        "top_churn_driver": str(
            risk_table['TopChurnDriver'].value_counts().index[0]
        )
    }
    summary.update(_tier_counts(risk_table))
    return summary


def get_paginated_risk_table(risk_table: pd.DataFrame,
                              page: int = 1,
                              page_size: int = 20,
                              risk_tier: Optional[str] = None,
                              min_probability: float = 0.0,
                              sort_by: str = 'RevenueAtRisk') -> Dict:
    """
    Return paginated customer risk table with optional filtering.
    """

    filtered = risk_table.copy()

    if risk_tier:
        filtered = filtered[filtered['RiskTier'] == risk_tier]

    if min_probability > 0:
        filtered = filtered[filtered['ChurnProbability'] >= min_probability]

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False)

    total = len(filtered)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size

    page_data = filtered.iloc[start:end]

        
    customers = []
    for idx, row in page_data.iterrows():
        customers.append({
            "customer_index": int(idx),
            "customer_id": int(row['CustomerID']) if pd.notna(row.get('CustomerID')) else None,
            "risk_tier": str(row['RiskTier']),
            "churn_probability": float(row['ChurnProbability']),
            "frequency": float(row.get('Frequency', 0)),
            "monetary": float(row.get('Monetary', 0)),
            "avg_order_value": float(row.get('AvgOrderValue', 0)),
            "unique_products": float(row.get('UniqueProducts', 0)),
            "avg_quantity": float(row.get('AvgQuantity', 0)),
            "days_active": float(row.get('DaysActive', 0)),
            "orders_per_day": float(row.get('OrdersPerDay', 0)),
            "clv": float(row.get('CLV', 0)),
            "revenue_at_risk": float(row.get('RevenueAtRisk', 0)),
            "top_churn_driver": str(row.get('TopChurnDriver', '')),
            "top_driver_direction": str(row.get('TopDriverDirection', '')),
            "explanation": str(row.get('Explanation', '')),
            "shap_row_index": int(row['ShapRowIndex']) if pd.notna(row.get('ShapRowIndex')) else None
        })
        
        
    result = {
        "total_customers": total,
        "total_revenue_at_risk": float(filtered['RevenueAtRisk'].sum()),
        "customers": customers,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }
    result.update(_tier_counts(filtered))
    return result


def predict_single_customer(features: Dict,
                              model,
                              explainer,
                              threshold: float) -> Dict:
    """
    Run churn prediction for a new customer with custom feature values.
    Feature order MUST match settings.FEATURE_COLS (the model's training order):
    Frequency, Monetary, AvgOrderValue, UniqueProducts, AvgQuantity, DaysActive, OrdersPerDay
    """

    feature_values = [
        features.get('frequency', 0),
        features.get('monetary', 0),
        features.get('avg_order_value', 0),
        features.get('unique_products', 0),
        features.get('avg_quantity', 0),
        features.get('days_active', 0),
        features.get('orders_per_day', 0)
    ]

    X = np.array(feature_values).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=settings.FEATURE_COLS)

    prob = float(model.predict_proba(X_df)[0, 1])
    predicted_churn = bool(prob >= threshold)

    # 4-tier risk classification — thresholds chosen to roughly mirror
    # the distribution used to build customer_risk_table.csv
    # 4-tier risk classification, matching the thresholds used to build
    # customer_risk_table.csv (verified against actual data distribution)
    if prob >= 0.70:
        risk_tier = "⚫ Extreme Risk"
    elif prob >= 0.50:
        risk_tier = "🔴 High Risk"
    elif prob >= 0.20:
        risk_tier = "🟡 Medium Risk"
    else:
        risk_tier = "🟢 Low Risk"

    try:
        shap_values = explainer(X_df)
        shap_array = shap_values.values[0]

        top_factors = []
        shap_pairs = sorted(
            zip(settings.FEATURE_COLS, shap_array),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature, shap_val in shap_pairs[:5]:
            top_factors.append({
                "feature": feature,
                "shap_value": float(shap_val),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk"
            })
    except Exception:
        top_factors = []

    return {
        "churn_probability": prob,
        "risk_tier": risk_tier,
        "predicted_churn": predicted_churn,
        "top_risk_factors": top_factors,
        "explanation": f"Churn probability: {prob:.1%}. "
                       f"Top factor: {top_factors[0]['feature'] if top_factors else 'unknown'}."
    }


def get_customer_by_index(risk_table: pd.DataFrame,
                           shap_values_df: pd.DataFrame,
                           customer_index: int) -> Tuple[pd.Series, pd.Series]:
    """
    Get customer and matching SHAP row by risk-table position.
    IMPORTANT: shap_values.csv is not row-aligned with customer_risk_table.csv —
    the join must go through the ShapRowIndex column.
    """

    if customer_index >= len(risk_table) or customer_index < 0:
        raise ValueError(
            f"Customer index {customer_index} out of range "
            f"(0-{len(risk_table)-1})"
        )

    customer_row = risk_table.iloc[customer_index]

    shap_row_index = int(customer_row['ShapRowIndex'])
    if 0 <= shap_row_index < len(shap_values_df):
        shap_row = shap_values_df.iloc[shap_row_index]
    else:
        shap_row = pd.Series()

    return customer_row, shap_row