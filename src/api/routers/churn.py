"""
Churn prediction API endpoints.

Endpoints:
- GET /churn/summary — dashboard summary statistics
- GET /churn/customers — paginated customer risk table
- GET /churn/customers/{index} — single customer detail
- POST /churn/predict — predict churn for new customer data
- GET /churn/feature-importance — global SHAP feature importance
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.api.dependencies import (
    get_risk_table, get_shap_values, get_churn_model,
    get_shap_explainer, get_optimal_threshold, get_feature_importance
)
from src.api.models.churn import (
    CustomerFeatures, ChurnPredictionResponse,
    CustomerRiskTableResponse, ChurnSummaryResponse
)
from src.api.services.churn_service import (
    get_churn_summary, get_paginated_risk_table,
    predict_single_customer, get_customer_by_index
)

router = APIRouter(prefix="/churn", tags=["Churn Prediction"])


@router.get("/summary", response_model=ChurnSummaryResponse)
async def get_summary():
    """Get high-level churn summary statistics for the dashboard."""
    risk_table = get_risk_table()
    return get_churn_summary(risk_table)


@router.get("/customers", response_model=CustomerRiskTableResponse)
async def get_customers(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Customers per page"),
    risk_tier: Optional[str] = Query(
        None,
        description="Filter by risk tier, e.g. '⚫ Extreme Risk', '🔴 High Risk', '🟡 Medium Risk', '🟢 Low Risk'"
    ),
    min_probability: float = Query(0.0, ge=0.0, le=1.0),
    sort_by: str = Query("RevenueAtRisk", description="Sort column")
):
    """
    Get paginated customer risk table with optional filtering.
    """
    risk_table = get_risk_table()

    return get_paginated_risk_table(
        risk_table, page, page_size,
        risk_tier, min_probability, sort_by
    )


@router.get("/customers/{customer_index}")
async def get_customer_detail(customer_index: int):
    """
    Get detailed information for a specific customer, including
    top SHAP factors (correctly joined via ShapRowIndex).
    """

    risk_table = get_risk_table()
    shap_values_df = get_shap_values()

    try:
        customer_row, shap_row = get_customer_by_index(
            risk_table, shap_values_df, customer_index
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    feature_cols = [
        'Frequency', 'Monetary', 'AvgOrderValue',
        'UniqueProducts', 'AvgQuantity', 'DaysActive', 'OrdersPerDay'
    ]

    top_factors = []
    if len(shap_row) > 0:
        for feature in feature_cols:
            shap_col = f'shap_{feature}'
            if shap_col in shap_row.index:
                top_factors.append({
                    "feature": feature,
                    "shap_value": float(shap_row[shap_col]),
                    "feature_value": float(customer_row.get(feature, 0))
                })
        top_factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        top_factors = top_factors[:5]

    return {
        "customer_index": customer_index,
        "customer_id": int(customer_row['CustomerID']) if pd.notna(customer_row.get('CustomerID')) else None, # type: ignore
        "risk_tier": str(customer_row.get('RiskTier', 'Unknown')),
        "churn_probability": float(customer_row.get('ChurnProbability', 0)),
        "predicted_churn": bool(customer_row.get('PredictedChurn', False)),
        "frequency": float(customer_row.get('Frequency', 0)),
        "monetary": float(customer_row.get('Monetary', 0)),
        "avg_order_value": float(customer_row.get('AvgOrderValue', 0)),
        "unique_products": float(customer_row.get('UniqueProducts', 0)),
        "days_active": float(customer_row.get('DaysActive', 0)),
        "clv": float(customer_row.get('CLV', 0)),
        "revenue_at_risk": float(customer_row.get('RevenueAtRisk', 0)),
        "top_churn_driver": str(customer_row.get('TopChurnDriver', '')),
        "explanation": str(customer_row.get('Explanation', '')),
        "top_shap_factors": top_factors
    }


@router.post("/predict", response_model=ChurnPredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a new customer with custom feature values.

    Returns churn probability, 4-tier risk classification, top SHAP
    risk factors, and a plain-text explanation. Useful for scoring
    customers not already in the dataset.
    """

    model = get_churn_model()
    explainer = get_shap_explainer()
    threshold = get_optimal_threshold()

    try:
        result = predict_single_customer(
            customer.model_dump(),
            model, explainer, threshold
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    return {
        "customer_index": None,
        "churn_probability": result['churn_probability'],
        "risk_tier": result['risk_tier'],
        "predicted_churn": result['predicted_churn'],
        "top_risk_factors": result['top_risk_factors'],
        "explanation": result['explanation'],
        "clv_estimate": None,
        "revenue_at_risk": None
    }


@router.get("/feature-importance")
async def feature_importance():
    """
    Get global SHAP feature importance across all customers.
    """
    importance_df = get_feature_importance()
    return {"features": importance_df.to_dict(orient='records')}