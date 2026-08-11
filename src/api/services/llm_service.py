"""
LLM narration service layer.

Notes on real method signatures (confirmed against source, not the plan doc):
- ChurnNarrator.explain_customer expects raw customer_row/shap_row Series,
  same format as churn_service.get_customer_by_index already returns.
- WinBackGenerator.generate_for_customer requires clv_median (float) and
  optional retail_df (for favorite-product lookup) — both computed here.
- DemandNarrator.explain_product_forecast requires trend_direction, sourced
  from seasonality_analysis.csv (not from all_forecasts.csv).
- LLM calls are slow (2-15s) — no caching on results themselves, only on
  the loaded model/client objects (handled in dependencies.py).
"""

import pandas as pd
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.api.config import settings
from src.api.services.churn_service import get_customer_by_index


def get_trend_direction(seasonality_df: pd.DataFrame, stock_code: str) -> str:
    """Look up a product's trend direction from seasonality_analysis.csv"""
    row = seasonality_df[seasonality_df['StockCode'] == stock_code]
    if len(row) == 0:
        return "Stable"
    return str(row.iloc[0].get('TrendDirection', 'Stable'))


def explain_churn_customer(customer_index: int,
                            risk_table: pd.DataFrame,
                            shap_values_df: pd.DataFrame,
                            narrator) -> Dict:
    """Generate LLM explanation for a customer's churn risk"""

    customer_row, shap_row = get_customer_by_index(
        risk_table, shap_values_df, customer_index
    )

    explanation = narrator.explain_customer(
        customer_row, shap_row, settings.FEATURE_COLS
    )

    return {
        "customer_index": customer_index,
        "churn_probability": float(customer_row.get('ChurnProbability', 0)),
        "risk_tier": str(customer_row.get('RiskTier', 'Unknown')),
        "explanation": explanation
    }


def generate_winback_message(customer_index: int,
                              risk_table: pd.DataFrame,
                              shap_values_df: pd.DataFrame,
                              generator,
                              retail_df: pd.DataFrame = None) -> Dict:
    """Generate personalized win-back message for a customer"""

    customer_row, shap_row = get_customer_by_index(
        risk_table, shap_values_df, customer_index
    )

    clv_median = float(risk_table['CLV'].median())

    result = generator.generate_for_customer(
        customer_row, shap_row, settings.FEATURE_COLS,
        clv_median, retail_df
    )


    return {
        "customer_index": customer_index,
        "risk_tier": result['risk_tier'],
        "churn_probability": result['churn_probability'],
        "favorite_product": result.get('favorite_product'),
        "explanation": result['explanation'],
        "email_subject": result['email_subject'],
        "email_body": result['email_body'],
        "recommended_incentive": result['recommended_incentive'],
        "urgency": result['urgency']
    }


def get_daily_briefing(risk_table: pd.DataFrame,
                        alerts_df: pd.DataFrame,
                        churn_narrator,
                        demand_narrator) -> Dict:
    """Generate combined daily briefing for churn and inventory"""

    high_risk = risk_table[risk_table['RiskTier'].isin(['🔴 High Risk', '⚫ Extreme Risk'])]

    churn_briefing = ""
    if len(high_risk) > 0:
        churn_briefing = churn_narrator.summarize_segment('🔴 High Risk', high_risk)

    inventory_briefing = demand_narrator.generate_daily_briefing(alerts_df)

    return {
        "churn_briefing": churn_briefing,
        "inventory_briefing": inventory_briefing,
        "high_risk_customers": len(high_risk),
        "stockout_products": int((alerts_df['AlertLevel'] == 'critical').sum())
    }


def answer_nl_query(question: str, engine) -> Dict:
    """Answer a natural language query about the retail data"""
    result = engine.answer(question)
    return {
        "question": result.get('question', question),
        "route": result.get('route', 'UNKNOWN'),
        "answer": result.get('answer', '')
    }


def explain_demand_product(stock_code: str,
                            alerts_df: pd.DataFrame,
                            seasonality_df: pd.DataFrame,
                            narrator) -> Dict:
    """Generate LLM explanation for a product's inventory/forecast situation"""

    product_alert = alerts_df[alerts_df['StockCode'] == stock_code]
    if len(product_alert) == 0:
        raise ValueError(f"Product {stock_code} not found in alerts")

    alert_row = product_alert.iloc[0]
    trend_direction = get_trend_direction(seasonality_df, stock_code)

    explanation = narrator.explain_product_forecast(
        alert_row, trend_direction, forecast_weeks=4
    )

    return {
        "stock_code": stock_code,
        "alert_status": str(alert_row.get('AlertStatus', 'Unknown')),
        "trend_direction": trend_direction,
        "explanation": explanation
    }


def get_segment_summaries(risk_table: pd.DataFrame, narrator) -> Dict[str, str]:
    """Generate an LLM summary for each risk tier separately, matching
    the original Streamlit loop over all 4 tiers."""
    
    tiers = ['⚫ Extreme Risk', '🔴 High Risk', '🟡 Medium Risk', '🟢 Low Risk']
    summaries = {}

    for tier in tiers:
        tier_df = risk_table[risk_table['RiskTier'] == tier]
        if len(tier_df) > 0:
            summaries[tier] = narrator.summarize_segment(tier, tier_df)

    return summaries


def get_weekly_report(forecasts_df: pd.DataFrame,
                       folds_df: pd.DataFrame,
                       narrator) -> str:
    """Generate the weekly demand report via DemandNarrator.generate_weekly_report"""
    return narrator.generate_weekly_report(forecasts_df, folds_df)