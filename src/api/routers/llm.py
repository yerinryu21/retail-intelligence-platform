"""
LLM narration API endpoints.

Important: these endpoints call Ollama, which takes 2-15 seconds per call.
Always show a loading state in the frontend.

Endpoints:
- POST /llm/explain/churn/{customer_index} — explain churn risk
- POST /llm/winback/{customer_index} — generate win-back email
- POST /llm/query — natural language query
- GET /llm/briefing — daily business briefing
- POST /llm/explain/demand/{stock_code} — explain product forecast
"""

from fastapi import APIRouter, HTTPException
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.api.dependencies import (
    get_risk_table, get_shap_values, get_alerts, get_seasonality_analysis,
    get_retail_data, get_churn_narrator, get_demand_narrator,
    get_query_engine, get_winback_generator, get_forecasts, get_folds
)
from src.api.models.llm import (
    NLQueryRequest, NLQueryResponse,
    WinBackResponse, DailyBriefingResponse, ChurnExplanationResponse
)
from src.api.services.llm_service import (
    explain_churn_customer, generate_winback_message,
    get_daily_briefing, answer_nl_query, explain_demand_product,
    get_segment_summaries, get_weekly_report
)
router = APIRouter(prefix="/llm", tags=["LLM Narration"])


@router.post("/explain/churn/{customer_index}", response_model=ChurnExplanationResponse)
async def explain_churn(customer_index: int):
    """
    Generate plain-English explanation for a customer's churn risk.
    Uses SHAP values (correctly joined via ShapRowIndex) via Ollama Llama3.
    Note: Takes 3-8 seconds due to LLM inference.
    """

    risk_table = get_risk_table()
    shap_values_df = get_shap_values()
    narrator = get_churn_narrator()

    try:
        result = explain_churn_customer(customer_index, risk_table, shap_values_df, narrator)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    return result


@router.post("/winback/{customer_index}", response_model=WinBackResponse)
async def generate_winback(customer_index: int):
    """
    Generate a personalized win-back email for an at-risk customer,
    including favorite-product lookup from the full transaction log.
    Note: Takes 5-15 seconds due to multiple LLM calls.
    """

    risk_table = get_risk_table()
    shap_values_df = get_shap_values()
    generator = get_winback_generator()
    retail_df = get_retail_data()

    try:
        result = generate_winback_message(
            customer_index, risk_table, shap_values_df, generator, retail_df
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Win-back generation failed: {str(e)}")

    return result


@router.post("/query", response_model=NLQueryResponse)
async def natural_language_query(request: NLQueryRequest):
    """
    Answer a plain-English question about customers or inventory.
    Note: Takes 3-8 seconds due to LLM inference.
    """

    engine = get_query_engine()

    try:
        result = answer_nl_query(request.question, engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return result


@router.get("/briefing", response_model=DailyBriefingResponse)
async def get_briefing():
    """
    Generate today's business briefing covering churn (High + Extreme risk)
    and inventory status.
    Note: Takes 8-20 seconds due to multiple LLM calls.
    """

    risk_table = get_risk_table()
    alerts = get_alerts()
    churn_narrator = get_churn_narrator()
    demand_narrator = get_demand_narrator()

    try:
        result = get_daily_briefing(risk_table, alerts, churn_narrator, demand_narrator)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Briefing generation failed: {str(e)}")

    return result


@router.post("/explain/demand/{stock_code}")
async def explain_demand(stock_code: str):
    """
    Generate plain-English explanation for a product's inventory/forecast
    situation, using trend direction from seasonality_analysis.csv.
    Note: Takes 3-8 seconds due to LLM inference.
    """

    alerts = get_alerts()
    seasonality = get_seasonality_analysis()
    narrator = get_demand_narrator()

    try:
        result = explain_demand_product(stock_code.upper(), alerts, seasonality, narrator)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    return result

@router.get("/segment-summaries")
async def segment_summaries():
    """
    Generate an LLM summary for each of the 4 churn risk tiers separately.
    Note: Takes several seconds per tier due to LLM inference (up to ~20s total).
    """

    risk_table = get_risk_table()
    narrator = get_churn_narrator()

    try:
        result = get_segment_summaries(risk_table, narrator)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segment summary generation failed: {str(e)}")

    return {"summaries": result}


@router.get("/weekly-report")
async def weekly_report():
    """
    Generate the weekly demand narrative report across all tracked products.
    Note: Takes several seconds due to LLM inference over the full forecast set.
    """

    forecasts = get_forecasts()
    folds = get_folds()
    narrator = get_demand_narrator()

    try:
        result = get_weekly_report(forecasts, folds, narrator)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weekly report generation failed: {str(e)}")

    return {"report": result}