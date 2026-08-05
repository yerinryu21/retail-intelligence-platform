"""
HTTP client for the Retail Intelligence Platform API.
Streamlit uses this to communicate with the FastAPI backend
instead of importing and calling model/data functions directly.
"""


import requests
import pandas as pd
import streamlit as st
from typing import Dict, Optional

API_BASE_URL = "http://localhost:8000"


class RetailAPIClient:
    """HTTP client for the Retail Intelligence API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running: `python src/api/main.py`")
            return {}
        except requests.exceptions.Timeout:
            st.error("API request timed out. Please try again.")
            return {}
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get('detail', str(e))
            except Exception:
                detail = str(e)
            st.error(f"API error: {detail}")
            return {}

    def _post(self, endpoint: str, data: Dict = None) -> Dict:
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", json=data, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API.")
            return {}
        except requests.exceptions.Timeout:
            st.warning("LLM response is taking longer than expected...")
            return {}
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get('detail', str(e))
            except Exception:
                detail = str(e)
            st.error(f"API error: {detail}")
            return {}

    # ── Health ─────────────────────────────────────────────────────
    def health_check(self) -> Dict:
        return self._get("/health/")

    def models_health(self) -> Dict:
        return self._get("/health/models")
    
    
    # ── Churn ──────────────────────────────────────────────────────
    def get_churn_summary(self) -> Dict:
        return self._get("/churn/summary")

    def get_customers(self, page: int = 1, page_size: int = 20,
                       risk_tier: Optional[str] = None,
                       min_probability: float = 0.0,
                       sort_by: str = "RevenueAtRisk") -> Dict:
        params = {"page": page, "page_size": page_size,
                  "min_probability": min_probability, "sort_by": sort_by}
        if risk_tier:
            params["risk_tier"] = risk_tier
        return self._get("/churn/customers", params=params)


    def get_all_customers(self) -> pd.DataFrame:
        """
        Fetch every customer by paging through /churn/customers
        (server caps page_size at 100), stitched into one DataFrame
        shaped like customer_risk_table.csv for chart compatibility.
        """
        first_page = self.get_customers(page=1, page_size=100)
        if not first_page:
            return pd.DataFrame()

        total_pages = first_page.get("total_pages", 1)
        all_customers = list(first_page.get("customers", []))

        for page_num in range(2, total_pages + 1):
            page_data = self.get_customers(page=page_num, page_size=100)
            all_customers.extend(page_data.get("customers", []))

        df = pd.DataFrame(all_customers)

        column_map = {
            "customer_index": "CustomerIndex",
            "customer_id": "CustomerID",
            "risk_tier": "RiskTier",
            "churn_probability": "ChurnProbability",
            "frequency": "Frequency",
            "monetary": "Monetary",
            "avg_order_value": "AvgOrderValue",
            "unique_products": "UniqueProducts",
            "avg_quantity": "AvgQuantity",
            "days_active": "DaysActive",
            "orders_per_day": "OrdersPerDay",
            "clv": "CLV",
            "revenue_at_risk": "RevenueAtRisk",
            "top_churn_driver": "TopChurnDriver",
            "top_driver_direction": "TopDriverDirection",
            "explanation": "Explanation",
            "shap_row_index": "ShapRowIndex"
        }
        df = df.rename(columns=column_map)
        if "CustomerIndex" in df.columns:
            df = df.set_index("CustomerIndex")
        return df

    def get_customer_detail(self, customer_index: int) -> Dict:
        return self._get(f"/churn/customers/{customer_index}")

    def predict_churn(self, features: Dict) -> Dict:
        return self._post("/churn/predict", data=features)

    def get_feature_importance(self) -> Dict:
        return self._get("/churn/feature-importance")

    # ── Demand ─────────────────────────────────────────────────────
    def get_demand_summary(self) -> Dict:
        return self._get("/demand/summary")

    def get_alerts(self, alert_level: Optional[str] = None,
                    sort_by: str = "UrgencyRank") -> Dict:
        params = {"sort_by": sort_by}
        if alert_level:
            params["alert_level"] = alert_level
        return self._get("/demand/alerts", params=params)

    def get_products(self) -> Dict:
        return self._get("/demand/products")

    def get_product_detail(self, stock_code: str) -> Dict:
        return self._get(f"/demand/products/{stock_code}")

    def get_backtest_results(self, stock_code: Optional[str] = None) -> Dict:
        params = {}
        if stock_code:
            params["stock_code"] = stock_code
        return self._get("/demand/backtest", params=params)

    # ── LLM ────────────────────────────────────────────────────────
    def explain_churn(self, customer_index: int) -> Dict:
        return self._post(f"/llm/explain/churn/{customer_index}")

    def generate_winback(self, customer_index: int) -> Dict:
        return self._post(f"/llm/winback/{customer_index}")

    def query(self, question: str) -> Dict:
        return self._post("/llm/query", data={"question": question})

    def get_briefing(self) -> Dict:
        return self._get("/llm/briefing")

    def explain_demand(self, stock_code: str) -> Dict:
        return self._post(f"/llm/explain/demand/{stock_code}")

    def get_segment_summaries(self) -> Dict:
        return self._get("/llm/segment-summaries")

    def get_weekly_report(self) -> Dict:
        return self._get("/llm/weekly-report")


@st.cache_resource
def get_api_client() -> RetailAPIClient:
    """Cached API client — created once per Streamlit session"""
    return RetailAPIClient()