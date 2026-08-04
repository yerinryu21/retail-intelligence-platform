import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module3_llm.llm_client import RetailLLMClient
from src.module3_llm.prompt_templates import (
    NL_QUERY_ROUTER_TEMPLATE,
    NL_CHURN_QUERY_TEMPLATE,
    NL_DEMAND_QUERY_TEMPLATE
)


class QueryEngine:
    """
    Natural language query interface for the Retail Intelligence Platform.

    Flow:
    1. Route question to CHURN, DEMAND, BOTH, or UNKNOWN
    2. Prepare relevant context data from the appropriate module
    3. Generate grounded answer using LLM + data context

    This is grounded generation — the LLM can only answer using data
    provided in the prompt, not from its training data. This prevents
    hallucination and keeps answers relevant.
    """

    def __init__(self,
                 risk_table_path: str = 'data/processed/customer_risk_table.csv',
                 alerts_path: str = 'data/processed/inventory_alerts.csv',
                 folds_path: str = 'data/processed/walkforward_folds.csv',
                 llm_client: RetailLLMClient = None):

        self.client = llm_client or RetailLLMClient()

        print("Loading data for query engine...")
        self.risk_table = pd.read_csv(risk_table_path)
        self.alerts = pd.read_csv(alerts_path)
        self.folds = pd.read_csv(folds_path)
        print("Query engine ready")

    def _route_question(self, question: str) -> str:
        route = self.client.generate_from_template(
            NL_QUERY_ROUTER_TEMPLATE,
            {'question': question},
            max_words=5
        )

        route = route.strip().upper()

        for valid_route in ['CHURN', 'DEMAND', 'BOTH', 'UNKNOWN']:
            if valid_route in route:
                return valid_route

        return 'UNKNOWN'
    def _prepare_churn_context(self) -> dict:
        """Prepare churn data summary for query context"""

        total = len(self.risk_table)
        extreme_risk = (self.risk_table['RiskTier'] == '⚫ Extreme Risk').sum()
        high_risk = (self.risk_table['RiskTier'] == '🔴 High Risk').sum()
        medium_risk = (self.risk_table['RiskTier'] == '🟡 Medium Risk').sum()
        low_risk = (self.risk_table['RiskTier'] == '🟢 Low Risk').sum()
        total_revenue = self.risk_table['RevenueAtRisk'].sum()
        top_driver = self.risk_table['TopChurnDriver'].value_counts().index[0]
        avg_prob = self.risk_table['ChurnProbability'].mean()

        top_customers = self.risk_table.nlargest(3, 'ChurnProbability')
        top_customer_str = '\n'.join([
            f"  - Customer {row.name}: "
            f"{row['ChurnProbability']:.1%} risk, "
            f"£{row['CLV']:.2f} CLV, "
            f"{row['RiskTier']}"
            for _, row in top_customers.iterrows()
        ])

        return {
            'total_customers': total,
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'extreme_risk_count': extreme_risk,
            'total_revenue_at_risk': total_revenue,
            'top_churn_driver': top_driver,
            'avg_churn_prob': avg_prob,
            'top_customers': top_customer_str
        }

    def _prepare_demand_context(self) -> dict:
        """Prepare demand data summary for query context"""

        total_products = len(self.alerts)
        stockout_count = (self.alerts['AlertLevel'] == 'critical').sum()
        reorder_count = (self.alerts['AlertLevel'] == 'warning').sum()
        unreliable_count = (self.alerts['AlertLevel'] == 'unreliable').sum()
        avg_mape = self.folds['Prophet_MAPE'].mean()

        total_forecasted = self.alerts['ForecastedDemand_Reorder'].sum()

        top_demand = self.alerts.nlargest(3, 'ForecastedDemand_Reorder')
        top_products_str = '\n'.join([
            f"  - {row['StockCode']} ({str(row['Description'])[:30]}): "
            f"{row['ForecastedDemand_Reorder']:.0f} units forecast"
            for _, row in top_demand.iterrows()
        ])

        stockout_products = self.alerts[self.alerts['AlertLevel'] == 'critical']
        if len(stockout_products) > 0:
            stockout_str = '\n'.join([
                f"  - {row['StockCode']}: "
                f"stock={row['EstimatedCurrentStock']:.0f}, "
                f"need={row['ForecastedDemand_Reorder']:.0f}"
                for _, row in stockout_products.iterrows()
            ])
        else:
            stockout_str = "  None — all products adequately stocked"

        return {
            'total_products': total_products,
            'stockout_count': int(stockout_count),
            'reorder_count': int(reorder_count),
            'unreliable_count': int(unreliable_count),
            'avg_mape': float(avg_mape),
            'total_forecasted': float(total_forecasted),
            'top_products': top_products_str,
            'stockout_products': stockout_str
        }
        
    def answer(self, question: str) -> dict:
        """
        Answer a plain-English question about the retail data.
        """

        route = self._route_question(question)

        if route == 'CHURN':
            context = self._prepare_churn_context()
            answer_text = self.client.generate_from_template(
                NL_CHURN_QUERY_TEMPLATE,
                {**context, 'question': question},
                max_words=100
            )

        elif route == 'DEMAND':
            context = self._prepare_demand_context()
            answer_text = self.client.generate_from_template(
                NL_DEMAND_QUERY_TEMPLATE,
                {**context, 'question': question},
                max_words=100
            )

        elif route == 'BOTH':
            churn_context = self._prepare_churn_context()
            demand_context = self._prepare_demand_context()
            
        
            extreme_risk_df = self.risk_table[self.risk_table['RiskTier'] == '⚫ Extreme Risk']
            extreme_risk_median_freq = extreme_risk_df['Frequency'].median() if len(extreme_risk_df) > 0 else None
            extreme_risk_note = (
                "Extreme risk customers are mostly one-time buyers with no repeat-purchase history "
                "(median orders per customer: {:.0f}), making them a lower retention priority than "
                "High risk customers despite the higher churn probability.".format(extreme_risk_median_freq)
                if extreme_risk_median_freq is not None and extreme_risk_median_freq <= 1
                else "Extreme risk customers show elevated churn probability."
                )
            
            combined_prompt = f"""You are a retail analytics assistant for an online retail store.

Customer Data:
- Total customers: {churn_context['total_customers']}
- Extreme risk customers: {churn_context['extreme_risk_count']}
- Note on extreme risk customers: {extreme_risk_note}
- High risk customers: {churn_context['high_risk_count']}
- Total revenue at risk across ALL customer tiers combined: £{churn_context['total_revenue_at_risk']:.2f}

Inventory Data:
- Products tracked: {demand_context['total_products']}
- Stockout risk products: {demand_context['stockout_count']}
- Reorder needed: {demand_context['reorder_count']}
- Unreliable forecasts: {demand_context['unreliable_count']}

Question: {question}

Answer in 3 sentences using only the data above.
Be specific with numbers."""

            answer_text = self.client.generate(combined_prompt, max_words=100)

        else:
            answer_text = ("I don't have data to answer that question. "
                            "I can answer questions about customer churn risk, "
                            "inventory levels, demand forecasts, and reorder recommendations.")

        return {
            'question': question,
            'route': route,
            'answer': answer_text
        }