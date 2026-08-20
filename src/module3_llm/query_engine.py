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
    def _prepare_churn_context(self, top_n: int = 10) -> dict:
        """Prepare churn data summary for query context"""

        total = len(self.risk_table)
        extreme_risk = (self.risk_table['RiskTier'] == '⚫ Extreme Risk').sum()
        high_risk = (self.risk_table['RiskTier'] == '🔴 High Risk').sum()
        medium_risk = (self.risk_table['RiskTier'] == '🟡 Medium Risk').sum()
        low_risk = (self.risk_table['RiskTier'] == '🟢 Low Risk').sum()
        total_revenue = self.risk_table['RevenueAtRisk'].sum()
        extreme_revenue = self.risk_table[self.risk_table['RiskTier'] == '⚫ Extreme Risk']['RevenueAtRisk'].sum()
        top_driver = self.risk_table['TopChurnDriver'].value_counts().index[0]
        avg_prob = self.risk_table['ChurnProbability'].mean()

        tier_map = {
            '⚫ Extreme Risk': 'Extreme Risk',
            '🔴 High Risk': 'High Risk',
            '🟡 Medium Risk': 'Medium Risk',
            '🟢 Low Risk': 'Low Risk'
        }

        top_customers = self.risk_table.nlargest(top_n, 'ChurnProbability')
        top_customer_str = '\n'.join([
            f"{i+1}. Customer {idx}: "
            f"{row['ChurnProbability']:.1%} risk, "
            f"£{row['CLV']:,.0f} CLV, "
            f"{tier_map.get(row['RiskTier'], row['RiskTier'])}"
            for i, (idx, row) in enumerate(top_customers.iterrows())
        ])

        # Top N highest CLV customers for value-based queries
        top_clv = self.risk_table.nlargest(top_n, 'CLV')
        top_clv_str = '\n'.join([
            f"{i+1}. Customer {idx}: "
            f"£{row['CLV']:,.0f} CLV, "
            f"{row['ChurnProbability']:.1%} risk, "
            f"{tier_map.get(row['RiskTier'], row['RiskTier'])}"
            for i, (idx, row) in enumerate(top_clv.iterrows())
        ])

        # Top N highest revenue at risk customers among high and extreme risk (churn probability >= 50%)
        high_extreme_risk_df = self.risk_table[self.risk_table['ChurnProbability'] >= 0.5]
        # Remove duplicate indices to ensure each customer appears once
        unique_customers = high_extreme_risk_df[~high_extreme_risk_df.index.duplicated(keep='first')]
        top_revenue = unique_customers.sort_values('RevenueAtRisk', ascending=False).head(top_n)
        top_revenue_str = '\n'.join([
            f"{i+1}. Customer {idx}: "
            f"£{row['RevenueAtRisk']:,.0f} revenue at risk, "
            f"{row['ChurnProbability']:.1%} churn risk, "
            f"£{row['CLV']:,.0f} CLV, "
            f"{tier_map.get(row['RiskTier'], row['RiskTier'])}"
            for i, (idx, row) in enumerate(top_revenue.iterrows())
        ])

        return {
            'total_customers': total,
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'extreme_risk_count': extreme_risk,
            'total_revenue_at_risk': total_revenue,
            'extreme_revenue_at_risk': extreme_revenue,
            'top_churn_driver': top_driver,
            'avg_churn_prob': avg_prob,
            'top_customers': top_customer_str,
            'top_clv_customers': top_clv_str,
            'top_revenue_customers': top_revenue_str
        }

    def _prepare_demand_context(self, top_n: int = 15) -> dict:
        """Prepare demand data summary for query context"""

        # Filter out unreliable products for ranking lists
        reliable_alerts = self.alerts[self.alerts['AlertLevel'] != 'unreliable']

        total_products = len(self.alerts)
        stockout_count = (self.alerts['AlertLevel'] == 'critical').sum()
        reorder_count = (self.alerts['AlertLevel'] == 'warning').sum()
        unreliable_count = (self.alerts['AlertLevel'] == 'unreliable').sum()
        avg_mape = self.folds['Prophet_MAPE'].mean()

        total_forecasted = self.alerts['ForecastedDemand_Reorder'].sum()

        # Use reliable alerts for top demand ranking, limited to exactly top_n for precise selection
        top_demand = reliable_alerts.nlargest(top_n, 'ForecastedDemand_Reorder')
        # Format with numbering (1., 2., 3., ...) so LLM can present as-is
        top_products_lines = []
        for i, (_, row) in enumerate(top_demand.iterrows(), start=1):
            top_products_lines.append(
                f"{i}. {row['StockCode']} ({str(row['Description'])[:30]}): {row['ForecastedDemand_Reorder']:.0f} units forecast"
            )
        top_products_str = '\n'.join(top_products_lines)

        # Stockout products: reliable critical alerts, sorted by shortfall (need - stock) descending
        stockout_candidates = reliable_alerts[reliable_alerts['AlertLevel'] == 'critical'].copy()
        # Compute shortfall (how much need exceeds current stock)
        stockout_candidates['shortfall'] = stockout_candidates['ForecastedDemand_Reorder'] - stockout_candidates['EstimatedCurrentStock']
        stockout_sorted = stockout_candidates.sort_values('shortfall', ascending=False)
        # Take top N
        stockout_top = stockout_sorted.head(top_n)
        # Format with numbering - use natural business language (not key=value)
        stockout_lines = []
        for i, (_, row) in enumerate(stockout_top.iterrows(), start=1):
            stockout_lines.append(
                f"{i}. {row['StockCode']} ({str(row['Description'])[:30]}): "
                f"{row['EstimatedCurrentStock']:.0f} in stock, {row['ForecastedDemand_Reorder']:.0f} needed"
            )
        stockout_str = '\n'.join(stockout_lines) if stockout_lines else "  None — all products adequately stocked"

        # Unreliable products: sorted by StockCode for deterministic ordering
        unreliable_products = self.alerts[self.alerts['AlertLevel'] == 'unreliable'].copy()
        if len(unreliable_products) > 0:
            unreliable_products = unreliable_products.sort_values('StockCode')
            unreliable_lines = []
            for i, (_, row) in enumerate(unreliable_products.iterrows(), start=1):
                unreliable_lines.append(
                    f"{i}. {row['StockCode']} ({str(row['Description'])[:30]})"
                )
            unreliable_str = '\n'.join(unreliable_lines)
        else:
            unreliable_str = "  None — all products have reliable forecasts"

        return {
            'total_products': total_products,
            'stockout_count': int(stockout_count),
            'reorder_count': int(reorder_count),
            'unreliable_count': int(unreliable_count),
            'unreliable_products': unreliable_str,
            'avg_mape': float(avg_mape),
            'total_forecasted': float(total_forecasted),
            'top_products': top_products_str,
            'stockout_products': stockout_str
        }
        
    def _extract_top_n_from_question(self, question: str) -> int:
        """
        Extract a leading integer N from questions like 'what 5 products...', 'top 3 products',
        or 'top 10 high risk'. Defaults to 5 if no number found or parsing fails.
        """
        import re
        # Look for a number near words like product, item, top, or general top N patterns
        patterns = [
            r'what\s+(\d+)\s+products',
            r'which\s+(\d+)\s+products',
            r'top\s+(\d+)\s+products',
            r'(\d+)\s+products',
            r'(\d+)\s+items',
            r'top\s+(\d+)',  # General top N pattern
            # ADDED FOR "which N customers..." PHRASING
            r'what\s+(\d+)\s+customers',
            r'which\s+(\d+)\s+customers',
            r'top\s+(\d+)\s+customers',
            r'(\d+)\s+customers',
            r'which\s+(\d+)\s+customers\s+have',
            r'which\s+(\d+)\s+customers\s+we',
            r'the\s+(\d+)\s+customers\s+who',
            r'the\s+(\d+)\s+customers\s+that',
            r'(\d+)\s+customers\s+have',
            r'(\d+)\s+customers\s+we',
        ]
        for pat in patterns:
            m = re.search(pat, question, re.IGNORECASE)
            if m:
                try:
                    n = int(m.group(1))
                    if n > 0:
                        return n
                except ValueError:
                    pass
        # Default fallback
        return 5

    def _classify_churn_question(self, question: str) -> str:
        """
        Deterministically classify churn questions to determine which context list to serve.
        Returns one of: 'highest_risk', 'highest_clv', 'highest_revenue_at_risk', 'highest_priority'
        """
        import re
        question_lower = question.lower().strip()

        # Check for highest revenue at risk queries
        if re.search(r'top\s+\d+\s+highest\s+revenue\s+at\s+risk', question_lower) or \
           re.search(r'top\s+\d+\s+revenue\s+at\s+risk', question_lower):
            return 'highest_revenue_at_risk'

        # Check for highest CLV queries
        if re.search(r'top\s+\d+\s+highest\s+clv', question_lower) or \
           re.search(r'top\s+\d+\s+clv', question_lower) or \
           re.search(r'top\s+\d+\s+highest\s+lifetime\s+value', question_lower) or \
           re.search(r'which\s+(\d+)\s+customers\s+have\s+the\s+highest\s+clv', question_lower) or \
           re.search(r'which\s+(\d+)\s+customers\s+have\s+the\s+highest\s+lifetime\s+value', question_lower) or \
           re.search(r'the\s+(\d+)\s+customers\s+who\s+have\s+the\s+highest\s+clv', question_lower) or \
           re.search(r'the\s+(\d+)\s+customers\s+that\s+have\s+the\s+highest\s+clv', question_lower) or \
           re.search(r'which\s+(\d+)\s+customers\s+we\s+.*?the\s+highest\s+clv', question_lower) or \
           re.search(r'which\s+(\d+)\s+customers\s+we\s+.*?the\s+highest\s+lifetime\s+value', question_lower):
            return 'highest_clv'

        # Check for priority/retention queries (Type 2)
        priority_indicators = [
            r'priorit',
            r'retention\s+(?:focus|effort|target)',
            r'who\s+should\s+we\s+(?:focus on|target|prioritize)',
            r'priority\s+customer',
            r'highest\s+priority'
        ]
        for pattern in priority_indicators:
            if re.search(pattern, question_lower):
                return 'highest_priority'

        # Check for highest risk queries (including variations like 'high risk') - Type 1
        if re.search(r'top\s+\d+\s+highest?\s+risk', question_lower):
            return 'highest_risk'

        # Default to highest risk for general risk questions (Type 1)
        if 'risk' in question_lower:
            return 'highest_risk'

        # Fallback (should not happen for valid churn questions)
        return 'highest_risk'

    def answer(self, question: str) -> dict:
        """
        Answer a plain-English question about the retail data.
        """

        route = self._route_question(question)

        if route == 'CHURN':
            # Extract top N from question for churn queries as well
            original_top_n = self._extract_top_n_from_question(question)
            # Cap the list size at 15 for chat readability
            MAX_DISPLAY_ITEMS = 15
            display_top_n = min(original_top_n, MAX_DISPLAY_ITEMS)
            context = self._prepare_churn_context(top_n=display_top_n)
            # Deterministically classify the question to determine which list to serve
            question_type = self._classify_churn_question(question)

            # Build a simplified prompt that just presents the appropriate list
            if question_type == 'highest_risk':
                list_to_present = context['top_customers']
                list_name = f"Top {display_top_n} highest risk customers"
            elif question_type == 'highest_clv':
                list_to_present = context['top_clv_customers']
                list_name = f"Top {display_top_n} highest CLV customers"
            elif question_type == 'highest_revenue_at_risk':
                list_to_present = context['top_revenue_customers']
                list_name = f"Top {display_top_n} highest revenue at risk customers"
            elif question_type == 'highest_priority':
                # For priority questions, use revenue-at-risk list (combines risk and value)
                list_to_present = context['top_revenue_customers']
                list_name = f"Top {display_top_n} highest priority customers"
            else:
                # Fallback - should not happen
                list_to_present = context['top_customers']
                list_name = f"Top {display_top_n} highest risk customers"

            # Create a direct instruction prompt
            prompt = f"""You are a customer analytics assistant.

{list_name}:
{list_to_present}

User question: {question}

You MUST present the list EXACTLY as it appears above, preserving all formatting including newlines and spacing.
Do not add any introductory text, commentary, or explanations.
Do not modify, re-order, filter, or remove any items.
Do not combine multiple items onto fewer lines.
Present each item on its own line exactly as shown above."""

            answer_text = self.client.generate_from_template(
                """You are a customer analytics assistant.

{context}

User question: {question}

You MUST present the list EXACTLY as it appears above, preserving all formatting including newlines and spacing.
Do not add any introductory text, commentary, or explanations.
Do not modify, re-order, filter, or remove any items.
Do not combine multiple items onto fewer lines.
Present each item on its own line exactly as shown above.""",
                {'context': f"{list_name}:\n{list_to_present}", 'question': question},
                max_words=200
            )

            # Add note if original request was for more than we displayed
            if original_top_n > MAX_DISPLAY_ITEMS:
                answer_text += "\n\nNote: Showing top " + str(MAX_DISPLAY_ITEMS) + " of " + str(original_top_n) + " requested. For the full list, please export the data or request a smaller range (e.g., \"top 10\")."

        elif route == 'DEMAND':
            # Determine N from question for precise top-N selection
            top_n = self._extract_top_n_from_question(question)
            context = self._prepare_demand_context(top_n=top_n)
            answer_text = self.client.generate_from_template(
                NL_DEMAND_QUERY_TEMPLATE,
                {**context, 'question': question},
                max_words=100
            )

        elif route == 'BOTH':
            # Determine N from question for both churn and demand sides
            top_n = self._extract_top_n_from_question(question)
            churn_context = self._prepare_churn_context(top_n=top_n)
            demand_context = self._prepare_demand_context(top_n=top_n)


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