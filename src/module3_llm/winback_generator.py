import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module3_llm.llm_client import RetailLLMClient
from src.module3_llm.churn_narrator import ChurnNarrator


class WinBackGenerator:
    """
    Generates personalized win-back messages for at-risk customers.

    Incentive tiers are based on churn probability (using this project's
    actual tier boundaries: Low <20%, Medium 20-50%, High 50-70%,
    Extreme >70%) combined with CLV percentile within this dataset
    (median ~£816, since CLV is heavily right-skewed).

    Favorite product is looked up from real transaction data via
    CustomerID (added to customer_risk_table.csv after fixing the
    Week 2/3 pipeline gap that had dropped it).
    """

    def __init__(self, llm_client: RetailLLMClient = None):
        self.client = llm_client or RetailLLMClient()
        self.narrator = ChurnNarrator(self.client)

    def _get_favorite_product(self, customer_id, retail_df: pd.DataFrame) -> str:
        """
        Find this customer's most-purchased product by total quantity,
        using real transaction data.
        """
        customer_purchases = retail_df[retail_df['CustomerID'] == customer_id]

        if len(customer_purchases) == 0:
            return None

        product_totals = customer_purchases.groupby('Description')['Quantity'].sum()
        top_product = product_totals.idxmax()

        return str(top_product).strip()

    def generate_for_customer(self,
                               customer_row: pd.Series,
                               shap_row: pd.Series,
                               feature_cols: list,
                               clv_median: float,
                               retail_df: pd.DataFrame = None) -> dict:
        """
        Generate a complete win-back package for one customer.
        """

        risk_tier = customer_row.get('RiskTier', '🟡 Medium Risk')
        churn_prob = customer_row.get('ChurnProbability', 0.5)
        clv = customer_row.get('CLV', 0)
        is_high_value = clv >= clv_median

        if churn_prob >= 0.70 and is_high_value:
            incentive = "20% discount on next order"
            urgency = "Send immediately"
        elif churn_prob >= 0.70:
            incentive = "15% discount on next order"
            urgency = "Send within 24 hours"
        elif churn_prob >= 0.50:
            incentive = "15% discount on next order"
            urgency = "Send within 3 days"
        elif churn_prob >= 0.20:
            incentive = "Free shipping on next order"
            urgency = "Send within 1 week"
        else:
            incentive = "Early access to new arrivals"
            urgency = "No urgency — maintain satisfaction"

        favorite_product = None
        if retail_df is not None:
            customer_id = customer_row.get('CustomerID')
            if customer_id is not None:
                favorite_product = self._get_favorite_product(customer_id, retail_df)

        subject_prompt = f"""Write a short, personalized email subject line
for a customer win-back email for an online retail store.

Incentive being offered: {incentive}

Requirements:
- Maximum 8 words
- Warm and personal, not salesy
- Do not mention the discount amount in the subject
- Do not use words like 'urgent' or 'last chance'
- Do not invent or use a customer name
- Do not invent or use a store name, brand name, or placeholder like "[Store Name]"

Write only the subject line, nothing else."""

        subject = self.client.generate(subject_prompt, max_words=12)
        subject = subject.strip().strip('"').strip("'")

        email_body = self.narrator.generate_winback(
            customer_row, shap_row, feature_cols, risk_tier, favorite_product
        )

        return {
            'risk_tier': risk_tier,
            'churn_probability': churn_prob,
            'clv': clv,
            'favorite_product': favorite_product,
            'explanation': self.narrator.explain_customer(
                customer_row, shap_row, feature_cols
            ),
            'email_subject': subject,
            'email_body': email_body,
            'recommended_incentive': incentive,
            'urgency': urgency
        }

    def generate_batch(self,
                        risk_table: pd.DataFrame,
                        shap_values_df: pd.DataFrame,
                        feature_cols: list,
                        n_customers: int = 5,
                        tier_filter: str = '🔴 High Risk',
                        retail_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate win-back messages for top N customers in a tier,
        ranked by revenue at risk.
        """

        clv_median = risk_table['CLV'].median()

        tier_customers = risk_table[
            risk_table['RiskTier'] == tier_filter
        ].sort_values('RevenueAtRisk', ascending=False).head(n_customers)

        print(f"Generating win-back messages for top {len(tier_customers)} "
              f"{tier_filter} customers...")

        results = []

        for i, (idx, customer_row) in enumerate(tier_customers.iterrows()):
            print(f"  Customer {i+1}/{len(tier_customers)}...")

            shap_row = (shap_values_df.iloc[idx]
                        if idx < len(shap_values_df)
                        else pd.Series())

            winback = self.generate_for_customer(
                customer_row, shap_row, feature_cols, clv_median, retail_df
            )

            results.append({
                'CustomerIndex': idx,
                'RiskTier': winback['risk_tier'],
                'ChurnProbability': winback['churn_probability'],
                'CLV': winback['clv'],
                'FavoriteProduct': winback['favorite_product'],
                'Explanation': winback['explanation'],
                'EmailSubject': winback['email_subject'],
                'EmailBody': winback['email_body'],
                'RecommendedIncentive': winback['recommended_incentive'],
                'Urgency': winback['urgency']
            })

        results_df = pd.DataFrame(results)

        output_path = f'data/processed/winback_messages_{tier_filter.split()[-2] if len(tier_filter.split()) > 1 else "batch"}.csv'
        results_df.to_csv(output_path, index=False)
        print(f"Saved {len(results_df)} win-back messages to {output_path}")

        return results_df