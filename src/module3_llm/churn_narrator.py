import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module3_llm.llm_client import RetailLLMClient
from src.module3_llm.prompt_templates import (
    CHURN_EXPLANATION_TEMPLATE,
    CHURN_WIN_BACK_TEMPLATE,
    CHURN_SEGMENT_SUMMARY_TEMPLATE
)


class ChurnNarrator:
    """
    Converts churn model outputs into plain-English business narratives.

    Note: Recency was removed from the feature set in Week 2 due to
    data leakage, so it never appears anywhere in this module.
    DaysActive measures the span between a customer's first and last
    purchase — not recency, and not tenure/loyalty.
    """

    def __init__(self, llm_client: RetailLLMClient = None):
        self.client = llm_client or RetailLLMClient()

    def _purchase_span_label(self, days_active_val: float) -> str:
        if days_active_val <= 1:
            return "a single-day purchase pattern (all orders placed close together)"
        elif days_active_val <= 30:
            return "a narrow purchase window (under a month between first and last order)"
        else:
            return "a wide purchase window (over a month between first and last order)"

    def _risk_tone_instruction(self, probability: float, frequency: float = 0) -> str:
        if probability < 0.20:
            return (
                "This customer is NOT at risk. You must say this customer shows "
                "low churn risk. Do not use words like 'moderate', 'concerning', "
                "'significant', 'lost interest', or similar alarming language."
            )
        elif probability < 0.50:
            return (
                "This customer is a mild-to-moderate watch point, not an urgent "
                "concern. Describe the risk factors calmly."
            )
        elif probability < 0.70:
            return (
                "This customer is a real, high-priority risk worth active "
                "attention. Describe the risk factors as serious."
            )
        else:
            if frequency >= 5:
                return (
                    "This customer has a very high churn probability, and has "
                    "placed several orders, showing a real repeat-purchase "
                    "relationship. Describe the risk as serious and call "
                    "retention a genuine priority."
                )
            elif frequency > 1:
                return (
                    "This customer has a very high churn probability and has "
                    "placed only a couple of orders — some repeat behavior, but "
                    "thin evidence of a strong relationship. Describe the risk "
                    "as serious, but note that retention priority here is "
                    "moderate, not as strong as for a customer with many orders."
                )
            else:
                return (
                    "This customer has a very high churn probability but only "
                    "one order on record, showing no repeat-purchase "
                    "relationship. Describe the risk as high, but state clearly "
                    "that this is not a strong retention priority."
                )

    def _segment_tone_instruction(self, avg_probability: float, one_time_buyer_segment: bool) -> str:
        if avg_probability < 0.20:
            return (
                "This segment is NOT at risk. Do not describe them as at risk of "
                "defecting. Recommend maintaining satisfaction, not active retention spend."
            )
        elif avg_probability < 0.50:
            return "This is a mild-to-moderate watch-point segment."
        elif avg_probability < 0.70:
            return (
                "This is the TOP retention priority segment. Recommend active "
                "retention investment such as loyalty offers or personalized outreach."
            )
        else:
            if one_time_buyer_segment:
                return (
                    "This segment has a high churn probability, but nearly all "
                    "customers are one-time buyers with no repeat-purchase history. "
                    "You MUST recommend AGAINST heavy retention investment here, "
                    "and state it is lower-ROI than the High Risk segment, despite "
                    "the higher probability."
                )
            else:
                return (
                    "This segment has a high churn probability and includes "
                    "customers with some repeat-purchase history, so active "
                    "retention investment is worthwhile."
                )

    def _format_shap_factors(self,
                              customer_row: pd.Series,
                              shap_row: pd.Series,
                              feature_cols: list,
                              top_n: int = 3) -> str:

        feature_descriptions = {
            'Frequency': 'Number of orders placed',
            'Monetary': 'Total spend',
            'AvgOrderValue': 'Average order value',
            'UniqueProducts': 'Number of unique products bought',
            'AvgQuantity': 'Average quantity per order',
            'DaysActive': 'Span between first and last purchase (days)',
            'OrdersPerDay': 'Order frequency rate'
        }

        shap_df = pd.DataFrame({
            'feature': feature_cols,
            'value': [customer_row.get(f, 0) for f in feature_cols],
            'shap_value': [shap_row.get(f'shap_{f}', 0) for f in feature_cols]
        }).sort_values('shap_value', key=abs, ascending=False)

        top_factors = shap_df[shap_df['shap_value'] > 0].head(top_n)
        if len(top_factors) == 0:
            top_factors = shap_df.head(top_n)

        factor_lines = []
        for i, (_, row) in enumerate(top_factors.iterrows()):
            description = feature_descriptions.get(row['feature'], row['feature'])
            direction = "increases" if row['shap_value'] > 0 else "decreases"
            if row['feature'] == 'DaysActive':
                value_display = self._purchase_span_label(row['value'])
            else:
                value_display = f"{row['value']:.1f}"
            factor_lines.append(
                f"{i+1}. {description}: {value_display} — {direction} churn risk"
            )

        return '\n'.join(factor_lines)

    def explain_customer(self,
                          customer_row: pd.Series,
                          shap_row: pd.Series,
                          feature_cols: list) -> str:

        shap_factors = self._format_shap_factors(customer_row, shap_row, feature_cols)
        purchase_span_label = self._purchase_span_label(customer_row.get('DaysActive', 0))

        risk_tone_instruction = self._risk_tone_instruction(
            customer_row.get('ChurnProbability', 0),
            customer_row.get('Frequency', 0)
        )

        return self.client.generate_from_template(
            CHURN_EXPLANATION_TEMPLATE,
            {
                'frequency': customer_row.get('Frequency', 0),
                'monetary': customer_row.get('Monetary', 0),
                'avg_order_value': customer_row.get('AvgOrderValue', 0),
                'unique_products': customer_row.get('UniqueProducts', 0),
                'purchase_span_label': purchase_span_label,
                'churn_probability': customer_row.get('ChurnProbability', 0),
                'risk_tone_instruction': risk_tone_instruction,
                'shap_factors': shap_factors
            },
            max_words=80
        )

    def generate_winback(self,
                          customer_row: pd.Series,
                          shap_row: pd.Series,
                          feature_cols: list,
                          risk_tier: str,
                          favorite_product: str = None) -> str:

        shap_df = pd.DataFrame({
            'feature': feature_cols,
            'shap_value': [shap_row.get(f'shap_{f}', 0) for f in feature_cols]
        }).sort_values('shap_value', ascending=False)

        primary_factor = shap_df.iloc[0]['feature'] if len(shap_df) > 0 else 'inactivity'

        feature_to_reason = {
            'Frequency': 'infrequent purchasing behavior',
            'Monetary': 'declining spend',
            'AvgOrderValue': 'declining order values',
            'UniqueProducts': 'narrowing product interest',
            'DaysActive': 'a short purchase history so far',
            'OrdersPerDay': 'slowing purchase rate'
        }

        primary_reason = feature_to_reason.get(primary_factor, 'reduced engagement')
        purchase_span_label = self._purchase_span_label(customer_row.get('DaysActive', 0))

        return self.client.generate_from_template(
            CHURN_WIN_BACK_TEMPLATE,
            {
                'risk_tier': risk_tier,
                'avg_order_value': customer_row.get('AvgOrderValue', 0),
                'purchase_span_label': purchase_span_label,
                'favorite_product': favorite_product if favorite_product else "unknown",
                'primary_risk_factor': primary_reason
            },
            max_words=120
        )

    def summarize_segment(self,
                           risk_tier: str,
                           segment_df: pd.DataFrame) -> str:

        customer_count = len(segment_df)
        revenue_at_risk = segment_df['RevenueAtRisk'].sum()
        avg_monetary = segment_df['Monetary'].mean()
        median_frequency = segment_df['Frequency'].median()
        p75_frequency = segment_df['Frequency'].quantile(0.75)
        mean_days_active = segment_df['DaysActive'].mean()
        avg_churn_probability = segment_df['ChurnProbability'].mean()

        top_driver = (
            segment_df['TopChurnDriver'].value_counts().index[0]
            if len(segment_df) > 0 else 'Unknown'
        )

        feature_to_description = {
            'Frequency': 'low purchase frequency',
            'Monetary': 'declining spend',
            'AvgOrderValue': 'falling order values',
            'UniqueProducts': 'narrowing product interests',
            'DaysActive': 'a short span between first and last purchase',
            'OrdersPerDay': 'slowing purchase rate'
        }
        top_driver_description = feature_to_description.get(top_driver, top_driver)

        if median_frequency <= 1 and p75_frequency <= 1 and mean_days_active < 10:
            customer_behavior_context = (
                "Nearly all customers in this segment made only one purchase and never "
                "returned — they have no established repeat-purchase relationship "
                "with the business."
            )
        else:
            customer_behavior_context = (
                "While frequency is low on average, a meaningful share of customers in "
                "this segment have purchased more than once or have a longer span between "
                "their first and last purchase, showing some real repeat-purchase behavior."
            )

        one_time_buyer_segment = (median_frequency <= 1 and p75_frequency <= 1 and mean_days_active < 10)
        segment_tone_instruction = self._segment_tone_instruction(avg_churn_probability, one_time_buyer_segment)

        return self.client.generate_from_template(
            CHURN_SEGMENT_SUMMARY_TEMPLATE,
            {
                'risk_tier': risk_tier,
                'customer_count': customer_count,
                'avg_churn_probability': avg_churn_probability,
                'revenue_at_risk': revenue_at_risk,
                'avg_monetary': avg_monetary,
                'top_churn_driver': top_driver_description,
                'customer_behavior_context': customer_behavior_context,
                'segment_tone_instruction': segment_tone_instruction,
            },
            max_words=80
        )

    def narrate_all_customers(self,
                               risk_table: pd.DataFrame,
                               shap_values_df: pd.DataFrame,
                               feature_cols: list) -> pd.DataFrame:

        print(f"Generating LLM explanations for {len(risk_table)} customers...")
        explanations = []

        for i in range(len(risk_table)):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(risk_table)} customers...")

            customer_row = risk_table.iloc[i]
            shap_row = shap_values_df.iloc[i] if i < len(shap_values_df) else pd.Series()

            explanation = self.explain_customer(customer_row, shap_row, feature_cols)
            explanations.append(explanation)

        risk_table = risk_table.copy()
        risk_table['LLM_Explanation'] = explanations

        print(f"Generated {len(explanations)} LLM explanations")
        return risk_table


def run_churn_narration():
    print("Loading data...")
    risk_table = pd.read_csv('data/processed/customer_risk_table.csv')
    shap_values_df = pd.read_csv('data/processed/shap_values.csv')

    feature_cols = [
        'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts',
        'AvgQuantity', 'DaysActive', 'OrdersPerDay'
    ]

    risk_tiers = ['⚫ Extreme Risk', '🔴 High Risk', '🟡 Medium Risk', '🟢 Low Risk']

    narrator = ChurnNarrator()

    print("\nTesting on one sample customer per tier...")
    for tier in risk_tiers:
        tier_customers = risk_table[risk_table['RiskTier'] == tier]
        if len(tier_customers) == 0:
            continue

        sample_idx = tier_customers.index[0]
        customer_row = risk_table.loc[sample_idx]
        shap_row = shap_values_df.iloc[sample_idx] if sample_idx < len(shap_values_df) else pd.Series()

        explanation = narrator.explain_customer(customer_row, shap_row, feature_cols)

        print(f"\n{tier} Customer:")
        print(f"  Churn Prob: {customer_row['ChurnProbability']:.1%}")
        print(f"  Explanation: {explanation}")

    print("\nGenerating segment summaries...")
    segment_summaries = {}
    for tier in risk_tiers:
        tier_df = risk_table[risk_table['RiskTier'] == tier]
        if len(tier_df) > 0:
            summary = narrator.summarize_segment(tier, tier_df)
            segment_summaries[tier] = summary
            print(f"\n{tier} Summary:")
            print(f"  {summary}")

    pd.DataFrame([
        {'RiskTier': k, 'Summary': v} for k, v in segment_summaries.items()
    ]).to_csv('data/processed/segment_summaries.csv', index=False)
    print("\nSegment summaries saved")

    return narrator, segment_summaries


if __name__ == "__main__":
    narrator, summaries = run_churn_narration()