import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.module3_llm.llm_client import RetailLLMClient
from src.module3_llm.prompt_templates import (
    DEMAND_FORECAST_SUMMARY_TEMPLATE,
    DEMAND_ALERT_EXPLANATION_TEMPLATE,
    DEMAND_WEEKLY_REPORT_TEMPLATE
)


class DemandNarrator:
    """
    Converts demand forecast outputs into plain-English business narratives.

    Note: the underlying Prophet model is trend-only — yearly seasonality,
    weekly seasonality, and UK holidays were all disabled in Week 4. Prompts
    must never reference seasonality or holiday effects.

    Two products (23843, 23166) are permanently flagged low-reliability
    (DataQualityFlag != 'OK', AlertLevel == 'unreliable'). These must never
    be narrated with confident specific numbers.
    """

    def __init__(self, llm_client: RetailLLMClient = None):
        self.client = llm_client or RetailLLMClient()

    def _reliability_instruction(self, is_unreliable: bool) -> str:
        if is_unreliable:
            return (
                "This product's forecast has been flagged as low reliability due to "
                "sparse or spike-driven sales history. Do NOT state the forecasted "
                "demand number with confidence. Instead, say the forecast for this "
                "product cannot be trusted and recommend manual review by a staff member."
            )
        else:
            return (
                "This forecast is considered reliable. You may state the forecasted "
                "demand number normally."
            )

    def explain_product_forecast(self,
                                  alert_row: pd.Series,
                                  trend_direction: str,
                                  forecast_weeks: int = 4) -> str:
        """
        Generate plain-English explanation for a product's forecast.

        Parameters:
        - alert_row: one row from inventory_alerts.csv
        - trend_direction: 'Growing' or 'Declining', from seasonality_analysis.csv
        - forecast_weeks: number of weeks the forecast covers
        """

        is_unreliable = str(alert_row.get('DataQualityFlag', 'OK')) != 'OK'
        reliability_instruction = self._reliability_instruction(is_unreliable)

        return self.client.generate_from_template(
            DEMAND_FORECAST_SUMMARY_TEMPLATE,
            {
                'product_name': str(alert_row.get('Description', 'Unknown'))[:50],
                'stock_code': str(alert_row.get('StockCode', 'Unknown')),
                'current_stock': float(alert_row.get('EstimatedCurrentStock', 0)),
                'forecast_weeks': forecast_weeks,
                'forecasted_demand': float(alert_row.get('ForecastedDemand_Reorder', 0)),
                'lower_bound': float(alert_row.get('DemandLower_Reorder', 0)),
                'upper_bound': float(alert_row.get('DemandUpper_Reorder', 0)),
                'alert_status': str(alert_row.get('AlertStatus', 'Unknown')),
                'trend_direction': trend_direction,
                'reliability_instruction': reliability_instruction
            },
            max_words=80
        )
    
    def generate_daily_briefing(self, alerts_df: pd.DataFrame) -> str:
        """
        Generate a daily inventory briefing covering all products.
        """

        stockout_count = (alerts_df['AlertLevel'] == 'critical').sum()
        reorder_count = (alerts_df['AlertLevel'] == 'warning').sum()
        adequate_count = (alerts_df['AlertLevel'] == 'ok').sum()
        unreliable_count = (alerts_df['AlertLevel'] == 'unreliable').sum()

        critical_products = alerts_df[alerts_df['AlertLevel'] == 'critical']

        if len(critical_products) > 0:
            most_urgent = critical_products.iloc[0]
            most_urgent_product = (
                f"{most_urgent['StockCode']} "
                f"({str(most_urgent['Description'])[:30]})"
            )
            most_urgent_reason = (
                f"stock ({most_urgent['EstimatedCurrentStock']:.0f}) "
                f"below reorder point ({most_urgent['ReorderPoint']:.0f})"
            )
        elif len(alerts_df[alerts_df['AlertLevel'] == 'warning']) > 0:
            warning_product = alerts_df[alerts_df['AlertLevel'] == 'warning'].iloc[0]
            most_urgent_product = (
                f"{warning_product['StockCode']} "
                f"({str(warning_product['Description'])[:30]})"
            )
            most_urgent_reason = "approaching reorder point"
        else:
            most_urgent_product = "None"
            most_urgent_reason = "all products adequately stocked"

        return self.client.generate_from_template(
            DEMAND_ALERT_EXPLANATION_TEMPLATE,
            {
                'stockout_count': int(stockout_count),
                'reorder_count': int(reorder_count),
                'adequate_count': int(adequate_count),
                'unreliable_count': int(unreliable_count),
                'most_urgent_product': most_urgent_product,
                'most_urgent_reason': most_urgent_reason
            },
            max_words=80
        )
        
    def generate_weekly_report(self,
                                forecasts_df: pd.DataFrame,
                                folds_df: pd.DataFrame) -> str:
        """
        Generate weekly demand summary report.
        """

        # Filter out unreliable products for the report
        reliable_forecasts = forecasts_df[forecasts_df['DataQualityFlag'] == 'OK'].copy()

        future = reliable_forecasts[reliable_forecasts['IsFuture'] == True].copy()
        next_week = future.groupby('StockCode').first().reset_index()

        total_products = len(next_week)

        historical = reliable_forecasts[~reliable_forecasts['IsFuture']].copy()
        last_week = historical.groupby('StockCode').last().reset_index()

        merged = next_week.merge(
            last_week[['StockCode', 'yhat']].rename(columns={'yhat': 'last_week_yhat'}),
            on='StockCode', how='left'
        )

        merged['WeeklyChange'] = merged['yhat'] - merged['last_week_yhat']
        growing = (merged['WeeklyChange'] > 0).sum()
        declining = (merged['WeeklyChange'] <= 0).sum()

        # Note: merged is already reliable-only because it's built from reliable_forecasts
        top_product_row = merged.nlargest(1, 'yhat').iloc[0]
        bottom_product_row = merged.nsmallest(1, 'yhat').iloc[0]

        # Safely get description, fallback to stock code, then truncate
        top_desc = top_product_row.get('Description', top_product_row['StockCode'])
        bottom_desc = bottom_product_row.get('Description', bottom_product_row['StockCode'])
        top_product = str(top_desc)[:40]
        bottom_product = str(bottom_desc)[:40]

        avg_mape = folds_df['Prophet_MAPE'].mean()

        return self.client.generate_from_template(
            DEMAND_WEEKLY_REPORT_TEMPLATE,
            {
                'total_products': total_products,
                'growing_products': int(growing),
                'declining_products': int(declining),
                'top_product': top_product,
                'top_demand': float(top_product_row['yhat']),
                'bottom_product': bottom_product,
                'bottom_demand': float(bottom_product_row['yhat']),
                'avg_mape': float(avg_mape)
            },
            max_words=100
        )