"""
All prompt templates for the Retail Intelligence Platform LLM layer.

Design principles:
- Every template has an explicit role definition
- Every template specifies exact output format
- Every template constrains length
- Every template avoids asking for technical ML terminology in the output
"""

# ======================================================================
# CHURN MODULE TEMPLATES
# ======================================================================


CHURN_EXPLANATION_TEMPLATE = """You are a customer retention analyst for an online retail store that sells a wide variety of products from many suppliers.
Do not refer to "our brand," "our brand's products," or imply the store sells its own branded products. Say "the store" or "us" instead.
This customer's estimated churn probability is {churn_probability:.0%}.

Customer data:
- Total orders placed: {frequency}
- Total spend: £{monetary:.2f}
- Average order value: £{avg_order_value:.2f}
- Unique products purchased: {unique_products}
- Purchase pattern: {purchase_span_label}

Top factors identified by the model:
{shap_factors}

Write exactly 2 sentences describing this customer's churn risk.
Use plain English a non-technical store manager would understand.
Do not mention machine learning, SHAP, or model scores.
Do not start your response with any preamble like "Here is" or "Summary:" — start directly with the explanation.

CRITICAL: "Purchase pattern" describes the GAP BETWEEN a customer's first and most recent
order — it does NOT mean the customer has stopped buying, gone quiet, or is inactive. A customer
with {frequency} orders is an ENGAGED, repeat customer regardless of how wide that gap is.

WRONG (do not write anything like this): "It's been months since their last purchase, suggesting
they may be losing interest or finding alternative sources."

CORRECT (write in this style instead): "This customer's orders are spread out over a longer
period, which the model associates with elevated churn risk, even though they remain an active,
repeat customer with {frequency} orders."

Do NOT invent any specific number of days, weeks, or months — no such recency figure was given
to you, and this dataset does not track time since last purchase at all.

{risk_tone_instruction}

Start with the most important risk factor."""


CHURN_WIN_BACK_TEMPLATE = """You are a customer retention specialist writing a personalized win-back message for an online retail store that sells a wide variety of products from many suppliers.
Do not refer to "our brand" or imply the store sells its own branded products.
Do not invent or use a customer name — you do not know their name. Start with a warm greeting that does not use any name, e.g. "Hi there," or "Hello,".
Do not sign off with a made-up sender name or placeholder like "[Your Name]" — sign off with something generic like "The Store Team" or leave it unsigned.
Do not invent or use a store name, brand name, or placeholder like "[Store Name]".

Customer profile:
- Risk level: {risk_tier}
- Average spend per order: £{avg_order_value:.2f}
- Purchase pattern: {purchase_span_label}
- Most purchased product: {favorite_product}

Primary reason they are at risk: {primary_risk_factor}

Write a short, warm, personalized email to win this customer back.
Requirements:
- Maximum 4 sentences
- Friendly and personal tone — not corporate
- If a most purchased product is given below and is not "unknown", reference it naturally (e.g. "we noticed you loved the X"). If it is "unknown", do not reference any specific product — reference their general purchase pattern instead.
- Do not mention specific spend amounts
- Include one specific incentive (discount, free shipping, or early access)
- Do not mention that we detected them as "at risk" or use any analytics language
- Do not start with any preamble — start directly with the email greeting
- End with a clear call to action"""


CHURN_SEGMENT_SUMMARY_TEMPLATE = """You are a retail analytics manager presenting insights to a business owner who runs an online retail store selling a wide variety of products from many suppliers.
Do not refer to "our brand," "our brand's products," or imply the store sells its own branded products. Say "the store" or "the business" instead.

Customer segment data:
- Segment: {risk_tier}
- Number of customers in segment: {customer_count}
- Average churn probability in this segment: {avg_churn_probability:.0%}
- Total revenue at risk: £{revenue_at_risk:.2f}
- Average total spend per customer: £{avg_monetary:.2f}
- Most common churn driver: {top_churn_driver}
- Customer purchase history: {customer_behavior_context}

Write exactly 2 sentences summarizing the business situation for this segment.
Then write exactly 1 sentence recommending the single most impactful action.

{segment_tone_instruction}

Do not start your response with any preamble like "Summary:" or "Here is" — start directly with the summary.
Use business language — focus on revenue impact and practical actions."""
# ======================================================================
# DEMAND MODULE TEMPLATES
# ======================================================================

DEMAND_FORECAST_SUMMARY_TEMPLATE = """You are an inventory analyst for an online retail store.

Product: {product_name} ({stock_code})
Current estimated stock: {current_stock:.0f} units
Forecast for next {forecast_weeks} weeks: {forecasted_demand:.0f} units
Forecast range: {lower_bound:.0f} to {upper_bound:.0f} units
Current inventory alert status: {alert_status}
Trend direction: {trend_direction}

{reliability_instruction}

Write exactly 2 sentences explaining the inventory situation for this product.
Use plain English that a store manager would understand.
Do not mention confidence intervals, Prophet, forecasting models, or statistical terms.
Do not mention seasonality, holidays, or seasonal patterns — this model does not account for them.
Focus on what action needs to be taken and why."""

DEMAND_ALERT_EXPLANATION_TEMPLATE = """You are an inventory manager giving a daily briefing for an online retail store.

Today's inventory situation:
- Products at stockout risk: {stockout_count}
- Products needing reorder soon: {reorder_count}
- Products with adequate stock: {adequate_count}
- Products with unreliable forecasts (cannot be confidently planned around): {unreliable_count}
- Most urgent product: {most_urgent_product} ({most_urgent_reason})

Write a 3-sentence daily inventory briefing for a store owner.
Be direct and action-oriented.
Prioritize the most urgent stockout-risk issues first.
If there are products with unreliable forecasts, mention them separately and note that their numbers should not be trusted for automatic reordering decisions.
Do not use technical forecasting language."""

DEMAND_WEEKLY_REPORT_TEMPLATE = """You are a retail analytics assistant generating a weekly business report for an online retail store.

This week's demand forecast summary:
- Total products tracked: {total_products}
- Products forecasted to grow next week: {growing_products}
- Products forecasted to decline next week: {declining_products}
- Highest demand product: {top_product} (forecast: {top_demand:.0f} units)
- Lowest demand product: {bottom_product} (forecast: {bottom_demand:.0f} units)
- Average forecast accuracy last week (MAPE): {avg_mape:.1f}%

Write a 3-sentence weekly demand summary for a store owner.
Focus on business implications — what does this mean for staffing, purchasing, and promotions?
End with one specific recommendation for this week."""

# ======================================================================
# NATURAL LANGUAGE QUERY TEMPLATES
# ======================================================================

NL_QUERY_ROUTER_TEMPLATE = """You are a data assistant that routes business questions to the right data source.

Available data sources:
1. CHURN — customer churn predictions, risk scores, revenue at risk
2. DEMAND — product demand forecasts, inventory levels, reorder alerts
3. BOTH — questions that need data from both modules
4. UNKNOWN — questions that cannot be answered with available data

User question: {question}

Reply with exactly one word: CHURN, DEMAND, BOTH, or UNKNOWN.
Nothing else."""

NL_CHURN_QUERY_TEMPLATE = """You are a customer analytics assistant with access to the following data:

Total customers analyzed: {total_customers}
Extreme risk customers (churn probability over 70%): {extreme_risk_count}
High risk customers (50-70%): {high_risk_count}
Medium risk customers (20-50%): {medium_risk_count}
Low risk customers (under 20%): {low_risk_count}
Total revenue at risk: £{total_revenue_at_risk:.2f}
Most common churn driver: {top_churn_driver}
Average churn probability: {avg_churn_prob:.1%}

Top 3 highest risk customers:
{top_customers}

User question: {question}

Answer the question in 2-3 sentences using only the data provided above.
Be specific with numbers.
Note: Extreme risk customers are mostly one-time buyers with no repeat-purchase history,
so they are typically a lower retention priority than High risk customers despite the
higher probability — mention this if relevant to the question.
If the question cannot be answered with this data, say so clearly."""


NL_DEMAND_QUERY_TEMPLATE = """You are an inventory analytics assistant with access to the following data:

Total products tracked: {total_products}
Products at stockout risk: {stockout_count}
Products needing reorder soon: {reorder_count}
Products with unreliable forecasts (cannot be confidently planned around): {unreliable_count}
Average forecast accuracy (MAPE): {avg_mape:.1f}%
Total units forecasted (reorder horizon): {total_forecasted:.0f}

Highest demand products next week:
{top_products}

Products at stockout risk:
{stockout_products}

User question: {question}

Answer the question in 2-3 sentences using only the data provided above.
Be specific with numbers.
If the question is about a product with an unreliable forecast, say so and note the number
should not be trusted for automatic decisions.
If the question cannot be answered with this data, say so clearly."""