# Retail Intelligence Platform

## Week 1: Data Foundation

### Overview

Week 1 built the data foundation both Module 1 (churn) and Module 2 (demand) rely on: from raw Excel ingestion, through exploration and cleaning, into two parallel feature tables — one customer-level (for churn), one product-time-level (for demand forecasting). Along the way, two real bugs were caught and fixed (a duplicate multi-index error in demand aggregation, and a date-normalization bug in the churn Recency calculation), and the 90-day churn threshold — initially adopted as a generic default — was validated directly against actual customer return behavior.

**Pipeline:**

Online Retail.xlsx
        │
        ▼
Data Exploration
        │
        ▼
Data Cleaning
        │
        ▼
Clean Retail Dataset
        │
   ┌────┴────┐
   ▼         ▼
Module 1    Module 2
(Customer)  (Demand)
   │         │
   ▼         ▼
customer_    weekly_
features     demand
   │         │
   ▼         ▼
Verified     Exploratory
Dataset      Analysis



**Final outputs:** `data/processed/clean_retail.csv` (~354,000 rows), `data/processed/customer_features.csv` (3,920 customers), `data/processed/weekly_demand.csv` (20 products × 54 weeks)

---

### Day 1 — Environment + Project Setup

**Deliverables:** Conda environment (`retail_intelligence`, Python 3.10), full project folder structure, all libraries installed and verified (pandas, xgboost, shap, prophet, streamlit, fastapi, langchain, ollama), Git initialized and pushed to GitHub.

No modeling work this day — pure infrastructure setup, confirmed via a 7-point test script (`src/shared/test_setup.py`) covering every core library plus a live Ollama connection test.

---

### Day 2 — Data Exploration

**File:** `notebooks/01_data_exploration.ipynb`

Answered 10 exploratory questions on the raw dataset: shape/date range, unique customers/products, country distribution, cancelled orders, negative quantities/prices, duplicates, missing data, top-selling products, and typical transaction shape.

**Key findings:**
- 541,909 rows, 8 columns, spanning 2010-12-01 to 2011-12-09 (373 days)
- 24.9% of rows (135,080) missing `CustomerID`
- 9,288 cancelled orders (`InvoiceNo` starting with 'C')
- 10,624 rows with negative quantities; 2,517 with zero/negative prices
- 5,268 exact duplicate rows
- 91.4% of transactions from the United Kingdom
- Extreme outliers present (quantity range: -80,995 to 80,995; price range: -11,062.06 to 38,970.00)

These findings directly informed the cleaning rules applied on Day 3.

---

### Day 3 — Data Cleaning

**File:** `src/shared/data_cleaning.py`
**Notebook:** `notebooks/02_data_cleaning_verification.ipynb`

**Cleaning steps applied:**
1. Parse `InvoiceDate` to datetime
2. Remove cancelled orders
3. Remove rows with missing `CustomerID`
4. Remove negative/zero quantities
5. Remove negative/zero prices
6. Remove rows with missing `Description`
7. Add derived columns: `TotalPrice`, `Year`, `Month`, `DayOfWeek`, `Week`, `Date`
8. Filter to United Kingdom only

**Output:** `data/processed/clean_retail.csv` — 354,321 rows (34.6% removed from raw), verified via 5 explicit assertions (no cancelled orders, no missing CustomerIDs, no negative quantities/prices, all derived columns present).

---

### Day 4 — Customer-Level Feature Engineering (Churn)

**File:** `src/module1_churn/feature_engineering.py`
**Notebook:** `notebooks/03_customer_features_verification.ipynb`

Built RFM (Recency, Frequency, Monetary) features plus behavioral features (`AvgOrderValue`, `UniqueProducts`, `AvgQuantity`, `DaysActive`, `OrdersPerDay`) at the customer level, with a binary `Churned` label: no purchase in the 90 days before the dataset's snapshot date (2011-12-09).

**Output:** `data/processed/customer_features.csv` — 3,920 customers, 33.3% churn rate (2,613 active / 1,307 churned).

**Bug caught — Recency date normalization:** 28 customers showed `Recency = -1`, all of whom made their last purchase on the dataset's final calendar day but with a timestamp later than midnight. Root cause: the snapshot timestamp (`2011-12-09 00:00:00`) was compared directly against full purchase timestamps rather than calendar dates. Fixed with `.dt.normalize()` on both sides before subtraction, correctly producing `Recency = 0` for same-day purchases.

**Class imbalance analysis:** 33.3%/66.7% split is a ~2:1 ratio — moderate imbalance, enough that plain accuracy would be misleading (a model predicting "always active" would score ~67% while catching zero real churners). This directly informed the Week 2 decision to prioritize Precision/Recall/F1/PR-AUC over accuracy, with class-weighting applied during training.

**Churn threshold validation (tutor-prompted):** the 90-day cutoff was originally a generic e-commerce default, not derived from this dataset. Validated it two ways:
- Repeat customers (65.6% of the base) have a median reorder gap of 53.4 days, with the 75th percentile at ~92 days — 90 days sits close to where "normal" reorder behavior ends.
- Directly tested return behavior across 12,637 measured purchase-to-purchase gaps: only 12.4% exceeded 90 days, meaning a customer going quiet for 90+ days fails to return 87.6% of the time. This confirms 90 days is a reasonably strong, evidence-backed signal rather than an arbitrary number.
- **Known limitation:** this is a single global cutoff; purchase rhythms vary by customer (repeat-customer reorder gaps range from days to a full year). A per-customer adaptive threshold would be a future improvement.

**Distribution check:** histograms comparing churned vs. active customers across all six behavioral features confirmed clear separation — churned customers cluster at high Recency, low Frequency/Monetary/AvgOrderValue/UniqueProducts, and shorter DaysActive — validating that the label is behaviorally meaningful, not arbitrary.

---

### Day 5 — Product/Time-Level Feature Engineering (Demand)

**File:** `src/module2_demand/feature_engineering.py`
**Notebook:** `notebooks/04_demand_exploration.ipynb`

Aggregated the top 20 products (by total quantity) into weekly time series: quantity, revenue, order count, and unique customer count per product per week, with missing weeks explicitly filled as zero.

**Output:** `data/processed/weekly_demand.csv` — 20 products × 54 weeks, 1,080 rows.

**Bug caught — non-unique multi-index:** initial version grouped by `(StockCode, Description, Week)`, but some products had inconsistent `Description` values over time (e.g., StockCode 22197 appeared as both "POPCORN HOLDER" and "SMALL POPCORN HOLDER"), creating duplicate `(StockCode, Week)` pairs that broke the `.reindex()` step used to fill missing weeks. Fixed by grouping on `(StockCode, Week)` only, then attaching each product's most frequent `Description` as a separate step.

**Finding — product selection method was flawed (tutor-prompted):** ranking by raw total volume surfaced products dominated by single bulk orders rather than organic demand. StockCode 23843 sold 80,995 units total, but was active in only 1.85% of the 54 weeks — essentially one giant order. StockCode 23166 showed a similar pattern (59.3% weeks active, but coefficient of variation of 7.09 — extremely spiky).

**Improved selection method developed:** ranking by `PctWeeksActive` (% of weeks with any sales, descending) and `CoeffOfVariation` (std/mean of weekly quantity, ascending) surfaces genuinely consistent products instead — e.g., StockCode 85099B, 84946, and 22178, all active in ~98% of weeks with low relative variability (0.6–0.9).

**Action carried into Week 4:** use consistency-based ranking, not raw total volume, when selecting the final product set for Prophet forecasting.

---

### Week 1 File Summary

| File | Purpose |
|---|---|
| `data/raw/Online Retail.xlsx` | Original UCI dataset (not committed — see `.gitignore`) |
| `notebooks/01_data_exploration.ipynb` | Raw data exploration (10 questions) |
| `src/shared/data_cleaning.py` | Cleaning pipeline → `clean_retail.csv` |
| `notebooks/02_data_cleaning_verification.ipynb` | Assertion-based cleaning verification |
| `src/module1_churn/feature_engineering.py` | Customer RFM features + churn label |
| `notebooks/03_customer_features_verification.ipynb` | Churn label validation, class imbalance, 90-day threshold test |
| `src/module2_demand/feature_engineering.py` | Weekly product demand aggregation |
| `notebooks/04_demand_exploration.ipynb` | Demand time series exploration, product selection method fix |
| `data/processed/clean_retail.csv` | Cleaned transaction-level data |
| `data/processed/customer_features.csv` | Customer-level churn features |
| `data/processed/weekly_demand.csv` | Weekly product-level demand data |



## Week 2: Customer Churn Classification Model

### Overview

Week 2 built a complete churn prediction pipeline on top of Week 1's cleaned customer feature table: from a stratified train/test split, through baseline modeling, XGBoost training and tuning, and finally into business-facing threshold optimization and revenue-at-risk analysis. Along the way, three real bugs were caught and fixed (a data leakage issue, a raw data bug inherited from 
Week 1, and a library version mismatch), each of which would have silently produced invalid results if left unaddressed.

**Pipeline:**

**Final model:** tuned XGBoost, `models/churn_model_tuned.pkl`, using threshold 0.20 (`models/optimal_threshold.npy`)

---

### Day 1 — Data Preparation + Train/Test Split

**File:** `src/module1_churn/data_preparation.py`
**Notebook:** `notebooks/05_class_imbalance_analysis.ipynb`

The original plan called for a time-based split (train on transactions before Sep 2011, test on transactions after). This was found to be incompatible with how Week 1's features were built — `customer_features.csv` computes every feature and the `Churned` label relative to a single fixed snapshot date (2011-12-09), not multiple time windows. Splitting by transaction date on top of a single-snapshot table produced a churn rate mismatch between train (43%) and test (2.8%), since it was accidentally splitting customers by a variable that nearly determines the label itself.

**Pipeline**
Transactions
        │
        ▼
Customer Features
        │
        ▼
Train/Test Split
        │
        ▼
Dummy Baseline
        │
        ▼
Logistic Regression
        │
        ▼
XGBoost
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Threshold Optimization
        │
        ▼
Revenue-at-Risk Analysis





**Fix:** switched to a stratified random split (`train_test_split(...,stratify=y)`), which preserves the same churn rate across train and test (33.4% vs 33.3%).

**Class imbalance finding:** 33.3% churn rate (2,613 active / 1,307 churned), a 2:1 ratio — moderate imbalance, enough to make accuracy a misleading metric.

---

### Day 2 — Baseline Model + Evaluation Framework

**Files:** `src/shared/evaluation.py`, `src/module1_churn/baseline_model.py`
**Notebook:** `notebooks/06_baseline_evaluation.ipynb`

Built a reusable evaluation function (precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix) used by every model trained for the rest of the project. Trained a Dummy Classifier (majority-class baseline) and a Logistic Regression model for comparison.

**Bug caught — Recency/Churned leakage:** the first Logistic Regression run scored a suspicious PR-AUC of 1.0 (99.6% precision, 100% recall). Traced to `Recency` being derived from the same `LastPurchaseDate` field and the same 90-day cutoff as the `Churned` label itself — the model was decoding the label's own definition rather than learning behavior patterns. Fixed by removing `Recency` from the model's feature set (kept in the data for display purposes only).

**Results after fix:**

**Results after fix:**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|----------:|-------:|---:|--------:|-------:|
| Dummy Classifier | 0.00 | 0.00 | 0.00 | 0.50 | 0.33 |
| Logistic Regression | 0.54 | 0.81 | 0.65 | 0.80 | 0.62 |


### Day 3 — XGBoost + Class Imbalance Handling

**File:** `src/module1_churn/xgboost_model.py`
**Notebook:** `notebooks/07_xgboost_evaluation.ipynb`

Trained XGBoost with two imbalance-handling strategies: `scale_pos_weight` (cost-weighted training) and SMOTE (synthetic oversampling).

**Results:**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|----------:|-------:|---:|--------:|-------:|
| XGBoost (`scale_pos_weight`) | 0.52 | 0.76 | 0.61 | 0.78 | 0.578 |
| XGBoost (SMOTE) | 0.52 | 0.77 | 0.62 | 0.77 | 0.565 |

**Finding:** both default-hyperparameter XGBoost variants underperformed the Logistic Regression baseline (0.62). scale_pos_weight` narrowly beat SMOTE and was selected per the project's model-selection rule (highest PR-AUC). Hypothesized causes: untuned hyperparameters, a fairly linear underlying churn signal, and a relatively small training set (~3,100 rows) limiting the payoff of tree-ensemble complexity.

---

### Day 4 — Hyperparameter Tuning

**File:** `src/module1_churn/hyperparameter_tuning.py`
**Notebook:** `notebooks/08_tuning_results.ipynb`

Ran `RandomizedSearchCV` (50 combinations, 5-fold cross-validation) over n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, and gamma.

**Bug caught — scikit-learn API mismatch:** the first run produced `nan` for all 250 fold/parameter combinations. Traceback showed `make_scorer()`'s `needs_proba` argument had been deprecated in scikit-learn 1.7.2 in favor of `response_method='predict_proba'`. The script still completed and printed a "best" result despite every score being invalid — a reminder to check for `nan` explicitly rather than assuming a clean run produced valid results. 

Fixed by updating the scorer syntax.

**Result after fix:**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|----------:|-------:|---:|--------:|-------:|
| XGBoost (Tuned) | 0.51 | 0.85 | 0.64 | 0.80 | 0.604 |

Best cross-validated PR-AUC: 0.6118 (close to the 0.604 test-set PR-AUC, suggesting the tuned settings generalized rather than overfitting to the CV folds). Best parameters found: `max_depth=4, learning_rate=0.01, n_estimators=200, subsample=0.9, min_child_weight=1, gamma=0.1, colsample_bytree=1.0` — a relatively conservative, anti-overfitting configuration given the dataset size.

**Decision:** tuning narrowed but did not close the gap to Logistic Regression (0.604 vs 0.62). XGBoost was selected as the production model anyway, primarily because Week 3's SHAP explainability step works natively and efficiently with tree-based models, and the performance gap is small enough that this tradeoff is reasonable. Logistic Regression remains documented as the baseline comparison rather than discarded from the project narrative.

---

### Day 5 — Threshold Tuning + Business Cost Analysis

**Files:** `src/module1_churn/threshold_analysis.py`, 
`src/module1_churn/clv_analysis.py`
**Notebook:** `notebooks/09_threshold_business_analysis.ipynb`

**Threshold optimization:** swept thresholds from 0.10–0.90 using assumed costs of $5 per false positive and $50 per false negative. The cost-optimal threshold (0.20) reduces total business cost by 38% versus the default 0.5 ($1,820 vs $2,955), by deliberately trading precision for recall — appropriate given that missing a churner is 10x more costly than a false alarm.

**Revenue-at-risk:** using threshold 0.20, 603 of 784 test customers are flagged as at-risk, representing $559,896.14 in CLV. A 50% intervention success rate implies ~$279,948 in potential savings.

**CLV outlier investigation:** CLV distribution is heavily right-skewed (mean $1,907 vs median $680, max $81,025), consistent with the wholesale-buyer pattern found in Week 1. Spot-checking the top 10 CLV customers found only 1 is a predicted churner.

**Key finding — structural, not coincidental:** high-CLV customers are unlikely to be flagged as high-risk because CLV (total historical spend) and the model's risk signal share overlapping underlying features (Monetary,Frequency, DaysActive) — sustained high spending mechanically looks like low churn risk to the model. This means the revenue-at-risk figure isn't inflated by misclassified outliers, but also means probability-only prioritization may underweight the business's most valuable customers; a combined risk-and-CLV prioritization approach would be more complete.

**Documented limitations:**
- The $5/$50 costs are flat, illustrative assumptions. The false-negative side could be made more accurate using each customer's actual CLV (already computed); the false-positive side would require real campaign cost data not present in this dataset.

- The cost formula doesn't account for true positives at all — it assumes every correctly-flagged churner who receives an offer is retained, which is unrealistic and inconsistent with the CLV analysis's own 50% retention assumption elsewhere.

- The CLV formula (`Monetary / 12 * 12`, which simplifies to just `Monetary`) is a placeholder, not a true forward-looking lifetime value projection.


---

### Week 2 File Summary

|## Week 2 File Summary

| File | Purpose |
|------|---------|
| `src/module1_churn/data_preparation.py` | Stratified train/test split |
| `src/shared/evaluation.py` | Reusable model evaluation functions |
| `src/module1_churn/baseline_model.py` | Dummy + Logistic Regression baselines |
| `src/module1_churn/xgboost_model.py` | XGBoost (scale_pos_weight vs SMOTE) |
| `src/module1_churn/hyperparameter_tuning.py` | RandomizedSearchCV tuning |
| `src/module1_churn/threshold_analysis.py` | Cost-based threshold optimization |
| `src/module1_churn/clv_analysis.py` | CLV + revenue-at-risk calculation |
| `models/churn_model_tuned.pkl` | Final production model |
| `models/optimal_threshold.npy` | Selected decision threshold (0.20) |
| `data/processed/churn_predictions.csv` | Per-customer predictions + CLV |



## Week 3: SHAP Explainability + Risk Segmentation + Streamlit Tab 1

### Overview

Week 3 turned the Week 2 churn model into something explainable and actionable: 
SHAP explainability at both the global and individual level, a 4-tier risk segmentation system with revenue-at-risk quantification, and a working Streamlit dashboard (Tab 1) tying it all together. Several real bugs were found and fixed  along the way — a SHAP/XGBoost version compatibility crash, two rounds of a risk-level inconsistency bug between different parts of the codebase, and a 
SHAP-value indexing mismatch in the dashboard — plus one genuine analytical finding: the highest-probability churn segment is not the highest-priority segment for retention spend.


**Pipeline:**

churn_model_tuned.pkl (Week 2)
        │
        ▼
Global SHAP Explainability
(Day 1)
        │
        ▼
Individual SHAP Explanations
(Day 2)
        │
        ▼
4-Tier Risk Segmentation
+ Revenue-at-Risk
(Day 3)
        │
        ▼
customer_risk_table.csv
        │
        ├──────────────┐
        ▼              ▼
Streamlit        Streamlit
Layout           Charts
(Day 4)          (Day 5)

**Final outputs:** `models/shap_explainer.pkl`, `data/processed/feature_importance.csv`, 
`data/processed/shap_values.csv`, `data/processed/customer_risk_table.csv`, 
working `src/app.py` (Streamlit Tab 1: Churn Prediction)

---

### Day 1 — SHAP Global Explainability

**File:** `src/module1_churn/shap_explainability.py`
**Notebook:** `notebooks/10_shap_global_analysis.ipynb`

Built `compute_shap_values()`, `plot_global_feature_importance()` (bar, beeswarm, heatmap), and `get_feature_importance_table()` using `shap.TreeExplainer` on the tuned XGBoost model.


**Key finding:** `DaysActive` is the dominant churn driver (mean |SHAP| = 0.846), nearly 3x the next feature (`UniqueProducts`, 0.285). High `DaysActive` decreases churn risk (longer-tenured customers are stickier).

**Surprising result:** `Frequency` and `AvgOrderValue` — the two "classic" RFM churn signals — ranked last (6th and 7th of 7). Investigated rather than accepted at face value: confirmed `Frequency` has real variance (not a broken feature, range 1–209), then found it correlates 0.739 with `UniqueProducts` (rank #2). 
Conclusion: not a bug — `UniqueProducts` absorbs `Frequency`'s signal in the tree-based model, a known effect of correlated features in SHAP importance.

---

### Day 2 — SHAP Individual Customer Explanations

**File:** `src/module1_churn/shap_explainability.py` (same file, extended)
**Notebook:** `notebooks/11_individual_shap_explanations.ipynb`

Built `explain_single_customer()` (waterfall plot + ranked driver/retention breakdown per customer) and generate_natural_language_explanation()` (template-based plain-English summary).


**Limitation noted (addressed further in Day 3):** one-time/burst buyers (e.g. `DaysActive=1`, `Frequency=1`) get the same HIGH RISK flag and win-back treatment as lapsed regulars, despite likely needing different retention strategies.

---

### Day 3 — Risk Segmentation + Revenue at Risk

**File:** `src/module1_churn/risk_segmentation.py`
**Notebook:** `notebooks/12_risk_segmentation.ipynb`

Built `assign_risk_tier()` and `build_customer_risk_table()` — full customer-level table combining churn probability, risk tier, CLV estimate (`Monetary × 1.2`), revenue at risk, top SHAP driver, and natural-language explanation per customer.

**Enhancement — 4th tier added:** original 3 tiers (Low/Medium/High) left 55% of customers in one undifferentiated "High Risk" bucket. Added an `Extreme Risk` tier (probability ≥ 0.70) to enable finer prioritization. Required updating both `assign_risk_tier()` and `generate_natural_language_explanation()` in tandem to avoid reintroducing a tier-mismatch bug.

**Key finding — probability ≠ priority:** directly compared Extreme Risk (n=75) against High Risk (n=359) across Frequency, DaysActive, and Monetary — Extreme Risk confirmed to be an almost uniform group of one-time, same-day buyers with the lowest revenue at risk of any tier despite the highest churn probability. High Risk, despite lower average probability, is a substantially more valuable population. **Practical implication:** retention spend is better directed at High and Medium Risk tiers; Extreme Risk customers are weak win-back candidates.

---

### Day 4 — Streamlit Tab 1: Layout + Data Tables

**File:** `src/app.py`

Built the dashboard skeleton: page config, sidebar navigation, KPI metrics row (5 columns for 4-tier system), filters (risk tier multiselect, probability slider, revenue minimum), customer risk table, and individual customer deep-dive section.


---

### Day 5 — Streamlit Tab 1: Charts + SHAP Visualizations

**File:** `src/app.py` (extended)

Added: risk tier donut chart, revenue-at-risk bar chart, churn probability histogram, global SHAP feature importance bar, DaysActive-vs-Monetary scatter,and a live per-customer SHAP driver bar chart tied to the deep-dive dropdown.


---

### Week 3 File Summary

| File | Purpose |
|---|---|
| `src/module1_churn/shap_explainability.py` | Global + individual SHAP, natural language explanations |
| `src/module1_churn/risk_segmentation.py` | 4-tier risk segmentation, CLV, revenue at risk |
| `notebooks/10_shap_global_analysis.ipynb` | Global SHAP interpretation (Day 1) |
| `notebooks/11_individual_shap_explanations.ipynb` | Individual SHAP testing, narrative bug fixes (Day 2) |
| `notebooks/12_risk_segmentation.ipynb` | Risk tier analysis, Extreme vs High Risk comparison (Day 3) |
| `src/app.py` | Streamlit Tab 1 — layout, filters, table, charts, deep dive (Days 4–5) |
| `models/shap_explainer.pkl` | Saved SHAP TreeExplainer |
| `data/processed/feature_importance.csv` | Global SHAP feature importance table |
| `data/processed/shap_values.csv` | Per-customer SHAP values, original row order |
| `data/processed/customer_risk_table.csv` | Full risk table with tiers, CLV, revenue at risk, `ShapRowIndex` |


## Week 4: Demand Forecasting (Prophet) + Uncertainty Quantification + Baseline Comparison

### Overview

Week 4 built the demand forecasting module for all 20 top products using Prophet. Five separate bugs were found and fixed across the week — an unstable yearly seasonality + holiday config, a weekly seasonality config that turned out to be modeling noise (all 
training dates fell on the same weekday), a relative uncertainty metric hidden by zero-clipping, a stock/demand horizon mismatch, and a MAPE near-zero blowup — plus a custom data-quality flag added to separate genuinely forecastable products from sparse/spike-driven ones. The week closes with a full evaluation table and two products flagged for deeper investigation in Week 5's walk-forward validation.

**Pipeline:**

weekly_demand.csv (Week 1)
        │
        ▼
Single-Product Prophet Fit
+ Config Debugging
(Day 1)
        │
        ▼
Multi-Product Pipeline
+ Data Quality Flag
(Day 2)
        │
        ▼
Uncertainty Quantification
+ Inventory Risk
(Day 3)
        │
        ▼
Naive + Seasonal Naive
Baseline Comparison
(Day 4)
        │
        ▼
final_evaluation_table.csv
(Day 5)

**Final outputs:** `models/prophet_models/*.pkl` (20 models), `data/processed/all_forecasts.csv`, `data/processed/seasonality_analysis.csv`, `data/processed/uncertainty_metrics.csv`, `data/processed/inventory_risk.csv`, `data/processed/baseline_comparison.csv`, `data/processed/final_evaluation_table.csv`

---

### Day 1 — Single-Product Prophet Fit + Config Debugging

**File:** `src/module2_demand/prophet_model.py`
**Notebook:** `notebooks/13_prophet_exploration.ipynb`

Built `prepare_prophet_data()` and `train_prophet_model()`, first tested on product 
21915 (RED HARMONICA IN BOX, high-variance/spiky demand).

**Bug found and fixed:** default config (`yearly_seasonality=True` + UK holidays) produced deeply negative 8-week forecasts (-848 to -1938 units). Diagnosed via components plot: with only ~54 weeks (~1 year) of history, yearly seasonality had no repeated cycle to validate against and fit the shape of one specific year instead of a real pattern; UK holidays were estimated from single occurrences. Fixed by disabling both — final model: trend + weekly seasonality only.

---

### Day 2 — Multi-Product Pipeline + Data Quality Flag

**File:** `src/module2_demand/forecast_pipeline.py`
**Notebook:** `notebooks/14_multi_product_forecasts.ipynb`

Scaled Day 1's model to all 20 products, saving forecasts and trained models.

**Fix 1 — TrendChange% division-by-near-zero:** original formula divided by `trend_start`, which is near-zero for many products, producing values like 28,427%. Fixed by dividing by each product's historical mean demand instead.

**Fix 2 — data quality flag added:** built `assess_forecastability()` to flag products with sparse/spike-driven demand (likely wholesale/reseller orders) that Prophet can't meaningfully model. Initial OR-based threshold (zero% > 40% OR spike ratio > 15) incorrectly flagged product 23203, which had a perfectly normal forecast; switched to AND after verifying, correctly narrowing the flag to products 23843 and 23166.

**Fix 3 — weekly seasonality bug:** confirmed via `demand['Week'].dt.day_name().unique()` that every training date falls on Monday 
— meaning "weekly seasonality" had no real day-of-week variation to learn from, and forecast dates (defaulting to Sunday via `freq='W'`) landed on an unconstrained part of the fitted curve, crashing forecasts toward implausible values (product 84077: historical mean ~910, forecast ~100). Fixed by disabling `weekly_seasonality` and switching to `freq='W-MON'`. Final model: **trend only** — the only component genuinely supported by this dataset's weekly-aggregated, single-weekday structure.

---

### Day 3 — Uncertainty Quantification + Inventory Risk

**File:** `src/module2_demand/uncertainty_analysis.py`
**Notebook:** `notebooks/15_uncertainty_analysis.ipynb`

Built relative uncertainty metrics and a synthetic inventory risk assessment per 
product.

**Fix 1 — uncertainty hidden by clipping:** dividing IntervalWidth by `yhat` silently reported 0% relative uncertainty for product 23166, whose forecast is clipped to exactly 0 — the opposite of the truth, since it's one of the least reliable forecasts in the dataset. Fixed by dividing by historical mean demand instead; 23166 correctly moved to the second-highest uncertainty ranking.

**Fix 2 — inventory risk misclassification:** the same clipping issue caused 23166 to show as "Adequate" stock status. Fixed by adding an explicit check: any product with `DataQualityFlag != 'OK'` is labeled "Unreliable Forecast" rather than receiving a computed status.

**Fix 3 — stock/demand horizon mismatch:** synthetic CurrentStock assumed 3 weeks of average demand while ForecastedDemand summed 4 weeks, structurally inflating "Low Stock" classifications (80% of products). Aligned both to 4 weeks, moving the distribution to 60% Low Stock / 30% Adequate / 10% Unreliable Forecast. Remaining 60% is a genuine pattern (many products have Growing trend, which a backward-looking stock average naturally undershoots), not a further bug — CurrentStock itself remains synthetic throughout, a real dataset limitation.

---

### Day 4 — Naive + Seasonal Naive Baseline Comparison

**File:** `src/module2_demand/baseline_models.py`
**Notebook:** `notebooks/16_baseline_comparison.ipynb`

Built naive (repeat last value) and seasonal naive (same week last year) baselines, compared against Prophet on a held-out 8-week window per product.

**Key finding — average and win-rate disagree:** Prophet's average MAE beats naive (336.1 vs 379.5) once the 2 flagged products are excluded, but Prophet only wins head-to-head on 7 of 18 products. A few large wins pull the average down while most products see small losses to naive — both numbers are reported rather than picking the more flattering one.

**Investigated — product 21977:** biggest single loss to naive (-150 MAE). Confirmed via direct comparison: Prophet forecast a flat ~385-391 units/week (trend extrapolation) while actual demand stayed 58-357; naive's flat 146 was closer to reality. Cause: trend-only model has no mechanism to temper a Growing trend when recent momentum doesn't continue. Not fixed — deferred to Week 5's walk-forward validation rather than tuning against a single test window.

---

### Day 5 — Full Evaluation Table

**File:** `src/module2_demand/model_evaluation.py`
**Notebook:** `notebooks/17_model_evaluation_summary.ipynb`

Merged all four prior CSVs into one evaluation table; added mean/median improvement reporting split by data quality flag.

**Fix — MAPE near-zero blowup:** MAPE only excluded exact-zero actual weeks, not small non-zero ones, causing product 21915 to show 3517% MAPE from a single near-zero-demand week. Fixed by excluding weeks below a 5-unit threshold; 21915's MAPE corrected to 470.2%, Prophet's overall average MAPE corrected from a misleading 292.2% to 122.9%.

**What the numbers actually say (18 OK products):**
- Win rate: beats naive 7/18, beats seasonal naive 13/18
- Improvement over naive: mean -13.1%, median -3.2% (roughly a wash)
- Improvement over seasonal naive: mean -1.6%, median +27.6% (large mean/median gap — a few products with extreme percentage swings pull the mean down even though most products see a real improvement)

**In short:** Prophet's trend-only model modestly underperforms naive on the typical product, but clearly outperforms seasonal naive on 72% of products, with the median product showing a 27.6% error reduction.

**Investigated — product 15036:** largest negative outlier vs. seasonal naive (-312.3%). Not a bug — seasonal naive's MAE was unusually low on this single test window, either a real seasonal pattern or a lucky match. Deferred to Week 5, same reasoning as 21977.

---

### Week 4 File Summary

| File | Purpose |
|---|---|
| `src/module2_demand/prophet_model.py` | Single-product Prophet fit, trend-only config |
| `src/module2_demand/forecast_pipeline.py` | Multi-product batch pipeline, data quality flag |
| `src/module2_demand/uncertainty_analysis.py` | Relative uncertainty, inventory risk assessment |
| `src/module2_demand/baseline_models.py` | Naive + seasonal naive baselines, comparison metrics |
| `src/module2_demand/model_evaluation.py` | Merged evaluation table, mean/median summary |
| `notebooks/13_prophet_exploration.ipynb` | Single-product testing, seasonality debugging (Day 1) |
| `notebooks/14_multi_product_forecasts.ipynb` | Batch pipeline validation, weekly seasonality fix (Day 2) |
| `notebooks/15_uncertainty_analysis.ipynb` | Uncertainty + inventory risk analysis (Day 3) |
| `notebooks/16_baseline_comparison.ipynb` | Baseline comparison, 21977 investigation (Day 4) |
| `notebooks/17_model_evaluation_summary.ipynb` | Final evaluation table, 15036 investigation (Day 5) |
| `models/prophet_models/*.pkl` | 20 trained Prophet models |
| `data/processed/final_evaluation_table.csv` | Combined metrics: MAE, MAPE, uncertainty, trend, risk |



## Week 5: Backtesting + Inventory Alerts + Streamlit Tab 2

### Overview

Week 5 validated the Week 4 demand forecasting model through walk-forward backtesting, converted forecast outputs into an actionable inventory alert system, and completed the Streamlit dashboard with a fully interactive Tab 2. Multi-window testing resolved one open question from Week 4 (product 21977's single-window underperformance did not hold up) and confirmed another (product 15036), while identifying a new consistent underperformer (product 22086).

---

### Day 1 — Walk-Forward Validation Framework

**File:** `src/module2_demand/walk_forward_validation.py`
**Notebook:** `notebooks/18_walkforward_results.ipynb`

Built a 5-fold walk-forward validation engine across all 20 products, using 4-week test windows with an expanding training window per fold. Compared Prophet against naive and seasonal naive baselines for each fold.

**Key finding:** product 21977, previously flagged as underperforming in Week 4's single-window test, wins 60% of folds under multi-window testing — resolved as a one-window fluke rather than a real pattern. Product 15036 confirmed as a consistent underperformer (0% win rate across all 5 folds). Product 22086 identified as a new consistent underperformer with the same pattern.

---

### Day 2 — Backtest Analysis + Error Deep Dive

**File:** `src/module2_demand/backtest_analysis.py`
**Notebook:** `notebooks/19_backtest_insights.ipynb`

Built error pattern analysis, four backtest visualizations (error heatmap, fold comparison, MAPE distribution, win rate by product), and an auto-generated plain-text backtest report.

**Key results (18 products):** Prophet beats naive baseline in 47.8% of folds and seasonal naive in 70.0% of folds. Median MAPE of 73.4% reported alongside mean MAPE of 136.0%, with the gap traced to a mix of low-demand weeks and a small number of genuinely large misses.

Interview Q&A notebook written covering walk-forward methodology, interpretation of win rates against both baselines, the MAPE mean/median gap, and future improvement directions.

---

### Day 3 — Inventory Alert System

**File:** `src/module2_demand/inventory_alerts.py`
**Notebook:** `notebooks/20_inventory_alerts.ipynb`

Built safety stock calculation (Z-score × forecast error std × √lead time) and a five-tier inventory alert system (Stockout Risk, Reorder Soon, Overstock, Adequate, Unreliable Forecast), using per-product forecast error estimated from the walk-forward fold results.

**Results:** 11 of 20 products flagged Stockout Risk, 6 Adequate, 2 Unreliable Forecast, 1 Reorder Soon. Alert logic verified against three visualizations (status distribution, stock-vs-demand-vs-reorder-point, suggested reorder quantities), with product 22197 identified as carrying both the largest suggested reorder quantity and the highest forecast uncertainty among reliable products.

---

### Day 4 — Streamlit Tab 2: Layout + Data Tables

**File:** `src/app.py`

Built the Tab 2 skeleton: four KPI metrics, a filterable and downloadable inventory alert table, and a per-product forecast deep-dive section with historical/forecast chart, confidence intervals, and product-level summary panel.

---

### Day 5 — Streamlit Tab 2: Charts + Backtesting Results

**File:** `src/app.py` (extended)

Added three overview charts (alert status distribution, suggested reorder quantities, MAPE distribution histogram) and a model validation section presenting walk-forward win rates and MAE comparison against both baselines, plus a sidebar summary of live model performance metrics.

---

### Week 5 File Summary

| File | Purpose |
|---|---|
| `src/module2_demand/walk_forward_validation.py` | 5-fold walk-forward validation engine, per-product summary |
| `src/module2_demand/backtest_analysis.py` | Error pattern analysis, backtest visualizations, report generation |
| `src/module2_demand/inventory_alerts.py` | Safety stock calculation, 5-tier alert system, alert visualizations |
| `notebooks/18_walkforward_results.ipynb` | Walk-forward results exploration (Day 1) |
| `notebooks/19_backtest_insights.ipynb` | Backtest deep dive, interview Q&A (Day 2) |
| `notebooks/20_inventory_alerts.ipynb` | Alert table and chart verification (Day 3) |
| `src/app.py` | Streamlit Tab 2 — layout, alerts, charts, backtest results (Days 4-5) |
| `data/processed/walkforward_folds.csv` | Fold-level backtest results, all products |
| `data/processed/walkforward_summary.csv` | Per-product walk-forward summary |
| `data/processed/inventory_alerts.csv` | Full inventory alert table |


## Week 6: LLM Narration Layer (Ollama + LangChain)

### Overview

Week 6 added a local LLM layer on top of the churn and demand models built in Weeks 2–5, translating structured model outputs into plain-English business narratives. This included customer churn explanations, demand summaries, a natural language query interface, and personalized win-back messages, all running locally via Ollama and integrated into the Streamlit dashboard.

### Pipeline

```text
Customer Risk Table (Week 3) + Demand Alerts (Week 5)
              │
              ▼
    RetailLLMClient (Ollama + LangChain)
    + Prompt Template Library
              │
      ┌───────┼────────┬─────────────┐
      ▼       ▼        ▼             ▼
  Churn    Demand    Query        Win-Back
Narrator  Narrator   Engine      Generator
      │       │        │             │
      └───────┴────────┴─────────────┘
              ▼
   Streamlit Dashboard (AI Insights)
```

---

### Day 1 — Ollama Connection + Prompt Engineering

**Files:** `src/module3_llm/llm_client.py`, `src/module3_llm/prompt_templates.py`

**Notebook:** `notebooks/20_prompt_template_testing.ipynb`

Built `RetailLLMClient`, a reusable Ollama + LangChain connection layer with retry logic and response cleaning. Created eight prompt templates covering churn, demand, inventory briefings, and natural-language queries.

---

### Day 2 — Churn Narration

**File:** `src/module3_llm/churn_narrator.py`

Built `ChurnNarrator`, which converts SHAP values into plain-English customer explanations, generates segment summaries, and creates personalized win-back messages.

---

### Day 3 — Demand Narration

**File:** `src/module3_llm/demand_narrator.py`

Built `DemandNarrator`, which translates demand forecasts into product summaries, daily inventory briefings, and weekly business reports.

---

### Day 4 — Natural Language Query Interface

**File:** `src/module3_llm/query_engine.py`

Built `QueryEngine`, which routes natural-language questions to churn, demand, or combined data sources and generates grounded answers from real platform outputs.

---

### Day 5 — Win-Back Generator

**File:** `src/module3_llm/winback_generator.py`

Built `WinBackGenerator`, which creates personalized emails, incentives, urgency levels, and recommendations based on churn probability and customer value.

---

### Day 6 — Streamlit Integration

**File:** `src/app.py`

Integrated the full LLM layer into the dashboard, adding AI customer explanations, win-back emails, daily business briefings, natural-language queries, and weekly reports.

---

### Week 6 File Summary

| File | Purpose |
|---|---|
| `src/module3_llm/llm_client.py` | Ollama connection, retry logic, and response cleaning |
| `src/module3_llm/prompt_templates.py` | Prompt template library |
| `src/module3_llm/churn_narrator.py` | Customer and segment churn explanations |
| `src/module3_llm/demand_narrator.py` | Forecast summaries and inventory reports |
| `src/module3_llm/query_engine.py` | Natural-language query routing |
| `src/module3_llm/winback_generator.py` | Personalized win-back packages |
| `notebooks/20_prompt_template_testing.ipynb` | Prompt testing notebook |
| `src/app.py` | Streamlit integration |


## Week 7: FastAPI Backend + Full Streamlit Integration

### Overview

Week 7 separated the Streamlit dashboard from direct model and data access by introducing a FastAPI backend. All churn, demand, and LLM narration logic built in Weeks 2–6 was wrapped in versioned REST endpoints with Pydantic request/response validation, dependency-injected caching, and interactive Swagger documentation. The Streamlit frontend was then fully migrated to consume this API via an HTTP client rather than importing model functions or reading CSVs directly, completing the transition from a single-script prototype to a properly layered application (frontend → API → business logic → data/models).

### Pipeline

```text
customer_risk_table.csv, shap_values.csv, all_forecasts.csv,
inventory_alerts.csv, walkforward_folds.csv, walkforward_summary.csv,
seasonality_analysis.csv, clean_retail.csv, model .pkl files
              │
              ▼
   FastAPI Backend (src/api/)
   config → dependencies (cached loaders) → services → routers
              │
      ┌───────┼────────┬─────────────┐
      ▼       ▼        ▼             ▼
  /health   /churn   /demand       /llm
              │
              ▼
   RetailAPIClient (src/api_client.py)
              │
              ▼
   Streamlit Dashboard (3 tabs, HTTP-driven)
```

---

### Day 1 — FastAPI Skeleton

**Files:** `src/api/config.py`, `src/api/dependencies.py`, `src/api/models/churn.py`, `src/api/models/demand.py`, `src/api/models/llm.py`, `src/api/routers/health.py`, `src/api/main.py`

Built the FastAPI project structure, centralized configuration, Pydantic request/response models, and cached dependency loaders for all data and model files. Corrected the churn risk tier enum to the actual 4-tier system (Extreme, High, Medium, Low) and the model's real 7-feature set (no `Recency`), both verified against production data rather than assumed. Health check endpoints confirmed all models and data load correctly through the live API.

---

### Day 2 — Churn Prediction Endpoints

**Files:** `src/api/services/churn_service.py`, `src/api/routers/churn.py`

Built `/churn/summary`, `/churn/customers` (paginated, filterable), `/churn/customers/{index}`, `/churn/predict`, and `/churn/feature-importance`. Discovered and corrected a row-misalignment between `customer_risk_table.csv` and `shap_values.csv`, resolved via the `ShapRowIndex` join column rather than direct positional indexing — verified by cross-checking SHAP output against the pre-written explanation text for the same customer. Fixed a numpy int64 JSON serialization error and a Pydantic response-model type mismatch.

---

### Day 3 — Demand Forecasting Endpoints

**Files:** `src/api/services/demand_service.py`, `src/api/routers/demand.py`

Built `/demand/summary`, `/demand/alerts`, `/demand/products`, `/demand/products/{stock_code}`, and `/demand/backtest`. Confirmed all 20 tracked products exist in both `all_forecasts.csv` and `inventory_alerts.csv`, so missing-product lookups now fail loudly (404) rather than silently degrading. Fixed a NaN/Infinity JSON serialization error in aggregate backtest statistics caused by zero-demand folds, adding a `folds_with_infinite_mape` transparency field to the response.

---

### Day 4 — LLM Narration Endpoints

**Files:** `src/api/services/llm_service.py`, `src/api/routers/llm.py`

Built `/llm/explain/churn/{index}`, `/llm/winback/{index}`, `/llm/query`, `/llm/briefing`, `/llm/explain/demand/{stock_code}`, `/llm/segment-summaries`, and `/llm/weekly-report`. Verified real method signatures for `ChurnNarrator`, `DemandNarrator`, `QueryEngine`, and `WinBackGenerator` directly from source rather than assuming, uncovering that win-back generation requires a computed `clv_median` and the full retail transaction log, and that demand explanations require `trend_direction` sourced from `seasonality_analysis.csv`. All endpoints tested live against Ollama with real generated output.

---

### Day 5 — Streamlit Migration

**Files:** `src/api_client.py`, `src/app.py`

Built `RetailAPIClient`, an HTTP client wrapping every backend endpoint with error handling for connection failures, timeouts, and API errors. Migrated all three Streamlit tabs (Churn Prediction, Demand Forecast, AI Insights) from direct CSV/model access to API calls, including a full-dataset pagination helper for chart compatibility. Restored feature parity for the walk-forward per-product summary view, which had been dropped from an earlier endpoint iteration, by extending `/demand/backtest` to include `product_summary` alongside fold-level detail.

---

### Day 6 — Buffer, Polish, and Review

Removed temporary diagnostic scripts used during data verification. Fixed emoji artifacts appearing mid-sentence in LLM-generated segment summaries and a duplicate-text rendering bug in the AI Insights tab. Disabled misleading delta-arrow coloring on KPI metrics that were not representing directional change. Verified both the API and Streamlit app start cleanly and all three tabs render without error.

---

### Week 7 File Summary

| File | Purpose |
|---|---|
| `src/api/config.py` | Centralized API settings, data/model paths |
| `src/api/dependencies.py` | Cached loaders for data, models, and LLM components |
| `src/api/models/churn.py` | Pydantic models for churn endpoints |
| `src/api/models/demand.py` | Pydantic models for demand endpoints |
| `src/api/models/llm.py` | Pydantic models for LLM endpoints |
| `src/api/services/churn_service.py` | Churn business logic, SHAP row-join handling |
| `src/api/services/demand_service.py` | Demand business logic, forecast/alert joins |
| `src/api/services/llm_service.py` | LLM narration business logic |
| `src/api/routers/health.py` | Health check endpoints |
| `src/api/routers/churn.py` | Churn prediction endpoints |
| `src/api/routers/demand.py` | Demand forecasting endpoints |
| `src/api/routers/llm.py` | LLM narration endpoints |
| `src/api/main.py` | FastAPI app entry point, CORS, lifespan startup |
| `src/api_client.py` | HTTP client for Streamlit-to-API communication |
| `src/app.py` | Streamlit dashboard, fully API-driven |
