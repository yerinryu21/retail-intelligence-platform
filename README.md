# Retail Intelligence Platform

## Project Overview
A retail intelligence platform built on the UCI Online Retail dataset (~541K UK 
e-commerce transactions), combining customer churn prediction, product demand 
forecasting, and an LLM narration layer. This README documents the development 
process week by week, module by module.

---

## Week 1 — Data Foundation

Week 1 focused on building a clean, reliable data foundation that both the churn 
prediction module (Module 1) and demand forecasting module (Module 2) build on top of.

### Environment Setup
- Python 3.10, managed via Conda (`retail_intelligence` environment)
- Core libraries: pandas, numpy, scikit-learn, xgboost, shap, prophet, 
  langchain, streamlit, fastapi, plotly
- Local LLM via Ollama (llama3, nomic-embed-text)
- Version control: Git + GitHub

---

### File-by-File Breakdown

#### `data/raw/Online Retail.xlsx`
The original, unmodified UCI Online Retail dataset — 541,909 transaction-level 
rows, 8 columns (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, 
UnitPrice, CustomerID, Country). Not committed to Git (excluded via `.gitignore`) 
since raw data shouldn't live in version control.

#### `notebooks/01_data_exploration.ipynb`
**Purpose:** First look at the raw dataset before any cleaning. Answers 10 
exploratory questions: dataset shape/date range, unique customers/products, 
country distribution, cancelled orders, negative quantities/prices, duplicates, 
missing data, top-selling products, and typical transaction shape.

**Key findings:**
- 24.9% of rows (135,080) have missing CustomerID
- 9,288 rows are cancelled orders (InvoiceNo starting with 'C')
- 10,624 rows have negative quantities; 2,517 have zero/negative prices
- 5,268 exact duplicate rows
- 91.4% of transactions are from the United Kingdom
- Extreme outliers present (quantity range: -80,995 to 80,995)

**Why it matters:** These findings directly informed the cleaning rules applied 
in `data_cleaning.py`. Without this step, data quality issues would silently 
corrupt every downstream model.

#### `src/shared/data_cleaning.py`
**Purpose:** Transforms the raw Excel file into a single clean transaction-level 
dataset used by both Module 1 and Module 2.

**Cleaning steps applied:**
1. Parse InvoiceDate to datetime
2. Remove cancelled orders (InvoiceNo starting with 'C')
3. Remove rows with missing CustomerID
4. Remove negative/zero quantities
5. Remove negative/zero prices
6. Remove rows with missing Description
7. Add derived columns: `TotalPrice`, `Year`, `Month`, `DayOfWeek`, `Week`, `Date`
8. Filter to United Kingdom only (91%+ of data, keeps analysis consistent)

**Output:** `data/processed/clean_retail.csv` — ~354,000 rows, the single source 
of truth all later feature engineering builds from.

#### `notebooks/02_data_cleaning_verification.ipynb`
**Purpose:** Confirms the cleaning script worked correctly via explicit 
assertions (no cancelled orders remain, no missing CustomerIDs, no negative 
quantities/prices, all derived columns present). This is a safety check — if 
any assertion fails, it means the cleaning pipeline has a bug that would 
silently corrupt downstream modeling.

---

#### `src/module1_churn/feature_engineering.py`
**Purpose:** Transforms the clean transaction-level data into one row per 
customer — the format the churn classifier needs. Computes RFM (Recency, 
Frequency, Monetary) features plus behavioral features (AvgOrderValue, 
UniqueProducts, AvgQuantity, DaysActive, OrdersPerDay), and assigns a binary 
`Churned` label.

**Churn label definition:** A customer is labeled churned if their last 
purchase was more than 90 days before the dataset's snapshot date (2011-12-09).

**Why 90 days, specifically:** Initially adopted as a common e-commerce default, 
then validated against this dataset directly — see `03_customer_features_verification.ipynb` 
for the evidence.

**Output:** `data/processed/customer_features.csv` — 3,920 customers, 11 columns, 
33.3% churn rate.

#### `notebooks/03_customer_features_verification.ipynb`
**Purpose:** Validates that the customer-level feature table and churn label 
are both structurally correct and behaviorally meaningful. Contains three 
analyses:

1. **Distribution comparison (churned vs. active):** Histograms across Recency, 
   Frequency, Monetary, AvgOrderValue, UniqueProducts, and DaysActive confirm 
   churned customers behave distinctly differently from active ones — validating 
   that the label captures real signal, not noise.

2. **Class imbalance analysis:** Churn rate is 33.3% (2,613 active vs. 1,307 
   churned) — a ~2:1 ratio. This is moderate imbalance, not severe, but enough 
   that plain accuracy would be misleading (a model predicting "always active" 
   would score ~67% while catching zero real churners). This directly informs 
   the evaluation metrics chosen for Week 2: Precision, Recall, F1, and PR-AUC 
   will be prioritized over accuracy, with class-weighting applied during training.

3. **Churn threshold validation:** Tested whether 90 days is actually a 
   meaningful cutoff, using two approaches:
   - Repeat customers (65.6% of the base) have a median reorder gap of 53.4 
     days, with the 75th percentile at ~92 days — 90 days sits near where 
     "normal" reorder behavior ends and "unusual silence" begins.
   - Directly tested return behavior: of 12,637 measured purchase-to-purchase 
     gaps across all customers, only 12.4% exceeded 90 days — meaning when a 
     customer goes quiet for 90+ days, they fail to return 87.6% of the time. 
     This confirms 90 days is a reasonably strong, evidence-backed churn signal.
   - **Known limitation:** this is a single global cutoff; purchase rhythms 
     vary by customer (some reorder every 2 weeks, others once a year). A 
     future improvement would use a per-customer adaptive threshold.

---

#### `src/module2_demand/feature_engineering.py`
**Purpose:** Transforms clean transaction data into weekly product-level demand 
time series — the format Prophet forecasting needs in Week 4.

**Process:** Selects top N products by total quantity sold, aggregates to 
weekly totals (quantity, revenue, orders, unique customers), and fills any 
missing weeks with zero (critical for time series — gaps must be explicit, 
not absent).

**Bug encountered and fixed:** Initial version grouped by `(StockCode, 
Description, Week)`, but some products had multiple inconsistent Description 
values over time (e.g., StockCode 22197 appeared as both "POPCORN HOLDER" and 
"SMALL POPCORN HOLDER"), creating duplicate `(StockCode, Week)` combinations 
that broke the reindex step. Fixed by grouping on `(StockCode, Week)` only, 
then attaching each product's most frequent Description as a separate step.

**Output:** `data/processed/weekly_demand.csv` — 20 products × 54 weeks, 
1,080 rows.

#### `notebooks/04_demand_exploration.ipynb`
**Purpose:** Visualizes weekly demand for top products and evaluates whether 
"top by volume" is actually a good product selection method for forecasting.

**Key finding:** Selecting products by raw total volume is misleading. Two of 
the top 5 products by volume (StockCode 23843, 23166) turned out to be flat 
for nearly the entire year with one massive single-week spike — clearly bulk/ 
wholesale orders, not organic recurring demand. StockCode 23843 sold 80,995 
units total, but 98%+ of that came from a single week (active in only 1.85% 
of all 54 weeks).

**Improved selection method developed:** Ranking products by `PctWeeksActive` 
(% of weeks with any sales) and `CoeffOfVariation` (std/mean of weekly quantity) 
surfaces genuinely consistent, forecastable products (e.g., StockCode 85099B, 
84946, 22178 — all active in ~98% of weeks with low relative variability), 
instead of volume-dominant outliers.

**Action for Week 4:** Use consistency-based ranking, not raw total volume, 
when selecting the final product set for Prophet forecasting.