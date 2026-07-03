import pandas as pd
import numpy as np

def build_demand_features(df: pd.DataFrame,
                         top_n_products: int = 20) -> pd.DataFrame:
    """
    Transform transaction-level data into weekly product demand time series.
    """
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Step 1 — Select top N products by total quantity sold
    top_products = (
        df.groupby('StockCode')['Quantity']
        .sum()
        .sort_values(ascending=False)
        .head(top_n_products)
        .index.tolist()
    )
    
    print(f"Selected top {top_n_products} products by volume")
    print(f"Top 5: {top_products[:5]}")
    
    # Step 2 — Filter to top products only
    df_top = df[df['StockCode'].isin(top_products)].copy()
    
    # Step 3 — Aggregate to weekly level
    df_top['Week'] = df_top['InvoiceDate'].dt.to_period('W').dt.start_time
    
    weekly_demand = (
        df_top.groupby(['StockCode', 'Week'])
        .agg(
            TotalQuantity=('Quantity', 'sum'),
            TotalRevenue=('TotalPrice', 'sum'),
            NumOrders=('InvoiceNo', 'nunique'),
            NumCustomers=('CustomerID', 'nunique')
        )
        .reset_index()
    )

    # Diagnostic: check for duplicates
    dupes = weekly_demand.duplicated(subset=['StockCode', 'Week'], keep=False)
    print(f"Duplicate StockCode/Week rows: {dupes.sum()}")
    if dupes.sum() > 0:
        print(weekly_demand[dupes].sort_values(['StockCode', 'Week']).head(20))
    
    # Step 4 — Fill missing weeks with 0
    all_weeks = pd.date_range(
        start=df_top['Week'].min(),
        end=df_top['Week'].max(),
        freq='W-MON'
    )
    
    full_index = pd.MultiIndex.from_product(
        [top_products, all_weeks],
        names=['StockCode', 'Week']
    )
    
    weekly_demand = (
        weekly_demand
        .set_index(['StockCode', 'Week'])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    
    product_descriptions = (
        df_top.groupby('StockCode')['Description']
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    weekly_demand = weekly_demand.merge(product_descriptions, on='StockCode', how='left')
    
    
    print(f"\n📊 Demand features built:")
    print(f"   Products: {weekly_demand['StockCode'].nunique()}")
    print(f"   Weeks: {weekly_demand['Week'].nunique()}")
    print(f"   Total rows: {len(weekly_demand):,}")
    print(f"   Date range: {weekly_demand['Week'].min()} to {weekly_demand['Week'].max()}")
    
    return weekly_demand

if __name__ == "__main__":
    df = pd.read_csv('data/processed/clean_retail.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    demand = build_demand_features(df)
    demand.to_csv('data/processed/weekly_demand.csv', index=False)
    print("\n✅ Saved to data/processed/weekly_demand.csv")