import pandas as pd
import numpy as np

def build_customer_features(df: pd.DataFrame, 
                           snapshot_date: str = '2011-12-09',
                           churn_days: int = 90) -> pd.DataFrame:
    """
    Transform transaction-level data into customer-level features.
    
    Parameters:
    - df: cleaned transaction dataframe
    - snapshot_date: the date we're observing from
    - churn_days: days of inactivity = churned
    
    Returns: one row per customer with features + churn label
    """
    
    snapshot = pd.Timestamp(snapshot_date)
    churn_cutoff = snapshot - pd.Timedelta(days=churn_days)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print(f"Snapshot date: {snapshot.date()}")
    print(f"Churn cutoff: {churn_cutoff.date()} (no purchase since this date = churned)")
    
    # --- RFM Features ---
    
    last_purchase = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    last_purchase.columns = ['CustomerID', 'LastPurchaseDate']
    last_purchase['Recency'] = (snapshot - last_purchase['LastPurchaseDate']).dt.days
    
    frequency = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    frequency.columns = ['CustomerID', 'Frequency']
    
    monetary = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
    monetary.columns = ['CustomerID', 'Monetary']
    
    aov = df.groupby('CustomerID').apply(
        lambda x: x.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    ).reset_index()
    aov.columns = ['CustomerID', 'AvgOrderValue']
    
    # --- Behavioral Features ---
    
    unique_products = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
    unique_products.columns = ['CustomerID', 'UniqueProducts']
    
    avg_quantity = df.groupby('CustomerID')['Quantity'].mean().reset_index()
    avg_quantity.columns = ['CustomerID', 'AvgQuantity']
    
    first_purchase = df.groupby('CustomerID')['InvoiceDate'].min().reset_index()
    first_purchase.columns = ['CustomerID', 'FirstPurchaseDate']
    
    days_active = last_purchase.merge(first_purchase, on='CustomerID')
    days_active['DaysActive'] = (
        days_active['LastPurchaseDate'] - days_active['FirstPurchaseDate']
    ).dt.days
    
    days_active = days_active.merge(frequency, on='CustomerID')
    days_active['OrdersPerDay'] = days_active.apply(
        lambda x: x['Frequency'] / max(x['DaysActive'], 1), axis=1
    )
    
    # --- Merge all features ---
    customers = last_purchase[['CustomerID', 'LastPurchaseDate', 'Recency']]
    customers = customers.merge(frequency, on='CustomerID')
    customers = customers.merge(monetary, on='CustomerID')
    customers = customers.merge(aov, on='CustomerID')
    customers = customers.merge(unique_products, on='CustomerID')
    customers = customers.merge(avg_quantity, on='CustomerID')
    customers = customers.merge(
        days_active[['CustomerID', 'DaysActive', 'OrdersPerDay']], 
        on='CustomerID'
    )
    
    # --- Churn Label ---
    customers['Churned'] = (
        customers['LastPurchaseDate'] < churn_cutoff
    ).astype(int)
    
    # --- Summary ---
    print(f"\n📊 Customer features built:")
    print(f"   Total customers: {len(customers):,}")
    print(f"   Churned: {customers['Churned'].sum():,} ({customers['Churned'].mean():.1%})")
    print(f"   Active: {(customers['Churned']==0).sum():,} ({(customers['Churned']==0).mean():.1%})")
    print(f"\nFeature columns: {customers.columns.tolist()}")
    
    return customers

if __name__ == "__main__":
    df = pd.read_csv('data/processed/clean_retail.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    customers = build_customer_features(df)
    customers.to_csv('data/processed/customer_features.csv', index=False)
    print("\n✅ Saved to data/processed/customer_features.csv")