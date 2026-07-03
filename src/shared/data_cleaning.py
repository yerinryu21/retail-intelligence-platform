import pandas as pd
import numpy as np

def load_raw_data(path: str = 'data/raw/Online Retail.xlsx') -> pd.DataFrame:
    """Load raw UCI Online Retail dataset"""
    print("Loading raw data...")
    df = pd.read_excel(path)
    print(f"Raw shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for UCI Online Retail dataset.
    Returns cleaned dataframe ready for both churn and demand modules.
    """
    print("\nStarting cleaning pipeline...")
    original_rows = len(df)
    
    # Step 1 — Parse dates
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"✅ Parsed dates")
    
    # Step 2 — Remove cancelled orders (InvoiceNo starting with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    print(f"✅ Removed cancelled orders: {original_rows - len(df):,} rows removed")
    
    # Step 3 — Remove rows with missing CustomerID
    before = len(df)
    df = df.dropna(subset=['CustomerID'])
    print(f"✅ Removed missing CustomerID: {before - len(df):,} rows removed")
    
    # Step 4 — Remove negative or zero quantities
    before = len(df)
    df = df[df['Quantity'] > 0]
    print(f"✅ Removed negative/zero quantities: {before - len(df):,} rows removed")
    
    # Step 5 — Remove negative or zero prices
    before = len(df)
    df = df[df['UnitPrice'] > 0]
    print(f"✅ Removed negative/zero prices: {before - len(df):,} rows removed")
    
    # Step 6 — Remove rows with missing Description
    before = len(df)
    df = df.dropna(subset=['Description'])
    print(f"✅ Removed missing descriptions: {before - len(df):,} rows removed")
    
    # Step 7 — Add useful derived columns
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Week'] = df['InvoiceDate'].dt.isocalendar().week.astype(int)
    df['Date'] = df['InvoiceDate'].dt.date
    print(f"✅ Added derived columns: TotalPrice, Year, Month, DayOfWeek, Week, Date")
    
    # Step 8 — Clean CustomerID type
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    
    # Step 9 — Focus on UK only
    before = len(df)
    df = df[df['Country'] == 'United Kingdom']
    print(f"✅ Filtered to UK only: {before - len(df):,} rows removed")
    
    # Final summary
    print(f"\n📊 Cleaning complete:")
    print(f"   Original rows: {original_rows:,}")
    print(f"   Clean rows: {len(df):,}")
    print(f"   Rows removed: {original_rows - len(df):,} ({(original_rows - len(df))/original_rows:.1%})")
    print(f"   Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    print(f"   Unique customers: {df['CustomerID'].nunique():,}")
    print(f"   Unique products: {df['StockCode'].nunique():,}")
    
    return df

def save_clean_data(df: pd.DataFrame, path: str = 'data/processed/clean_retail.csv'):
    """Save cleaned dataframe"""
    df.to_csv(path, index=False)
    print(f"\n✅ Saved clean data to {path}")

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean)