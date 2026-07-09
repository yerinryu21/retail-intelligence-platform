import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_churn_data(customer_features_path: str = 'data/processed/customer_features.csv'):
    """
    Prepare customer features for churn modeling.

    Note: Week 1's feature engineering already computes every feature and the
    Churned label relative to a single fixed snapshot date (2011-12-09), using
    each customer's full transaction history up to that point. That means
    leakage is already prevented at the feature-engineering stage — there's no
    "future" information baked into any row.

    Because of that, we do NOT need (or want) a time-based split here. Splitting
    by transaction date on top of an already-snapshotted table just partitions
    customers by an unrelated condition (recent transaction timing) that happens
    to correlate strongly with the churn label — which is what caused the
    43% vs 2.8% churn rate mismatch.

    Instead we use a stratified random split, which keeps the churn rate
    consistent between train and test.
    """

    print("Loading data...")
    customers = pd.read_csv(customer_features_path)

    feature_cols = [
        'Frequency', 'Monetary',
        'AvgOrderValue', 'UniqueProducts',
        'AvgQuantity', 'DaysActive', 'OrdersPerDay'
    ]
    target_col = 'Churned'

    customers['CustomerID'] = customers['CustomerID'].astype(str)

    # ── Handle any remaining missing values ───────────────────────
    for col in feature_cols:
        customers[col] = customers[col].fillna(customers[col].median())

    X = customers[feature_cols]
    y = customers[target_col]

    # ── Stratified random split ────────────────────────────────────
    # stratify=y ensures train and test have the same churn rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ── Scale features ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n📊 Data preparation complete:")
    print(f"   Training set: {X_train.shape[0]:,} customers")
    print(f"   Test set: {X_test.shape[0]:,} customers")
    print(f"   Features: {len(feature_cols)}")
    print(f"\n   Training churn rate: {y_train.mean():.1%}")
    print(f"   Test churn rate: {y_test.mean():.1%}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler
    }


if __name__ == "__main__":
    data = prepare_churn_data()
    print("\n✅ Data preparation complete")
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")