import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

def analyze_error_patterns(fold_results: pd.DataFrame) -> dict:
    """
    Deep analysis of forecasting errors across folds.
    Assumes fold_results is already filtered to DataQualityFlag == 'OK'.
    """

    print("="*60)
    print("ERROR PATTERN ANALYSIS (OK products only)")
    print("="*60)

    insights = {}

    # ── 1. Error by fold ─────────────────────────────────────────
    fold_mae = fold_results.groupby('Fold')['Prophet_MAE'].agg(['mean', 'std'])

    print("\n📊 Error by fold:")
    for fold, row in fold_mae.iterrows():
        print(f"  Fold {fold}: MAE = {row['mean']:.2f} ± {row['std']:.2f}")

    insights['fold_mae'] = fold_mae

    # ── 2. Best and worst products by average MAE ─────────────────
    product_mae = fold_results.groupby('StockCode')['Prophet_MAE'].agg(['mean', 'median'])

    best_products = product_mae['mean'].nsmallest(3)
    worst_products = product_mae['mean'].nlargest(3)

    print(f"\n🏆 Most predictable products (lowest mean MAE):")
    for code, mae in best_products.items():
        print(f"  {code}: mean MAE = {mae:.2f}")

    print(f"\n⚠️ Least predictable products (highest mean MAE):")
    for code, mae in worst_products.items():
        print(f"  {code}: mean MAE = {mae:.2f}")

    insights['best_products'] = best_products
    insights['worst_products'] = worst_products
    insights['product_mae'] = product_mae

    # ── 3. Win rate — report by product, not just overall ─────────
    win_rate_by_product = fold_results.groupby('StockCode').agg(
        WinRateVsNaive=('BeatNaive', 'mean'),
        WinRateVsSeasonal=('BeatSeasonal', 'mean')
    ).sort_values('WinRateVsNaive')

    overall_win_naive = fold_results['BeatNaive'].mean()
    overall_win_seasonal = fold_results['BeatSeasonal'].mean()

    always_loses_naive = win_rate_by_product[win_rate_by_product['WinRateVsNaive'] == 0]

    print(f"\n🎯 Overall Prophet win rates:")
    print(f"  vs Naive baseline:    {overall_win_naive:.1%}")
    print(f"  vs Seasonal naive:    {overall_win_seasonal:.1%}")

    if len(always_loses_naive) > 0:
        print(f"\n🔴 Products that NEVER beat naive (0% win rate, all folds):")
        for code in always_loses_naive.index:
            print(f"  {code}")

    insights['win_rate_naive'] = overall_win_naive
    insights['win_rate_seasonal'] = overall_win_seasonal
    insights['win_rate_by_product'] = win_rate_by_product
    insights['always_loses_naive'] = list(always_loses_naive.index)

    # ── 4. MAPE distribution ───────────────────────────────────────
    # MAPE values already computed with the 5-unit min_actual guard in
    # walk_forward_validation.py — just reading the saved results here,
    # not recalculating.
    valid_mape = fold_results['Prophet_MAPE'].dropna()

    print(f"\n📉 MAPE distribution (mean vs median — Week 4 found these can disagree):")
    print(f"  Median MAPE: {valid_mape.median():.1f}%")
    print(f"  Mean MAPE:   {valid_mape.mean():.1f}%")
    print(f"  25th pct:    {valid_mape.quantile(0.25):.1f}%")
    print(f"  75th pct:    {valid_mape.quantile(0.75):.1f}%")

    insights['mape_stats'] = {
        'median': valid_mape.median(),
        'mean': valid_mape.mean(),
        'p25': valid_mape.quantile(0.25),
        'p75': valid_mape.quantile(0.75)
    }

    return insights


def plot_backtest_deep_dive(fold_results: pd.DataFrame,
                             save_dir: str = 'notebooks/demand_plots') -> None:
    """Comprehensive backtest visualizations, OK products only."""

    os.makedirs(save_dir, exist_ok=True)

    # ── Plot 1: Error heatmap — product vs fold ────────────────────
    pivot = fold_results.pivot_table(
        values='Prophet_MAE',
        index='StockCode',
        columns='Fold',
        aggfunc='mean'
    )

    fig1 = px.imshow(
        pivot,
        title='Forecast Error Heatmap: Product × Fold (MAE, OK products only)',
        labels={'x': 'Fold', 'y': 'Product', 'color': 'MAE'},
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    fig1.update_layout(height=600)
    fig1.write_html(f'{save_dir}/error_heatmap.html')
    fig1.show()
    print("✅ Saved: error_heatmap.html")

    # ── Plot 2: Prophet vs baselines across folds ──────────────────
    fold_avg = fold_results.groupby('Fold').agg(
        Prophet_MAE=('Prophet_MAE', 'mean'),
        Naive_MAE=('Naive_MAE', 'mean'),
        Seasonal_MAE=('Seasonal_MAE', 'mean')
    ).reset_index()

    fig2 = go.Figure()
    for model, color in [('Prophet_MAE', '#1f77b4'),
                          ('Naive_MAE', '#d62728'),
                          ('Seasonal_MAE', '#ff7f0e')]:
        label = model.replace('_MAE', '').replace('_', ' ')
        fig2.add_trace(go.Scatter(
            x=fold_avg['Fold'], y=fold_avg[model],
            mode='lines+markers', name=label,
            line=dict(color=color, width=2), marker=dict(size=8)
        ))

    fig2.update_layout(
        title='Average MAE Across Folds: Prophet vs Baselines (OK products only)',
        xaxis_title='Fold (chronological)',
        yaxis_title='Mean Absolute Error (units)',
        height=450
    )
    fig2.write_html(f'{save_dir}/fold_comparison.html')
    fig2.show()
    print("✅ Saved: fold_comparison.html")

    # ── Plot 3: MAPE distribution box plot ─────────────────────────
    mape_long = fold_results.melt(
        id_vars=['StockCode', 'Fold'],
        value_vars=['Prophet_MAPE', 'Naive_MAPE', 'Seasonal_MAPE'],
        var_name='Model', value_name='MAPE'
    ).dropna()

    mape_long['Model'] = mape_long['Model'].str.replace('_MAPE', '')

    fig3 = px.box(
        mape_long, x='Model', y='MAPE',
        title='MAPE Distribution: Prophet vs Baselines (OK products only)',
        color='Model',
        color_discrete_map={'Prophet': '#1f77b4', 'Naive': '#d62728', 'Seasonal': '#ff7f0e'},
        labels={'MAPE': 'MAPE (%)'}
    )
    fig3.update_layout(height=450, showlegend=False)
    fig3.write_html(f'{save_dir}/mape_distribution.html')
    fig3.show()
    print("✅ Saved: mape_distribution.html")

    # ── Plot 4: Win rate by product ────────────────────────────────
    win_rates = fold_results.groupby('StockCode').agg(
        WinRateNaive=('BeatNaive', 'mean'),
        WinRateSeasonal=('BeatSeasonal', 'mean')
    ).reset_index()

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='vs Naive', x=win_rates['StockCode'],
                           y=win_rates['WinRateNaive'], marker_color='#1f77b4'))
    fig4.add_trace(go.Bar(name='vs Seasonal', x=win_rates['StockCode'],
                           y=win_rates['WinRateSeasonal'], marker_color='#2ca02c'))
    fig4.add_hline(y=0.5, line_dash='dash', line_color='red',
                   annotation_text='50% win rate threshold')
    fig4.update_layout(
        barmode='group',
        title='Prophet Win Rate by Product (OK products only)',
        yaxis_title='Win Rate', yaxis_tickformat='.0%', height=450
    )
    fig4.write_html(f'{save_dir}/win_rates.html')
    fig4.show()
    print("✅ Saved: win_rates.html")


def generate_backtest_report(fold_results: pd.DataFrame, insights: dict) -> str:
    """Plain-text backtest report for README, mean AND median reported per Week 4 finding."""

    win_naive = insights['win_rate_naive']
    win_seasonal = insights['win_rate_seasonal']
    mape = insights['mape_stats']
    product_mae = insights['product_mae']
    best = list(insights['best_products'].index[:2])
    worst = list(insights['worst_products'].index[:1])
    always_loses = insights['always_loses_naive']

    n_products = fold_results['StockCode'].nunique()

    report = f"""
## Demand Forecasting — Walk-Forward Validation Results

**Methodology:** 5-fold walk-forward validation, 4-week test windows, trend-only
Prophet model (weekly/yearly seasonality and holidays disabled — confirmed
non-signal for this dataset in Week 4). {n_products} products evaluated after
excluding 2 low-reliability products (sparse/spike-driven demand).

**Key Results:**
- Prophet beats naive baseline in {win_naive:.0%} of folds
- Prophet beats seasonal naive in {win_seasonal:.0%} of folds
- Median MAPE: {mape['median']:.1f}% | Mean MAPE: {mape['mean']:.1f}%
  (reported together — Week 4 found these can tell different stories)
- MAPE interquartile range: {mape['p25']:.1f}% — {mape['p75']:.1f}%

**Most predictable products:** {', '.join(best)}
**Hardest to forecast:** {', '.join(worst)}

**Products that never beat naive across any fold:** {', '.join(always_loses) if always_loses else 'None'}
(consistent underperformance, not single-window noise — candidates for
Prophet config tuning rather than the trend-only default)

**Conclusion:** Prophet consistently outperforms seasonal naive across nearly
every fold and product. Against plain naive, results are closer and vary by
product — a minority of products (see above) show reliable, multi-window
underperformance and are worth targeted investigation rather than blanket
model tuning.
"""

    print(report)
    with open('data/processed/backtest_report.txt', 'w') as f:
        f.write(report)
    print("✅ Report saved to data/processed/backtest_report.txt")

    return report


if __name__ == "__main__":
    fold_results = pd.read_csv('data/processed/walkforward_folds.csv')

    # Filter to OK products from the start — not bolted on after,
    # per Day 1's carryover lesson.
    fold_results_ok = fold_results[fold_results['DataQualityFlag'] == 'OK'].copy()

    insights = analyze_error_patterns(fold_results_ok)
    plot_backtest_deep_dive(fold_results_ok)
    report = generate_backtest_report(fold_results_ok, insights)