import pandas as pd
import matplotlib.pyplot as plt
from models.utils_metrics import mad, mae, rmse, mape

import pmdarima as pm

def arima_sarima_forecast(series, train_end_year, test_end_year, seasonal=True, m=12, plot_dir="img/"):
    train = series[series.index.year <= train_end_year]
    test = series[(series.index.year > train_end_year) & (series.index.year <= test_end_year)]

    model = pm.auto_arima(train, seasonal=seasonal, m=m, stepwise=True, suppress_warnings=True, error_action="ignore")
    forecast = pd.Series(model.predict(n_periods=len(test)), index=test.index)
    actual = test.loc[forecast.index]

    # Save results
    results = {
        "mad": mad(actual, forecast),
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "mape": mape(actual, forecast),
        "forecast": forecast,
        "actual": actual,
        "forecast_ma": forecast.rolling(window=12).mean()
    }

    # Plot (without MA)
    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Train", color="grey", alpha=0.6, linewidth=1)
    plt.plot(actual, label="Actual", color="midnightblue", alpha=0.9, linewidth=1.5)
    plt.plot(forecast, label="Forecast", color="tomato", linewidth=1.1)

    plt.title(f"SARIMA Forecast vs Actual ({train_end_year+1}-{test_end_year})", fontweight='bold', fontsize=16)
    plt.xlabel("")
    plt.ylabel("Total Energy Consumption\n(in Quadrillion BTU)", fontweight='bold', fontsize=12)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(pd.Timestamp("2011-01-01"), pd.Timestamp("2024-12-31"))
    ax.set_xticks(pd.date_range(start="2011-01-01", end="2024-12-31", freq="YS"))
    ax.set_xticklabels([str(y.year) for y in pd.date_range("2011", "2024", freq="YS")], rotation=0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}sarima_forecast_vs_actual.png")
    plt.close()

    return results





