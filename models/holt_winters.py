# models/holt_winters.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from models.utils_metrics import mad, mae, rmse, mape

def holt_winters_forecast(series, train_end_year, test_end_year, seasonal='add', plot_dir="img/"):
    # Split train/test
    train = series[(series.index.year <= train_end_year)]
    test = series[(series.index.year > train_end_year) & (series.index.year <= test_end_year)]

    model = ExponentialSmoothing(train, trend='add', seasonal=seasonal, seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(len(test))

    # Metrics
    actual = test.loc[forecast.index]
    results = {
        "mad": mad(actual, forecast),
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "mape": mape(actual, forecast),
        "forecast": forecast,
        "actual": actual,
    }

    # Plot: Forecast vs Actual
    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Train", color="grey", alpha=0.6)
    plt.plot(actual, label="Actual", color="blue", alpha=0.7)
    plt.plot(forecast, label="Forecast", color="red", linestyle="--")
    plt.title(f"Holt-Winters Forecast vs Actual ({train_end_year+1}-{test_end_year})")
    plt.xlabel("Date")
    plt.ylabel("Total Primary Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}holt_winters_forecast_vs_actual.png")
    plt.close()

    return results
