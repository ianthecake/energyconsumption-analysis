# models/arima_model.py

import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from models.utils_metrics import mad, mae, rmse, mape

def arima_sarima_forecast(series, train_end_year, test_end_year, seasonal=True, m=12, plot_dir="img/"):
    # Split train/test
    train = series[(series.index.year <= train_end_year)]
    test = series[(series.index.year > train_end_year) & (series.index.year <= test_end_year)]

    # ARIMA/SARIMA Modell mit automatischer Parameterauswahl
    model = pm.auto_arima(
        train,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    forecast = pd.Series(model.predict(n_periods=len(test)), index=test.index)

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
    plt.plot(forecast, label="Forecast", color="green", linestyle="--")
    plt.title(f"SARIMA Forecast vs Actual ({train_end_year+1}-{test_end_year})")
    plt.xlabel("Date")
    plt.ylabel("Total Primary Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}sarima_forecast_vs_actual.png")
    plt.close()

    return results
