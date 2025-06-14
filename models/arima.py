import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from models.utils_metrics import mad, mae, rmse, mape

def arima_sarima_forecast(series, train_end_year, test_end_year, seasonal=True, m=12, plot_dir="img/"):
    train = series[(series.index.year <= train_end_year)]
    test = series[(series.index.year > train_end_year) & (series.index.year <= test_end_year)]

    model = pm.auto_arima(
        train,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    forecast = pd.Series(model.predict(n_periods=len(test)), index=test.index)

    actual = test.loc[forecast.index]
    results = {
        "mad": mad(actual, forecast),
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "mape": mape(actual, forecast),
        "forecast": forecast,
        "actual": actual,
    }

    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Train", color="grey", alpha=0.6, linewidth=1)
    plt.plot(actual, label="Actual", color="midnightblue", alpha=0.9, linestyle="-", linewidth=1.5)
    plt.plot(forecast, label="Forecast", color="tomato", linestyle="-", linewidth=1.1)

    plt.title(f"SARIMA Forecast vs Actual ({train_end_year+1}-{test_end_year})", fontweight='bold', fontsize=16, pad=20)
    plt.xlabel("")
    plt.ylabel("Total Energy Consumption\n(in Quadrillion BTU)", fontweight='bold', fontsize=12)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    years = pd.date_range(start=train.index.min(), end=forecast.index.max(), freq="YS")
    year_labels = [y for y in years.year if (y % 4 == 0 or y == years.year[0] or y == years.year[-1])]
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in year_labels])
    ax.set_xticklabels([str(y) for y in year_labels], rotation=0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}sarima_forecast_vs_actual.png")
    plt.close()

    return results
