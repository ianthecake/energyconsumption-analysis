# analysis/event_impact_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from models.utils_metrics import mad, mae, rmse, mape

def event_error_analysis(actual, forecast, events, plot_dir="img/", model_name="Modell"):
    """
    events: dict, z.B. {
        "Covid (2020-2021)": ("2020-01-01", "2021-12-31"),
        "Russland-Krieg (2022-)": ("2022-01-01", "2024-12-31")
    }
    actual, forecast: pd.Series (gleicher Index)
    """

    error_results = {}

    for event, (start, end) in events.items():
        # Zeitraum extrahieren
        period_actual = actual.loc[(actual.index >= start) & (actual.index <= end)]
        period_forecast = forecast.loc[(forecast.index >= start) & (forecast.index <= end)]

        if len(period_actual) == 0:
            print(f"Keine Testdaten für Zeitraum {event}")
            continue

        result = {
            "MAD": mad(period_actual, period_forecast),
            "MAE": mae(period_actual, period_forecast),
            "RMSE": rmse(period_actual, period_forecast),
            "MAPE": mape(period_actual, period_forecast),
        }
        error_results[event] = result

        # Plot Forecast vs Actual für das Event
        plt.figure(figsize=(8, 3))
        plt.plot(period_actual, label="Actual", color="blue", alpha=0.7)
        plt.plot(period_forecast, label="Forecast", color="red", linestyle="--")
        plt.title(f"{model_name} Forecast vs Actual: {event}")
        plt.xlabel("Date")
        plt.ylabel("Total Primary Energy Consumption")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}{model_name.lower()}_forecast_vs_actual_{event.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')}.png")
        plt.close()

    return pd.DataFrame(error_results).T

