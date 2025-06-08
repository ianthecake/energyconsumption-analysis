import pandas as pd
import matplotlib.pyplot as plt
from models.utils_metrics import mad, mae, rmse, mape
import matplotlib.dates as mdates

def event_error_analysis(actual, forecast, events, plot_dir="img/", model_name="Modell"):
    error_results = {}

    for event, (start, end) in events.items():
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

        plt.figure(figsize=(8, 3))
        plt.plot(period_actual, label="Actual", color="midnightblue", alpha=0.7, linewidth=3)
        plt.plot(period_forecast, label="Forecast", color="tomato", linewidth=3, linestyle="--", alpha=0.9)

        plt.title(f"{model_name} Forecast vs Actual: {event}", fontweight='bold', fontsize=16, pad=20)
        plt.xlabel("")

        plt.ylabel("Energy Consumption\n(in Quadrillion BTU)", fontsize=10, fontweight='bold')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=5))  # alle 2 Monate
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        plt.setp(ax.get_xticklabels(), fontsize=10)

        plt.legend()
        plt.tight_layout()
        fname = f"{plot_dir}{model_name.lower()}_forecast_vs_actual_{event.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')}.png"
        plt.savefig(fname)
        plt.close()

    return pd.DataFrame(error_results).T
