# analysis/error_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from models.utils_metrics import mad, mae, rmse, mape

def error_over_time(actual, forecast, plot_dir="img/", model_name="Modell"):
    df = pd.DataFrame({"actual": actual, "forecast": forecast})
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Fehler pro Jahr
    metrics_year = df.groupby("year").apply(lambda g: pd.Series({
        "MAD": mad(g["actual"], g["forecast"]),
        "MAE": mae(g["actual"], g["forecast"]),
        "RMSE": rmse(g["actual"], g["forecast"]),
        "MAPE": mape(g["actual"], g["forecast"]),
    }))

    # Fehler pro Monat (über alle Jahre)
    metrics_month = df.groupby("month").apply(lambda g: pd.Series({
        "MAD": mad(g["actual"], g["forecast"]),
        "MAE": mae(g["actual"], g["forecast"]),
        "RMSE": rmse(g["actual"], g["forecast"]),
        "MAPE": mape(g["actual"], g["forecast"]),
    }))

    # Plot MAPE pro Jahr
    plt.figure(figsize=(10, 4))
    plt.bar(metrics_year.index.astype(str), metrics_year["MAPE"])
    plt.title(f"MAPE pro Jahr – {model_name}")
    plt.xlabel("Jahr")
    plt.ylabel("MAPE (%)")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}{model_name.lower()}_mape_per_year.png")
    plt.close()

    # Plot MAPE pro Monat
    plt.figure(figsize=(10, 4))
    month_names = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
    plt.bar([month_names[m-1] for m in metrics_month.index], metrics_month["MAPE"])
    plt.title(f"MAPE pro Monat – {model_name}")
    plt.xlabel("Monat")
    plt.ylabel("MAPE (%)")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}{model_name.lower()}_mape_per_month.png")
    plt.close()

    return metrics_year, metrics_month
