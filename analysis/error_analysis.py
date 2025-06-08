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

    # Plot MAPE pro Jahr – mit farb- und alpha-codierten Balken
    plt.figure(figsize=(10, 4))
    cmap = plt.cm.Blues
    mape_vals = metrics_year["MAPE"].values
    norm = plt.Normalize(mape_vals.min(), mape_vals.max())
    bar_colors = [cmap(norm(m)) for m in mape_vals]
    bar_alphas = [0.7 + 0.3 * norm(m) for m in mape_vals]

    bars = []
    for idx, (color, alpha) in enumerate(zip(bar_colors, bar_alphas)):
        bar = plt.bar(str(metrics_year.index[idx]), mape_vals[idx], color=color, alpha=alpha)
        bars.append(bar)

    # Titel und Achsenbeschriftung fett und größer
    plt.title(f"MAPE per year – {model_name}", fontweight='bold', fontsize=16, pad=20)
    #plt.xlabel("Year", fontweight='bold', fontsize=12)
    plt.xlabel("")
    plt.ylabel("MAPE (%)", fontweight='bold', fontsize=12)

    # Rechten und oberen Rahmen entfernen
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}{model_name.lower()}_mape_per_year.png")
    plt.close()

    # Plot MAPE pro Monat – mit farb- und alpha-codierten Balken
    plt.figure(figsize=(10, 4))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mape_vals = metrics_month["MAPE"].values
    norm = plt.Normalize(mape_vals.min(), mape_vals.max())
    bar_colors = [cmap(norm(m)) for m in mape_vals]
    bar_alphas = [0.7 + 0.3 * norm(m) for m in mape_vals]

    bars = []
    for idx, (color, alpha) in enumerate(zip(bar_colors, bar_alphas)):
        bar = plt.bar(month_names[metrics_month.index[idx]-1], mape_vals[idx], color=color, alpha=alpha)
        bars.append(bar)

    plt.title(f"MAPE pro Monat – {model_name}", fontweight='bold', fontsize=16, pad=20)
    #plt.xlabel("Monat", fontweight='bold', fontsize=12)
    plt.xlabel("")
    plt.ylabel("MAPE (%)", fontweight='bold', fontsize=12)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}{model_name.lower()}_mape_per_month.png")
    plt.close()

    return metrics_year, metrics_month
