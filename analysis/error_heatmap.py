import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.utils_metrics import mape

def mape_heatmap(actual, forecast, plot_dir="img/", model_name="Modell"):

    df = pd.DataFrame({"actual": actual, "forecast": forecast})
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(
        index="year", columns="month",
        values=["actual", "forecast"],
        aggfunc="mean"
    )

    mape_matrix = np.full((len(pivot.index), 12), np.nan)
    for i, year in enumerate(pivot.index):
        for j in range(12):
            a = df[(df["year"] == year) & (df["month"] == j+1)]["actual"]
            f = df[(df["year"] == year) & (df["month"] == j+1)]["forecast"]
            if len(a) > 0 and len(f) > 0:
                mape_matrix[i, j] = mape(a, f)

    plt.figure(figsize=(12, 6))
    plt.imshow(mape_matrix, aspect="auto", cmap="Reds", interpolation="none")
    plt.colorbar(label="MAPE (%)")
    plt.xticks(np.arange(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.yticks(np.arange(len(pivot.index)), [str(y) for y in pivot.index])
    plt.title(f"MAPE-Heatmap pro Jahr & Monat ({model_name})", fontweight='bold', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}{model_name.lower()}_mape_heatmap.png")
    plt.close()

    df_heatmap = pd.DataFrame(mape_matrix, index=pivot.index, columns=range(1, 13))
    return df_heatmap
