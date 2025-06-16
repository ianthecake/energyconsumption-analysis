import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


IMG_DIR = "img/expl/"
os.makedirs(IMG_DIR, exist_ok=True)


# Hilfsfunktion: einheitlicher Plot-Stil

def apply_plot_style(ax, title):
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("Total Energy Consumption\n(in Quadrillion BTU)", fontweight='bold', fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()








def plot_time_series(series):
    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Energy Consumption", color="midnightblue", alpha=0.9, linewidth=1.5)
    ax = plt.gca()
    apply_plot_style(ax, "Time Series of Total Primary Energy Consumption")
    plt.savefig(os.path.join(IMG_DIR, "time_series.png"))
    plt.close()


def plot_moving_average(series, window=12):
    rolling = series.rolling(window=window).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Original", color="grey", alpha=0.5, linewidth=1)
    plt.plot(rolling, label=f"{window}-Month Moving Average", color="tomato", linewidth=1.5)
    ax = plt.gca()
    apply_plot_style(ax, f"{window}-Month Moving Average")
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, "moving_average.png"))
    plt.close()


def plot_monthly_seasonality(series):
    df = series.copy().to_frame("value")
    df["month"] = df.index.month
    df["year"] = df.index.year
    monthly_avg = df.groupby("month")["value"].mean()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=monthly_avg.index, y=monthly_avg.values, color="midnightblue")
    ax = plt.gca()
    ax.set_xticks(range(0, 12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    apply_plot_style(ax, "Average Energy Consumption per Month")
    plt.savefig(os.path.join(IMG_DIR, "monthly_seasonality.png"))
    plt.close()


def plot_boxplot_by_month(series):
    df = series.copy().to_frame("value")
    df["month"] = df.index.month
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="month",
        y="value",
        data=df,
        color="steelblue",
        medianprops={"color": "white", "linewidth": 1}
    )
    ax = plt.gca()
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    apply_plot_style(ax, "Monthly Energy Consumption Distribution (Boxplot)")
    plt.savefig(os.path.join(IMG_DIR, "boxplot_by_month.png"))
    plt.close()


def plot_histogram(series):
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(12, 6))
    plt.hist(series, bins=20, color="midnightblue", edgecolor='white', alpha=0.8)

    # Statistik
    mean = series.mean()
    median = series.median()
    q10 = series.quantile(0.10)
    q90 = series.quantile(0.90)

    ax = plt.gca()

    # Linien einzeichnen
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='green', linestyle='-.', linewidth=2, label=f'Median: {median:.2f}')
    ax.axvline(q10, color='purple', linestyle=':', linewidth=2, label=f'Q10: {q10:.2f}')
    ax.axvline(q90, color='purple', linestyle=':', linewidth=2, label=f'Q90: {q90:.2f}')

    # Titel & Achsen
    ax.set_title("Distribution of US energy consumption", fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel("Energy consumption (in Quadrillion BTU)")
    ax.set_ylabel("Month count", fontweight='bold', fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "histogram.png"))
    plt.close()




def plot_acf_pacf(series, lags=40):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, ax=ax[0], lags=lags, alpha=0.05)
    plot_pacf(series, ax=ax[1], lags=lags, alpha=0.05, method='ywm')
    ax[0].set_title("Autocorrelation (ACF)", fontweight='bold')
    ax[1].set_title("Partial Autocorrelation (PACF)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "acf_pacf.png"))
    plt.close()


def plot_decomposition(series, model='additive', freq=12):
    decomposition = seasonal_decompose(series, model=model, period=freq)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"decomposition_{model}.png"))
    plt.close()


def plot_combined_moving_averages(actual, hw_forecast, arima_forecast, plot_dir="img/"):
    actual_ma = actual.rolling(window=12).mean()
    hw_ma = hw_forecast.rolling(window=12).mean()
    arima_ma = arima_forecast.rolling(window=12).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(actual_ma, label="Moving Average (Actual)", color="midnightblue", linewidth=2)
    plt.plot(hw_ma, label="Moving Average (Holt-Winters)", color="tomato", linestyle="--", linewidth=2)
    plt.plot(arima_ma, label="Moving Average (SARIMA)", color="orange", linestyle="--", linewidth=2)

    plt.title("Vergleich der Moving Averages", fontweight='bold', fontsize=16)
    plt.xlabel("")
    plt.ylabel("12-Monats Moving Average\nTotal Energy Consumption", fontweight='bold', fontsize=12)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2024-12-31"))
    ax.set_xticks(pd.date_range(start="2015-01-01", end="2024-12-31", freq="YS"))
    ax.set_xticklabels([str(y.year) for y in pd.date_range("2015", "2024", freq="YS")], rotation=0)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}combined_forecast_ma_comparison.png")
    plt.close()



def run_all_explorations(series):
    plot_time_series(series)
    plot_moving_average(series)
    plot_monthly_seasonality(series)
    plot_boxplot_by_month(series)
    plot_histogram(series)
    plot_acf_pacf(series)
    plot_decomposition(series)
