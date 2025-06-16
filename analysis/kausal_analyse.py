import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis.exploration import apply_plot_style

IMG_DIR = "img/kausalitaet/"
os.makedirs(IMG_DIR, exist_ok=True)


def plot_energy_vs_temperature(energy_series, temp_series):

    #energy_series = energy_series.loc["2000-01":"2010-12"]
    #temp_series = temp_series.loc["2000-01":"2010-12"]

    temp_series["average_temp"] = (5/9) * (temp_series["average_temp"] - 32)

    energy_norm = (energy_series - energy_series.min()) / (energy_series.max() - energy_series.min())
    temp_norm = (temp_series - temp_series.min()) / (temp_series.max() - temp_series.min())

    plt.figure(figsize=(12, 6))
    plt.plot(energy_norm, label="Energy Consumption (normalized)", color="midnightblue", alpha=0.9, linewidth=2.5)
    plt.plot(temp_norm, label="Avg Temperature (normalized)", color="tomato", alpha=0.8, linewidth=2.5)
    ax = plt.gca()
    apply_plot_style(ax, "Energy Consumption vs. Temperature (normalized)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, "energy_vs_temperature.png"))
    plt.close()


def plot_averages_comparison(energy_series, temp_series, window=12):
    df = pd.DataFrame({
        'energy': energy_series,
        'temp': temp_series
    })
    df['month'] = df.index.month

    monthly_energy_avg = df.groupby('month')['energy'].mean().sort_index()
    monthly_temp_avg = df.groupby('month')['temp'].mean().sort_index()

    energy_avg_norm = (monthly_energy_avg - monthly_energy_avg.min()) / (monthly_energy_avg.max() - monthly_energy_avg.min())
    temp_avg_norm = (monthly_temp_avg - monthly_temp_avg.min()) / (monthly_temp_avg.max() - monthly_temp_avg.min())

    plt.figure(figsize=(12, 6))
    plt.plot(energy_avg_norm, label='Energy (normalized)', color='midnightblue', linewidth=4)
    plt.plot(temp_avg_norm, label='Temperature (normalized)', color='tomato', linewidth=4)
    plt.xticks(ticks=np.arange(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.xlabel("")
    plt.ylabel("Normalized Value (temperature and energy consumption)", fontsize=10, fontweight="bold")
    plt.title(f"monthly average temperature vs. energy consumption", fontsize=16, fontweight="bold", pad=20)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(IMG_DIR, "monthly_temp_vs_energy_consumption.png"))
    plt.close()