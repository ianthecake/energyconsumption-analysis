# main.py
# https://www.kaggle.com/datasets/justinrwong/average-monthly-temperature-by-us-state?resource=download
import pandas as pd
from models.holt_winters import holt_winters_forecast
from models.arima import arima_sarima_forecast
from analysis.event_impact_analysis import event_error_analysis
from analysis.error_analysis import error_over_time
from analysis.model_comparison import model_comparison_table
from analysis.error_heatmap import mape_heatmap
from analysis import exploration
from analysis import kausal_analyse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



DATA_PATH = "data/energyconsumption.csv"
TEMP_PATH = "data/usa_monthly_temperature_1950-2022.csv"
COLUMN = "Total Primary Energy Consumption"
TRAIN_END_YEAR = 2014
TEST_END_YEAR = 2024


def load_series(path, col):
    df = pd.read_csv(path, sep=";", decimal=",")
    df['Month'] = pd.to_datetime(df['Month'], format='%Y %B')
    df = df.sort_values('Month')
    df.set_index('Month', inplace=True)
    series = df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
    series = pd.to_numeric(series, errors='coerce').interpolate().dropna()
    series = series.sort_index()
    series = series.asfreq('MS')
    return series

def load_temperature_series(path):
    df = pd.read_csv(path)
    df = df[df['year'].between(1973, 2022)]
    df_grouped = df.groupby(['year', 'month'])['average_temp'].mean().reset_index()
    df_grouped['Month'] = pd.to_datetime(df_grouped['year'].astype(str) + '-' + df_grouped['month'].astype(str))
    df_grouped = df_grouped.sort_values('Month')
    df_grouped.set_index('Month', inplace=True)
    series = df_grouped['average_temp'].asfreq('MS')
    return series


if __name__ == "__main__":
    series = load_series(DATA_PATH, COLUMN)
    temp_series = load_temperature_series(TEMP_PATH)

    #hw_results = holt_winters_forecast(series, TRAIN_END_YEAR, TEST_END_YEAR)
    #sarima_results = arima_sarima_forecast(series, TRAIN_END_YEAR, TEST_END_YEAR)

    #exploration.plot_combined_moving_averages(
    #    actual=hw_results["actual"],
    #    hw_forecast=hw_results["forecast"],
    #    arima_forecast=sarima_results["forecast"]
    #)

    #print("Mean total energy consumption:", series.mean())

    #sub_series = series.loc["2000-01":"2010-12"]
    #sub_tempseries = temp_series.loc["2000-01":"2010-12"]

    #exploration.plot_histogram(series)
    #kausal_analyse.plot_energy_vs_temperature(series, temp_series)
    kausal_analyse.plot_averages_comparison(series, temp_series)

    #exploration.run_all_explorations(series)

'''
    # Holt-Winters Modell
    hw_results = holt_winters_forecast(series, TRAIN_END_YEAR, TEST_END_YEAR)
    print("Holt-Winters Results:")
    print(f"  MAD:  {hw_results['mad']:.2f}")
    print(f"  MAE:  {hw_results['mae']:.2f}")
    print(f"  RMSE: {hw_results['rmse']:.2f}")
    print(f"  MAPE: {hw_results['mape']:.2f}%")

    # SARIMA Modell
    sarima_results = arima_sarima_forecast(series, TRAIN_END_YEAR, TEST_END_YEAR)
    print("\nSARIMA Results:")
    print(f"  MAD:  {sarima_results['mad']:.2f}")
    print(f"  MAE:  {sarima_results['mae']:.2f}")
    print(f"  RMSE: {sarima_results['rmse']:.2f}")
    print(f"  MAPE: {sarima_results['mape']:.2f}%")

    # Events definieren (anpassbar)
    events = {
        "Covid (2020-2021)": ("2020-01-01", "2021-12-31"),
        "Russland-Krieg (2022-)": ("2022-01-01", "2024-12-31"),
        # Du kannst beliebige weitere Events ergänzen
    }

    print("\nEvent-basierte Fehleranalyse Holt-Winters:")
    df_hw_events = event_error_analysis(hw_results["actual"], hw_results["forecast"], events, model_name="HoltWinters")
    print(df_hw_events)

    print("\nEvent-basierte Fehleranalyse SARIMA:")
    df_sarima_events = event_error_analysis(sarima_results["actual"], sarima_results["forecast"], events,
                                            model_name="SARIMA")
    print(df_sarima_events)



    # Nach den Modellberechnungen:
    metrics_year_hw, metrics_month_hw = error_over_time(hw_results["actual"], hw_results["forecast"],
                                                        model_name="HoltWinters")
    metrics_year_sarima, metrics_month_sarima = error_over_time(sarima_results["actual"], sarima_results["forecast"],
                                                                model_name="SARIMA")

    print("\nHolt-Winters Fehler pro Jahr:\n", metrics_year_hw)
    print("\nSARIMA Fehler pro Jahr:\n", metrics_year_sarima)

    df_compare = model_comparison_table(
        hw_results, sarima_results,
        model_names=["Holt-Winters", "SARIMA"]
    )
    print("\nModellvergleich (Gesamt):\n", df_compare)
    df_compare.to_csv("img/model_comparison.csv")

    df_hw_heatmap = mape_heatmap(hw_results["actual"], hw_results["forecast"], model_name="HoltWinters")
    df_sarima_heatmap = mape_heatmap(sarima_results["actual"], sarima_results["forecast"], model_name="SARIMA")
    '''




