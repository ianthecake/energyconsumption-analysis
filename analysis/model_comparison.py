# analysis/model_comparison.py

import pandas as pd

def model_comparison_table(*results, model_names=None):
    """
    results: beliebig viele Ergebnisdicts, jeweils mit keys 'mad', 'mae', 'rmse', 'mape'
    model_names: optionale Liste von Modellnamen (in gleicher Reihenfolge wie results)
    Gibt einen DataFrame mit Modellvergleich zurück.
    """
    data = []
    for i, res in enumerate(results):
        name = model_names[i] if model_names else f"Modell_{i+1}"
        data.append({
            "Modell": name,
            "MAD": res["mad"],
            "MAE": res["mae"],
            "RMSE": res["rmse"],
            "MAPE (%)": res["mape"],
        })
    return pd.DataFrame(data).set_index("Modell")
