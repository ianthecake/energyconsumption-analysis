# analysis/model_comparison.py

import pandas as pd

def model_comparison_table(*results, model_names=None):

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
