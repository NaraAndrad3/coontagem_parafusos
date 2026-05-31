import numpy as np, pandas as pd


def evaluate_counts(results: list[dict], ground_truth: dict) -> tuple[pd.DataFrame, dict]:
    """
    Avalia a contagem prevista pelo sistema em relação ao valor real.

    Métricas:
    - erro absoluto por imagem
    - erro percentual por imagem
    - accuracy exata
    - MAE
    - RMSE
    - MAPE
    """

    rows = []

    for res in results:
        image_name = res["image"]
        predicted = res["count"]
        real = ground_truth.get(image_name)

        if real is None:
            continue

        abs_error = abs(real - predicted)

        perc_error = (abs_error / real) * 100 if real != 0 else 0

        rows.append({
            "imagem": image_name,
            "real": real,
            "predito": predicted,
            "erro_absoluto": abs_error,
            "erro_percentual": perc_error,
            "acerto_exato": real == predicted
        })

    df = pd.DataFrame(rows)

    if df.empty:
        metrics = {
            "accuracy_exata": 0,
            "mae": None,
            "rmse": None,
            "mape": None
        }

        return df, metrics

    errors = df["erro_absoluto"].values

    accuracy = df["acerto_exato"].mean()
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = df["erro_percentual"].mean()

    metrics = {
        "accuracy_exata": accuracy,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }

    return df, metrics