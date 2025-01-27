from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from eggshell import rise  # type: ignore

import plotting


def score_models(models, test: pd.DataFrame):
    # get predictions

    basic_metrics = defaultdict(dict)
    mae_by_label = defaultdict(dict)
    rmse_by_label = defaultdict(dict)

    for model_name, model in models.items():
        y_pred = model.predict(test.drop(columns=["generation", "expression"]))

        basic_metrics[model_name]["r2"] = r2_score(test["generation"], y_pred)
        basic_metrics[model_name]["mae"] = mean_absolute_error(
            test["generation"], y_pred
        )
        basic_metrics[model_name]["rmse"] = root_mean_squared_error(
            test["generation"], y_pred
        )

        by_value = defaultdict(list)
        for label, pred in zip(test["generation"], y_pred):
            by_value[label].append(pred)

        mae_by_label[model_name] = {
            int(label): mean_absolute_error([label] * len(pred), pred)
            for label, pred in by_value.items()
        }
        rmse_by_label[model_name] = {
            int(label): root_mean_squared_error([label] * len(pred), pred)
            for label, pred in by_value.items()
        }

    path_prefix = Path("viz")
    path_prefix.mkdir(parents=True, exist_ok=True)

    plotting.plot_metrics(
        path_prefix,
        basic_metrics,
        mae_by_label,
        rmse_by_label,
    )


def evaluate_predictions(
    three_from_start: pd.DataFrame, goal: rise.PyRecExpr
) -> pd.DataFrame:
    actual_distances = []
    stop_reasons = []
    reports = []
    for _, row in three_from_start[["expression", "pred"]].iterrows():
        expr = row["expression"]
        pred = row["pred"]
        actual_distance, stop_reason, report = rise.eqsat_check(
            rise.PyRecExpr(expr), goal, 8
        )
        actual_distances.append(actual_distance)
        stop_reasons.append(stop_reason)
        reports.append(report)
        print(
            f"Predicted Distance: {pred}\nMeasured Distance: {actual_distance}\nStop Reason: {stop_reason}\n---"
        )
    three_from_start["actual_distance"] = actual_distances
    three_from_start["stop_reason"] = stop_reasons
    three_from_start["report"] = reports

    three_from_start.to_csv("csv/three_from_start.csv")
    print("DONE!")
    return three_from_start
