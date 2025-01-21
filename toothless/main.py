from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import sklearn.model_selection

from eggshell import rise  # type: ignore

import plotting
import loading

NAMES_BLINDED = True
IGNORE_UNKNOWN = True


def score_models(models, test):
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


def main(update_csv: bool = False):
    if update_csv:
        loading.update_csv(NAMES_BLINDED, IGNORE_UNKNOWN)

    lang_manager = rise.PyLanguageManager(
        loading.symbols(NAMES_BLINDED), ignore_unknown=IGNORE_UNKNOWN
    )
    goal_data = pd.read_csv("csv/goal.csv")
    print("Data loading goal done")
    feature_names = lang_manager.feature_names_simple()

    test_size = 0.2
    random_state = 42
    goal_train, goal_test = sklearn.model_selection.train_test_split(
        goal_data, test_size=test_size, random_state=random_state
    )

    decision_tree = DecisionTreeRegressor().fit(
        goal_data.drop(columns=["generation", "expression"]), goal_data["generation"]
    )
    ridge = Ridge().fit(
        goal_data.drop(columns=["generation", "expression"]), goal_data["generation"]
    )
    dummy_regressor = DummyRegressor(strategy="median").fit(
        goal_data.drop(columns=["generation", "expression"]), goal_data["generation"]
    )
    print("Models fitted")
    models = {
        "DecisionTreeRegressor": decision_tree,
        "DummyRegressor": dummy_regressor,
        "Ridge": ridge,
    }

    score_models(models, goal_test)
    plotting.plot_decision_tree("DecisionTreeRegressor", feature_names, decision_tree)
    print("Models scored and plotted")

    start_data = pd.read_csv("csv/start.csv")
    start_train, start_test = sklearn.model_selection.train_test_split(
        goal_data, test_size=test_size, random_state=random_state
    )

    three_from_start = start_test.loc[start_test["generation"] == 3.0].head(100)
    three_from_start["pred"] = decision_tree.predict(
        three_from_start.drop(columns=["generation", "expression"])
    )
    goal = rise.PyRecExpr(
        "(lam (>> (>> (>> (>> (>> f1 (>> transpose (>> transpose transpose))) (>> transpose transpose)) transpose) (>> (>> transpose transpose) (>> (>> (>> (>> transpose transpose) transpose) transpose) (>> (>> transpose transpose) transpose)))) transpose) (lam f2 (lam f3 (lam (>> (>> f4 transpose) (>> (>> transpose transpose) transpose)) (lam f5 (lam x3 (app (app iterateStream (var f5)) (app (app map (var f4)) (app (app iterateStream (var f3)) (let x2 (var x3) (app (app map (lam mfu466 (app (var f2) (app (var f1) (var mfu466))))) (var x2))))))))))))"
    )
    del start_data
    del goal_data
    print("Start eqsat goal check\n\n")
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
