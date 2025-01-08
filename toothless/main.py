from collections import defaultdict
from pathlib import Path
import json
import itertools

import pandas as pd
import sklearn.ensemble
import sklearn.tree
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import sklearn.model_selection

from eggshell import rise

import plotting


IGNORE_UNKNOWN = True


def get_symbol_names(blinded):
    if blinded:
        return [
            "map",
            "mapSeq",
            "iterateStream",
            "split",
            "join",
            "transpose",
            "toMem",
        ]
    else:
        return [
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "x0",
            "x1",
            "x2",
            "x3",
            "map",
            "mapSeq",
            "iterateStream",
            "split",
            "join",
            "transpose",
            "toMem",
        ]


def load_data(data_path, symbol_names):
    with open(data_path) as f:
        dataset = json.load(f)

    featurizer = rise.PyFeaturizer(symbol_names, ignore_unknown=IGNORE_UNKNOWN)

    X = pd.DataFrame(
        rise.many_featurize_simple(
            [x["sample"] for x in dataset["sample_data"]], featurizer
        )
    )

    y = pd.Series([float(x["generation"]) for x in dataset["sample_data"]])

    return featurizer.feature_names_simple(), X, y


def run(dataset_name, data_path, names_blinded):
    prefix = (
        Path("viz")
        / f"{dataset_name}_{"names_blinded" if names_blinded else "with_names"}_term"
    )
    prefix.mkdir(parents=True, exist_ok=True)
    feature_names, X, y = load_data(data_path, get_symbol_names(names_blinded))

    basic_metrics = defaultdict(dict)
    mae_by_label = defaultdict(dict)
    rmse_by_label = defaultdict(dict)

    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    for model in [
        Ridge(alpha=0.5),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
    ]:
        model_type = type(model).__name__

        model.fit(X_train, y_train)

        # get predictions
        y_pred = model.predict(X_test)

        basic_metrics[model_type]["r2"] = r2_score(y_test, y_pred)
        basic_metrics[model_type]["mae"] = mean_absolute_error(y_test, y_pred)
        basic_metrics[model_type]["rmse"] = root_mean_squared_error(y_test, y_pred)

        by_value = defaultdict(list)
        for label, pred in zip(y_test, y_pred):
            by_value[label].append(pred)

        mae_by_label[model_type] = {
            int(label): mean_absolute_error([label] * len(pred), pred)
            for label, pred in by_value.items()
        }
        rmse_by_label[model_type] = {
            int(label): root_mean_squared_error([label] * len(pred), pred)
            for label, pred in by_value.items()
        }

        if isinstance(model, DecisionTreeRegressor):
            plotting.plot_decision_tree(prefix, feature_names, model)

        if isinstance(model, RandomForestRegressor):
            plotting.plot_random_forest(prefix, feature_names, model)

        if isinstance(model, Ridge):
            plotting.plot_ridge_regression(prefix, feature_names, model)

    plotting.plot_metrics(
        prefix,
        basic_metrics,
        mae_by_label,
        rmse_by_label,
    )


if __name__ == "__main__":
    # load and process data
    datasets = [
        ("start", Path("data/start_and_goal_no_new_vars/0.json")),
        ("goal", Path("data/start_and_goal_no_new_vars/1.json")),
    ]

    for (dataset_name, data_path), names_blinded in itertools.product(
        datasets, [True, False]
    ):
        run(dataset_name, data_path, names_blinded)
