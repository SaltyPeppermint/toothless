from collections import defaultdict
from pathlib import Path
import json
from multiprocessing import Pool
import logging

import sklearn.ensemble
import sklearn.tree
from sklearn import metrics
import wandb
import pandas as pd
import sklearn
import numpy as np

import eggshell
from eggshell import EggshellException
import wandb.sklearn
# from eggshell.rise import PyAst

DATA = Path("data/formatted.json")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# device = (
#     torch.device(0)
#     if torch.cuda.is_available()
#     else torch.device("mps")
#     if torch.backends.mps.is_available()
#     else None
# )

VARIABLE_NAMES = [
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


def pp(sample):
    try:
        sample_expr = eggshell.rise.PyAst(sample, featurizer)
    except EggshellException as err:
        print(err)
        raise ValueError(f"EggshellException! {err}")

    return sample_expr.feature_vec_simple(VARIABLE_NAMES)


def ridge_regression():
    model = sklearn.linear_model.Ridge(alpha=0.5)
    name = "Ridge Regression"
    return (model, name)


def svm():
    model = sklearn.svm.SVR()
    name = "SVM"
    return (model, name)


def random_forest():
    model = sklearn.ensemble.RandomForestRegressor()
    name = "Random Forest"
    return (model, name)


def decision_tree():
    model = sklearn.tree.DecisionTreeRegressor()
    name = "Decision Tree"
    return (model, name)


if __name__ == "__main__":
    # load and process data

    with open(DATA) as f:
        dataset = json.load(f)

    logger.info("DATA LOADED")

    featurizer = eggshell.rise.PyFeaturizer(VARIABLE_NAMES)

    X = pd.DataFrame(
        eggshell.rise.many_featurize_simple(
            [x["sample"] for x in dataset["sample_data"]], featurizer
        )
    )

    y = pd.Series([float(x["generation"]) for x in dataset["sample_data"]])

    feature_names = featurizer.feature_names_simple()

    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(X.head())

    logger.info("PP DONE")

    for model, model_type in [
        ridge_regression(),
        svm(),
        decision_tree(),
        random_forest(),
    ]:
        model.fit(X_train, y_train)

        # get predictions
        y_pred = model.predict(X_test)

        # start a new wandb run and add your model hyperparameters
        wandb.init(
            project="egraph-distance-measure",
            config=model.get_params(),
            tags=["classical", "simple_features", model_type],
        )

        # Add additional configs to wandb
        wandb.config.update(
            {"test_size": test_size, "train_len": len(X_train), "test_len": len(X_test)}
        )

        wandb.run.summary["r2"] = metrics.r2_score(y_test, y_pred)
        wandb.run.summary["mae"] = metrics.mean_absolute_error(y_test, y_pred)
        wandb.run.summary["rmse"] = metrics.root_mean_squared_error(y_test, y_pred)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            importances = model.feature_importances_
            std = np.std(
                [tree.feature_importances_ for tree in model.estimators_], axis=0
            )

        wandb.sklearn.plot_feature_importances(model, feature_names=feature_names)
        wandb.sklearn.plot_outlier_candidates(model, X_train, y_train)
        wandb.sklearn.plot_class_proportions(y_train, y_test)

        by_value = defaultdict(list)
        for label, pred in zip(y_test, y_pred):
            by_value[label].append(pred)

        by_value_mae = [
            [label, metrics.mean_absolute_error([label] * len(pred), pred)]
            for (label, pred) in by_value.items()
        ]
        table = wandb.Table(data=by_value_mae, columns=["label", "mae"])
        wandb.log(
            {
                "mae_by_label": wandb.plot.bar(
                    table, "label", "mae", title="Mean absolute error by label"
                )
            }
        )

        by_value_rmse = [
            [label, metrics.root_mean_squared_error([label] * len(pred), pred)]
            for (label, pred) in by_value.items()
        ]
        table = wandb.Table(data=by_value_rmse, columns=["label", "rmse"])
        wandb.log(
            {
                "rmse_by_label": wandb.plot.bar(
                    table, "label", "rmse", title="Root mean squared error by label"
                )
            }
        )

        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()
