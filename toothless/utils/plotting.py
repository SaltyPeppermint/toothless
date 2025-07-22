from pathlib import Path

import numpy as np
import plotly.express as px
import polars as pl


def plot_metrics(prefix, basic_metrics, mae_by_label, rmse_by_label):
    basic_metrics = pl.from_dict(basic_metrics)
    mae_by_label = pl.from_dict(mae_by_label)
    rmse_by_label = pl.from_dict(rmse_by_label)

    _plot_basic_metrics(prefix, basic_metrics)
    _plot_by_label(prefix, mae_by_label, "MAE")
    _plot_by_label(prefix, rmse_by_label, "RMSE")


def _plot_by_label(prefix, metrics, name):
    metrics.index.name = "Model"
    fig = px.bar(metrics, barmode="group", title=f"{name} by Label")
    fig.update_yaxes(title=name)
    fig.update_legends(title="Label")
    fig.write_image(f"{prefix}/{str.lower(name)}_by_label.png", scale=3)


def _plot_basic_metrics(prefix, metrics):
    fig = px.bar(metrics, y="r2")
    fig.update_xaxes(title="Model")
    fig.write_image(f"{prefix}/r2.png", scale=3)
    fig = px.bar(metrics, y="mae")
    fig.update_xaxes(title="Model")
    fig.write_image(f"{prefix}/mae.png", scale=3)
    fig = px.bar(metrics, y="rmse")
    fig.update_xaxes(title="Model")
    fig.write_image(f"{prefix}/rmse.png", scale=3)


def plot_ridge_regression(model_name, feature_names, regression):
    path_prefix = Path("viz") / model_name
    path_prefix.mkdir(parents=True, exist_ok=True)

    coef_df = pl.from_dict(
        {
            "Feature": feature_names,
            "Coefficient": regression.coef_,
        }
    ).sort(by="Coefficient", descending=True)
    fig = px.bar(
        coef_df,
        x="Feature",
        y="Coefficient",
        title="Ridge Regression Coefficients",
        labels={"Coefficient": "Coefficient", "Feature": "Feature"},
    )
    fig.write_image(f"{path_prefix}/ridge_regression_coefficients.png", scale=3)


def plot_random_forest(model_name, feature_names, model):
    path_prefix = Path("viz") / model_name
    path_prefix.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    std_devs = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    importance_df = pl.from_dict({"Feature": feature_names, "Importance": importances, "StdDev": std_devs}).sort(
        by="Importance", descending=True
    )

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Feature Importances in Random Forest",
        error_y="StdDev",
    )
    fig.write_image(f"{path_prefix}/random_forest_feature_importance.png", scale=3)


def plot_decision_tree(model_name, feature_names, model):
    path_prefix = Path("viz") / model_name
    path_prefix.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    importance_df = pl.from_dict({"Feature": feature_names, "Importance": importances}).sort(
        by="Importance", descending=True
    )

    fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importances in Decision Tree")
    fig.write_image(f"{path_prefix}/decision_tree_feature_importance.png", scale=3)
