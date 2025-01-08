import plotly.express as px
import pandas as pd
import numpy as np


def plot_metrics(
    prefix,
    basic_metrics,
    mae_by_label,
    rmse_by_label,
):
    basic_metrics = pd.DataFrame.from_dict(basic_metrics, orient="index").sort_index(
        axis=1
    )

    mae_by_label = pd.DataFrame.from_dict(mae_by_label, orient="index").sort_index(
        axis=1
    )

    rmse_by_label = pd.DataFrame.from_dict(rmse_by_label, orient="index").sort_index(
        axis=1
    )

    plot_basic_metrics(prefix, basic_metrics)
    plot_by_label(prefix, mae_by_label, "MAE")
    plot_by_label(prefix, rmse_by_label, "RMSE")


def plot_by_label(prefix, metrics, name):
    metrics.index.name = "Model"
    fig = px.bar(metrics, barmode="group", title=f"{name} by Label")
    fig.update_yaxes(title=name)
    fig.update_legends(title="Label")
    fig.write_image(f"{prefix}/{str.lower(name)}_by_label.png", scale=3)


def plot_basic_metrics(prefix, metrics):
    fig = px.bar(metrics, y="r2")
    fig.update_xaxes(title="Model")
    fig.write_image(f"{prefix}/r2.png", scale=3)
    fig = px.bar(metrics, y="mae")
    fig.update_xaxes(title="Model")
    fig.write_image(f"{prefix}/mae.png", scale=3)
    fig = px.bar(metrics, y="rmse")
    fig.update_xaxes(title="Model")
    fig.write_image(f"{prefix}/rmse.png", scale=3)


def plot_ridge_regression(prefix, feature_names, regression):
    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": regression.coef_,
        }
    ).sort_values(by="Coefficient", ascending=False)
    fig = px.bar(
        coef_df,
        x="Feature",
        y="Coefficient",
        title="Ridge Regression Coefficients",
        labels={"Coefficient": "Coefficient", "Feature": "Feature"},
    )
    fig.write_image(f"{prefix}/ridge_regression_coefficients.png", scale=3)


def plot_random_forest(prefix, feature_names, model):
    importances = model.feature_importances_
    std_devs = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
            "StdDev": std_devs,
        }
    ).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Feature Importances in Random Forest",
        error_y="StdDev",
    )
    fig.write_image(f"{prefix}/random_forest_feature_importance.png", scale=3)


def plot_decision_tree(prefix, feature_names, model):
    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Feature Importances in Decision Tree",
    )
    fig.write_image(f"{prefix}/decision_tree_feature_importance.png", scale=3)
