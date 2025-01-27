import json

import pandas as pd

from eggshell import rise  # type: ignore


DATASETS = {
    "start": "data/start_and_goal_multi_size/0.json",
    "goal": "data/start_and_goal_multi_size/1.json",
}


def load_data(
    data_path: str, var_names: list[str], ignore_unknown: bool
) -> pd.DataFrame:
    with open(data_path) as f:
        dataset = json.load(f)
    print("JSON loaded")

    exprs = rise.PyRecExpr.many_new([x["sample"] for x in dataset["sample_data"]])
    print("Expr PyAst gen done")
    data = pd.DataFrame(
        rise.many_featurize_simple(exprs, var_names, ignore_unknown),
        columns=rise.feature_names_simple(var_names),
    )
    print("Featurize done")

    data["generation"] = pd.Series([i["generation"] for i in dataset["sample_data"]])
    data["generation"] = data["generation"].astype(int)
    data["expression"] = pd.Series([str(i) for i in exprs])

    return data


def update_csv(var_names: list[str], ignore_unknown: bool = True):
    for name, path in DATASETS.items():
        data = load_data(path, var_names, ignore_unknown)
        data.to_csv(f"csv/{name}.csv", index=False)
