import json

import pandas as pd

from eggshell import rise  # type: ignore


DATASETS = {
    "start": "data/start_and_goal_multi_size/0.json",
    "goal": "data/start_and_goal_multi_size/1.json",
}


def symbols(blinded):
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


def load_data(data_path, lang_manager):
    with open(data_path) as f:
        dataset = json.load(f)
    print("JSON loaded")

    exprs = rise.PyRecExpr.many_new([x["sample"] for x in dataset["sample_data"]])
    print("Expr PyAst gen done")
    data = pd.DataFrame(lang_manager.many_featurize_simple(exprs, lang_manager))
    print("Featurize done")

    data["generation"] = pd.Series([i["generation"] for i in dataset["sample_data"]])
    data["generation"] = data["generation"].astype(float)
    data["expression"] = pd.Series([str(i) for i in exprs])

    return data


def update_csv(names_blinded: bool = True, ignore_unknown: bool = True):
    lang_manager = rise.PyLanguageManager(
        symbols(names_blinded), ignore_unknown=ignore_unknown
    )
    for name, path in DATASETS.items():
        data = load_data(path, lang_manager)
        data.to_csv(f"csv/{name}.csv", index=False)
