import json
from pathlib import Path

import polars as pl
from eggshell import rise

from toothless.utils.dist_helper import rank0print  # type: ignore

DATASETS = {
    "start": "data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0",
    "goal": "data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/1",
}


def update_cache(rank: int):
    rank0print(rank, "Updating Cache...")
    for name, path in DATASETS.items():
        data = load_df(Path(path), rank)
        data.write_parquet(f"cache/{name}.parquet")

    rank0print(rank, "Cache updated!")


def load_df(data_path: Path, rank: int) -> pl.DataFrame:
    all_dfs = []
    for data_file in sorted(Path(data_path).glob("*.json")):
        df = _load_fragment(data_file, rank)
        all_dfs.append(df)

    rank0print(rank, "All data fragments loading, now concating...")
    all_data = pl.concat(all_dfs, parallel=True)
    rank0print(rank, "Data concatenated")
    rank0print(rank, f"Estimated size: {all_data.estimated_size(unit='gb')} GB")
    return all_data


def _load_fragment(data_file: Path, rank: int) -> pl.DataFrame:
    with open(data_file) as f:
        json_content = json.load(f)

    exprs = rise.PyRecExpr.batch_new([x["sample"] for x in json_content["sample_data"]])
    # features = rise.PyRecExpr.batch_simple_features(exprs, var_names, ignore_unknown)
    # schema = rise.PyRecExpr.simple_feature_names(var_names, ignore_unknown)
    start_term = rise.PyRecExpr(json_content["start_expr"])

    # df = pl.DataFrame(features, schema=schema, orient="row")
    df = pl.DataFrame()

    expl_chain = pl.Series(
        name="explanation_chain",
        values=[[y["rec_expr"] for y in x["explanation"]["explanation_chain"]] for x in json_content["sample_data"]],
    )
    generation = pl.Series(name="generation", values=[i["generation"] for i in json_content["sample_data"]])
    goal_expr = pl.Series(name="goal_expr", values=[str(i) for i in exprs])
    df = df.with_columns([generation, expl_chain, goal_expr])
    df = df.with_columns(
        pl.col("explanation_chain").map_elements(lambda x: x[len(x) // 2], return_dtype=pl.String).alias("middle_expr")
    )
    df = df.with_columns(pl.lit(str(start_term)).alias("start_expr"))

    rank0print(rank, f"Loaded data fragment {data_file}")
    return df
