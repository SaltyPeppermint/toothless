import json
from pathlib import Path

import polars as pl
from eggshell import rise  # type: ignore


DATASETS = {
    "start": "data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0",
    "goal": "data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/1",
}


def update_cache():
    print("Updating Cache...")
    for name, path in DATASETS.items():
        data = load_df(Path(path))
        data.write_parquet(f"cache/{name}.parquet")

    print("Cache updated!")


def load_df(data_path: Path) -> pl.DataFrame:
    all_dfs = []
    for data_file in sorted(Path(data_path).glob("*.json")):
        df = _load_fragment(data_file)
        all_dfs.append(df)

    print("All data fragments loading, now concating...")
    all_data = pl.concat(all_dfs, parallel=True)
    print("Data concatenated")
    print(f"Estimated size: {all_data.estimated_size(unit='gb')} GB")
    return all_data


def _load_fragment(data_file: Path) -> pl.DataFrame:
    with open(data_file, encoding="utf-8") as f:
        json_content = json.load(f)

    exprs = [rise.RecExpr(x["sample"]) for x in json_content["sample_data"]]
    start_term = rise.RecExpr(json_content["start_expr"])

    rule_chains = []
    expr_chains = []
    for x in json_content["sample_data"]:
        r_c = []
        expl_c = []
        for y in x["explanation"]["explanation_chain"]:
            expl_c.append(y["rec_expr"])
            applied_rules = y["applied_rules"]
            if len(applied_rules) == 0:
                r_c.append("")
            else:
                r_c.append(applied_rules[0])
        rule_chains.append(r_c)
        expr_chains.append(expl_c)

    df = pl.DataFrame()
    df = df.with_columns(
        [
            pl.Series(name="generation", values=[i["generation"] for i in json_content["sample_data"]]),
            pl.Series(name="expr_chain", values=expr_chains),
            pl.Series(name="rules_chain", values=rule_chains),
            pl.Series(name="goal_expr", values=[str(i) for i in exprs]),
        ]
    )
    df = df.with_columns(
        pl.col("expr_chain").map_elements(lambda x: x[len(x) // 2], return_dtype=pl.String).alias("middle_expr")
    )
    df = df.with_columns(pl.lit(str(start_term)).alias("start_expr"))

    print(f"Loaded data fragment {data_file}")
    return df
