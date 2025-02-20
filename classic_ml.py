import sys
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection
import polars as pl


from toothless.utils.consts import VAR_NAMES, IGNORE_UNKNOWN, COLS_TO_DROP
import toothless.utils.loading as loading
import toothless.utils.eval as eval


def main(update_cache: bool = False):
    if update_cache:
        loading.update_cache(VAR_NAMES, IGNORE_UNKNOWN)

    goal_data = pl.read_parquet("cache/goal.parquet")

    print(goal_data.head())
    print("Data loading goal done")

    test_size = 0.2
    random_state = 42
    goal_train, goal_test = sklearn.model_selection.train_test_split(
        goal_data, test_size=test_size, random_state=random_state
    )

    # Add impossible to reach data points
    start_data = pl.read_parquet("cache/start.parquet")
    start_data = start_data.with_columns(
        (pl.when(pl.col("generation") == 3).then(0).otherwise(pl.col("generation"))).alias("generation")
    ).filter(pl.col("generation") == 0)

    start_train, start_test = sklearn.model_selection.train_test_split(
        start_data, test_size=test_size, random_state=random_state
    )

    # concat concrete distances and impossible to reach
    train = pl.concat([goal_train, start_train])
    test = pl.concat([goal_test, start_test])

    print(train["generation"].value_counts())

    # fit them all
    decision_tree = DecisionTreeClassifier().fit(train.drop(COLS_TO_DROP), train["generation"])
    ridge = RidgeClassifier().fit(train.drop(COLS_TO_DROP), train["generation"])
    dummy = DummyClassifier(strategy="most_frequent").fit(train.drop(COLS_TO_DROP), train["generation"])
    print("Models fitted")
    models = {
        "DecisionTree": decision_tree,
        "Dummy": dummy,
        "Ridge": ridge,
    }

    basic_metrics, mae_by_label, rmse_by_label = eval.score_models(models, test, COLS_TO_DROP)
    # path_prefix = Path("viz")
    # path_prefix.mkdir(parents=True, exist_ok=True)
    # plotting.plot_basic_metrics(path_prefix, basic_metrics)
    # plotting.plot_by_label(path_prefix, mae_by_label, "MAE")
    # plotting.plot_by_label(path_prefix, rmse_by_label, "RMSE")
    # plotting.plot_decision_tree("DecisionTree", feature_names, decision_tree)
    # print("Models scored and plotted")

    # Add 3 generations away terms

    start_test = start_test.with_columns(
        pl.Series(name="pred", values=decision_tree.predict(start_test.drop(COLS_TO_DROP)))
    )

    print(start_test["pred"].value_counts())

    # goal = rise.PyRecExpr(
    #     "(lam (>> (>> (>> (>> (>> f1 (>> transpose (>> transpose transpose))) (>> transpose transpose)) transpose) (>> (>> transpose transpose) (>> (>> (>> (>> transpose transpose) transpose) transpose) (>> (>> transpose transpose) transpose)))) transpose) (lam f2 (lam f3 (lam (>> (>> f4 transpose) (>> (>> transpose transpose) transpose)) (lam f5 (lam x3 (app (app iterateStream (var f5)) (app (app map (var f4)) (app (app iterateStream (var f3)) (let x2 (var x3) (app (app map (lam mfu466 (app (var f2) (app (var f1) (var mfu466))))) (var x2))))))))))))"
    # )
    # del start_data
    # del goal_data
    # print("Start eqsat goal check\n\n")
    # eval.evaluate_predictions(start_test, goal)


if __name__ == "__main__":
    update_cache = sys.argv[1] == "--update-cache" if len(sys.argv) >= 2 else False
    main(update_cache)
