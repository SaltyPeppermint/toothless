import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import sklearn.model_selection

from eggshell import rise  # type: ignore

import plotting
import loading
import eval

NAMES_BLINDED = True
IGNORE_UNKNOWN = True
VAR_NAMES = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "x0",
    "x1",
    "x2",
    "x3",
]


def main(update_csv: bool = False):
    if update_csv:
        loading.update_csv(VAR_NAMES, IGNORE_UNKNOWN)

    goal_data = pd.read_csv("csv/goal.csv")
    print("Data loading goal done")
    feature_names = rise.feature_names_simple(VAR_NAMES)

    test_size = 0.2
    random_state = 42
    goal_train, goal_test = sklearn.model_selection.train_test_split(
        goal_data, test_size=test_size, random_state=random_state
    )

    # Add impossible to reach data points
    start_data = pd.read_csv("csv/start.csv")
    start_data = start_data.loc[start_data["generation"] == 3]
    start_data["generation"] = 0

    start_train, start_test = sklearn.model_selection.train_test_split(
        start_data, test_size=test_size, random_state=random_state
    )

    # concat concrete distances and impossible to reach
    train = pd.concat([goal_train, start_train])
    test = pd.concat([goal_test, start_test])

    print(train["generation"].value_counts())

    # fit them all
    decision_tree = DecisionTreeClassifier().fit(
        train.drop(columns=["generation", "expression"]), train["generation"]
    )
    ridge = RidgeClassifier().fit(
        train.drop(columns=["generation", "expression"]), train["generation"]
    )
    dummy = DummyClassifier(strategy="most_frequent").fit(
        train.drop(columns=["generation", "expression"]), train["generation"]
    )
    print("Models fitted")
    models = {
        "DecisionTree": decision_tree,
        "Dummy": dummy,
        "Ridge": ridge,
    }

    eval.score_models(models, test)
    plotting.plot_decision_tree("DecisionTree", feature_names, decision_tree)
    print("Models scored and plotted")

    # three_from_start["pred"] = decision_tree.predict(
    #     three_from_start.drop(columns=["generation", "expression"])
    # )
    # goal = rise.PyRecExpr(
    #     "(lam (>> (>> (>> (>> (>> f1 (>> transpose (>> transpose transpose))) (>> transpose transpose)) transpose) (>> (>> transpose transpose) (>> (>> (>> (>> transpose transpose) transpose) transpose) (>> (>> transpose transpose) transpose)))) transpose) (lam f2 (lam f3 (lam (>> (>> f4 transpose) (>> (>> transpose transpose) transpose)) (lam f5 (lam x3 (app (app iterateStream (var f5)) (app (app map (var f4)) (app (app iterateStream (var f3)) (let x2 (var x3) (app (app map (lam mfu466 (app (var f2) (app (var f1) (var mfu466))))) (var x2))))))))))))"
    # )
    # del start_data
    # del goal_data
    # print("Start eqsat goal check\n\n")
    # eval.evaluate_predictions(three_from_start, goal)
