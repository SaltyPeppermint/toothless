import json
import sys
from pathlib import Path
import pprint
from collections import defaultdict

# from typing import DefaultDict
import statistics
from typing import DefaultDict

from tqdm.auto import tqdm

from eggshell import rise  # type: ignore

from toothless.inference import InferResult  # type: ignore


MAX_SAMPLES = 5000
ITER_LIMIT = 6


def check_tuple(sample: InferResult) -> dict[str, dict]:
    start = rise.RecExpr(sample.left)
    target = rise.RecExpr(sample.right)

    print(f"Start:\n{sample.left}")

    tuple_report = defaultdict(dict)

    report_str, goal_reached = rise.eqsat_check(start, target, iter_limit=ITER_LIMIT)
    tuple_report["baseline"]["report"] = json.loads(report_str)["report"]
    tuple_report["baseline"]["goal_reached"] = goal_reached

    i_limit = min(ITER_LIMIT, tuple_report["baseline"]["report"]["iterations"] - 1)

    # GT
    gt_guide_sketch = rise.Guide(sample.middle.replace("[var]", "?"))
    report_str_1, guide_stuff, goal_reached = rise.eqsat_guide_check(start, gt_guide_sketch, target, iter_limit=i_limit)
    tuple_report["gt"]["report"] = json.loads(report_str_1)["report"]
    tuple_report["gt"]["goal_reached"] = goal_reached
    if guide_stuff:
        report_str_2, gt_extracted_guide = guide_stuff
        tuple_report["gt"]["report_2"] = json.loads(report_str_2)["report"]
        print(f"GT Guide Sketchyfied:\n{gt_guide_sketch}")
        print(f"Extracted Guide for GT:\n{gt_extracted_guide}")

    # GENERATED
    generated_guide_sketch = rise.Guide(sample.generated.replace("[var]", "?"))
    report_str_1, guide_stuff, goal_reached = rise.eqsat_guide_check(
        start, generated_guide_sketch, target, iter_limit=i_limit
    )
    tuple_report["generated"]["report"] = json.loads(report_str_1)["report"]
    tuple_report["generated"]["goal_reached"] = goal_reached
    if guide_stuff:
        report_str_2, generated_extracted_guide = guide_stuff
        tuple_report["generated"]["report_2"] = json.loads(report_str_2)["report"]
        print(f"Generated Guide Sketchyfied:\n{generated_guide_sketch}")
        print(f"Extracted Guide for Generated:\n{generated_extracted_guide}")

    # SKETCHIFIED TARGET
    target_sketch = rise.Guide(sample.right.replace("[var]", "?"))
    print(f"Target Sketchyfied:\n{target_sketch}")
    # GT WITH GOAL AS SKETCH
    report_str_1, guide_stuff, gt_extracted_goal = rise.eqsat_two_guide_check(
        start, gt_guide_sketch, target_sketch, iter_limit=i_limit
    )
    tuple_report["gt_goal_sketch"]["report"] = json.loads(report_str_1)["report"]
    if guide_stuff:
        report_str_2, _ = guide_stuff
        tuple_report["gt_goal_sketch"]["report_2"] = json.loads(report_str_2)["report"]
    if gt_extracted_goal:
        tuple_report["gt_goal_sketch"]["goal_reached"] = True
        print(f"Extracted target for GT with SKETCHGOAL:\n{gt_extracted_goal}")
    else:
        tuple_report["gt_goal_sketch"]["goal_reached"] = False

    # GT WITH GOAL AS SKETCH
    report_str_1, guide_stuff, generated_extracted_target = rise.eqsat_two_guide_check(
        start, generated_guide_sketch, target_sketch, iter_limit=i_limit
    )
    tuple_report["generated_goal_sketch"]["report"] = json.loads(report_str_1)["report"]
    if guide_stuff:
        report_str_2, _ = guide_stuff
        tuple_report["generated_goal_sketch"]["report_2"] = json.loads(report_str_2)["report"]
    if generated_extracted_target:
        tuple_report["generated_goal_sketch"]["goal_reached"] = True
        print(f"Extracted target for Generated with SKETCHGOAL:\n{generated_extracted_target}")
    else:
        tuple_report["generated_goal_sketch"]["goal_reached"] = False

    print("---")

    return tuple_report


def parse_guided_report(report: dict, name: str, summary: dict, max_node: DefaultDict[str, int]):
    max_node[name] = report[name]["report"]["egraph_nodes"]
    if "Other" in report[name]["report"]["stop_reason"]:
        sr = "Goal found"
    else:
        sr = str(list(report[name]["report"]["stop_reason"].keys())[0])
    summary["stop_reasons"][name][sr] += 1

    if report[name]["goal_reached"]:
        summary["goal_reached"][name] += 1

    if "report_2" in report[name]:
        summary["guide_reached"][name] += 1
        max_node[name] = max(max_node[name], report[name]["report"]["egraph_nodes"])
        if max_node[name] < max_node["baseline"]:
            summary["guide_reduced_mem"][name] += 1


if __name__ == "__main__":
    folder = Path("models") / str(sys.argv[1])
    eval_path = folder / "eval"
    train_or_eval = str(sys.argv[2])
    usefulness_path = eval_path / f"usefulness_{train_or_eval}"

    with open(eval_path / f"{str(train_or_eval)}_gen_triples.json", encoding="utf-8") as f:
        eval_tuples = InferResult.from_list(json.load(f))

    n_samples = min(len(eval_tuples), MAX_SAMPLES)

    usefulness_path.mkdir(parents=True, exist_ok=True)

    report_path = usefulness_path / "reports.json"

    if not report_path.exists() or str(sys.argv[3]) == "--force":
        reports = []
        for tuple in tqdm(eval_tuples[:n_samples], desc=f"Evaluating {n_samples} samples"):
            reports.append(check_tuple(tuple))
        with open(report_path, mode="w", encoding="utf-8") as f:
            json.dump(reports, f)
    else:
        with open(report_path, encoding="utf-8") as f:
            reports = json.load(f)

    max_nodes = []

    guide_reached = defaultdict(int)
    guide_reduced_mem = defaultdict(int)
    goal_reached = defaultdict(int)
    summary = {
        "stop_reasons": {
            "baseline": defaultdict(int),
            "gt": defaultdict(int),
            "gt_goal_sketch": defaultdict(int),
            "generated": defaultdict(int),
            "generated_goal_sketch": defaultdict(int),
        },
        "guide_reached": defaultdict(int),
        "guide_reduced_mem": defaultdict(int),
        "goal_reached": defaultdict(int),
    }

    for report in reports:
        max_node = defaultdict(int)

        for name in summary["stop_reasons"].keys():
            parse_guided_report(report, name, summary, max_node)

        max_nodes.append(max_node)

    summary = {k: dict(v) for k, v in summary.items()}
    summary["stop_reasons"] = {k: dict(v) for k, v in summary["stop_reasons"].items()}

    with open(usefulness_path / "summary.json", mode="w", encoding="utf-8") as f:
        json.dump(summary, f)

    average_mem = {k: statistics.fmean([d[k] for d in max_nodes]) for k in max_nodes[0]}

    # with open(usefulness_path / "average_mem.json", mode="w", encoding="utf-8") as f:
    #     json.dump(average_mem, f)

    print("---\nSummary:")
    pprint.pp(summary)
    print("---\nAverage Memory:")
    print(average_mem)
