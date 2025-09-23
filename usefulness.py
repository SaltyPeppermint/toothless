import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict
import statistics

from tqdm.auto import tqdm

from eggshell import rise  # type: ignore

from toothless.inference import InferResult  # type: ignore


MAX_SAMPLES = 5
ITER_LIMIT = 6


def check_tuple(sample: InferResult) -> dict[str, dict]:
    left = rise.RecExpr(sample.left)
    right = rise.RecExpr(sample.right)

    tuple_report = defaultdict(dict)

    report_str, reached = rise.eqsat_check(left, right, iter_limit=ITER_LIMIT)
    tuple_report["baseline"]["report"] = json.loads(report_str)["report"]
    tuple_report["baseline"]["reached"] = reached

    i_limit = min(ITER_LIMIT, tuple_report["baseline"]["report"]["iterations"] - 1)

    middle_str = sample.middle.replace("[var]", "?")
    report_str_1, report_str_2, reached = rise.eqsat_guide_check(
        left, right, rise.Guide(middle_str), iter_limit=i_limit
    )
    tuple_report["middle"]["report"] = json.loads(report_str_1)["report"]
    tuple_report["middle"]["reached"] = reached
    if reached:
        tuple_report["middle"]["report_2"] = json.loads(report_str_2)["report"]

    right_str = sample.right.replace("[var]", "?")
    report_str_1, report_str_2, reached = rise.eqsat_two_guide_check(
        left, rise.Guide(right_str), rise.Guide(middle_str), iter_limit=i_limit
    )
    tuple_report["middle_as_sketch"]["report"] = json.loads(report_str_1)["report"]
    tuple_report["middle_as_sketch"]["reached"] = reached
    if reached is not None:
        tuple_report["middle_as_sketch"]["report_2"] = json.loads(report_str_2)["report"]

    report_str_1, report_str_2, reached = rise.eqsat_guide_check(
        left, right, rise.Guide(sample.generated), iter_limit=i_limit
    )
    tuple_report["generated"]["report"] = json.loads(report_str_1)["report"]
    tuple_report["generated"]["reached"] = reached
    if reached:
        tuple_report["generated"]["report_2"] = json.loads(report_str_2)["report"]

    return tuple_report


def parse_guided_report(
    report: dict,
    name: str,
    stop_reasons: dict[str, DefaultDict[str, int]],
    goal_reached: DefaultDict[str, int],
    guide_reached: DefaultDict[str, int],
    reached_after_guide: DefaultDict[str, int],
    guide_reduced_mem: DefaultDict[str, int],
    max_node: dict[str, int],
):
    max_node[name] = report[name]["report"]["egraph_nodes"]
    if "Other" in report[name]["report"]["stop_reason"]:
        sr = "Goal found"
    else:
        sr = str(list(report[name]["report"]["stop_reason"].keys())[0])
    stop_reasons[name][sr] += 1

    if report[name]["reached"]:
        goal_reached[name] += 1

    if "report_2" in report[name]:
        guide_reached[name] += 1
        max_node[name] = max(max_node[name], report[name]["report"]["egraph_nodes"])
        reached_after_guide[name] += 1
        if max_node[name] < max_node["baseline"]:
            guide_reduced_mem[name] += 1


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
    stop_reasons: dict[str, DefaultDict[str, int]] = {
        "baseline": defaultdict(int),
        "middle": defaultdict(int),
        "middle_as_sketch": defaultdict(int),
        "generated": defaultdict(int),
    }
    guide_reached = defaultdict(int)
    reached_after_guide = defaultdict(int)
    guide_reduced_mem = defaultdict(int)
    goal_reached = defaultdict(int)

    for report in reports:
        max_node = {}

        parse_guided_report(
            report,
            "baseline",
            stop_reasons,
            goal_reached,
            guide_reached,
            reached_after_guide,
            guide_reduced_mem,
            max_node,
        )
        parse_guided_report(
            report,
            "middle",
            stop_reasons,
            goal_reached,
            guide_reached,
            reached_after_guide,
            guide_reduced_mem,
            max_node,
        )
        parse_guided_report(
            report,
            "middle_as_sketch",
            stop_reasons,
            goal_reached,
            guide_reached,
            reached_after_guide,
            guide_reduced_mem,
            max_node,
        )
        parse_guided_report(
            report,
            "generated",
            stop_reasons,
            goal_reached,
            guide_reached,
            reached_after_guide,
            guide_reduced_mem,
            max_node,
        )

        max_nodes.append(max_node)

    stop_reasons_dict = {k: dict(v) for k, v in stop_reasons.items()}

    with open(usefulness_path / "node_counts.json", mode="w", encoding="utf-8") as f:
        json.dump(max_nodes, f)

    with open(usefulness_path / "stop_reasons.json", mode="w", encoding="utf-8") as f:
        json.dump(stop_reasons_dict, f)

    with open(usefulness_path / "goal_reached.json", mode="w", encoding="utf-8") as f:
        json.dump(goal_reached, f)

    with open(usefulness_path / "guide_reached.json", mode="w", encoding="utf-8") as f:
        json.dump(guide_reached, f)

    with open(usefulness_path / "reached_after_guide.json", mode="w", encoding="utf-8") as f:
        json.dump(reached_after_guide, f)

    with open(usefulness_path / "guide_reduced_mem.json", mode="w", encoding="utf-8") as f:
        json.dump(guide_reduced_mem, f)

    average_mem = {k: statistics.fmean([d[k] for d in max_nodes]) for k in max_nodes[0]}

    with open(usefulness_path / "average_mem.json", mode="w", encoding="utf-8") as f:
        json.dump(average_mem, f)

    print("---\nSTOP REASONS")
    print(stop_reasons)
    print("---\nGOALS REACHED")
    print(goal_reached)
    print("---\nGUIDES REACHED")
    print(guide_reached)
    print("---\nREACHED AFTER GUIDE")
    print(reached_after_guide)
    print("---\nGUIDE REDUCED MEM")
    print(guide_reduced_mem)
    print("---\nAVERAGE MEM")
    print(average_mem)
