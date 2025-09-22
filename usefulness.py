import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict
import statistics

from tqdm.auto import tqdm

from eggshell import rise  # type: ignore

from toothless.inference import InferResult  # type: ignore


MAX_SAMPLES = 5000
ITER_LIMIT = 6


def check_tuple(sample: InferResult) -> dict[str, dict]:
    left = rise.RecExpr(sample.left)
    right = rise.RecExpr(sample.right)

    tuple_report = {}

    report_str = rise.eqsat_check(left, right, iter_limit=ITER_LIMIT)
    tuple_report["baseline"] = json.loads(report_str)["report"]

    i_limit = min(ITER_LIMIT, tuple_report["baseline"]["iterations"] - 1)

    middle_str = sample.middle.replace("[var]", "?")
    report_str_1, report_str_2 = rise.eqsat_guide_check(left, right, rise.Guide(middle_str), iter_limit=i_limit)
    tuple_report["middle"] = json.loads(report_str_1)["report"]
    if report_str_2 is not None:
        tuple_report["middle_2"] = json.loads(report_str_2)["report"]

    report_str_1, report_str_2 = rise.eqsat_guide_check(left, right, rise.Guide(sample.generated), iter_limit=i_limit)
    tuple_report["generated"] = json.loads(report_str_1)["report"]
    if report_str_2 is not None:
        tuple_report["generated_2"] = json.loads(report_str_2)["report"]

    return tuple_report


def parse_guided_report(
    report: dict,
    name: str,
    stop_reasons: dict[str, DefaultDict[str, int]],
    guide_found: dict[str, int],
    reached_after_guide: dict[str, int],
    guide_reduced_mem: dict[str, int],
    max_node: dict[str, int],
):
    max_node[name] = report[name]["egraph_nodes"]
    if "Other" in report[name]["stop_reason"]:
        sr = "Goal found"
    else:
        sr = str(list(report[name]["stop_reason"].keys())[0])
    stop_reasons[name][sr] += 1

    name_2 = f"{name}_2"
    if name_2 in report:
        guide_found[name] += 1
        max_node[name] = max(max_node[name], report[name_2]["egraph_nodes"])
        if "Goal found" in list(report[name_2]["stop_reason"].keys())[0]:
            stop_reasons[name]["Goal Found"] += 1
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

    if not report_path.exists():
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
        "generated": defaultdict(int),
    }
    guide_found = {"middle": 0, "generated": 0}
    reached_after_guide = {"middle": 0, "generated": 0}
    guide_reduced_mem = {"middle": 0, "generated": 0}

    for report in reports:
        max_node = {}

        parse_guided_report(
            report, "baseline", stop_reasons, guide_found, reached_after_guide, guide_reduced_mem, max_node
        )
        parse_guided_report(
            report, "middle", stop_reasons, guide_found, reached_after_guide, guide_reduced_mem, max_node
        )
        parse_guided_report(
            report, "generated", stop_reasons, guide_found, reached_after_guide, guide_reduced_mem, max_node
        )

        max_nodes.append(max_node)

    stop_reasons_dict = {k: dict(v) for k, v in stop_reasons.items()}

    with open(usefulness_path / "node_counts.json", mode="w", encoding="utf-8") as f:
        json.dump(max_nodes, f)

    with open(usefulness_path / "stop_reasons.json", mode="w", encoding="utf-8") as f:
        json.dump(stop_reasons_dict, f)

    with open(usefulness_path / "guide_found.json", mode="w", encoding="utf-8") as f:
        json.dump(guide_found, f)

    with open(usefulness_path / "reached_after_guide.json", mode="w", encoding="utf-8") as f:
        json.dump(reached_after_guide, f)

    with open(usefulness_path / "guide_reduced_mem.json", mode="w", encoding="utf-8") as f:
        json.dump(guide_reduced_mem, f)

    average_mem = {k: statistics.fmean([d[k] for d in max_nodes]) for k in max_nodes[0]}

    with open(usefulness_path / "average_mem.json", mode="w", encoding="utf-8") as f:
        json.dump(average_mem, f)

    print("---\nSTOP REASONS")
    print(stop_reasons)
    print("---\nGUIDES FOUND")
    print(guide_found)
    print("---\nREACHED AFTER GUIDE")
    print(reached_after_guide)
    print("---\nGUIDE REDUCED MEM")
    print(guide_reduced_mem)
    print("---\nAVERAGE MEM")
    print(average_mem)
