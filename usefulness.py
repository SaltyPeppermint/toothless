import json
from pathlib import Path
from collections import defaultdict
import statistics

from tqdm.auto import tqdm

from eggshell import rise  # type: ignore

from toothless.inference import InferResult  # type: ignore


MAX_SAMPLES = 50000
TIME_LIMIT = 10.0
BATCH_SIZE = 1
WORKER_THREADS = 2

MODEL_PATH = Path("models/25-08-05_13:25:52")
EVAL_PATH = MODEL_PATH / "eval"
USEFULNESS_PATH = EVAL_PATH / "usefulness"

RUN_EQSAT = False


def eqsat_check(left: rise.RecExpr, right: rise.RecExpr, name: str) -> dict[str, dict]:
    report_str, _ = rise.eqsat_guide_check(left, right, time_limit=TIME_LIMIT, guides=[])
    return {name: json.loads(report_str)}


def eqsat_guide_check(left: rise.RecExpr, right: rise.RecExpr, guide: rise.RecExpr, name: str) -> dict[str, dict]:
    report_str, guide_used = rise.eqsat_guide_check(
        left, right, time_limit=TIME_LIMIT, guides=[guide] if guide is not None else []
    )
    report = {name: json.loads(report_str)}
    if guide_used:
        report_str, _ = rise.eqsat_guide_check(guide, right, time_limit=TIME_LIMIT)
        report[f"{name}_2"] = json.loads(report_str)
    return report


def eqsat_rules_check(left: rise.RecExpr, right: rise.RecExpr, ordered_rules: list[str], name: str) -> dict[str, dict]:
    ordered_rules = [r for r in ordered_rules if r]
    report_str = rise.eqsat_ordered_rules_check(left, right, time_limit=TIME_LIMIT, ordered_rules=ordered_rules)
    return {name: json.loads(report_str)}


def check_tuple(sample: InferResult) -> dict[str, dict]:
    left = rise.RecExpr(sample.left)
    right = rise.RecExpr(sample.right)

    tuple_report = eqsat_check(left, right, "baseline")
    tuple_report |= eqsat_rules_check(left, right, sample.rules_chain, "rules_chain")
    tuple_report |= eqsat_guide_check(left, right, rise.RecExpr(sample.middle), "middle")
    tuple_report |= eqsat_guide_check(left, right, rise.RecExpr(sample.generated), "generated")

    return tuple_report


if __name__ == "__main__":
    with open(EVAL_PATH / "train_gen_triples.json", encoding="utf-8") as f:
        eval_tuples = InferResult.from_list(json.load(f))

    n_samples = min(len(eval_tuples), MAX_SAMPLES)

    USEFULNESS_PATH.mkdir(parents=True, exist_ok=True)

    if RUN_EQSAT:
        reports = []
        for tuple in tqdm(eval_tuples[:n_samples], desc=f"Evaluating {n_samples} samples"):
            reports.append(check_tuple(tuple))
        with open(USEFULNESS_PATH / "reports_usefulness.json", mode="w", encoding="utf-8") as f:
            json.dump(reports, f)
    else:
        with open(USEFULNESS_PATH / "reports_usefulness.json", encoding="utf-8") as f:
            reports = json.load(f)

    max_nodes = []
    stop_reasons = {
        "baseline": defaultdict(int),
        "middle": defaultdict(int),
        "generated": defaultdict(int),
        "rules_chain": defaultdict(int),
    }
    reached_after_guide = {"middle": 0, "generated": 0}
    guide_reduced_mem = {"middle": 0, "generated": 0, "rules_chain": 0}

    for report in reports:
        max_node = {}

        max_node["baseline"] = report["baseline"]["report"]["egraph_nodes"]
        sr = list(report["baseline"]["report"]["stop_reason"])[0]
        stop_reasons["baseline"][sr] += 1

        max_node["middle"] = report["middle"]["report"]["egraph_nodes"]
        sr = list(report["middle"]["report"]["stop_reason"])[0]
        stop_reasons["middle"][sr] += 1

        if "middle_2" in report:
            max_node["middle"] = max(max_node["middle"], report["middle_2"]["report"]["egraph_nodes"])
            reached_after_guide["middle"] += 1
            if max_node["middle"] < max_node["baseline"]:
                guide_reduced_mem["middle"] += 1

        max_node["rules_chain"] = report["rules_chain"]["report"]["egraph_nodes"]
        sr = list(report["rules_chain"]["report"]["stop_reason"])[0]
        stop_reasons["rules_chain"][sr] += 1

        if max_node["rules_chain"] < max_node["baseline"]:
            guide_reduced_mem["rules_chain"] += 1

        max_node["generated"] = report["generated"]["report"]["egraph_nodes"]
        sr = list(report["generated"]["report"]["stop_reason"])[0]
        stop_reasons["generated"][sr] += 1

        if "generated_2" in report:
            max_node["generated"] = max(max_node["generated"], report["generated_2"]["report"]["egraph_nodes"])
            reached_after_guide["generated"] += 1
            if max_node["generated"] < max_node["baseline"]:
                guide_reduced_mem["generated"] += 1

        max_nodes.append(max_node)

    stop_reasons = {k: dict(v) for k, v in stop_reasons.items()}

    with open(USEFULNESS_PATH / "node_counts.json", mode="w", encoding="utf-8") as f:
        json.dump(max_nodes, f)

    with open(USEFULNESS_PATH / "stop_reasons.json", mode="w", encoding="utf-8") as f:
        json.dump(stop_reasons, f)

    with open(USEFULNESS_PATH / "reached_after_guide.json", mode="w", encoding="utf-8") as f:
        json.dump(reached_after_guide, f)

    with open(USEFULNESS_PATH / "guide_reduced_mem.json", mode="w", encoding="utf-8") as f:
        json.dump(guide_reduced_mem, f)

    average_mem = {k: statistics.fmean([d[k] for d in max_nodes]) for k in max_nodes[0]}

    with open(USEFULNESS_PATH / "average_mem.json", mode="w", encoding="utf-8") as f:
        json.dump(average_mem, f)

    # print("---\nMAX NODE COUNT")
    # print(max_nodes)
    print("---\nSTOP REASONS")
    print(stop_reasons)
    print("---\nREACHED AFTER GUIDE")
    print(reached_after_guide)
    print("---\nGUIDE REDUCED MEM")
    print(guide_reduced_mem)
    print("---\nAVERAGE MEM")
    print(average_mem)
