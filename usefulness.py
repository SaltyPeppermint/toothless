from dataclasses import dataclass
import json
from collections import defaultdict

from dataclass_wizard import JSONWizard
from tqdm.auto import tqdm

from eggshell import rise  # type: ignore


SAMPLES = 100


@dataclass
class CheckResult(JSONWizard):
    report: dict
    generation: int


if __name__ == "__main__":
    with open("eval_gen_tripples.json", encoding="utf-8") as f:
        eval_tuples = json.load(f)

    reports = []
    for tuple in tqdm(eval_tuples[:SAMPLES], desc=f"Evaluating usefulness on {SAMPLES} of eval set"):
        tuple_reports = {}

        left = rise.RecExpr(tuple["left"])
        right = rise.RecExpr(tuple["right"])

        report, generation, _ = rise.eqsat_check(left, right, time_limit=10.0)
        tuple_reports["baseline"] = CheckResult(json.loads(report), generation)

        middle = rise.RecExpr(tuple["middle"])
        report, generation, guide_used = rise.eqsat_check(left, right, time_limit=10.0, guides=[middle])
        tuple_reports["middle"] = CheckResult(json.loads(report), generation)
        if guide_used:
            report, generation, _ = rise.eqsat_check(middle, right, time_limit=10.0)
            tuple_reports["middle_2"] = CheckResult(json.loads(report), generation)

        generated = rise.RecExpr(tuple["generated"])
        report, generation, guide_used = rise.eqsat_check(left, right, time_limit=10.0, guides=[middle])
        tuple_reports["generated"] = CheckResult(json.loads(report), generation)
        if guide_used:
            report, generation, _ = rise.eqsat_check(middle, right, time_limit=10.0)
            tuple_reports["generated_2"] = CheckResult(json.loads(report), generation)

        reports.append(tuple_reports)

    node_counts = {"baseline": 0, "middle": 0, "generated": 0}
    stop_reasons = {"baseline": defaultdict(int), "middle": defaultdict(int), "generated": defaultdict(int)}
    reached_after_guide = {"middle": 0, "generated": 0}

    for report in reports:
        node_counts["baseline"] += report["baseline"].report["egraph_nodes"]
        stop_reasons["baseline"][report["baseline"].report["stop_reason"]] += 1

        node_counts["middle"] += report["middle"].report["egraph_nodes"]
        stop_reasons["middle"][report["middle"].report["stop_reason"]] += 1

        if "middle_2" in report:
            node_counts["middle"] = max(node_counts["middle"], report["middle_2"].report["egraph_nodes"])
            reached_after_guide["middle"] += 1

        node_counts["generated"] += report["generated"].report["egraph_nodes"]
        stop_reasons["generated"][report["generated"].report["stop_reason"]] += 1

        if "generated_2" in report:
            node_counts["generated"] = max(node_counts["generated"], report["generated_2"].report["egraph_nodes"])
            reached_after_guide["generated"] += 1

    with open("node_counts.json", mode="w", encoding="utf-8") as f:
        json.dump(node_counts, f)

    with open("stop_reasons.json", mode="w", encoding="utf-8") as f:
        json.dump(stop_reasons, f)

    with open("reached_after_guide.json", mode="w", encoding="utf-8") as f:
        json.dump(reached_after_guide, f)

    print(node_counts)
    print(stop_reasons)
    print(reached_after_guide)
