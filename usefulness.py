import json
from collections import defaultdict
# from multiprocessing import Pool, Manager


from tqdm.auto import tqdm


from eggshell import rise  # type: ignore


MAX_SAMPLES = 50000
TIME_LIMIT = 10.0
BATCH_SIZE = 1
WORKER_THREADS = 2


def eqsat_check(
    left: rise.RecExpr, right: rise.RecExpr, name: str, guide: rise.RecExpr | None = None
) -> dict[str, dict]:
    report_str, _, guide_used = rise.eqsat_check(
        left,
        right,
        time_limit=TIME_LIMIT,
        guides=[guide] if guide is not None else [],
    )
    report = {name: json.loads(report_str)}
    if guide_used and guide is not None:
        report_str, _, _ = rise.eqsat_check(guide, right, time_limit=TIME_LIMIT)
        report[f"{name}_2"] = json.loads(report_str)
    return report


def check_tuple(sample: dict[str, str]) -> dict[str, dict]:
    left = rise.RecExpr(sample["left"])
    right = rise.RecExpr(sample["right"])

    tuple_report = eqsat_check(left, right, "baseline")
    tuple_report |= eqsat_check(left, right, "middle", rise.RecExpr(sample["middle"]))
    tuple_report |= eqsat_check(left, right, "generated", rise.RecExpr(sample["generated"]))

    return tuple_report


if __name__ == "__main__":
    with open("eval_gen_tripples.json", encoding="utf-8") as f:
        eval_tuples = json.load(f)

    n_samples = min(len(eval_tuples), MAX_SAMPLES)

    reports = []

    for tuple in tqdm(eval_tuples[:n_samples], desc=f"Evaluating {n_samples} samples"):
        reports.append(check_tuple(tuple))

    # with Manager() as manager:
    #     shared_list = manager.list(eval_tuples[:n_samples])

    #     with Pool(processes=WORKER_THREADS) as pool:
    #         for r in tqdm(pool.imap_unordered(check_tuple, shared_list, chunksize=BATCH_SIZE)):
    #             reports.append(r)

    max_nodes = []
    stop_reasons = {"baseline": defaultdict(int), "middle": defaultdict(int), "generated": defaultdict(int)}
    reached_after_guide = {"middle": 0, "generated": 0}
    guide_reduced_mem = {"middle": 0, "generated": 0}

    for report in reports:
        max_node = {}

        max_node["baseline"] = report["baseline"]["report"]["egraph_nodes"]
        stop_reasons["baseline"][str(report["baseline"]["report"]["stop_reason"])] += 1

        max_node["middle"] = report["middle"]["report"]["egraph_nodes"]
        stop_reasons["middle"][str(report["middle"]["report"]["stop_reason"])] += 1

        if "middle_2" in report:
            max_node["middle"] = max(max_node["middle"], report["middle_2"]["report"]["egraph_nodes"])
            reached_after_guide["middle"] += 1
            if max_node["middle"] < max_node["baseline"]:
                guide_reduced_mem["middle"] += 1

        max_node["generated"] = report["generated"]["report"]["egraph_nodes"]
        stop_reasons["generated"][str(report["generated"]["report"]["stop_reason"])] += 1

        if "generated_2" in report:
            max_node["generated"] = max(max_node["generated"], report["generated_2"]["report"]["egraph_nodes"])
            reached_after_guide["generated"] += 1
            if max_node["generated"] < max_node["baseline"]:
                guide_reduced_mem["generated"] += 1

        max_nodes.append(max_node)

    with open("node_counts.json", mode="w", encoding="utf-8") as f:
        json.dump(max_nodes, f)

    with open("stop_reasons.json", mode="w", encoding="utf-8") as f:
        json.dump(stop_reasons, f)

    with open("reached_after_guide.json", mode="w", encoding="utf-8") as f:
        json.dump(reached_after_guide, f)

    with open("guide_reduced_mem.json", mode="w", encoding="utf-8") as f:
        json.dump(guide_reduced_mem, f)

    print("---\nMAX NODE COUNT")
    print(max_nodes)
    print("---\nSTOP REASONS")
    print(stop_reasons)
    print("---\nREACHED AFTER GUIDE")
    print(reached_after_guide)
    print("---\nGUIDE REDUCED MEM")
    print(guide_reduced_mem)

    print("Averages")
    s = [x["baseline"] for x in max_nodes]
    print(f"Baseline: {sum(s) / len(s)}")
    s = [x["middle"] for x in max_nodes]
    print(f"Middle: {sum(s) / len(s)}")
    s = [x["generated"] for x in max_nodes]
    print(f"Generated: {sum(s) / len(s)}")
