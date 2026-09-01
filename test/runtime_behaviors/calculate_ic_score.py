import argparse
import ast
import json
import math
from collections import deque

from calculate_runtime_behaviors import is_output_correct


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the IC score from saved runtime-behavior results."
    )
    parser.add_argument(
        "--result_path",
        "--results_path",
        dest="result_path",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "codenetmut", "humaneval"),
        default="auto",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    return args


def read_jsonl(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path!r} at line {line_number}."
                ) from exc
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise ValueError("The result file contains no examples.")
    return items


def infer_dataset(path, items):
    if "codenetmut" in path.lower():
        return "codenetmut"
    if "humaneval" in path.lower():
        return "humaneval"
    if any(not isinstance(item.get("output_true"), str) for item in items):
        return "codenetmut"
    return "humaneval"


def get_predicted_trace(item):
    if "agent_trace" in item:
        return item["agent_trace"]
    if "predicted_trace" in item:
        return item["predicted_trace"]
    if "ce_trace" in item:
        return item["ce_trace"]
    raise KeyError("Each result must contain agent_trace or predicted_trace.")


def parse_ground_truth(value):
    if value == "Nil":
        return "Nil"
    if "set()" in value:
        return ast.literal_eval(value.replace("set()", "{}"))
    if value == "[inf]":
        return [math.inf]
    if value == "[(11, 0, inf)]":
        return [(11, 0, math.inf)]
    if value == "[(0, deque([]))]":
        return [(0, deque())]
    if value == (
        "[deque([')']), deque(['(', ')', ')']), "
        "deque(['(', '(', ')', ')', ')']), "
        "deque(['(', '(', '(', ')', ')', ')', '(']), "
        "deque(['(', '(', '(', ')', ')', ')', '(', ')']), "
        "deque(['(', '(', '(', ')', ')', ')', '(', ')', ')'])]"
    ):
        return [
            deque([")"]),
            deque(["(", ")", ")"]),
            deque(["(", "(", ")", ")", ")"]),
            deque(["(", "(", "(", ")", ")", ")", "("]),
            deque(["(", "(", "(", ")", ")", ")", "(", ")"]),
            deque(["(", "(", "(", ")", ")", ")", "(", ")", ")"]),
        ]
    return ast.literal_eval(value)


def check_ccp(agent_trace, task):
    target_line = task["lineno"] - 1
    was_executed = target_line in [entry.get("line") for entry in agent_trace]
    return was_executed == bool(task["exe_or_not"])


def check_psp(agent_trace, task):
    ground_truth = parse_ground_truth(task["value"])
    target_line = task["lineno"] - 1
    target_variable = task["var"]
    predictions = [
        entry["program_states"][target_variable]
        for entry in agent_trace
        if entry.get("line") == target_line
        and target_variable in entry.get("program_states", {})
    ]

    if ground_truth == "Nil":
        return not predictions
    if not predictions:
        return False

    for expected in ground_truth:
        for prediction in predictions:
            if isinstance(expected, float) and isinstance(prediction, (int, float)):
                if math.isclose(expected, prediction):
                    return True
            elif expected == prediction:
                return True
    return False


def find_next(lines, target_line, function_length):
    next_lines = []
    for index in range(len(lines) - 1):
        if lines[index] != target_line:
            continue
        next_line = lines[index + 1]
        if next_line == "completion":
            continue
        if isinstance(next_line, int) and next_line < function_length:
            next_lines.append(next_line)
    return next_lines


def check_epp(agent_trace, task, function_length):
    target_line = task["lineno"] - 1
    expected_next_lines = task["next_line"]
    trace_lines = [entry.get("line") for entry in agent_trace]

    if not task["exe_or_not"]:
        if expected_next_lines != [-1]:
            raise ValueError("A non-executed statement must use next_line=[-1].")
        return target_line not in trace_lines

    if target_line not in trace_lines:
        return False

    predicted_next_lines = find_next(trace_lines, target_line, function_length)
    if expected_next_lines == [-1]:
        return not predicted_next_lines
    return any(line + 1 in expected_next_lines for line in predicted_next_lines)


def ic_contribution(ccp, psp, epp, op):
    if ccp and psp and epp and op:
        return 1.0
    if ccp and psp and epp:
        return 0.5
    if ccp and psp:
        return 0.25
    if ccp:
        return 0.125
    return 0.0


def main():
    args = parse_args()
    items = read_jsonl(args.result_path, args.limit)
    dataset = (
        infer_dataset(args.result_path, items)
        if args.dataset == "auto"
        else args.dataset
    )

    ic_score = 0.0
    ccp_correct = 0
    psp_correct = 0
    epp_correct = 0
    op_correct = 0
    task_count = 0

    for item in items:
        agent_trace = get_predicted_trace(item)
        op = is_output_correct(item, dataset)
        op_correct += int(op)
        function_length = len(item["function"].splitlines())

        for task in item["statements_and_vars"]:
            ccp = check_ccp(agent_trace, task)
            psp = check_psp(agent_trace, task)
            epp = check_epp(agent_trace, task, function_length)
            ccp_correct += int(ccp)
            psp_correct += int(psp)
            epp_correct += int(epp)
            ic_score += ic_contribution(ccp, psp, epp, op)
            task_count += 1

    if task_count == 0:
        raise ValueError("The result file contains no IC evaluation tasks.")

    print(f"Examples: {len(items)}")
    print(f"Tasks: {task_count}")
    print(f"CCP: {ccp_correct / task_count:.4f}")
    print(f"PSP: {psp_correct / task_count:.4f}")
    print(f"EPP: {epp_correct / task_count:.4f}")
    print(f"OP: {op_correct / len(items):.4f}")
    print(f"IC Score: {ic_score / task_count:.4f}")


if __name__ == "__main__":
    main()
