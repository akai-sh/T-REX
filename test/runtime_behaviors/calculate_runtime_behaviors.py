import argparse
import ast
import json
import re

from sklearn.metrics import f1_score, precision_score, recall_score

from utils import execute_code_with_trace


def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def get_code(item):
    code = item.get("code", item.get("program"))
    if code is None:
        raise KeyError("Each result must contain 'code' or 'program'.")
    return code


def get_top_level_definition_lines(code):
    tree = ast.parse(code)
    return {
        node.lineno - 1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def get_coverage_trace_lines(item, code):
    if "actual_trace" in item:
        return [
            trace_entry["line"]
            for trace_entry in item["actual_trace"]
            if isinstance(trace_entry.get("line"), int)
        ]

    raw_trace, _, _, _ = execute_code_with_trace(code)
    top_level_definition_lines = get_top_level_definition_lines(code)
    actual_trace_lines = []
    for trace_entry in raw_trace:
        match = re.match(r"<line>\s*<(\d+)>", trace_entry)
        if match:
            line_number = int(match.group(1))
            if line_number not in top_level_definition_lines:
                actual_trace_lines.append(line_number)
    return actual_trace_lines


def get_predicted_trace(item):
    if "predicted_trace" in item:
        return item["predicted_trace"]
    return item.get("agent_trace", [])[1:]


def get_predicted_trace_lines(predicted_trace):
    return [
        trace_entry["line"]
        for trace_entry in predicted_trace
        if isinstance(trace_entry.get("line"), int)
    ]


def coverage_scores(code, predicted_trace_lines, actual_trace_lines):
    line_range = range(len(code.split("\n")))
    predicted_coverage = [
        1 if line_number in predicted_trace_lines else 0
        for line_number in line_range
    ]
    actual_coverage = [
        1 if line_number in actual_trace_lines else 0
        for line_number in line_range
    ]

    # Preserve the P/R argument order used by the paper's evaluation code.
    precision = precision_score(
        predicted_coverage, actual_coverage, zero_division=0
    )
    recall = recall_score(
        predicted_coverage, actual_coverage, zero_division=0
    )
    f1 = f1_score(predicted_coverage, actual_coverage, zero_division=0)
    exact_match = float(predicted_coverage == actual_coverage)
    return precision, recall, f1, exact_match


def parse_literal(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def normalize_output(value):
    if not isinstance(value, str):
        return value

    value = value.strip()
    if value.startswith("print(") and value.endswith(")"):
        value = value[6:-1]

    parsed_value = parse_literal(value)
    if isinstance(parsed_value, tuple):
        return list(parsed_value)
    return parsed_value


def is_output_correct(item, dataset):
    if item.get("finished") is False or not isinstance(item.get("output"), dict):
        return False

    predicted_output = item["output"].get("cur_line")
    expected_output = item.get("output_true")
    if expected_output == "(3 + 5 + 7, 3 * 5 * 7)":
        expected_output = "(15,105)"

    if predicted_output == expected_output:
        return True

    if dataset == "codenetmut":
        predicted_output = normalize_output(predicted_output)
        expected_output = normalize_output(expected_output)
        return predicted_output == expected_output

    return predicted_output == parse_literal(expected_output)


def format_percentage(value):
    return f"{value * 100:.1f}"


def main():
    args = parse_args()
    dataset = args.dataset
    if dataset == "auto":
        dataset = "codenetmut" if "codenetmut" in args.result_path.lower() else "humaneval"

    with open(args.result_path, "r", encoding="utf-8") as result_file:
        items = [json.loads(line) for line in result_file if line.strip()]

    if not items:
        raise ValueError("The result file contains no examples.")

    coverage_precision = []
    coverage_recall = []
    coverage_f1 = []
    coverage_exact_match = []
    output_exact_match = []

    for item in items:
        code = get_code(item)
        actual_trace_lines = get_coverage_trace_lines(item, code)
        predicted_trace = get_predicted_trace(item)
        predicted_trace_lines = get_predicted_trace_lines(predicted_trace)

        precision, recall, f1, exact_match = coverage_scores(
            code, predicted_trace_lines, actual_trace_lines
        )
        coverage_precision.append(precision)
        coverage_recall.append(recall)
        coverage_f1.append(f1)
        coverage_exact_match.append(exact_match)
        output_exact_match.append(float(is_output_correct(item, dataset)))

    sample_count = len(items)
    print(
        "Code Coverage (CCP) - P: "
        f"{format_percentage(sum(coverage_precision) / sample_count)}"
    )
    print(
        "Code Coverage (CCP) - R: "
        f"{format_percentage(sum(coverage_recall) / sample_count)}"
    )
    print(
        "Code Coverage (CCP) - F1: "
        f"{format_percentage(sum(coverage_f1) / sample_count)}"
    )
    print(
        "Code Coverage (CCP) - A_EM: "
        f"{format_percentage(sum(coverage_exact_match) / sample_count)}"
    )
    print(
        "Program Output - A_EM: "
        f"{format_percentage(sum(output_exact_match) / sample_count)}"
    )


if __name__ == "__main__":
    main()
