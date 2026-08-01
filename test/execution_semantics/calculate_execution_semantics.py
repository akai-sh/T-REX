import argparse
import ast
import json
import re
from collections import defaultdict


PREDICTION_PATTERN = re.compile(
    r"Line:\s*(\d+)\s*"
    r"Analysis:.*?"
    r"Check:\s*(.*?)\s*"
    r"Next statement:\s*(Line\s+\d+|line\s+\d+|\w+)",
    re.DOTALL,
)

TABLE_3_CATEGORIES = {
    "S1": "expressions",
    "S2": "variable assignments",
    "Seq.": "sequential/completion flow",
    "S3": "if-statements",
    "S4": "for/while-statements",
    "S5": "method calls",
    "Branch": "branch flow",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        "--results_path",
        dest="result_path",
        required=True,
    )
    return parser.parse_args()


def parse_prediction(item):
    model_output = item.get("model_output", item.get("gpt_output", ""))
    match = PREDICTION_PATTERN.search(model_output)
    if match is None:
        return None

    line_number, state_text, next_statement_text = match.groups()
    try:
        state = ast.literal_eval(state_text.strip().strip("`"))
    except (SyntaxError, ValueError):
        return None

    if next_statement_text.lower() == "completion":
        next_line = "completion"
    else:
        try:
            next_line = int(
                re.sub(r"^line\s+", "", next_statement_text, flags=re.IGNORECASE)
            )
        except ValueError:
            return None

    return [[int(line_number), state]], next_line


def classify_statement(statement):
    statement = statement.strip()
    if statement.startswith("while"):
        return "S4"
    if statement.startswith("for"):
        return "S4"
    if statement.startswith(("if", "elif")):
        return "S3"
    if any(operator in statement for operator in ("+", "-", "*", "/", "%")):
        return "S1"
    if statement.startswith("def"):
        return None
    if "=" in statement:
        return "S2"
    if "(" in statement and statement.endswith(")"):
        return "S5"
    return None


def flow_category(item):
    if item["next_line"] == "completion":
        return "Seq."
    if item["next_line"] == item["start_line"] + 1:
        return "Seq."
    return "Branch"


def percentage(numerator, denominator):
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.1f}"


def main():
    args = parse_args()

    total = 0
    ns_correct = 0
    ps_correct = 0
    joint_correct = 0
    category_totals = defaultdict(int)
    category_joint_correct = defaultdict(int)

    with open(args.result_path, "r", encoding="utf-8") as result_file:
        for line in result_file:
            if not line.strip():
                continue

            item = json.loads(line)
            total += 1

            prediction = parse_prediction(item)
            if prediction is None:
                is_ns_correct = False
                is_ps_correct = False
            else:
                predicted_values, predicted_next_line = prediction
                is_ns_correct = predicted_next_line == item["next_line"]
                is_ps_correct = predicted_values == item["subsequent_values"]

            is_joint_correct = is_ns_correct and is_ps_correct
            ns_correct += is_ns_correct
            ps_correct += is_ps_correct
            joint_correct += is_joint_correct

            flow = flow_category(item)
            category_totals[flow] += 1
            category_joint_correct[flow] += is_joint_correct

            code_lines = item["code"].splitlines()
            start_line = item["start_line"]
            if 1 <= start_line <= len(code_lines):
                statement_category = classify_statement(code_lines[start_line - 1])
                if statement_category is not None:
                    category_totals[statement_category] += 1
                    category_joint_correct[statement_category] += is_joint_correct

    print(f"A_NS (next-statement accuracy): {percentage(ns_correct, total)}")
    print(f"A_PS (program-state accuracy): {percentage(ps_correct, total)}")
    print(f"A_NS+PS (joint accuracy): {percentage(joint_correct, total)}")
    for category, description in TABLE_3_CATEGORIES.items():
        print(
            f"{category} ({description}): "
            f"{percentage(category_joint_correct[category], category_totals[category])}"
        )


if __name__ == "__main__":
    main()
