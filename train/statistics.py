#!/usr/bin/env python3
"""Calculate the Table 1 statistics with the artifact's original logic."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STATEMENT_MODULE_DIR = SCRIPT_DIR.parent / "test" / "execution_semantics"
sys.path.insert(0, str(STATEMENT_MODULE_DIR))

from python_statement import get_python_statement_classification  # noqa: E402


STATEMENT_TYPE_MAPPING = {
    "variable": "Variable Assignment (S2)",
    "expression": "Expression (S1)",
    "import": "Expression (S1)",
    "return": "Expression (S1)",
    "annotation": "Expression (S1)",
    "try": "Expression (S1)",
    "finally": "Expression (S1)",
    "if": "If Statement (S3)",
    "for": "For/While Loop (S4)",
    "while": "For/While Loop (S4)",
    "break": "For/While Loop (S4)",
    "continue": "For/While Loop (S4)",
    "function": "Method Calls (S5)",
    "method": "Method Calls (S5)",
}

STATEMENT_TYPE_ORDER = (
    "Variable Assignment (S2)",
    "Expression (S1)",
    "If Statement (S3)",
    "For/While Loop (S4)",
    "Method Calls (S5)",
)

ERROR_TYPE_ORDER = (
    "TypeError",
    "ZeroDivisionError",
    "NameError",
    "IndexError",
    "KeyError",
    "ValueError",
    "UnboundLocalError",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Table 1 statistics from train and train_excep."
    )
    parser.add_argument(
        "--train_data",
        type=Path,
        default=SCRIPT_DIR / "train.jsonl",
        help="Path to train.jsonl (default: train.jsonl beside this script).",
    )
    parser.add_argument(
        "--train_excep_data",
        type=Path,
        default=SCRIPT_DIR / "train_excep.jsonl",
        help="Path to train_excep.jsonl (default: train_excep.jsonl beside this script).",
    )
    return parser.parse_args()


def loads_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def calculate_statement_types(file_path):
    category_counts = defaultdict(int)
    data = loads_jsonl_file(file_path)

    for data_item in data:
        code = data_item["code"]
        line_no = data_item["start_line"]
        current_statement = code.split("\n")[line_no - 1]
        statement_type = get_python_statement_classification(
            current_statement.strip()
        )

        table1_type = STATEMENT_TYPE_MAPPING.get(statement_type)
        if table1_type is not None:
            category_counts[table1_type] += 1

    total = sum(category_counts.values())
    print("\nStatement Type Analysis:")
    print("------------------------")
    for category in STATEMENT_TYPE_ORDER:
        count = category_counts[category]
        percentage = count / total * 100
        print(f"{category}: {count}/{total}={percentage:.2f}%")

def calculate_error_types(file_path):
    category_counts = defaultdict(int)
    sum_number = 0
    data = loads_jsonl_file(file_path)

    for data_item in data:
        output = data_item["output"]
        pattern = (
            r"Line:\s*(\d+)\s*Analysis:.*?Check:\s*(.*?)\s*"
            r"Next statement:\s*(\d+|\w+)"
        )
        matches = re.findall(pattern, output, re.DOTALL)
        match = matches[0]
        if match[2] == "error":
            check = match[1]
            error_type = next(
                (error_type for error_type in ERROR_TYPE_ORDER if error_type in check),
                None,
            )
            if error_type is not None:
                category_counts[error_type] += 1
                sum_number += 1

    print("\nError Type Analysis:")
    print("------------------------")
    for category in ERROR_TYPE_ORDER:
        count = category_counts[category]
        percentage = count / sum_number * 100
        print(f"{category}: {count}/{sum_number}={percentage:.2f}%")

def main():
    args = parse_args()
    calculate_statement_types(args.train_data)
    calculate_error_types(args.train_excep_data)


if __name__ == "__main__":
    main()
