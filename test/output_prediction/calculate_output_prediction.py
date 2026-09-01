import argparse
import ast
import json
import re


ANSWER_PATTERN = re.compile(r"\[ANSWER\](.*?)\[/ANSWER\]", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate exact-match output accuracy for Table 5."
    )
    parser.add_argument(
        "--result_path",
        "--results_path",
        dest="result_path",
        required=True,
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


def parse_literal(value):
    if not isinstance(value, str):
        return value
    if value == "(3 + 5 + 7, 3 * 5 * 7)":
        value = "(15, 105)"
    return ast.literal_eval(value)


def ground_truth(item):
    if "output_true" in item:
        return item["output_true"]
    if "output" in item:
        return parse_literal(item["output"])
    raise KeyError("Each result must contain output_true or output.")


def predicted_output(model_output):
    match = ANSWER_PATTERN.search(model_output)
    if match is None:
        raise ValueError("Missing [ANSWER]...[/ANSWER] block.")
    answer = match.group(1).strip()
    if "==" not in answer:
        raise ValueError("The answer block does not contain an assertion.")
    return ast.literal_eval(answer.split("==", 1)[1].strip())


def equivalent(prediction, expected):
    return prediction == expected


def main():
    args = parse_args()
    items = read_jsonl(args.result_path, args.limit)
    correct = 0
    parse_failures = 0

    for item in items:
        try:
            prediction = predicted_output(item["model_output"])
            expected = ground_truth(item)
        except (KeyError, SyntaxError, TypeError, ValueError):
            parse_failures += 1
            continue
        correct += int(equivalent(prediction, expected))

    print(f"Correct: {correct}")
    print(f"Total: {len(items)}")
    print(f"Accuracy: {correct / len(items) * 100:.1f}%")


if __name__ == "__main__":
    main()
