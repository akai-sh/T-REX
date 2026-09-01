import argparse
import json
from collections import defaultdict, deque

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import execute_code_with_trace, parse_trace, program_execute


RESULT_FIELDS = (
    "source_id",
    "code",
    "buggy_code",
    "root_cause_statement",
    "root_cause_idx",
    "crash_statement",
    "crash_statement_idx",
    "error_type",
    "exe_start",
    "func_info",
    "ans_len",
    "agent_trace",
)


def read_jsonl(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path!r} at line {line_number}"
                ) from exc
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise RuntimeError(f"No records found in {path!r}.")
    return items


def sample_key(item):
    """Identify a sample, including duplicate source IDs."""
    return (
        item.get("source_id"),
        item.get("code"),
        item.get("buggy_code"),
        item.get("root_cause_idx"),
        item.get("crash_statement_idx"),
        item.get("ans_len"),
        json.dumps(
            item.get("func_info"),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def select_saved_predictions(data_items, saved_items, saved_path):
    predictions_by_sample = defaultdict(deque)
    for item in saved_items:
        predictions_by_sample[sample_key(item)].append(item)

    selected = []
    for data_index, data_item in enumerate(data_items, start=1):
        candidates = predictions_by_sample[sample_key(data_item)]
        if not candidates:
            raise ValueError(
                f"No prediction in {saved_path!r} matches buggy-data record "
                f"{data_index} (source_id={data_item.get('source_id')!r})."
            )
        selected.append(candidates.popleft())

    return selected


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    executor = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return executor, tokenizer


def run_predictions(data_items, executor, tokenizer, args):
    predictions = []
    for item_index, data_item in tqdm(
        enumerate(data_items, start=1),
        total=len(data_items),
        desc="Running bug detector",
    ):
        # program_execute reads `code`; this task evaluates the mutated program.
        runtime_item = dict(data_item)
        runtime_item["code"] = data_item["buggy_code"]
        try:
            trace, _, _, _, _ = program_execute(
                runtime_item,
                executor,
                tokenizer,
                args,
                prompt_mode="normal",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Executor failed for buggy-data record {item_index} "
                f"(source_id={data_item.get('source_id')!r})."
            ) from exc

        prediction = dict(data_item)
        prediction["agent_trace"] = trace
        predictions.append(prediction)
    return predictions


def normalize_predictions(data_items, prediction_items):
    normalized = []
    for item_index, (data_item, prediction_item) in enumerate(
        zip(data_items, prediction_items),
        start=1,
    ):
        item = {
            field: data_item.get(field)
            for field in RESULT_FIELDS
            if field != "agent_trace"
        }
        item["agent_trace"] = prediction_item.get("agent_trace")

        required_fields = [
            field
            for field in RESULT_FIELDS
            if field != "error_type"
        ]
        missing = [field for field in required_fields if item.get(field) is None]
        if missing:
            raise ValueError(
                f"Result record {item_index} is missing: "
                f"{', '.join(missing)}"
            )
        normalized.append(item)
    return normalized


def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as destination:
        for item in items:
            destination.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} result(s) to {path}")


def add_trace_evaluation(items):
    for item_index, item in enumerate(items, start=1):
        try:
            trace_output, _, _, _ = execute_code_with_trace(item["buggy_code"])
            parsed_trace = parse_trace(trace_output)[1]
            if not parsed_trace:
                raise ValueError("the reconstructed true trace is empty")

            true_trace = [
                {"line": trace[0] - 1, "program_states": trace[1]}
                for trace in parsed_trace
            ]
            def_lines = [
                f"def {func_name}("
                for func_name in item.get("func_info", {})
            ]
            no_check_lines = [
                line_number
                for line_number, code_line in enumerate(
                    item["buggy_code"].split("\n")
                )
                if any(def_line in code_line for def_line in def_lines)
            ]

            execute_correct = 1
            for true_state, agent_state in zip(
                true_trace,
                item["agent_trace"][1:],
            ):
                if true_state["line"] in no_check_lines:
                    true_state["line"] = agent_state["line"]
                if true_state != agent_state:
                    break
                execute_correct += 1

            item["true_trace"] = true_trace
            item["execute_correct"] = execute_correct
        except Exception as exc:
            raise RuntimeError(
                f"Trace evaluation failed for record {item_index} "
                f"(source_id={item.get('source_id')!r})."
            ) from exc


def calculate_fault_location_accuracy(items):
    correct = 0
    missing_fault_lines = 0
    for item in items:
        root_cause_line = item["root_cause_idx"] - 1
        true_trace_lines = [state["line"] for state in item["true_trace"]]
        try:
            reverse_index = true_trace_lines[::-1].index(root_cause_line)
        except ValueError:
            missing_fault_lines += 1
            continue

        trace_length_to_fault = len(true_trace_lines) - reverse_index
        if item["execute_correct"] >= trace_length_to_fault:
            correct += 1

    accuracy = correct / len(items)
    return correct, len(items), accuracy, missing_fault_lines


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run or load bug-detection predictions, verify them against "
            "buggy_data, "
            "and calculate root-cause localization accuracy."
        )
    )
    parser.add_argument(
        "--buggy_data",
        required=True,
        help="Buggy-data JSONL containing root_cause_idx",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--executor_model_path",
        help="Hugging Face model ID or local Reasoner checkpoint",
    )
    mode.add_argument(
        "--saved_result_path",
        help="Previously generated bug-detection result JSONL",
    )
    parser.add_argument(
        "--result_output",
        help=(
            "Write newly generated predictions, or the selected and normalized "
            "saved predictions, to this JSONL file"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run or evaluate only the first N buggy-data records",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    if args.executor_model_path and not args.result_output:
        parser.error("Model execution requires --result_output.")
    return args


def main():
    args = parse_args()
    data_items = read_jsonl(args.buggy_data, args.limit)

    if args.saved_result_path:
        saved_items = read_jsonl(args.saved_result_path)
        prediction_items = select_saved_predictions(
            data_items,
            saved_items,
            args.saved_result_path,
        )
    else:
        executor, tokenizer = load_model(args.executor_model_path)
        prediction_items = run_predictions(
            data_items,
            executor,
            tokenizer,
            args,
        )

    normalized_items = normalize_predictions(data_items, prediction_items)
    if args.result_output:
        write_jsonl(args.result_output, normalized_items)

    add_trace_evaluation(normalized_items)
    correct, total, accuracy, missing = calculate_fault_location_accuracy(
        normalized_items
    )
    print(f"Records: {total}")
    print(f"Correct fault locations: {correct}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
