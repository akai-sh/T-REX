import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    calculate_confusion_matrix,
    execute_code_with_trace,
    parse_trace,
    program_execute,
)


def read_results(path, limit=None):
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


def write_results(path, items, has_exception):
    fields = [
        "source_id",
        "code",
        "exe_start",
        "func_info",
        "ans_len",
    ]
    if has_exception:
        fields[2:2] = ["error_line", "error_type"]
    fields.extend(["agent_trace"])

    with open(path, "w", encoding="utf-8") as destination:
        for item_index, item in enumerate(items, start=1):
            missing = [
                field
                for field in fields
                if field != "error_info" and item.get(field) is None
            ]
            if missing:
                raise ValueError(
                    f"Cannot save record {item_index} to {path!r}; missing: "
                    f"{', '.join(missing)}"
                )
            result = {field: item.get(field) for field in fields}
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} result(s) to {path}")


def run_executor(path, executor, tokenizer, args):
    items = read_results(path, args.limit)
    executed_items = []
    prediction_failures = 0

    for item_index, item in tqdm(
        enumerate(items),
        total=len(items),
        desc="Running executor",
    ):
        try:
            trace, _, error, error_info, _ = program_execute(
                item,
                executor,
                tokenizer,
                args,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Executor failed for {path!r} record {item_index + 1} "
                f"(source_id={item.get('source_id')!r})"
            ) from exc

        if error is not None:
            prediction_failures += 1
            tqdm.write(
                f"Prediction failed for {path!r} record {item_index + 1} "
                f"(source_id={item.get('source_id')!r}): {error}; "
                f"details={error_info!r}"
            )

        item["agent_trace"] = trace
        item["error_info"] = error_info
        executed_items.append(item)

    print(
        f"Completed {len(items)} record(s) from {path}; "
        f"prediction failures: {prediction_failures}"
    )
    return executed_items


def add_trace_evaluation(items, source_name):
    for item_index, item in enumerate(items, start=1):
        try:
            trace_output, _, _, _ = execute_code_with_trace(item["code"])
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
                for line_number, code_line in enumerate(item["code"].split("\n"))
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

            item["true_trace_len"] = len(true_trace)
            item["execute_correct"] = execute_correct
        except Exception as exc:
            raise RuntimeError(
                f"Trace evaluation failed for {source_name!r} record "
                f"{item_index} (source_id={item.get('source_id')!r})"
            ) from exc


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exception detection either from current-schema saved "
            "results or by running an executor model."
        )
    )
    parser.add_argument(
        "--executor_model_path",
        help="Hugging Face model ID or local executor checkpoint",
    )
    parser.add_argument(
        "--excep_data",
        help="Current-schema JSONL containing programs with exceptions",
    )
    parser.add_argument(
        "--n_excep_data",
        help="Current-schema JSONL containing programs without exceptions",
    )
    parser.add_argument(
        "--excep_result_path",
        help="Current-schema saved results for programs with exceptions",
    )
    parser.add_argument(
        "--n_excep_result_path",
        help="Current-schema saved results for programs without exceptions",
    )
    parser.add_argument(
        "--excep_result_output",
        help="Write newly generated exception predictions to this JSONL file",
    )
    parser.add_argument(
        "--n_excep_result_output",
        help=(
            "Write newly generated no-exception predictions to this JSONL "
            "file"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run or evaluate only the first N records from each input file",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")

    result_mode = bool(args.excep_result_path or args.n_excep_result_path)
    run_mode = bool(
        args.executor_model_path
        or args.excep_data
        or args.n_excep_data
        or args.excep_result_output
        or args.n_excep_result_output
    )
    if result_mode and run_mode:
        parser.error(
            "Choose either saved-result arguments or model/data arguments, "
            "not both."
        )
    if result_mode:
        if not args.excep_result_path or not args.n_excep_result_path:
            parser.error(
                "Saved-result mode requires --excep_result_path and "
                "--n_excep_result_path."
            )
    elif not all(
        (args.executor_model_path, args.excep_data, args.n_excep_data)
    ):
        parser.error(
            "Run mode requires --executor_model_path, --excep_data, and "
            "--n_excep_data."
        )
    return args, result_mode


def main():
    args, result_mode = parse_args()

    if result_mode:
        exception_items = read_results(args.excep_result_path, args.limit)
        no_exception_items = read_results(args.n_excep_result_path, args.limit)
        exception_source = args.excep_result_path
        no_exception_source = args.n_excep_result_path
    else:
        executor, tokenizer = load_model(args.executor_model_path)
        exception_items = run_executor(
            args.excep_data,
            executor,
            tokenizer,
            args,
        )
        no_exception_items = run_executor(
            args.n_excep_data,
            executor,
            tokenizer,
            args,
        )
        if args.excep_result_output:
            write_results(
                args.excep_result_output,
                exception_items,
                has_exception=True,
            )
        if args.n_excep_result_output:
            write_results(
                args.n_excep_result_output,
                no_exception_items,
                has_exception=False,
            )
        exception_source = args.excep_data
        no_exception_source = args.n_excep_data

    add_trace_evaluation(exception_items, exception_source)
    add_trace_evaluation(no_exception_items, no_exception_source)

    tp, fn, fp, tn, accuracy = calculate_confusion_matrix(
        exception_items,
        no_exception_items,
    )
    print(f"Exception records: {len(exception_items)}")
    print(f"No-exception records: {len(no_exception_items)}")
    print(f"TP: {tp}")
    print(f"FN: {fn}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
