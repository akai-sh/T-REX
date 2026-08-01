#!/usr/bin/env python3
"""Minimal model-backed smoke test for the T-REX artifact.

This script loads a fine-tuned T-REX executor model, runs one tiny function
execution task, and checks that the executor finishes with the expected output.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "ling031001/T-REX-qwen2.5-coder-14b"
DEFAULT_TASK_PATH = Path(__file__).with_name("kick_the_tires_example.jsonl")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PM_CRMS_DIR = PROJECT_ROOT / "test" / "PM_CRMs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal T-REX model smoke test.")
    parser.add_argument(
        "--executor_model_path",
        default=DEFAULT_MODEL,
        help="Hugging Face repo id or local path for the fine-tuned executor model.",
    )
    parser.add_argument(
        "--task_path",
        default=str(DEFAULT_TASK_PATH),
        help="Path to a JSONL file containing at least one T-REX task.",
    )
    parser.add_argument(
        "--task_index",
        default=0,
        type=int,
        help="Zero-based index of the task to run from the JSONL file.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype passed to transformers when loading the model.",
    )
    return parser.parse_args()


def load_task(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as task_file:
        tasks = [json.loads(line) for line in task_file if line.strip()]

    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    if index < 0 or index >= len(tasks):
        raise IndexError(f"task_index {index} is out of range for {len(tasks)} tasks")

    task = tasks[index]
    normalize_entry_call(task)
    return task


def normalize_entry_call(task: dict[str, Any]) -> None:
    """Use a literal entry-point call so program_execute can read parameters."""
    entry_point = task.get("entry_point")
    raw_input = task.get("input")
    function = task.get("function")

    if entry_point and raw_input and function:
        task["code"] = f"{function.rstrip()}\n{entry_point}{raw_input}\n"
        return

    task["code"] = task["code"].rstrip() + "\n"


def resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def load_executor(model_path: str, dtype_name: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_torch_dtype(dtype_name),
        device_map="auto",
        trust_remote_code=True,
    )
    if "llama" in model_path.lower() and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def parse_expected_output(raw_output: Any) -> Any:
    if isinstance(raw_output, list):
        return tuple(raw_output)
    return raw_output


def parse_model_output(raw_output: str) -> Any:
    return ast.literal_eval(raw_output.strip())


def extract_return_expression(output: Any) -> str:
    if output and output.get("cur_line", "").strip():
        return output["cur_line"].strip()

    if output and output.get("code"):
        for line in reversed(output["code"].splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped

    raise AssertionError(f"T-REX executor returned no parseable final output. Output: {output!r}")


def compact_effective_trace(trace: Any) -> list[dict[str, Any]]:
    effective_trace = []
    previous_states: dict[str, Any] = {}

    for step in trace:
        states = step.get("program_states")
        if not states:
            continue

        changed_states = {
            key: value
            for key, value in states.items()
            if previous_states.get(key) != value
        }
        previous_states = states

        if "ma" in changed_states or "mi" in changed_states:
            effective_trace.append(
                {
                    "line": step.get("line"),
                    "program_states": states,
                }
            )

    return effective_trace


def run_trex_executor(
    task: dict[str, Any],
    model: Any,
    tokenizer: Any,
    model_path: str,
) -> tuple[Any, bool, Any, Any, Any]:
    sys.path.insert(0, str(PM_CRMS_DIR))
    from utils import program_execute

    executor_args = SimpleNamespace(
        executor_model_path=model_path,
        variant="sft",
        num_sequences=1,
    )
    executor_args.args = executor_args

    trace, finished, error, error_info, output = program_execute(
        task,
        model,
        "",
        "",
        tokenizer,
        "",
        executor_args,
    )
    return trace, finished, output, error, error_info


def print_executor_result(trace: Any, output: Any) -> None:
    print("Effective executor trace:")
    print(json.dumps(compact_effective_trace(trace), indent=2, ensure_ascii=False, default=str))
    print("Final executor output:")
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


def assert_executor_output(task: dict[str, Any], finished: bool, output: Any) -> Any:
    if not finished:
        raise AssertionError(f"T-REX executor did not finish. Output: {output!r}")

    return_expression = extract_return_expression(output)
    observed = parse_model_output(return_expression)
    expected = parse_expected_output(task["output_true"])
    if observed != expected:
        raise AssertionError(f"Expected output {expected!r}, got {observed!r}")
    return observed


def main() -> None:
    args = parse_args()
    task_path = Path(args.task_path)
    task = load_task(task_path, args.task_index)

    print(f"Loaded task_id={task.get('task_id')} from {task_path}")
    print("Task code:")
    print(task["code"].rstrip())
    print(f"Loading executor model: {args.executor_model_path}")
    model, tokenizer = load_executor(args.executor_model_path, args.torch_dtype)

    print(f"Running T-REX executor on entry point: {task.get('entry_point')}")
    trace, finished, output, error, error_info = run_trex_executor(
        task,
        model,
        tokenizer,
        args.executor_model_path,
    )
    print_executor_result(trace, output)
    if error is not None:
        raise RuntimeError(f"T-REX executor failed: {error}; details: {error_info}")

    return_expression = extract_return_expression(output)
    observed = assert_executor_output(task, finished, output)

    print(f"Return expression: {return_expression}")
    print(f"Observed output: {observed}")
    print(f"Effective trace steps: {len(compact_effective_trace(trace))}")
    print("Kick-the-Tires passed.")


if __name__ == "__main__":
    main()
