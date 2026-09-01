import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm


PROMPT_TEMPLATE = '''Code:

{code_with_line_number}

Current program state: 
Suppose that the code is going to execute <line {line_number}> statement ({statement}). The current values of the variables are {program_state}.    

Number of subsequent executed statements: 1

Values of variables after executing subsequent statements: 

<line {line_number}> {program_state_after}
<line {next_line}> (this statement will NOT be executed due to the number of subsequent executed statements)

Instruction:
Given the code, current program state, and the values of variables after executing subsequent statements, please analyze each statement's execution WITHIN THE NUMBER of subsequent executed statements. For each executable statement, first indicate the line number and describe the variable values after executing this statement. Then, if there's a jump after execution or the entire program ends after the statement execution, indicate the next statement or that the code execution has finished. 

Provide your analysis in the following format for each statement:
Line: [Line number of the executed statement shown in the subsequent statements]
Analysis: [Step-by-step explanation of the execution progress in a single paragraph without Enumeration and Enumeration Symbols]
Check: [Values of the variables in dictionary format based on the analysis if the statement is executable]
Next statement:[The line number of the statement to be executed next or the label 'completion' if the entire program ends after the statement execution]'''


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate grounded program-execution explanations with GPT-4o-mini."
        )
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only records not already present in the output file",
    )
    parser.add_argument(
        "--include_teacher_prompt",
        action="store_true",
        help="Also save the grounded API prompt in teacher_prompt",
    )
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be greater than zero.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    if Path(args.input_path).resolve() == Path(args.output_path).resolve():
        parser.error("--input_path and --output_path must be different files.")
    return args


def sample_key(item):
    return item.get("source_id"), item.get("sub_id"), item.get("id")


def read_jsonl(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path!r} at line {line_number}."
                ) from exc
            items.append(item)
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise ValueError("The input file contains no records.")
    return items


def read_completed_keys(path):
    completed = set()
    if not Path(path).exists():
        return completed
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                completed.add(sample_key(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path!r} at line {line_number}."
                ) from exc
    return completed


def add_line_numbers(code):
    return "\n".join(
        f"<line {line_number}>{line}"
        for line_number, line in enumerate(code.splitlines(), 1)
    )


def build_teacher_prompt(item):
    required_fields = (
        "code",
        "start_line",
        "initial_value",
        "subsequent_values",
        "next_line",
    )
    missing = [field for field in required_fields if field not in item]
    if missing:
        raise ValueError(
            f"Sample {sample_key(item)} is missing: {', '.join(missing)}"
        )

    code_lines = item["code"].splitlines()
    line_number = item["start_line"]
    if not isinstance(line_number, int) or not 1 <= line_number <= len(code_lines):
        raise ValueError(
            f"Sample {sample_key(item)} has invalid start_line={line_number!r}."
        )

    subsequent_values = item["subsequent_values"]
    if not isinstance(subsequent_values, list) or len(subsequent_values) != 1:
        raise ValueError(
            f"Sample {sample_key(item)} must contain one subsequent state."
        )
    executed_line, program_state_after = subsequent_values[0]
    if executed_line != line_number:
        raise ValueError(
            f"Sample {sample_key(item)} has inconsistent executed line: "
            f"{executed_line!r} != {line_number!r}."
        )

    return PROMPT_TEMPLATE.format(
        code_with_line_number=add_line_numbers(item["code"]),
        line_number=line_number,
        statement=code_lines[line_number - 1].strip(),
        program_state=item["initial_value"],
        program_state_after=program_state_after,
        next_line=item["next_line"],
    )


async def generate_one(client, model, item, include_teacher_prompt):
    teacher_prompt = build_teacher_prompt(item)
    response = await client.responses.create(
        model=model,
        input=teacher_prompt,
    )
    explanation = response.output_text
    if not explanation or not explanation.strip():
        raise RuntimeError(f"Empty response for sample {sample_key(item)}.")

    result = dict(item)
    result["output"] = explanation.strip()
    if include_teacher_prompt:
        result["teacher_prompt"] = teacher_prompt
    return result


async def run(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running this script.")

    items = read_jsonl(args.input_path, args.limit)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"{output_path} already exists; use --resume or choose another output."
        )
    completed = read_completed_keys(output_path) if args.resume else set()
    pending = [item for item in items if sample_key(item) not in completed]
    if not pending:
        print("No records remain to be generated.")
        return

    client = AsyncOpenAI()
    mode = "a" if args.resume else "x"
    generated_count = 0
    with output_path.open(mode, encoding="utf-8") as destination:
        for start in tqdm(
            range(0, len(pending), args.concurrency),
            desc="Generating explanations",
        ):
            batch = pending[start : start + args.concurrency]
            results = await asyncio.gather(
                *(
                    generate_one(
                        client,
                        args.model,
                        item,
                        args.include_teacher_prompt,
                    )
                    for item in batch
                ),
                return_exceptions=True,
            )

            failures = []
            for item, result in zip(batch, results):
                if isinstance(result, BaseException):
                    failures.append((sample_key(item), result))
                    continue
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                destination.flush()
                generated_count += 1

            if failures:
                details = "; ".join(
                    f"{key}: {error!r}" for key, error in failures
                )
                raise RuntimeError(
                    f"{len(failures)} API request(s) failed. Successful records "
                    f"were saved; rerun with --resume. {details}"
                )

    print(f"Generated {generated_count} explanation(s) in {output_path}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
