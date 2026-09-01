import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = '''You are given a Python function and test code containing inputs to the function. Complete the assertion in the test code by just replacing the question marks with your answers, considering executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Do NOT modify any other parts of the test code. Think step by step. Your thoughts should be concise and contain no more than 10 steps. For loop logics, express your thoughts without detailing each iteration. Provide your answer as the completed code snippet in [ANSWER] and [/ANSWER] tags, following the example.

[PYTHON]
def g(s):
    return s + "a"

assert g("x9j") == ??
[/PYTHON]
[THOUGHT]
1. The function g is defined, which takes a single argument s. Then g is called with argument s="x9j".
2. In the function, s is concatenated with "a" and returned.
3. Thus the return value should be "x9ja", and the question marks should be replaced with "x9ja".
[/THOUGHT]
[ANSWER]
assert g("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{function}
{invocation}
[/PYTHON]
[THOUGHT]'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run direct program-output prediction for Table 5."
    )
    parser.add_argument(
        "--model_path",
        "--executor_model_path",
        dest="model_path",
        required=True,
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument(
        "--result_path",
        "--results_path",
        dest="result_path",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero.")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be greater than zero.")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero.")
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
        raise ValueError("The input file contains no examples.")
    return items


def build_prompt(item):
    invocation = f"assert {item['entry_point']}{item['input']} == ??"
    return PROMPT_TEMPLATE.format(
        function=item["function"],
        invocation=invocation,
    )


def batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def format_model_input(tokenizer, prompt, model_path):
    if "llama" in model_path.lower():
        return prompt
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are Qwen, created by Alibaba Cloud. "
                    "You are a helpful assistant."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )


def main():
    args = parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = read_jsonl(args.data_path, args.limit)
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with result_path.open("w", encoding="utf-8") as destination:
        for batch in tqdm(
            batches(items, args.batch_size),
            total=(len(items) + args.batch_size - 1) // args.batch_size,
            desc="Predicting outputs",
        ):
            prompts = [build_prompt(item) for item in batch]
            model_inputs = [
                format_model_input(tokenizer, prompt, args.model_path)
                for prompt in prompts
            ]
            encoded = tokenizer(
                model_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
            )

            input_length = encoded.input_ids.shape[1]
            responses = tokenizer.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
            )
            for item, prompt, response in zip(batch, prompts, responses):
                result = dict(item)
                result["prompt"] = prompt
                result["model_output"] = response
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved {len(items)} result(s) to {result_path}")


if __name__ == "__main__":
    main()
