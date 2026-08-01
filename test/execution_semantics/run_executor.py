import argparse
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def read_file(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

parser = argparse.ArgumentParser()
parser.add_argument("--executor_model_path", default='', type=str)
parser.add_argument("--results_path", default='', type=str)
parser.add_argument("--data_path", default='', type=str)
parser.add_argument("--batch_size", default=8, type=int)
args = parser.parse_args()

print(args.executor_model_path)
print(args.results_path)
print(args.data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    args.executor_model_path,
    padding_side="left",
    trust_remote_code=True
)


executor = AutoModelForCausalLM.from_pretrained(
    args.executor_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

if 'llama' in args.executor_model_path or 'Llama' in args.executor_model_path:
    tokenizer.pad_token = tokenizer.eos_token

def process_batch(batch):

    if 'llama' in args.executor_model_path or 'Llama' in args.executor_model_path:
        inputs = [item["prompt"] for item in batch]
    else:
        inputs = [
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": item["prompt"]}
            ], add_generation_prompt=True, tokenize=False)
            for item in batch
        ]

    inputs = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(executor.device)

    outputs = executor.generate(
        **inputs,
        max_new_tokens=10240,
        do_sample=True,
        temperature=0.8,
        # num_return_sequences=args.num_sequences,
    )

    batch_responses = []
    for i in range(len(batch)):
        sample_outputs = outputs[i]

        decoded_responses = [
            tokenizer.decode(sample_outputs[len(inputs.input_ids[i]):], skip_special_tokens=True)
        ]
        batch_responses.append(decoded_responses)

    return batch_responses


test_data = read_file(args.data_path)

progress_bar = tqdm(total=len(test_data), desc="Processing items")

with open(args.results_path, "a+") as result_file:
    for batch in batch_generator(test_data, args.batch_size):
        responses = process_batch(batch)

        for item, item_responses in zip(batch, responses):
            assert len(item_responses)==1
            item["model_output"] = item_responses[0]
            result_file.write(json.dumps(item) + "\n")

        progress_bar.update(len(batch))

progress_bar.close()