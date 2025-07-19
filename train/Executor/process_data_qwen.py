import argparse
import json
import random
from datasets import load_dataset, Dataset

random.seed(20241204)
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    default='./../../data/train/Executor/sft.jsonl',
                    type=str,
                    help="path of input data")
parser.add_argument("--save_path",
                    default='./../../data/train/Executor/sft_formated.jsonl',
                    type=str,
                    help="path to save processed data")
args = parser.parse_args()

lines = []

with open(args.data_path, 'r', encoding='utf-8') as file:
    for line in file:
        lines.append(json.loads(line.strip()))
random.shuffle(lines)

with open(args.save_path,'w')as w:
    for line in lines:
        item={
    "messages":[
         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
         {"role": "user", "content": line['prompt']},
         {"role": "assistant", "content": line['output']}
    ],
    "format": "chatml"
}
        w.write(json.dumps(item) + '\n')