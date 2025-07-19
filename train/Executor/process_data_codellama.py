import argparse
import json
import random
from datasets import Dataset

random.seed(20241204)
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    default='./../../data/train/Executor/sft.jsonl',
                    type=str,
                    help="path of input data")
parser.add_argument("--save_path",
                    default='./../../data/train/Executor/dataset_codellama',
                    type=str,
                    help="path to save processed data")
args = parser.parse_args()

lines = []

with open(args.data_path, 'r', encoding='utf-8') as file:
    for line in file:
        lines.append(json.loads(line.strip()))
random.shuffle(lines)

dataset_dict = {
    'prompt':[line['prompt'] for line in lines],
    'output': [line['output'] for line in lines]
}

dataset = Dataset.from_dict(dataset_dict)

dataset.save_to_disk(args.save_path)