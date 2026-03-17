import json
import re,ast
from python_statement import get_python_statement_classification
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--result_path",
                    default='',
                    type=str,
                    help="")
args = parser.parse_args()
def loads_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


# Initialize counters
category_counts = defaultdict(int)
category_correct = defaultdict(int)
data=loads_jsonl_file(args.result_path)
for data_item in data:
    code = data_item['code']
    line_no = data_item['start_line']
    current_statement = code.split('\n')[line_no - 1]
    statement_type = get_python_statement_classification(current_statement.strip())

    if statement_type == "None":
        print(f"Unclassified statement: {current_statement}")
        continue

    # Increment the count for this category
    category_counts[statement_type] += 1

    category_correct[statement_type] += 1


# Print the results
print("\nStatement Type Analysis:")
print("------------------------")
strp=''
for category, count in category_counts.items():
    correct = category_correct[category]
    print(f"{category}:")
    percentage = (correct / count) * 100
    print(f"{correct}/{count}={percentage:.2f}%")
    strp += f"{correct}/{count}={percentage:.2f}% , "

print(strp[:-2])

# Calculate and print overall statistics
total_statements = sum(category_counts.values())
total_correct = sum(category_correct.values())
print("Overall Statistics:")
print(f"Total statements analyzed: {total_statements}")
print(f"Total correct statements: {total_correct}")
print(f"Overall accuracy: {total_correct / total_statements:.2%}")
