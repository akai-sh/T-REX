import argparse
import os
import json

from joblib import load
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from utils import   program_execute, execute_code_with_trace, parse_trace, calculate_prefix_match

parser = argparse.ArgumentParser()

parser.add_argument("--executor_model_path",
                    default='',
                    type=str)
parser.add_argument("--reward_model_path",
                    default='',
                    type=str)
parser.add_argument("--rf_model_path",
                    default='',
                    type=str)
parser.add_argument("--num_sequences", default=32, type=int)
parser.add_argument("--results_path",
                    default='', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--data_path",
                    default='', type=str,
                    help="The test filename. Should contain the .jsonl files for this task.")
parser.add_argument("--variant",
                    default='', type=str,choices=["sft", "rvbs", "swa"])

args = parser.parse_args()
print(args.executor_model_path)
print(args.reward_model_path)
print(args.rf_model_path)
print(args.data_path)
print(args.results_path)

if args.variant=='rvbs':
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.executor_model_path,
        padding_side="left",
        trust_remote_code=True
    )
    tokenizer_qwen = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        padding_side="left",
        trust_remote_code=True
    )
    if 'llama' in args.executor_model_path or 'Llama' in args.executor_model_path:
        tokenizer.pad_token = tokenizer.eos_token

    executor = AutoModelForCausalLM.from_pretrained(
        args.executor_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path, num_labels=1, torch_dtype=torch.bfloat16
    )
    reward_model.to(device)

    rf_model = load(args.rf_model_path)
else :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.executor_model_path,
        padding_side="left",
        trust_remote_code=True
    )

    # 加载执行器模型
    executor = AutoModelForCausalLM.from_pretrained(
        args.executor_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if 'llama' in args.executor_model_path or 'Llama' in args.executor_model_path:
        tokenizer.pad_token = tokenizer.eos_token
#
if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)
executed_items = []

# run executor with SIPA
with open(args.data_path, 'r') as f,open(args.results_path+'/results.jsonl','w')as w:
    lines = f.readlines()

    for i, line in tqdm(enumerate(lines), desc="Processing"):
        item = json.loads(line)

        try:
            if args.variant!='rvbs':
                reward_model=rf_model=tokenizer_qwen=''
            trace, finished, error, error_info, output = program_execute(item, executor, reward_model, rf_model,
                                                                         tokenizer, tokenizer_qwen, args)

            item["agent_trace"] = trace
            item['output'] = output
            item['finished'] = finished
            json_line = json.dumps(item)
            w.write(json_line + '\n')
            executed_items.append(item)

        except:
            continue

# calculate prefix match
trace_output = []
trace_dicts = []
traced_functions = set()

for i, js in enumerate(executed_items):
    try:
        trace_output, wrong_flg, error_type, wrong_lineno = execute_code_with_trace(js['code'])
        trace_output = parse_trace(trace_output)[1]
        if trace_output == []:
            continue
        new_traces = []
        for tc in trace_output:
            new_trace = {}
            new_trace['line'] = tc[0] - 1
            new_trace['program_states'] = tc[1]
            new_traces.append(new_trace)

        js['true_trace'] = new_traces
        js['true_trace_len'] = len(new_traces)
        no_check_lines = []
        def_lines = []
        for func in js['func_info'].keys():
            def_lines.append(f'def {func}(')
        for i, line in enumerate(js['code'].split('\n')):
            for def_line in def_lines:
                if def_line in line:
                    no_check_lines.append(i)
        # print(no_check_lines)

        execute_correct = 1
        # for tc, at in zip(new_traces, js['ce_trace'][1:]):
        for tc, at in zip(new_traces, js['agent_trace'][1:]):
            # print("#############################################################")
            # print(f"\ttc:{tc}")
            # print(f"\tat:{at}\ttc==at:{tc==at}")
            if tc['line'] in no_check_lines:
                tc['line'] = at['line']
            if tc == at:
                execute_correct += 1
            else:
                break
        print(execute_correct)
        js['execute_correct'] = execute_correct
        jl = json.dumps(js)

    except Exception as e:
        continue

# calculate_prefix_match
avg_prefix, ratio_50, ratio_80, ratio_completion=calculate_prefix_match(executed_items)

print(f"Prefix: {avg_prefix}")
print(f"Completion 50%: {ratio_50}")
print(f"Completion 80%: {ratio_80}")
print(f"Completion : {ratio_completion}")

