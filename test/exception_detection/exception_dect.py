import json
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from utils import execute_code_with_trace,parse_trace,program_execute,calculate_fp_tn,calculate_tp_fn

parser = argparse.ArgumentParser()

parser.add_argument("--executor_model_path",
                    default='',
                    type=str)
parser.add_argument("--excep_data",
                    default='',
                    type=str,
                    help="The test filename. Should contain the .jsonl files for this task.")
parser.add_argument("--n_excep_data",
                    default='',
                    type=str,
                    help="The test filename. Should contain the .jsonl files for this task.")
args = parser.parse_args()
print(args.excep_data)
print(args.n_excep_data)
print(args.executor_model_path)


def execute_file(path):
    executed_items = []
    with open(path, 'r') as f:
        lines = f.readlines()

        for i, line in tqdm(enumerate(lines), desc="Processing"):
            item = json.loads(line)

            try:

                trace, finished, error, error_info, output = program_execute(item, executor, tokenizer, args)

                item["agent_trace"] = trace
                item['error_info'] = error_info
                executed_items.append(item)

            except:
                continue

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

        except Exception as e:
            continue
    return executed_items

# device
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

# run code with exception
executed_items=execute_file(args.excep_data)
tp,fn=calculate_tp_fn(executed_items)

# run code without exception
executed_items_n=execute_file(args.n_excep_data)
fp,tn=calculate_fp_tn(executed_items_n)

print(f"TP: {tp}")
print(f"FN: {fn}")
print(f"FP: {fp}")
print(f"TN: {tn}")
