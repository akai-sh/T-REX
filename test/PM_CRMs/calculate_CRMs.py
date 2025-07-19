import argparse
import json, ast
from collections import deque

import math
from math import inf

parser = argparse.ArgumentParser()
parser.add_argument("--result_path",
                    default='',
                    type=str,
                    help="")
args = parser.parse_args()

score=0.0
sum = 0
num_op=0
num_ccp=0
num_psp=0
num_epp=0

def find_next(numbers, line_no,n):
    result = []
    # 遍历列表
    for i in range(len(numbers) - 1):  # -1是因为要查找当前元素的后一个元素
        if numbers[i] == line_no:
            next_number = numbers[i + 1]
            if next_number=='completion':
                continue
            if next_number < n:
                result.append(next_number)
    return result

def check_op_humaneval(out, out_true):
    if out==out_true:
        return True
    elif ast.literal_eval(out_true) == out:
        return True
    return False

def check_op_codenet(out, out_true):
    if out == out_true:
        return True
    else:
        if 'print' in out:
            out = out[6:-1]
        try:
            out = ast.literal_eval(out)
            if type(out).__name__ == 'tuple':
                out = list(out)
            if out == out_true:
                return True
            elif out == ast.literal_eval(out_true):
                return True
            else:
                return False
        except:
            return False

def check_psp(agent_traces,task):
    answer=task['value']
    tg_line = task['lineno']
    tg_var = task['var']

    if answer == 'Nil':
        ground_truth = 'Nil'
    elif 'set()' in answer:
        ground_truth = ast.literal_eval(answer.replace('set()', '{}'))
    elif answer == '[inf]':
        ground_truth = [inf]
    elif answer == '[(11, 0, inf)]':
        ground_truth = [(11, 0, inf)]
    elif answer == '[(0, deque([]))]':
        ground_truth = (0, deque([]))
    elif answer == "[deque([')']), deque(['(', ')', ')']), deque(['(', '(', ')', ')', ')']), deque(['(', '(', '(', ')', ')', ')', '(']), deque(['(', '(', '(', ')', ')', ')', '(', ')']), deque(['(', '(', '(', ')', ')', ')', '(', ')', ')'])]":
        ground_truth = [deque([')']), deque(['(', ')', ')']), deque(['(', '(', ')', ')', ')']),
                        deque(['(', '(', '(', ')', ')', ')', '(']), deque(['(', '(', '(', ')', ')', ')', '(', ')']),
                        deque(['(', '(', '(', ')', ')', ')', '(', ')', ')'])]
    else:
        ground_truth = ast.literal_eval(answer)

    pred_vars = []
    for agent_trace in agent_traces:
        if 'program_states' not in agent_trace:
            continue
        if agent_trace['line'] == tg_line - 1 and tg_var in agent_trace['program_states']:
            pred_vars.append(agent_trace['program_states'][tg_var])

    if ground_truth == 'Nil':
        if pred_vars == []:
            return True
    else:
        if pred_vars == []:
            return False
        # tag = False
        for gt in ground_truth:
            for pred in pred_vars:
                if type(ground_truth[0]).__name__ == 'float' or type(ground_truth[0]).__name__ == 'double':
                    if math.isclose(gt, pred):
                        return True
                elif gt == pred:
                    return True
    return False

def check_ccp(agent_traces,task):
    tg_line = task['lineno']
    tg_var = task['var']

    ground_truth = task['exe_or_not']
    if ground_truth == True:
        gt=1
    if ground_truth == False:
        gt=0
    line_values = [agt['line'] for agt in agent_traces]
    if tg_line - 1 in line_values:
        ms=1
    else:
        ms=0
    if gt == ms:
        return True
    return False

def check_epp(agent_traces,task):
    tg_line = task['lineno']
    tg_var = task['var']

    ground_truth = task['next_line']
    line_values = [agt['line'] for agt in agent_traces]

    if task['exe_or_not'] == False:
        assert ground_truth == [-1]
        if tg_line - 1 not in line_values:
            return True
    else:
        if tg_line - 1 not in line_values:
            return False
        assert task['exe_or_not'] == True
        model_answer = find_next(line_values, tg_line - 1, len(item['function'].split('\n')))
        if ground_truth == [-1]:
            if model_answer == []:
                return True
        else:
            for ms in model_answer:
                if ms + 1 in ground_truth:
                    return True
    return False

with open(args.result_path, 'r') as f:
    results_lines = f.readlines()
    for l in results_lines:
        item = json.loads(l)
        if item['finished'] == True:
            out = item['output']['cur_line']
            out_true = item['output_true']
            if out_true == '(3 + 5 + 7, 3 * 5 * 7)':
                out_true = '(15,105)'
            if 'codenetmut' in args.result_path:
                op_flag = check_op_codenet(out, out_true)
            else:
                op_flag = check_op_humaneval(out, out_true)
        else:
            op_flag = False
        tasks = item['statements_and_vars']
        if op_flag:
            num_op+=1
        for task_idx, task in enumerate(tasks):
            sum+=1
            try:
                agent_trace=item['agent_trace']
            except:
                agent_trace = item['ce_trace']
            ccp_flag=check_psp(agent_trace,task)
            psp_flag=check_psp(agent_trace,task)
            epp_flag=check_epp(agent_trace,task)

            if ccp_flag: num_ccp+=1;
            if psp_flag: num_psp+=1;
            if epp_flag: num_epp+=1

            if ccp_flag and psp_flag and epp_flag and op_flag:
                score += 1
            elif ccp_flag and psp_flag and epp_flag and not op_flag:
                score += 0.5
            elif ccp_flag and psp_flag and not epp_flag and not op_flag:
                score += 0.25
            elif ccp_flag and not psp_flag and not epp_flag and not op_flag:
                score += 0.125

print(f"CCP: {num_ccp/sum}")
print(f"PSP: {num_psp/sum}")
print(f"EPP: {num_epp/sum}")
print(f"OP: {num_op/len(results_lines)}")
print(f"IC Score : {score/sum}")