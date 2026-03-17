import json
import re
import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--result_path",
                    default='',
                    type=str,
                    help="")
args = parser.parse_args()

def check_ns_ps_gpt(item):
    try:
        model_output = item['model_output']
    except:
        model_output = item['gpt_output']
    result_out = []
    nexts = []
    next_right = True
    special = False
    ns_flag = False
    ps_flag = False
    model_output=model_output.replace('*','').replace('#','').replace('-','').replace('`','')
    out_lins = re.findall(r"Line:\s*(\d+)", model_output)
    # if len(out_lins)>1:
    #     special=True
    #     return False,False
    pattern = r"Line:\s*(\d+)\s*Analysis:.*?Check:\s*({.*?}|`{.*?}`)\s*Next statement:\s*(Line\s+\d+|line\s+\d+|\w+)"

    # pattern = r"Line:\s*(\d+)\s*Analysis:.*?Check:\s*({.*?}|`{.*?}`)\s*Next statement:\s*(\d+|\w+)"
    # pattern = r"Line:\s*(\d+)\s*Analysis:.*?Check:\s*({.*?})\s*Next statement:\s*(\d+|\w+)"
    matches = re.findall(pattern, model_output, re.DOTALL)
    for match in matches[:1]:
        line_n = int(match[0])
        try:
            dic = ast.literal_eval(match[1])
        except:
            return False, False
            special = True
        result_out.append([line_n, dic])
        if match[2] == 'completion' or match[2] == 'Completion':
            next_line = 'completion'
        else:
            try:
                next_line = int(match[2].replace('Line ', '').replace('line ', ''))
            except:
                special = True
                return False, False
        nexts.append(next_line)
    if len(nexts) == 1:
        if nexts[0] != item['next_line']:
            next_right = False
    else:
        special = True
        return False, False
    if result_out == []:
        special = True
        return False, False
    if next_right and not special:
        ns_flag = True
    if result_out == item['subsequent_values'] and not special:
        ps_flag = True
    return ns_flag, ps_flag
def check_ns_ps(item):
    try:
        model_output = item['model_output']
    except:
        model_output = item['gpt_output']
    result_out = []
    nexts = []
    next_right = True
    special = False
    ns_flag = False
    ps_flag = False
    out_lins = re.findall(r"Line:\s*(\d+)", model_output)
    pattern = r"Line:\s*(\d+)\s*Analysis:.*?Check:\s*({.*?}|`{.*?}`)\s*Next statement:\s*(Line\s+\d+|line\s+\d+|\w+)"
    matches = re.findall(pattern, model_output, re.DOTALL)
    for match in matches[:1]:
        line_n = int(match[0])
        try:
            dic = ast.literal_eval(match[1].replace('`', ''))
        except:
            return False, False
        result_out.append([line_n, dic])
        if match[2] == 'completion' or match[2] == 'Completion':
            next_line = 'completion'
        else:
            try:
                next_line = int(match[2])
            except:
                return False, False
        nexts.append(next_line)
    if len(nexts) == 1:
        if nexts[0] != item['next_line']:
            next_right = False
    else:
        return False, False
    if result_out == []:
        return False, False
    if next_right and not special:
        ns_flag = True
    if result_out == item['subsequent_values'] and not special:
        ps_flag = True
    return ns_flag, ps_flag



num_ns=0
num_ps=0
num_ns_ps=0
with open(args.result_path, 'r') as r:
    lines=r.readlines()
    sum=len(lines)
    for l in lines:
        item=json.loads(l)
        ns_flag, ps_flag = check_ns_ps(item)
        if ns_flag:
            num_ns+=1
        if ps_flag:
            num_ps+=1
        if ns_flag and ps_flag:
            num_ns_ps+=1
print(f"ns: {num_ns/sum}")
print(f"ps: {num_ps/sum}")
print(f"ns_ps: {(num_ns_ps)/(sum)}")
