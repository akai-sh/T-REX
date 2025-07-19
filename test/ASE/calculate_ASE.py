import json
import re
import ast
import argparse

idx=0
parser = argparse.ArgumentParser()
parser.add_argument("--result_path",
                    default='',
                    type=str,
                    help="")
args = parser.parse_args()

with open(args.result_path, 'r') as r:
    lines=r.readlines()
    num_true = num_true_seq = num_true_branch = num_seq = num_branch = num_completion = num_true_completion = 0
    for line in lines:
        idx += 1
        result_out=[]
        nexts=[]
        next_right=True
        false_flag=False
        item=json.loads(line)
        model_output=item['model_output']
        out_lins = re.findall(r"Line:\s*(\d+)", model_output)
        if len(out_lins)>1:
            false_flag=True
        pattern = r"Line:\s*(\d+)\s*Analysis:.*?Check:\s*({.*?})\s*Next statement:\s*(\d+|\w+)"
        matches = re.findall(pattern, model_output, re.DOTALL)
        if len(matches)>1:
            false_flag = True
        for match in matches:
            line_n=int(match[0])
            try:
                dic=ast.literal_eval(match[1])
            except:
                false_flag=True
            result_out.append([line_n,dic])
            if match[2]=='completion'or match[2]=='Completion':
                next_line='completion'
            else:
                try:
                    next_line = int(match[2])
                except:
                    false_flag = True
            nexts.append(next_line)
        if len(nexts)==1:
            if nexts[0]!=item['next_line']:
                next_right = False
        else:
            false_flag=True
        if item['next_line']=='completion':
            num_completion+=1
        elif item['next_line']==item['start_line']+1:
            num_seq+=1
        else:
            num_branch+=1
        if result_out==[]:
            false_flag=True
        if result_out==item['subsequent_values'] and next_right and not false_flag:
            num_true+=1
            if item['next_line'] == 'completion':
                num_true_completion += 1
            elif item['next_line'] == item['start_line'] + 1:
                num_true_seq += 1
            else:
                num_true_branch += 1


    acc=num_true/len(lines)

    print(args.result_path)
    print("Accuracy:", acc)
    print('')
    print("Accuracy_seq:", (num_true_seq+num_true_completion)/(num_seq+num_completion))
    print('')
    print("Accuracy_branch:", num_true_branch/num_branch)
    print('')
    print("Sum:", len(lines))
    print("True num:", num_true)
