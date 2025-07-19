import sys
import io
import contextlib
import re
import ast

def one_step_execute(prompt,executor,tokenizer,args):
    if 'llama' in args.executor_model_path or 'Llama' in args.executor_model_path:
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=10240
        ).to(executor.device)
    else:
    # 1.
        inputs = [
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ], add_generation_prompt=True, tokenize=False)
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
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
        # num_return_sequences=args.num_sequences,
    )

    responses = [
        tokenizer.decode(seq[len(inputs.input_ids[0]):], skip_special_tokens=True)
        for seq in outputs
    ]

    assert len(responses)==1
    return responses[0]


def trace_lines(frame, event, arg):
    global last_lineno
    if frame.f_code.co_filename != "<string>":

        return

    if event == 'call':
        if last_lineno is not None:
            lineno = last_lineno
            local_vars = frame.f_back.f_locals.copy()
            state_parts = []
            program_states = {}
            for var in vars_order:
                if var in local_vars:
                    val = local_vars[var]
                    val_type = type(val).__name__
                    if val_type == 'map' or val_type == 'function':
                        continue
                    val_str = format_value(val)
                    program_states[var]=val
                    state_parts.append(f'{var}({val_type}) : {val_str}')
            if state_parts:
                state = ' <dictsep> '.join(state_parts)
                trace_dict = {"line": lineno, "program_states": program_states}
                trace_line = f'<line> <{lineno}> <state> {state} </state>'
                trace_output.append(trace_line)
            # else:
            #     trace_line = f'<line> <{lineno}> <state> </state>'
            #     trace_output.append(trace_line)
        code = frame.f_code
        func_def_line = code.co_firstlineno - 1  # line number start from 0
        if code not in traced_functions:
            local_vars = frame.f_locals.copy()  
            state_parts = []
            program_states={}
            for var in vars_order:
                if var in local_vars:
                    val = local_vars[var]
                    val_type = type(val).__name__
                    if val_type == 'map' or val_type == 'function':
                        continue
                    program_states[var]=val
                    val_str = format_value(val)
                    state_parts.append(f'{var}({val_type}) : {val_str}')
            if state_parts:
                state = ' <dictsep> '.join(state_parts)
                trace_dict = {"line": lineno, "program_states": program_states}
                trace_line = f'<line> <{lineno}> <state> {state} </state>'
                trace_output.append(trace_line)
            # else:
            #     trace_line = f'<line> <{func_def_line}> <state> </state>'
            #     trace_output.append(trace_line)
            traced_functions.add(code)
        last_lineno = func_def_line
        return trace_lines
    elif event == 'line':
        if last_lineno is not None:
            lineno = last_lineno
            local_vars = frame.f_locals.copy()
            state_parts = []
            program_states={}
            for var in vars_order:
                if var in local_vars:
                    val = local_vars[var]
                    val_type = type(val).__name__
                    if val_type == 'map' or val_type == 'function':
                        continue
                    program_states[var]=val
                    val_str = format_value(val)
                    state_parts.append(f'{var}({val_type}) : {val_str}')
            if state_parts:
                state = ' <dictsep> '.join(state_parts)
                trace_dict = {"line": lineno, "program_states": program_states}
                trace_line = f'<line> <{lineno}> <state> {state} </state>'
                trace_output.append(trace_line)
            # else:
            #     trace_line = f'<line> <{lineno}> <state> </state>'
            #     trace_output.append(trace_line)
        last_lineno = frame.f_lineno - 1
    elif event == 'return':
        if last_lineno is not None:
            lineno = last_lineno
            local_vars = frame.f_locals.copy()
            state_parts = []
            program_states={}
            for var in vars_order:
                if var in local_vars:
                    val = local_vars[var]
                    val_type = type(val).__name__
                    if val_type == 'map' or val_type == 'function':
                        continue
                    program_states[var]=val
                    val_str = format_value(val)
                    state_parts.append(f'{var}({val_type}) : {val_str}')
            if state_parts:
                state = ' <dictsep> '.join(state_parts)
                trace_dict = {"line": lineno, "program_states": program_states}
                trace_line = f'<line> <{lineno}> <state> {state} </state>'
                trace_output.append(trace_line)
            # else:
            #     trace_line = f'<line> <{lineno}> <state> </state>'
            #     trace_output.append(trace_line)
        last_lineno = frame.f_lineno - 1

    return trace_lines


def format_value(val):
    if isinstance(val, list):
        var_str = []
        for x in val:
            var_str.append(str(x) if type(x).__name__ != 'str' else f"'{x}'")
        return '[' + ','.join(var_str) + ']'
    elif callable(val):
        return '<function>'
    else:
        return str(val) if type(val).__name__ != 'str' else f"'{val}'"


def get_vars_order(code_str):
    var_lines = {}

    class VarVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # func name
            var_lines[node.name] = node.lineno - 1
            self.generic_visit(node)

        def visit_Assign(self, node):
            # assignment
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id not in var_lines:
                        var_lines[target.id] = node.lineno - 1
            self.generic_visit(node)

        def visit_For(self, node):
            # for
            if isinstance(node.target, ast.Name):
                if node.target.id not in var_lines:
                    var_lines[node.target.id] = node.lineno - 1
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                if node.id not in var_lines:
                    var_lines[node.id] = node.lineno - 1
            self.generic_visit(node)

    parsed_ast = ast.parse(code_str)
    VarVisitor().visit(parsed_ast)
    vars_order = sorted(var_lines.items(), key=lambda x: x[1])
    vars_order = [var for var, _ in vars_order]
    return vars_order


def remove_consecutive_duplicates(trace_list):
    new_trace = []
    previous_line = None
    for line in trace_list:
        if line != previous_line:
            new_trace.append(line)
        previous_line = line
    return new_trace


def execute_code_with_trace(code_str_input):
    global vars_order, last_lineno, trace_output, traced_functions
    code_str = code_str_input
    wrong_flg = False
    wrong_lineno = -1
    error_type = 'None error'

    # prepare envirment
    local_env = {}
    global_env = {'math': __import__('math'),'bisect':__import__('bisect'),'gcd':__import__('math').gcd,'factorial':__import__('math').factorial,'bisect_left':__import__('bisect').bisect_left,'deque':__import__('collections').deque,'itertools':__import__('itertools'),'groupby':__import__('itertools').groupby,'bisect_right':__import__('bisect').bisect_right}

    trace_output = []
    traced_functions = set()

    vars_order = get_vars_order(code_str)

    code_obj = compile(code_str, filename="<string>", mode="exec")

    last_lineno = None

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        sys.settrace(trace_lines)
        try:
            exec(code_obj, global_env, local_env)
            #
            if last_lineno is not None:
                lineno = last_lineno
                local_vars = local_env.copy()
                state_parts = []
                program_states = {}
                for var in vars_order:
                    if var in local_vars:
                        val = local_vars[var]
                        val_type = type(val).__name__
                        if val_type == 'map' or val_type == 'function':
                            continue
                        program_states[var] = val
                        val_str = format_value(val)
                        state_parts.append(f'{var}({val_type}) : {val_str}')
                if state_parts:
                    state = ' <dictsep> '.join(state_parts)
                    trace_line = f'<line> <{lineno}> <state> {state} </state>'
                    trace_dict = {"line":lineno,"program_states":program_states}
                    trace_output.append(trace_line)
                # else:
                #     trace_line = f'<line> <{lineno}> <state> </state>'
                #     trace_output.append(trace_line)
        except Exception as e:
            error_type = type(e).__name__
            trace_output.append(f'<line> <{last_lineno}> <error> <{error_type}> <info> {e}')
            # trace_output = 'error'
            wrong_flg = True
            wrong_lineno = last_lineno
        finally:
            sys.settrace(None)


    trace_output = remove_consecutive_duplicates(trace_output)
    return trace_output, wrong_flg, error_type, wrong_lineno

def parse_trace(trace):
    parsed_trace=[]
    line_numbers=[]
    for s in trace:
        if 'error' in s:
            continue
        if s.startswith('<line>'):
            number_match = re.search(r'<(\d+)>', s)
            number = int(number_match.group(1)) if number_match else None

            # extract content between <state> and </state> 
            state_match = re.search(r'<state>(.*?)</state>', s)
            state_content = state_match.group(1).strip() if state_match else ""

            dict_items = re.split(r'<dictsep>', state_content)

            result_dict = {}
            for item in dict_items:
                key_val = item.split(':', 1)
                if len(key_val) != 2:
                    continue
                key = key_val[0].strip()
                if 'set' in key:
                    return [], []
                match = re.match(r'^[^(]+', key)
                key=match.group(0)
                val = key_val[1].strip()

                try:
                    parsed_val = ast.literal_eval(val)
                except:
                    return [],[]

                result_dict[key] = parsed_val
            line_numbers.append(number+1)
            parsed_trace.append((number+1, result_dict))
        else:
            continue
    return line_numbers,parsed_trace

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("time out")

def add_linenum(code):
    new_lines=[]
    code_lines = code.split('\n')
    for line_num, line in enumerate(code_lines):
        line_nums_str = f'<line {line_num + 1}>'
        this_line = line_nums_str+line
        new_lines.append(this_line)
    return new_lines

def gen_prompt_excep(code, cur_line_num, cur_line, cur_states):
    prompt = f"""Code:
{code}

Current program state:
Suppose that the code is going to execute <line {cur_line_num + 1}> statement ({cur_line.strip()}). The current values of the variables are {cur_states}.

Number of subsequent executed statements: 1

Instruction:
Given the code, current program state, and the values of variables after executing subsequent statements, please analyze each statement's execution WITHIN THE NUMBER of subsequent executed statements. For each statement, first indicate the line number, then describe the variable values after executing this statement or the error, and finally point out the next statement to be executed. If the entire program ends after the statement execution, additionally indicate that the code execution has finished. If the statement encounters an error during execution, analyze its cause and type.
Provide your analysis in the following format for each statement:
Line: [Line number of the executed statement WITHIN execution steps]
Analysis: [Step-by-step explanation of the execution progress in a single paragraph without Enumeration and Enumeration Symbols]
Check: [Values of the variables in dictionary format based on the analysis or Error Type without any extra words]
Next statement:[The line number of the statement to be executed next or the label 'completion' if the entire program ends after the statement execution or label 'error' if  this statement encounters an error]"""
    return prompt

def extract_result_excep(result):
    pattern = r"Line:\s*(\d+)\s*Analysis:[\s\S]*?Check:\s*(.*?)\s*Next statement:\s*(\w+)"
    matches = re.findall(pattern, result, re.DOTALL)
    wrong_tag = None
    if matches == []:
        wrong_tag = 'extrace result error'
    for match in matches[:1]:
        if '{' in match[1]:
            try:
                cur_states = ast.literal_eval(match[1])
            except:
                wrong_tag = 'extrace result error'
        else:
            cur_states = match[1]

        if match[2] == 'completion' or match[2] == 'Completion':
            cur_line_num = 'completion'
        elif match[2] == 'error':
            cur_line_num = 'error'
        else:
            try:
                cur_line_num = int(match[2])
            except:
                wrong_tag = 'extrace result error'
    if wrong_tag == 'extrace result error' or wrong_tag =='match error: extrace_result':
        return -1, -1, wrong_tag

    return cur_line_num, cur_states, None


def read_params(cur_line, stack, func_name,funcs):
    match = re.search(rf'{func_name}\((.*?)\)', cur_line)
    func_state = ''
    wrong_tag = None
    if match:
        try:
            real_param_str = match.group(1)
            func_state_replace = f'{func_name}({real_param_str})'

            program_states = stack[-1]["program_states"]
            exec_states = program_states.copy()

            # cur_states = {}
            func_parms = funcs[func_name]['params']
            param_code_str = ','.join(func_parms) + '=' +real_param_str

            exec(param_code_str,{},exec_states)

            cur_states = {}
            for param in func_parms:
                cur_states[param]=exec_states[param]

            return cur_states, func_state_replace, wrong_tag
        except:
            wrong_tag = "read param exec error"
            return -1 ,-1,wrong_tag
    else:
        wrong_tag = 'read params match error'
        return -1, -1, wrong_tag

def find_first_matching_function(code: str, func_list: list):
    code = code.strip()

    wrong_flg = True
    wrong_info = None
    try:
        tree = ast.parse(code)
        wrong_flg = True
    except Exception as e:
        wrong_flg = False
        wrong_info = e

    if wrong_flg==False:
        if code.endswith(":"):
            code+='pass'
        if 'elif' in code:
            code = code.replace('elif','if')
        if code.startswith("else"):
            code = "if True:"+code[4:]
        try:
            tree = ast.parse(code)
            wrong_flg=True
        except Exception as e:
            wrong_flg = False
            wrong_info = e

    if wrong_flg==True:

        func_names = []
        # 遍历 AST 查找函数调用
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # 获取函数调用的名字
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id  # 普通函数名
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr  # 对象方法名
                func_names.append(func_name)
                # 如果函数名在给定的函数名列表中，返回第一个匹配的函数
        for fn in func_names[::-1]:
            if fn in func_list:
                return fn,wrong_flg,None

        return None,wrong_flg,None
    else:
        return None,wrong_flg,wrong_info

def program_execute(item ,executor,tokenizer,args):
    # trace
    trace = []

    code = item['code']
    codes = code.split('\n')
    formatted_code = '\n'.join(add_linenum(code))

    cur_line_num = len(codes)-2
    cur_line = codes[cur_line_num]
    trace.append({'line': 0, 'program_states': {}})
    trace.append({'line': cur_line_num})

    cur_states = {}

    # stack
    stack = [{'line': cur_line_num, 'program_states': cur_states.copy()}]

    entry_func_name = item['entry_point']
    funcs = item['func_info']

    error = None
    error_info = None

    output = None

    # start execute
    while 'completion' != cur_line_num and len(trace)<item['ans_len']+10:

        func_tag = False
        func_name,wrong_tag,wrong_info = find_first_matching_function(cur_line,funcs.keys())
        if wrong_tag==False:
            error = "find_first_matching_function error"
            error_info = {'cur_line':cur_line,'funcs':funcs.keys(),'wrong_info':wrong_info}
            break

        if func_name:
            func_tag=True
            func_line_num = funcs[func_name]['func_line']-1

        if func_tag:
            # update trace
            trace[-1]['program_states'] = cur_states.copy()
            trace.append({'line': func_line_num})

            cur_states, func_state, wrong_tag = read_params(cur_line, stack, func_name,funcs)

            if wrong_tag:
                error = wrong_tag
                error_info = {'cur_line': cur_line, 'stack': stack, 'func_name': func_name,"funcs":funcs}
                break

            trace[-1]['program_states'] = cur_states.copy()
            trace.append({'line': func_line_num})

            stack[-1]['func_state'] = func_state

            cur_line_num = func_line_num
            cur_line = codes[cur_line_num]

            stack.append({'line': cur_line_num, 'program_states': cur_states.copy()})

        elif 'return ' in cur_line:
            try:
                trace[-1]['program_states'] = cur_states.copy()
                trace.append({'line': cur_line_num})

                return_state = cur_line.replace('return ', '').strip()
                return_result = eval(return_state, {}, stack[-1]['program_states'])

                stack.pop()
                cur_line_num = stack[-1]["line"]
                cur_line = codes[cur_line_num]
                func_state = stack[-1]['func_state']

                cur_line = cur_line.replace(func_state, str(return_result))
                codes[cur_line_num] = cur_line
                code = '\n'.join(codes)
                formatted_code = '\n'.join(add_linenum(code))

                cur_states = stack[-1]["program_states"]
                trace[-1]['program_states'] = cur_states.copy()
            except Exception as e:
                error = 'elif return error'
                error_info = e
                break
        else:
            prompt = gen_prompt_excep(formatted_code, cur_line_num, cur_line, cur_states)

            response = one_step_execute(prompt, executor, tokenizer, args)
            cur_line_num, cur_states, wrong_tag = extract_result_excep(response)

            if wrong_tag:
                error = wrong_tag
                error_info = {'result': response}
                break

            if 'completion' == cur_line_num:
                output = {'cur_line':cur_line,'code':'\n'.join(codes)}

                trace[-1]['program_states'] = cur_states.copy()
                trace.append({'line': cur_line_num})
                break

            cur_line_num = cur_line_num - 1

            print(f"    Next Line:{cur_line_num}")
            print(f"    Program States:{cur_states}")

            trace[-1]['program_states'] = cur_states.copy()
            trace.append({'line': cur_line_num})

            stack[-1]["program_states"] = cur_states.copy()
            stack[-1]['line'] = cur_line_num


            cur_line = codes[cur_line_num]


    if cur_line_num=='completion' and error==None:
        finished=True
        return trace, finished, error,error_info,output
    elif cur_line_num!='completion' and error==None and len(trace)>=item['ans_len']+5:
        finished=False
        error = 'len(trace) >= ans_len'
        error_info = 'len(trace) >= ans_len'
        return trace,finished,error,error_info,output
    else:
        finished=False
        return trace, finished, error,error_info,output

def calculate_fp_tn(items):
    
    num = 0
    num_completion = 0
    for item in items:
        true_trace_len = item.get('true_trace_len', 1)
        execute_correct = item.get('execute_correct', 0)
        if execute_correct >= true_trace_len:
            num_completion += 1
        num += 1
    fp=num-num_completion
    tn=num_completion

    return  fp, tn

def calculate_tp_fn(items):
    num = 0
    num_true = 0
    for item in items:
        agent_trace=item['agent_trace']
        true_trace_len = item.get('true_trace_len', 1)
        execute_correct = item.get('execute_correct', 0)
        if agent_trace[-1]['line']=='error' and agent_trace[-2]['line']==item['error_line']-1 and item['error_type'] in agent_trace[-2]['program_states']:
            if execute_correct>=true_trace_len:
                num_true+=1
        num += 1
    fn = num - num_true
    tp = num_true

    return tp, fn






