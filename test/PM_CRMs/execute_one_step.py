import json
import re,ast
import numpy as np
import pandas as pd
from transformers import  pipeline
import torch
def extract_result(result):
    out_lins = re.findall(r"Line:\s*(\d+)", result)
    if len(out_lins) > 1:
        return None
    pattern = r"Line:\s*(\d+)\s*Analysis:.*?Check:\s*({.*?})\s*Next statement:\s*(\d+|\w+)"
    matches = re.findall(pattern, result, re.DOTALL)
    if len(matches) != 1:
        return None
    current_line = int(matches[0][0])
    try:
        program_state = ast.literal_eval(matches[0][1])
    except:
        return None
    if matches[0][2] == 'completion' or matches[0][2] == 'Completion':
        next_line = 'completion'
    else:
        try:
            next_line = int(matches[0][2])
        except:
            return None
    return (current_line, program_state, next_line)

def get_most_common(extracted):
    freq = {}
    for i,e in enumerate(extracted):
        flag=False
        if i==0:
            freq[i]=[]
        else:
            for k in freq.keys():
                if extracted[k]==extracted[i]:
                    freq[k].append(i)
                    flag=True
                    break
            if not flag:
                freq[i] = []
    return freq

def predict_best_group(new_sample_df, model):
    new_sample_df = compute_relative_features(new_sample_df)
    feature_cols = ['rel_max', 'rel_mean', 'inv_std', 'rel_size', 'rel_min_sq']
    features = new_sample_df[feature_cols]

    # 使用类别1的概率作为评分
    probas = model.predict_proba(features)
    scores = probas[:, 1]  # 获取正类的概率

    new_sample_df['score'] = scores
    best_group = new_sample_df.loc[new_sample_df['score'].idxmax()]
    return best_group['group_id'], new_sample_df

def compute_relative_features(df):
    # 计算每个 sample 的全局统计
    grouped = df.groupby('sample_id')
    global_stats = grouped[['max', 'min', 'mean', 'size']].agg({
        'max': ['max', 'min'],
        'min': ['max', 'min'],
        'mean': ['max', 'min'],
        'size': 'sum'
    })
    # 重命名列
    global_stats.columns = ['global_max_max', 'global_max_min',
                            'global_min_max', 'global_min_min',
                            'global_mean_max', 'global_mean_min',
                            'global_size_sum']
    df = df.merge(global_stats, on='sample_id', how='left')

    # 计算相对特征
    df['rel_max'] = (df['max'] - df['global_max_min']) / (df['global_max_max'] - df['global_max_min'] + 1e-6)
    df['rel_min'] = (df['min'] - df['global_min_min']) / (df['global_min_max'] - df['global_min_min'] + 1e-6)
    df['rel_mean'] = (df['mean'] - df['global_mean_min']) / (df['global_mean_max'] - df['global_mean_min'] + 1e-6)
    df['rel_size'] = df['size'] / df['global_size_sum']

    # 计算 group 内的标准差（假设有多个三元组，若只有一个则 std=0）
    # 这里简化为模拟标准差，可以根据实际数据调整
    # df['std'] = np.random.uniform(1, 10, size=len(df))  # 示例，实际数据请计算
    df['inv_std'] = 1 / (df['std'] + 1e-6)

    # 计算保留特征
    df['rel_min_sq'] = df['rel_min'] ** 2

    return df

def read_file(path):
    """读取包含多个JSON对象的文件"""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def batch_generator(data, batch_size):
    """生成指定大小的数据批次"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

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


def one_step_execute_RVBS(prompt,executor,reward_model,rf_model,tokenizer,tokenizer_qwen,args):
    if 'llama' in args.executor_model_path or 'Llama' in args.executor_model_path:
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(executor.device)
    else:
    # 1.
        inputs = [
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ], add_generation_prompt=True, tokenize=False)
        ]

        # 批量编码输入
        inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(executor.device)


    # 1.
    # inputs = [
    #     tokenizer.apply_chat_template([
    #         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #         {"role": "user", "content": prompt}
    #     ], add_generation_prompt=True, tokenize=False)
    # ]
    #
    # # 批量编码输入
    # inputs = tokenizer(
    #     inputs,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True,
    #     max_length=1024
    # ).to(executor.device)

    # 生成响应
    outputs = executor.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=args.num_sequences,
    )

    responses = [
        tokenizer.decode(seq[len(inputs.input_ids[0]):], skip_special_tokens=True)
        for seq in outputs
    ]
    # 2.
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model,
        # device="auto",
        device=reward_model.device,
        tokenizer=tokenizer_qwen,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1
    }
    texts = []
    for answer in responses:
        text = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        texts.append(text)
    test_texts = [tokenizer_qwen.apply_chat_template(text, tokenize=False, add_generation_prompt=False) for text in
                  texts]
    # outputs=
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    scores = [output[0]["score"] for output in pipe_outputs]

    # 3.
    extracted = [extract_result(answer) for answer in responses]
    freq = get_most_common(extracted)
    data=[]
    for k in freq.keys():
        group_scores = []
        group_scores.append(scores[k])
        for va in freq[k]:
            group_scores.append(scores[va])
        max_val = np.max(group_scores)
        min_val = np.min(group_scores)
        mean_val = np.mean(group_scores)
        std_val = np.std(group_scores)
        size_val = np.size(group_scores)
        result = extract_result(responses[k])

        data.append([0, k, max_val, min_val, mean_val, std_val, size_val])
    sample = pd.DataFrame(data, columns=['sample_id', 'group_id', 'max', 'min', 'mean', 'std', 'size'])
    best_group_id, sample_new = predict_best_group(sample, rf_model)
    final_answer = responses[int(best_group_id)]
    return final_answer


