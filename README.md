# T-REX
This repo provides the code for reproducing the experiments in T-REX, which teaching Large Language Models to reason about program execution
## prepare environments
```
conda create -n llmexecutor python=3.9
conda activate llmexecutor
cd T-REX
pip install -r requirements.txt
```
## prepare data
Please download the dataset from [here](https://drive.google.com/drive/folders/1mdfxpxl_PNjpo_cbHBQlQ7tQLmfJW55W?usp=drive_link) and place the `data` folder inside the `T-REX` folder 

**Table 1.** Statistics of statement and exception types in the training data.

| **Statement Types** | **%** | **Exception Types** | **%** |
|---|---|---|---|
| Variable Assignment (S₂) | 32.70% | TypeError | 16.50% |
| Expression (S₁) | 23.80% | ZeroDivisionError | 16.21% |
| If Statement (S₃) | 18.70% | NameError | 15.99% |
| For/While Loop (S₄) | 16.93% | IndexError | 15.42% |
| Method Calls (S₅) | 7.87% | KeyError | 14.94% |
| | | ValueError | 13.16% |
| | | UnboundLocalError | 7.78% |
## Training
We have fine-tuned: `meta-llama/CodeLlama-7b-Instruct-hf`, `meta-llama/CodeLlama-13b-Instruct-hf`, `Qwen/Qwen2.5-Coder-7B-Instruct`, `Qwen/Qwen2.5-Coder-14B-Instruct`
### SFT
#### CodeLlama:

```
cd train/Executor
# process data
python process_data_codellama.py --data_path ./../../data/train/Executor/sft.jsonl --save_path ./../../data/train/Executor/dataset_codellama
# CodeLlama-7b
python train_codellama.py --output_dir ./../../fine_tuned_models/codellama_7b_sft --config_name meta-llama/CodeLlama-7b-Instruct-hf --tokenizer_name meta-llama/CodeLlama-7b-Instruct-hf --model_name_or_path meta-llama/CodeLlama-7b-Instruct-hf --max_target_length 1024 --max_source_length 1024 --pad_to_max_length true --do_train true --learning_rate 1e-5 --lr_scheduler_type cosine --logging_steps 2 --num_train_epochs 3 --save_steps 1000 --per_device_train_batch_size 3 --overwrite_output_dir false --train_data ./../../data/train/Executor/dataset_codellama
# CodeLlama-13b
python train_codellama.py --output_dir ./../../fine_tuned_models/codellama_13b_sft --config_name meta-llama/CodeLlama-13b-Instruct-hf --tokenizer_name meta-llama/CodeLlama-13b-Instruct-hf --model_name_or_path meta-llama/CodeLlama-13b-Instruct-hf --max_target_length 1024 --max_source_length 1024 --pad_to_max_length true --do_train true --learning_rate 1e-5 --lr_scheduler_type cosine --logging_steps 2 --num_train_epochs 3 --save_steps 1000 --per_device_train_batch_size 3 --overwrite_output_dir false --train_data ./../../data/train/Executor/dataset_codellama
```

#### Qwen2.5-Coder:
```
# process data
python process_data_qwen.py --data_path ./../../data/train/Executor/sft.jsonl --save_path ./../../data/train/Executor/sft_formated_qwen.jsonl
python binarize_data.py --input_path ./../../data/train/Executor/sft_formated_qwen.jsonl --output_path ./../../data/train/Executor/sft_processed_qwen.jsonl
# Qwen2.5-Coder-7b
python train.py     --model_name_or_path  Qwen/Qwen2.5-Coder-7B-Instruct    --data_path ./../../data/train/Executor/sft_processed_qwen.jsonl.npy     --model_max_length 1280     --output_dir ./../../fine_tuned_models/qwen_7b_sft     --num_train_epochs 5     --per_device_train_batch_size 1    --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 50     --save_total_limit 1000    --learning_rate 1e-5    --weight_decay 0.0    --warmup_steps 100    --lr_scheduler_type "cosine"     --logging_strategy "steps"    --logging_steps 1     --report_to "tensorboard"     --bf16 False    --tf32 False     --fp16 True     --truncate_source True
# Qwen2.5-Coder-14b
python train.py     --model_name_or_path  Qwen/Qwen2.5-Coder-14B-Instruct    --data_path ./../../data/train/Executor/sft_processed_qwen.jsonl.npy     --model_max_length 1280     --output_dir ./../../fine_tuned_models/qwen_14b_sft     --num_train_epochs 5     --per_device_train_batch_size 1    --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 50     --save_total_limit 1000    --learning_rate 1e-5    --weight_decay 0.0    --warmup_steps 100    --lr_scheduler_type "cosine"     --logging_strategy "steps"    --logging_steps 1     --report_to "tensorboard"     --bf16 False    --tf32 False     --fp16 True     --truncate_source True 
```
## RQ1 Predicting execution semantics
### commend
```bash
for model in "codellama_7b" "codellama_13b" "qwen_7b" "qwen_14b"; do
  for dataset in "codenetmut" "humaneval"; do
    python run_executor.py   --executor_model_path "./../../fine_tuned_models/${model}_sft/checkpoint_xx"   --results_path "./../../results/ASE/${model}_sft_${dataset}.jsonl"   --data_path "./../../data/test/ASE/${dataset}.jsonl"
    python calculate_ASE.py --results_path "./../../results/ASE/${model}_sft_${dataset}.jsonl"
    python calculate_NS_PS.py --results_path "./../../results/ASE/${model}_sft_${dataset}.jsonl"
    python calculate_stratify_results.py --results_path "./../../results/ASE/${model}_sft_${dataset}.jsonl"
  done
done
```
### results
<img width="561" height="227" alt="image" src="https://github.com/user-attachments/assets/9cb69b27-c3ac-4a55-b33f-f6c648e38da8" />
<img width="561" height="189" alt="image" src="https://github.com/user-attachments/assets/1379e3ae-6cc8-469e-af33-1209bfd02cdb" />


## RQ2 and RQ5 Predicting runtime behaviors 
### commend
```bash
  for model in "codellama_7b" "codellama_13b" "qwen_7b" "qwen_14b"; do
    for dataset in "codenetmut" "humaneval"; do
      python SIPA.py   --executor_model_path "./../../fine_tuned_models/${model}_sft/checkpoint_xx"   --results_path "./../../results/PM_CRMs/${model}_sft_${dataset}"   --data_path "./../../data/test/PM_CRMs/${dataset}.jsonl"   --variant "sft"
      python calculate_CRMs.py   --result_path "./../../results/PM_CRMs/${model}_sft_${dataset}/results.jsonl"
    done
  done
```
### results
<img width="561" height="174" alt="image" src="https://github.com/user-attachments/assets/552b4c3b-3a6d-45df-9bf3-129235398b92" />
<img width="561" height="228" alt="image" src="https://github.com/user-attachments/assets/6b8b4233-4582-4cd1-b6cf-cdd267201070" />
<img width="561" height="228" alt="image" src="https://github.com/user-attachments/assets/30f9069f-54ee-4d00-938d-73c429e46b6c" />


## RQ3 Static Detection of Runtime Errors
### commend
```bash
cd Excep_dect
python exception_dect.py   --executor_model_path "./../../fine_tuned_models/qwen_14b_sft/checkpoint_xx"   --excep_data "./../../data/test/Excep_dect/excep.jsonl" --n_excep_data "./../../data/test/Excep_dect/n_excep.jsonl"
```
### results
<img width="286" height="141" alt="image" src="https://github.com/user-attachments/assets/2faadc7a-fcf2-4ab8-9d7b-37986660483d" />

## RQ4 Aiding Debugging
### commend
```bash
python SIPA.py   --executor_model_path "./../../fine_tuned_models/qwen_14b_sft/checkpoint_xx"   --results_path "./../../results/Bug_dect/qwen_14b_sft_buggy"   --data_path "./../../data/test/Bug_dect/buggy.jsonl"   --variant "sft"
```
### results
<img width="286" height="114" alt="image" src="https://github.com/user-attachments/assets/2d34629e-92ee-4ccd-bfa2-df35f2eb4b84" />
