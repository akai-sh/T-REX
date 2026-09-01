# T-REX
This repo provides the code for reproducing the main experiments in T-REX, which teaching Large Language Models to reason about program execution
## prepare environments
```
conda create -n llmexecutor python=3.9
conda activate llmexecutor
cd T-REX
pip install -r requirements.txt
```
## Prepare data

Run the following commands from the `T-REX` root directory:

```bash
pip install gdown==5.2.2
gdown --continue 1TYCVFsQLBsORvThXRI0yAJ8NR3SYpO7M -O trex-data-v1.0.tar.gz
echo "2949a4b3590e5b7dc752b4b55e72a77c5eb41418d4bad4db2ef819337f99a99b  trex-data-v1.0.tar.gz" | sha256sum -c -
tar -xzf trex-data-v1.0.tar.gz -C .
```

Alternatively, download `trex-data-v1.0.tar.gz` manually from [Google Drive](https://drive.google.com/file/d/1TYCVFsQLBsORvThXRI0yAJ8NR3SYpO7M/view?usp=drive_link), verify its SHA-256 checksum, and extract it into the `T-REX` root directory.

The extracted dataset should be located at `T-REX/data/`.

## Kick-the-Tires Test

We provide a lightweight smoke test to verify that the artifact is installed correctly and that the T-REX executor produces the expected result.

### Running the Test

Run the smoke test from the project root directory:

```bash
python test/kick_the_tires.py \
  --executor_model_path ling031001/T-REX-qwen2.5-coder-14b
```

> **Note:** The executor model will be downloaded from Hugging Face automatically if it is not already available locally.

### Test Example

The smoke test uses the following program:

```python
def f(x, a, b):
    ma = min(a, b)
    mi = max(0, a + b - x)
    return (ma, mi)

f(10, 7, 5)
```

The expected return value is:

```text
(5, 2)
```

### Expected Output

After running the kick-the-tires test, the output should look similar to the following:

```text
Loaded task_id=1623 from .../test/kick_the_tires_example.jsonl

Task code:
def f(x,a,b):
    ma  = min(a,b)
    mi = max(0,a+b-x)
    return (ma,mi)
f(10,7,5,)

Loading executor model: ling031001/T-REX-qwen2.5-coder-14b
Running T-REX executor on entry point: f

Effective executor trace:
...

Final executor output:
...

Return expression: (5, 2)
Observed output: (5, 2)
Effective trace steps: 2
Kick-the-Tires passed.
```

### Success Criteria

The key lines to verify are:

```text
Return expression: (5, 2)
Observed output: (5, 2)
Kick-the-Tires passed.
```

If these lines appear, the artifact has passed the kick-the-tires sanity test.

## Training

We fine-tuned the following models:

- `meta-llama/CodeLlama-7b-Instruct-hf`
- `meta-llama/CodeLlama-13b-Instruct-hf`
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `Qwen/Qwen2.5-Coder-14B-Instruct`

The fine-tuned checkpoints are available on Hugging Face:

- [`ling031001/T-REX-qwen2.5-coder-14b`](https://huggingface.co/ling031001/T-REX-qwen2.5-coder-14b)
- [`ling031001/T-REX-qwen2.5-coder-7b`](https://huggingface.co/ling031001/T-REX-qwen2.5-coder-7b)
- [`ling031001/T-REX-codellama-7b`](https://huggingface.co/ling031001/T-REX-codellama-7b)
- [`ling031001/T-REX-codellama-13b`](https://huggingface.co/ling031001/T-REX-codellama-13b)

You can load these checkpoints directly from Hugging Face or fine-tune the models yourself using the following command:

### CodeLlama:
```
cd train/Executor
# process data
python process_data_codellama.py --data_path ./../../data/train/sft.jsonl --save_path ./../../data/train/dataset_codellama
# CodeLlama-7b
python train_codellama.py --output_dir ./../../fine_tuned_models/codellama_7b_sft --config_name meta-llama/CodeLlama-7b-Instruct-hf --tokenizer_name meta-llama/CodeLlama-7b-Instruct-hf --model_name_or_path meta-llama/CodeLlama-7b-Instruct-hf --max_target_length 1024 --max_source_length 1024 --pad_to_max_length true --do_train true --learning_rate 1e-5 --lr_scheduler_type cosine --logging_steps 2 --num_train_epochs 3 --save_steps 1000 --per_device_train_batch_size 3 --overwrite_output_dir false --train_data ./../../data/train/dataset_codellama
# CodeLlama-13b
python train_codellama.py --output_dir ./../../fine_tuned_models/codellama_13b_sft --config_name meta-llama/CodeLlama-13b-Instruct-hf --tokenizer_name meta-llama/CodeLlama-13b-Instruct-hf --model_name_or_path meta-llama/CodeLlama-13b-Instruct-hf --max_target_length 1024 --max_source_length 1024 --pad_to_max_length true --do_train true --learning_rate 1e-5 --lr_scheduler_type cosine --logging_steps 2 --num_train_epochs 3 --save_steps 1000 --per_device_train_batch_size 3 --overwrite_output_dir false --train_data ./../../data/train/dataset_codellama
```

### Qwen2.5-Coder:
```
# process data
python process_data_qwen.py --data_path ./../../data/train/sft.jsonl --save_path ./../../data/train/sft_formated_qwen.jsonl
python binarize_data.py --input_path ./../../data/train/sft_formated_qwen.jsonl --output_path ./../../data/train/sft_processed_qwen.jsonl --tokenizer_path Qwen/Qwen2.5-Coder-14B-Instruct
# Qwen2.5-Coder-7b
python train_qwen.py     --model_name_or_path  Qwen/Qwen2.5-Coder-7B-Instruct    --data_path ./../../data/train/sft_processed_qwen.jsonl.npy     --model_max_length 1280     --output_dir ./../../fine_tuned_models/qwen_7b_sft     --num_train_epochs 5     --per_device_train_batch_size 1    --eval_strategy "no"     --save_strategy "steps"     --save_steps 50     --save_total_limit 1000    --learning_rate 1e-5    --weight_decay 0.0    --warmup_steps 100    --lr_scheduler_type "cosine"     --logging_strategy "steps"    --logging_steps 1     --report_to "none"     --bf16 False    --tf32 False     --fp16 True     --truncate_source True
# Qwen2.5-Coder-14b
python train_qwen.py     --model_name_or_path  Qwen/Qwen2.5-Coder-14B-Instruct    --data_path ./../../data/train/sft_processed_qwen.jsonl.npy     --model_max_length 1280     --output_dir ./../../fine_tuned_models/qwen_14b_sft     --num_train_epochs 5     --per_device_train_batch_size 1    --eval_strategy "no"     --save_strategy "steps"     --save_steps 50     --save_total_limit 1000    --learning_rate 1e-5    --weight_decay 0.0    --warmup_steps 100    --lr_scheduler_type "cosine"     --logging_strategy "steps"    --logging_steps 1     --report_to "none"     --bf16 False    --tf32 False     --fp16 True     --truncate_source True 
```

### Generating Execution Rationales

The following script uses the teacher model to generate execution rationales for the training data. Set the `OPENAI_API_KEY` environment variable before running it.

```bash
export OPENAI_API_KEY="your-api-key"

python train/generate_program_explanations.py \
  --input_path data/train/train.jsonl \
  --output_path data/train/train_gpt4omini.jsonl \
  --model gpt-4o-mini
```

## Evaluation Instructions 
### Training-Data Statistics (Table 1)

```bash
python train/statistics.py \
  --train_data data/train/train.jsonl \
  --train_excep_data data/train/train_excep.jsonl
```
statistics.py reports the statement-type and exception-type distributions shown in Table 1.

### Predicting Execution Semantics

The following example evaluates the Qwen2.5-Coder-14B executor on the CodeNetMut dataset.

For other models, use the corresponding checkpoint and replace `qwen_14b` in the result filename with `qwen_7b`, `codellama_7b`, or `codellama_13b`. To evaluate on HumanEval, replace `codenetmut` with `humaneval`.

```bash
cd test/execution_semantics
mkdir -p ../../results/execution_semantics

python run_executor.py \
  --executor_model_path ling031001/T-REX-qwen2.5-coder-14b \
  --results_path ../../results/execution_semantics/qwen_14b_sft_codenetmut.jsonl \
  --data_path ../../data/test/execution_semantics/codenetmut.jsonl

python calculate_execution_semantics.py \
  --result_path ../../results/execution_semantics/qwen_14b_sft_codenetmut.jsonl
```

The script reports the following metrics:

- **Table 2:** `A_NS` (next-statement accuracy), `A_PS` (program-state accuracy), and `A_NS+PS` (joint accuracy).
- **Table 3:** `S1` (expressions), `S2` (variable assignments), `Seq.` (sequential/completion flow), `S3` (if-statements), `S4` (for/while-statements), `S5` (method calls), and `Branch` (branch flow).

### Evaluating Explanation Quality

The following command evaluates the similarity between the executor-generated explanations and the reference explanations on the CodeNetMut single-step execution dataset. The result file (generated by the preceding execution-semantics) contains the reference explanation in `output` and the executor-generated explanation in `model_output`.

```bash
cd test/explanation_quality

python calculate_explanation_quality.py \
  --result_path qwen_14b_sft_codenetmut.jsonl \
  --bert_model_path FacebookAI/roberta-large \
  --device cuda:0
```

The script reports BERTScore F1, ROUGE-L, and BLEU-4. These metrics correspond to Figure 8 of the paper.

### Predicting Runtime Behaviors

The following example evaluates the Qwen2.5-Coder-14B executor on the CodeNetMut dataset.

For other models, use the corresponding checkpoint and replace `qwen_14b` in the result path with `qwen_7b`, `codellama_7b`, or `codellama_13b`. To evaluate on HumanEval, replace `codenetmut` with `humaneval`.

```bash
cd test/runtime_behaviors

python SIPA.py \
  --executor_model_path ling031001/T-REX-qwen2.5-coder-14b \
  --results_path "qwen_14b_sft_codenetmut" \
  --data_path "../../data/test/runtime_behaviors/codenetmut.jsonl" \
  --variant "sft"
```

`SIPA.py` saves the predictions to `results.jsonl` and reports the execution-trace metrics: Prefix, $A_{0.5}$, $A_{0.8}$, and $A_{1.0}$.

```bash
python calculate_runtime_behaviors.py \
  --result_path "qwen_14b_sft_codenetmut/results.jsonl"
```
`calculate_runtime_behaviors.py` reports the code-coverage metrics (P, R, F1, and $A_{EM}$) and program-output exact-match accuracy ($A_{EM}$). These metrics correspond to the columns in Table 4 of the paper.

```bash
python calculate_ic_score.py \
  --result_path "qwen_14b_sft_codenetmut/results.jsonl"
```
`calculate_ic_score.py` reports the IC Score in Figure 7 of the paper.

### Exception Detection and Fault Localization
#### Exception Detection
To rerun the model and save the predictions:
```bash
cd test/exception_detection

python exception_dect.py \
  --executor_model_path ling031001/T-REX-qwen2.5-coder-14b-excep \
  --excep_data ../../data/test/exception_detection/excep.jsonl \
  --n_excep_data ../../data/test/exception_detection/n_excep.jsonl \
  --excep_result_output result_excep.jsonl \
  --n_excep_result_output result_no_excep.jsonl
```

To calculate the metrics from saved predictions:
```bash
python exception_dect.py \
  --excep_result_path result_excep.jsonl \
  --n_excep_result_path result_no_excep.jsonl
```
`exception_dect.py` reports TP, FP, TN, and FN. These metrics correspond to the columns in Table 6 of the paper.


#### Fault Localization
To rerun the model and save the predictions:
```bash
cd test/exception_detection

python bug_dect.py \
  --buggy_data ../../data/test/exception_detection/buggy.jsonl \
  --executor_model_path ling031001/T-REX-qwen2.5-coder-14b \
  --result_output result_buggy.jsonl
```

To calculate the metric from saved predictions:
```bash
python bug_dect.py \
  --buggy_data ../../data/test/exception_detection/buggy.jsonl \
  --saved_result_path result_buggy.jsonl
```
bug_dect.py reports root-cause localization accuracy in Table 7.

### Output Prediction
The following example directly prompts the fine-tuned Qwen2.5-Coder-14B model to predict program outputs on CodeNetMut. This experiment corresponds to Table 5.

```bash
cd test/output_prediction

python run_output_prediction.py \
  --model_path ling031001/T-REX-qwen2.5-coder-14b \
  --data_path ../../data/test/output_prediction/codenetmut_test.jsonl \
  --result_path qwen_14b_sft_codenetmut.jsonl \
  --batch_size 8

python calculate_output_prediction.py \
  --result_path qwen_14b_sft_codenetmut.jsonl
```
