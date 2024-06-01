#!/bin/bash
#SBATCH -J eval_qwen
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1


TASK_PATH="$1"
MODEL_BASE="$2"
MODEL_PATH="$3"
CKPT="$4" # 使用实际的检查点名称替换CHECKPOINT_NAME
LOGS_BASE_PATH="$5"
DATASET="$6"
LORA_PATH="$7"

DATA_PATH=${TASK_PATH}/medical_test
mkdir -p ${LOGS_BASE_PATH}/${CKPT}-switch/${DATASET}


echo "Processing ${DATASET}"
srun -p medai_llm --quotatype=auto --gres=gpu:1 --output="${LOGS_BASE_PATH}/${CKPT}-switch/${DATASET}/infer.log" python -m ming.eval.model_diverse_gen_batch \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${LOGS_BASE_PATH}/${CKPT}-switch/${DATASET}/infer.jsonl \
    --temperature 0 \
    --conv-mode qwen \
    --resume \
    --lora_name_or_path ${LORA_PATH} \
    --switch-old-expert 

echo "Evaluating ${DATASET}"
srun -p medai_llm --quotatype=auto --output="${LOGS_BASE_PATH}/${CKPT}-switch/${DATASET}/eval.log" python -m ming.eval.eval_em \
    --input_file ${LOGS_BASE_PATH}/${CKPT}-switch/${DATASET}/infer.jsonl \
    --output_file ${LOGS_BASE_PATH}/${CKPT}-switch/${DATASET}/wrong.jsonl