#!/bin/bash
#SBATCH -J eval_qwen
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1


TASK_PATH="$1"
CKPT="$2" # 使用实际的检查点名称替换CHECKPOINT_NAME
LOGS_BASE_PATH="$3"
DATASET="$4"

DATA_PATH=${TASK_PATH}/test
mkdir -p ${LOGS_BASE_PATH}/${CKPT}/${DATASET}

if [[ $CKPT == *"chatgpt"* ]]; then
    api_name=chatgpt
fi 

echo "Conv mode: ${conv_mode}"
echo "Processing ${DATASET}"
srun -p medai_llm --quotatype=auto --output="${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.log" python -m ming.eval.model_diverse_gen_batch_api \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --temperature 0 \
    --api-name $api_name \
    --resume 

echo "Evaluating ${DATASET}"
srun -p medai_llm --quotatype=auto --output="${LOGS_BASE_PATH}/${CKPT}/${DATASET}/eval.log" python -m ming.eval.eval_em \
    --input_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --output_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/wrong.jsonl