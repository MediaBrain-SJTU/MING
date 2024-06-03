#!/bin/bash
#SBATCH -J eval_qwen
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1
bash ~/add_oss.sh

TASK_PATH="$1"
MODEL_BASE="$2"
MODEL_PATH="$3"
CKPT="$4" # 使用实际的检查点名称替换CHECKPOINT_NAME
LOGS_BASE_PATH="$5"
DATASET="$6"
NUM_CHUNK="$7"
CHUNK_ID="$8"

DATA_PATH=${TASK_PATH}/test
mkdir -p ${LOGS_BASE_PATH}/${CKPT}/${DATASET}


echo "Processing ${DATASET}"
srun -p medai_llm --quotatype=auto --gres=gpu:1 --output="${LOGS_BASE_PATH}/${CKPT}/${DATASET}/${NUM_CHUNK}_${CHUNK_ID}_infer.log" python -m ming.eval.model_diverse_gen_batch \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/${NUM_CHUNK}_${CHUNK_ID}_infer.jsonl \
    --temperature 0 \
    --num-chunks ${NUM_CHUNK} \
    --chunk-idx ${CHUNK_ID} \
    --conv-mode qwen \
    --resume 