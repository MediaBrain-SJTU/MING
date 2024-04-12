#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

NODE_NUM=${#GPULIST[@]}
CHUNKS_PER_NODE=2

CHUNKS=$((NODE_NUM * CHUNKS_PER_NODE))

LOG_DIR="$1"
domain="$2"
metric="$3"


echo "Processing ${domain}"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$((IDX % NODE_NUM))]} python -m ming.eval.get_attention \
        --model_base /home/cs/yangyuchen/yushengliao/Medical_LLM/download_models/models--Qwen--Qwen1.5-1.8B-Chat \
        --model_path /home/cs/yangyuchen/yushengliao/Medical_LLM/Medical_MOE/checkpoints/xsum-qwen1.5-1.8b-molora-4x1-r4a32-topk-fixbug \
        --load_molora \
        --input_file ${LOG_DIR}/${domain}/merge.jsonl \
        --output_file ${LOG_DIR}/attentions/${domain}/${CHUNKS}_${IDX}.csv \
        --metric ${metric} &> ${LOG_DIR}/attentions/${domain}/${CHUNKS}_${IDX}.log & 
done

wait

output_file=${LOG_DIR}/attentions/${domain}/merge.csv

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${LOG_DIR}/attentions/${domain}/${CHUNKS}_${IDX}.csv >> "$output_file"
done

echo "Evaluating ${domain}"
# Evaluate
python -m ming.eval.eval_em \
    --input_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/merge.jsonl \
    --output_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/wrong.jsonl &> "${LOGS_BASE_PATH}/${CKPT}/${DATASET}/eval.log"