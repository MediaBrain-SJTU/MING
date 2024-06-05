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
CKPT="$3" # 使用实际的检查点名称替换CHECKPOINT_NAME
LOGS_BASE_PATH="$4"
DATASET="$5"

DATA_PATH=${TASK_PATH}/test
mkdir -p ${LOGS_BASE_PATH}/${CKPT}/${DATASET}

if [[ "$CKPT" =~ "llama2" ]]; then
    CONV_MODE=llama2
elif [[ "$CKPT" =~ "llama3" ]]; then
    CONV_MODE=llama3
elif [[ "$CKPT" =~ "qwen1.5" ]]; then
    CONV_MODE=qwen
elif [[ "$CKPT" =~ "chatglm2" ]]; then
    CONV_MODE=chatglm2
elif [[ "$CKPT" =~ "chatglm3" ]]; then
    CONV_MODE=chatglm3
elif [[ "$CKPT" =~ "baichuan2" ]]; then
    CONV_MODE=baichuan2
elif [[ "$CKPT" =~ "huatuo2" ]]; then
    CONV_MODE=huatuo2
else
    CONV_MODE=qwen
fi

echo "Processing ${DATASET}"
echo "Use conv-mode as ${CONV_MODE}"
srun -p medai_llm --quotatype=spot --gres=gpu:1 --output="${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.log" python -m ming.eval.model_gen_vllm_batch \
    --model-path ${MODEL_BASE} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --temperature 0 \
    --conv-mode ${CONV_MODE} \
    --resume 

echo "Evaluating ${DATASET}"
srun -p medai_llm --quotatype=spot --output="${LOGS_BASE_PATH}/${CKPT}/${DATASET}/eval.log" python -m ming.eval.eval_em \
    --input_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --output_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/wrong.jsonl