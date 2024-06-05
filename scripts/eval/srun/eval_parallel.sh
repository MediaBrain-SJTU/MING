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
CKPT="$3" # 使用实际的检查点名称替换CHECKPOINT_NAME
LOGS_BASE_PATH="$4"
DATASET="$5"

DATA_PATH=${TASK_PATH}/test
mkdir -p ${LOGS_BASE_PATH}/${CKPT}/${DATASET}

if [[ $CKPT == *"llama2"* ]]; then
    conv_mode=llama2
elif [[ $CKPT == *"llama3_8b"* ]]; then 
    conv_mode=llama3
elif [[ $CKPT == *"qwen"* ]]; then 
    conv_mode=qwen 
elif [[ $CKPT == *"chatglm3"* ]]; then 
    conv_mode=chatglm3
elif [[ $CKPT == *"chatglm2"* ]]; then 
    conv_mode=chatglm2
elif [[ $CKPT == *"baichuan2"* ]]; then 
    conv_mode=baichuan2
elif [[ $CKPT == *"huatuogpt2"* ]]; then 
    conv_mode=huatuogpt2
fi 

echo "Conv mode: ${conv_mode}"
echo "Processing ${DATASET}"
srun -p medai_llm --quotatype=auto --gres=gpu:1 --output="${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.log" python -m ming.eval.model_diverse_gen_batch \
    --model-path ${MODEL_BASE} \
    --question-file ${DATA_PATH}/${DATASET}.json \
    --answers-file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --temperature 0 \
    --conv-mode $conv_mode \
    --resume 

echo "Evaluating ${DATASET}"
srun -p medai_llm --quotatype=auto --output="${LOGS_BASE_PATH}/${CKPT}/${DATASET}/eval.log" python -m ming.eval.eval_em \
    --input_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --output_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/wrong.jsonl