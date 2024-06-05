OSS_PATH=~/oss
TASK_PATH=datas
domain="CBLUE"

NUM_CHUNK=4

# TRAINING_DATA=cblue_16k
TRAINING_DATA=ming-moe-clinical-2stage_30k
CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_2_orthlora_freeze_base_w_attn_2epoch_lambda0001
LOGS_BASE_PATH=logs/${TRAINING_DATA}
MODEL_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/${TRAINING_DATA}-${CKPT}
MODEL_BASE=${OSS_PATH}/download_models/Qwen1.5-1.8B-Chat
# MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/my_models/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_fix_taia
LORA_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/ming-moe-clinical-v2-qwen1.5-1.8b-molora-r16a32_share_expert_2_fix

for ((CHUNK_ID=0; CHUNK_ID<${NUM_CHUNK}; CHUNK_ID++)); do
  sbatch ./scripts/eval/srun/eval_parallel_peft_batch_chunk_w_lora_init.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain ${LORA_PATH} ${NUM_CHUNK} ${CHUNK_ID} & sleep 1
done

wait

output_file=${LOGS_BASE_PATH}/${CKPT}/${domain}/infer.jsonl
> "$output_file"

# Loop through the indices and concatenate each file.
for CHUNK_ID in $(seq 0 $((NUM_CHUNK-1))); do
    cat ${LOGS_BASE_PATH}/${CKPT}/${domain}/${NUM_CHUNK}_${CHUNK_ID}_infer.jsonl >> "$output_file"
done


echo "Evaluating ${domain}"
srun -p medai_llm --quotatype=auto --output="${LOGS_BASE_PATH}/${CKPT}/${domain}/eval.log" python -m ming.eval.eval_em \
    --input_file ${LOGS_BASE_PATH}/${CKPT}/${domain}/infer.jsonl \
    --output_file ${LOGS_BASE_PATH}/${CKPT}/${domain}/wrong.jsonl


# TRAINING_DATA=ming-moe-clinical
# CKPT=qwen1.5-1.8b-molora-r16a32
# LOGS_BASE_PATH=./logs/${TRAINING_DATA}
# MODEL_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/${TRAINING_DATA}-${CKPT}
# sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_attn.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1


