OSS_PATH=/mnt/hwfile/medai/jiangshuyang.p
TASK_PATH=/mnt/hwfile/medai/liaoyusheng/datas
domains=("mmedbench_zh_cot" "medqa_mainland_cot")
# domains=("CBLUE" "cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
# domains=("CMB_cot" "CMExam_cot")
# domains=("CBLUE")


TRAINING_DATA=api_model
# TRAINING_DATA=cblue
LOGS_BASE_PATH=./logs/${TRAINING_DATA}

CKPT=chatgpt
# CKPT=baichuan2_7b
# CKPT=qwen1.5-7b-molora-r16a32_share_expert_2_orthlora_2epoch


# LORA_PATH=${OSS_PATH}/checkpoints/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_fix

for domain in "${domains[@]}"; do
    # sbatch ./scripts/eval/srun/eval_parallel_peft_batch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain $LORA_PATH & sleep 1
    sbatch ./scripts/eval/srun/eval_api.sh $TASK_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
    # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_attn.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
    # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_share.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
    # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_no_share.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
done

# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_4_fix
# MODEL_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/${TRAINING_DATA}-${CKPT}

# for domain in "${domains[@]}"; do
#     sbatch ./scripts/eval/srun/eval_parallel_peft_batch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_attn.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_share.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_no_share.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
# done

# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_2_orthlora
# MODEL_PATH=/mnt/hwfile/medai/jiangshuyang.p/checkpoints/${TRAINING_DATA}-${CKPT}

# for domain in "${domains[@]}"; do
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain $LORA_PATH & sleep 1
#     sbatch ./scripts/eval/srun/eval_parallel_peft_batch_switch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain $LORA_PATH & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_attn.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_share.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_no_share.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
# done

# MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/download_models/Qwen1.5-7B-Chat
# LOGS_BASE_PATH=./logs/base
# CKPT=qwen1.5-7b

# for domain in "${domains[@]}"; do
#     sbatch ./scripts/eval/srun/eval_parallel_vllm.sh $TASK_PATH $MODEL_BASE $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_share.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_attn.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
# done


