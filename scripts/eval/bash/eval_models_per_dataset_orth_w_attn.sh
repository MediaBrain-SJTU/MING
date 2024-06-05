OSS_PATH=~/oss
TASK_PATH=datas
domains=("cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
# domains=("clinical-test-200-v2")
# # domains=("CBLUE")
# domains=('image_analysis_new')
# TRAINING_DATA=cblue_16k
# TRAINING_DATA=ming-moe-clinical-2stage_30k
TRAINING_DATA=clinical_16k
CKPT=qwen1.5-7b-molora-r16a32_share_expert_2_orthlora_freeze_base_w_attn_3epoch_0001
LOGS_BASE_PATH=logs/${TRAINING_DATA}
MODEL_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/${TRAINING_DATA}-${CKPT}
# MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/my_models/ming-moe-clinical-v2-qwen1.5-1.8b-molora-r16a32_share_expert_2_fix_taia
MODEL_BASE=${OSS_PATH}/download_models/Qwen1.5-7B-Chat
LORA_PATH=${OSS_PATH}/checkpoints/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_fix

for domain in "${domains[@]}"; do
    sbatch ./scripts/eval/srun/eval_parallel_peft_batch_w_lora_init.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain $LORA_PATH & sleep 1
    # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_switch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain $LORA_PATH & sleep 1
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


# MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/download_models/Qwen1.5-7B-Chat
# LOGS_BASE_PATH=./logs/base
# CKPT=qwen1.5-7b

# for domain in "${domains[@]}"; do
#     sbatch ./scripts/eval/srun/eval_parallel_vllm.sh $TASK_PATH $MODEL_BASE $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_share.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
#     # sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_attn.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
# done


