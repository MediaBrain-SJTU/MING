OSS_PATH=~/oss
TASK_PATH=datas
domains=("PLE_Pharmacy_cot" "PLE_TCM_cot")
# domains=("CBLUE")
# domains=("clinical-test-200")
# domains=("image_analysis_new")

CKPT=llama2-13b
LOGS_BASE_PATH=logs/base
# MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/download_models/Qwen1.5-14B-Chat
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct/
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/llama-2/Llama-2-7b-chat-hf
MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/llama-2/Llama-2-13b-chat-hf

for domain in "${domains[@]}"; do
    sbatch ./scripts/eval/srun/eval_parallel_vllm_llama2.sh $TASK_PATH $MODEL_BASE $CKPT $LOGS_BASE_PATH $domain & sleep 1
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


