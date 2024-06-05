OSS_PATH=/mnt/hwfile/medai/jiangshuyang.p
TASK_PATH=/mnt/hwfile/medai/liaoyusheng/datas
domains=("mmedbench_zh_cot" "medqa_mainland_cot")
domains=("CMB_cot" "CMExam_cot" "cmmlu_cot")
# domains=("CBLUE" "cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
# domains=("CMB_cot" "CMExam_cot")
# domains=("CBLUE")

MODEL_BASE=/mnt/petrelfs/jiangshuyang.p/models/models--Qwen--Qwen1.5-7B-Chat
# MODEL_BASE=/mnt/petrelfs/jiangshuyang.p/models/llama2_7b_chat
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/Qwen1.5-14B-Chat
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/llama-2/Llama-2-13b-chat-hf
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/chatglm3-6b
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/chatglm2-6b
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/Baichuan2-7B-Chat
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/Baichuan2-13B-Chat
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/HuatuoGPT2-7B
# MODEL_BASE=/mnt/hwfile/medai/LLMModels/Model/HuatuoGPT2-13B
# MODEL_BASE=/mnt/petrelfs/jiangshuyang.p/models/Meta-Llama-3-8B-Instruct
# MODEL_BASE=/mnt/hwfile/medai/jiangshuyang.p/checkpoints/ming-moe-clinical-v2-qwen1.5-1.8b-molora-r16a32_share_expert_2_mergelora
# MODEL_BASE=/mnt/hwfile/medai/jiangshuyang.p/checkpoints/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_mergelora
TRAINING_DATA=base_model
# TRAINING_DATA=cblue
LOGS_BASE_PATH=./logs/${TRAINING_DATA}

CKPT=qwen1.5_7b
# CKPT=qwen1.5_14b
# CKPT=llama2_13b
# CKPT=llama3_8b
# CKPT=chatglm3_6b
# CKPT=chatglm2_6b
# CKPT=baichuan2_7b
# CKPT=baichuan2_13b
# CKPT=huatuogpt2_7b
# CKPT=huatuogpt2_13b
# CKPT=qwen1.5-7b-molora-r16a32_share_expert_2_orthlora_2epoch


# LORA_PATH=${OSS_PATH}/checkpoints/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_fix

for domain in "${domains[@]}"; do
    # sbatch ./scripts/eval/srun/eval_parallel_peft_batch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain $LORA_PATH & sleep 1
    sbatch ./scripts/eval/srun/eval_parallel.sh $TASK_PATH $MODEL_BASE $CKPT $LOGS_BASE_PATH $domain & sleep 1
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


