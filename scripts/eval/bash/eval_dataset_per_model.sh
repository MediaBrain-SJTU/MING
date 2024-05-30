OSS_PATH=~/oss
TASK_PATH=datas
# domains=("CBLUE" "CMB_cot" "CMExam_cot" "cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
# domains=("cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
domain="CBLUE"
# domains=("PLE_TCM_cot" "PLE_Pharmacy_cot")

# MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/download_models/Qwen1.5-1.8B-Chat
# CKPT=qwen1.5-1.8b
# LOGS_BASE_PATH=logs/base
# sbatch ./scripts/eval/srun/eval_parallel_vllm.sh $TASK_PATH $MODEL_BASE $CKPT $LOGS_BASE_PATH $domain & sleep 1

TRAINING_DATA=ming-moe-clinical
CKPT=qwen1.5-1.8b-lora-r16a32
LOGS_BASE_PATH=logs/${TRAINING_DATA}
MODEL_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/${TRAINING_DATA}-${CKPT}
MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/download_models/Qwen1.5-1.8B-Chat
# sbatch ./scripts/eval/srun/eval_parallel_peft_batch.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1

# TRAINING_DATA=ming-moe-clinical
# CKPT=qwen1.5-1.8b-molora-r16a32
# LOGS_BASE_PATH=./logs/${TRAINING_DATA}
# MODEL_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/${TRAINING_DATA}-${CKPT}
sbatch ./scripts/eval/srun/eval_parallel_peft_batch_only_attn.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1


