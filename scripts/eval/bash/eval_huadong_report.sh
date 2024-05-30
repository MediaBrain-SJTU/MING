# OSS_PATH=~/oss
# TASK_PATH=datas
# TRAINING_DATA=base
# # domains=("CBLUE" "MedQA_cot" "CMB_cot" "CMExam_cot" "mmlu_cot" "cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
# # domains=("huadong_report")
# domains=("CBLUE" "cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")

# LOGS_BASE_PATH=./logs/${TRAINING_DATA}
# # MODEL_BASE=/mnt/hwfile/medai/liaoyusheng/moe_saves/qwen1.5-moe-v2.1/full/sft/checkpoint-18000
# # MODEL_BASE=/mnt/petrelfs/liaoyusheng/download_models/Qwen1.5-MoE-A2.7B-Chat
# MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/download_models/Qwen1.5-1.8B-Chat

# for domain in "${domains[@]}"; do
#     CKPT="qwen1.5-1.8b"
#     sbatch ./scripts/eval/srun/eval_parallel_vllm.sh $TASK_PATH $MODEL_BASE $CKPT $LOGS_BASE_PATH $domain & sleep 1
# done

# sleep 10800

OSS_PATH=~/oss
TASK_PATH=datas
# domains=("CBLUE" "CMB_cot" "CMExam_cot" "cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
# domains=("cmmlu_cot" "ceval_cot" "PLE_Pharmacy_cot" "PLE_TCM_cot")
domains=("huadong_report_diagnosis")


MODEL_BASE=/mnt/petrelfs/liaoyusheng/oss/download_models/Qwen1.5-7B-Chat
TRAINING_DATA=ming-moe-clinical
CKPT="qwen1.5-7b-molora-r16a32"
LOGS_BASE_PATH=./logs/${TRAINING_DATA}
MODEL_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/${TRAINING_DATA}-${CKPT}

for domain in "${domains[@]}"; do
    sbatch ./scripts/eval/srun/eval_parallel_peft.sh $TASK_PATH $MODEL_BASE $MODEL_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
done




