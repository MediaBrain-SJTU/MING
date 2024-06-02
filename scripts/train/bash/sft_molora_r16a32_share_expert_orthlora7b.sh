OSS_PATH=/mnt/hwfile/medai/jiangshuyang.p
TASK_PATH=/mnt/hwfile/medai/jiangshuyang.p/datasets
# TRAINING_DATA=ming-moe-clinical-v2
TRAINING_DATA=cblue_16k
# TRAINING_DATA=cblue

# MODEL_BASE=/mnt/petrelfs/jiangshuyang.p/models/models--Qwen--Qwen1.5-1.8B-Chat
# MODEL_BASE=/mnt/hwfile/medai/jiangshuyang.p/checkpoints/ming-moe-clinical-v2-qwen1.5-1.8b-molora-r16a32_share_expert_2_mergelora
MODEL_BASE=/mnt/hwfile/medai/jiangshuyang.p/checkpoints/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_mergelora

# CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_2_orthlora_1epoch
CKPT=qwen1.5-7b-molora-r16a32_share_expert_2_orthlora_2epoch
SAVE_PATH=${OSS_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}
# LORA_PATH=${OSS_PATH}/checkpoints/ming-moe-clinical-v2-qwen1.5-1.8b-molora-r16a32_share_expert_2_fix
LORA_PATH=${OSS_PATH}/checkpoints/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_fix

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}
sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/train/srun/sft_molora_r16a32_share_orthlora_1epoch.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $LOGS_BASE_PATH $CKPT $LORA_PATH & sleep 1