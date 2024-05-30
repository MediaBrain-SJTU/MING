OSS_PATH=/mnt/petrelfs/liaoyusheng/oss
TASK_PATH=./datas/
TRAINING_DATA=ming-moe-clinical-v2

MODEL_BASE=${OSS_PATH}/download_models/Qwen1.5-1.8B-Chat

CKPT=qwen1.5-1.8b-molora-r16a32_share_expert_4_fix
SAVE_PATH=${OSS_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}
sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/train/srun/sft_molora_r16a32_share_expert_4.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $LOGS_BASE_PATH $CKPT & sleep 1