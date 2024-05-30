OSS_PATH=/mnt/petrelfs/liaoyusheng/oss
TASK_PATH=./datas/
TRAINING_DATA=ming-moe-nlp-v2

MODEL_BASE=${OSS_PATH}/download_models/Qwen1.5-14B-Chat

CKPT=qwen1.5-14b-lora-r16a32
SAVE_PATH=${OSS_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}
sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/train/srun/sft_lora_r16a32_cuda8_batch4.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $LOGS_BASE_PATH $CKPT & sleep 1

