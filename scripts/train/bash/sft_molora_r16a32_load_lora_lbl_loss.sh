OSS_PATH=/mnt/petrelfs/liaoyusheng/oss
TASK_PATH=./datas/
TRAINING_DATA=ming-moe-clinical

MODEL_BASE=${OSS_PATH}/download_models/Qwen1.5-1.8B-Chat

CKPT=qwen1.5-1.8b-molora-r16a32-load_lora_lbl_loss_new
SAVE_PATH=${OSS_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}
LORA_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/ming-moe-nlp-qwen1.5-1.8b-lora-r16a32

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}
sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/train/srun/sft_molora_r16a32_load_lora_lbl_loss.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $LOGS_BASE_PATH $CKPT $LORA_PATH & sleep 1