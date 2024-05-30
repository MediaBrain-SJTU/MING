# 设置文件路径
# file_path="/mnt/petrelfs/liaoyusheng/oss/checkpoints/ming-moe-clinical-qwen1.5-1.8b-lora-r16a32/adapter_model.safetensors"

# # 无限循环直到文件存在
# while true; do
#     if [ -f "$file_path" ]; then
#         echo "文件已找到: $file_path"
#         break  # 如果文件存在，跳出循环
#     else
#         echo "文件不存在，等待10分钟..."
#         sleep 300  # 等待10分钟 (600秒)
#     fi
# done

OSS_PATH=/mnt/petrelfs/liaoyusheng/oss
TASK_PATH=./datas/
TRAINING_DATA=ming-moe-clinical-v2

MODEL_BASE=${OSS_PATH}/download_models/Qwen1.5-1.8B-Chat

CKPT=qwen1.5-1.8b-molora-r16a32-load_900pt_lora-wo_lbl_loss-infer2
SAVE_PATH=${OSS_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}
LORA_PATH=/mnt/petrelfs/liaoyusheng/oss/checkpoints/ming-moe-nlp-v2-qwen1.5-1.8b-lora-r16a32/checkpoint-900

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}
sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/train/srun/sft_molora_r16a32_load_lora_infer2.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $LOGS_BASE_PATH $CKPT $LORA_PATH & sleep 1