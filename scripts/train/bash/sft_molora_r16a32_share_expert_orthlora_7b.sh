OSS_PATH=/mnt/petrelfs/liaoyusheng/oss
TASK_PATH=/mnt/petrelfs/liaoyusheng/projects/MING/datas
# TRAINING_DATA=cblue_16k
TRAINING_DATA=ming-moe-clinical-2stage_30k
# TRAINING_DATA=clinical_16k

# MODEL_BASE=/mnt/petrelfs/jiangshuyang.p/models/models--Qwen--Qwen1.5-1.8B-Chat
MODEL_BASE=${OSS_PATH}/my_models/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_fix_taia

CKPT=qwen1.5-7b-molora-r16a32_share_expert_2_orthlora_freeze_base_w_attn_3epoch_0001
SAVE_PATH=${OSS_PATH}/checkpoints/${TRAINING_DATA}-${CKPT}
LOGS_BASE_PATH=logs/${TRAINING_DATA}
LORA_PATH=${OSS_PATH}/checkpoints/ming-moe-clinical-v2-qwen1.5-7b-molora-r16a32_share_expert_2_fix

mkdir -p ${LOGS_BASE_PATH}
mkdir -p ${LOGS_BASE_PATH}/${CKPT}
sbatch -o ${LOGS_BASE_PATH}/${CKPT}/train.log scripts/train/srun/sft_molora_r16a32_share_orthlora_freeze_base_w_attn.sh $TASK_PATH $TRAINING_DATA $MODEL_BASE $SAVE_PATH $LOGS_BASE_PATH $CKPT $LORA_PATH & sleep 1