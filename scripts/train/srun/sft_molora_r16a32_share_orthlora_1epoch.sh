#!/bin/bash
#SBATCH -J sft_clinical_qwen
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

export LOGLEVEL=INFO
export NCCL_DEBUG=ERROR
###

TASK_PATH="$1"
TRAINING_DATA="$2"
MODEL_BASE="$3"
SAVE_PATH="$4"
LOGS_BASE_PATH="$5"
CKPT="$6"
LORA_PATH="$7"

export MASTER_PORT=$((RANDOM % 101 + 20000))
DATA_PATH=${TASK_PATH}/${TRAINING_DATA}

SCRIPT_PATH=$(realpath $0)
mkdir -p ${SAVE_PATH}
cp -rf ${SCRIPT_PATH} ${SAVE_PATH}

srun --jobid $SLURM_JOBID python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_id $MASTER_PORT --rdzv_backend c10d --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    ming/train/train_mem.py \
    --lora_attn_enable True --lora_r 16 --lora_alpha 32 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $MODEL_BASE \
    --train_data_path ${DATA_PATH}/train.json \
    --bf16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 2 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --lora_name_or_path ${LORA_PATH} \
    --dataloader_num_workers 1 \
    --use_orthogonal True \
    --lazy_preprocess True \
    --report_to wandb \
    --lamda_1 0.001 \
    --lamda_2 0.











