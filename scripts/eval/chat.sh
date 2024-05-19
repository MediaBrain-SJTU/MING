#!/bin/bash
#SBATCH -J ming_chat
#SBATCH --partition=medai_llm
#SBATCH -N2
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --time=120:00:00
###SBATCH --kill-on-bad-exit=1

srun --jobid $SLURM_JOBID python ming/serve/cli.py \
    --model_path /mnt/petrelfs/liaoyusheng/oss/download_models/MING-7B \
    --conv_template bloom \
    --max_new_token 512 \
    --beam_size 3 \
    --temperature 1.2