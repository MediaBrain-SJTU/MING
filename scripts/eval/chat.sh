# python ming/serve/cli.py \
#     --model_path /mnt/petrelfs/liaoyusheng/download_models/MING-MOE-1.8B \
#     --model_base /mnt/petrelfs/liaoyusheng/download_models/Qwen1.5-1.8B-Chat

srun -p medai_llm --quotatype="auto" --gres=gpu:1 python ming/serve/cli.py \
    --model_path /mnt/petrelfs/liaoyusheng/oss/download_models/MING-7B \
    --conv_template bloom \
    --max-new-token 512 \
    --beam-size 3 \
    --temperature 1.2