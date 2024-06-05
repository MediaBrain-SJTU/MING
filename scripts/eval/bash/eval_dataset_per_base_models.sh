OSS_PATH=~/oss
TASK_PATH=datas
# domains=("PLE_Pharmacy_cot" "PLE_TCM_cot")
# domain="ceval_cot"
domain="clinical-test-200-v2"
# domains=("image_analysis_new")

LOGS_BASE_PATH=logs/base
CKPTS=(
    "chatglm2-6b"
    # "chatglm3-6b"
    # "baichuan2-7b"
    # "baichuan2-13b"
    # "huatuo2-7b"
    # "huatuo2-13b"
)

for CKPT in "${CKPTS[@]}"; do
    sbatch ./scripts/eval/srun/eval_parallel_vllm_batch_auto_base.sh $TASK_PATH $CKPT $LOGS_BASE_PATH $domain & sleep 1
done
