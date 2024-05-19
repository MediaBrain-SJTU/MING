DATA_PATH=/mnt/petrelfs/liaoyusheng/projects/MING-MOE/datas/medical_moe/test
LOGS_BASE_PATH=/mnt/petrelfs/liaoyusheng/projects/MING-MOE/MING-MOE/logs/base
CKPT=gpt4-turbo
DATASETS=("PLE_TCM_cot")
CHUNKS=10

for DATASET in "${DATASETS[@]}"; do
    echo "processing ${DATASET}"
    mkdir -p ${LOGS_BASE_PATH}/${CKPT}/${DATASET}

    for IDX in $(seq 0 $((CHUNKS-1))); do
        srun -p medai --output="${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer_${CHUNKS}_${IDX}.log" python -m ming.eval.eval_gpt4 \
            --question-file ${DATA_PATH}/${DATASET}.json \
            --answers-file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer_${CHUNKS}_${IDX}.jsonl \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} \
            --resume  & sleep 1
    done

    wait
    output_file=${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer_${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    echo "evaluating ${DATASET}"
    srun -p medai --output=${LOGS_BASE_PATH}/${CKPT}/${DATASET}/eval.log python -m ming.eval.eval_em \
        --input_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
        --output_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer-wrong.jsonl  
done