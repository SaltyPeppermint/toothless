uv run infer.py \
    --infer_data "data/infer_data.json" \
    --folder "models/25-09-09-14:45:43" \
    --model-suffix "0" \
    --n-train-data 4096 \
    --n-eval-data 4096 \
    --batch-size 32
#torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601