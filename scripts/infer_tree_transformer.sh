uv run infer_tree_transformer.py \
    --infer_data "data/infer_data.json" \
    --folder "saved_models/25-04-25-2025_20:32:30" \
    --model-suffix "" \
    --n-train-data 5 \
    --n-eval-data 5
#torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601
