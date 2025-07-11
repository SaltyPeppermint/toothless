uv run infer_tree_transformer.py \
    --infer_data "data/infer_data.json" \
    --folder "saved_models/08-07-25-2025_12:04:38" \
    --model-suffix "_0" \
    --n-train-data 4096 \
    --n-eval-data 4096 \
    --batch-size 16
#torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601
