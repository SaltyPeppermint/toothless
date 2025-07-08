uv run infer_tree_transformer.py \
    --infer_data "data/infer_data.json" \
    --folder "saved_models/07-05-25-2025_14:58:20" \
    --model-suffix "_0" \
    --n-train-data 256 \
    --n-eval-data 256 \
    --batch-size 128
#torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601
