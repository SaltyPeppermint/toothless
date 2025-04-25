uv run torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 infer_tree_transformer.py \
    --infer_data "data/infer_data.json" \
    --folder "saved_models/22-04-25-2025_18:51:31"
