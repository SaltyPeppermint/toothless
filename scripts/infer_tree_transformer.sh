uv run torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 infer_tree_transformer.py \
    --data-path "data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0" \
    --output_dir "saved_models" \
    --infer_data "data/infer_data.json" \
    --weights_path "saved_models/04-04-25-2025_13:34:46/tree_transformer.pt" \
    --vocab_path "saved_models/04-04-25-2025_13:34:46/vocab.json"
