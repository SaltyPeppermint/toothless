uv run train_tree_transformer.py \
    --data-path "data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0" \
    --output_dir "saved_models" \
    --save_model_end True \
    --logging_steps 1 \
    --num-layers 12 \
    --sample-distance 2 \
    --epochs 1 \
    --warmup-steps 500 \
    --sample-limit 1000000
#torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601
