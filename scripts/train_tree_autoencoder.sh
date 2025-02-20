uv run torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 tree_autoencoder.py \
    --data-path "data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0" \
    --output_dir "cache/saved_models" \
    --num_train_epochs 5 \
    --tmax 30 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --logging_steps 1 \
    --save_model_end True
# --gradient_accumulation_steps 16 \
# --evaluation_strategy "no" \
# --report_to "none" \
# --save_strategy "steps" \
# --save_steps 1000 \
# --save_total_limit 10 \
# --warmup_ratio 0.01 \
# --lr_scheduler_type "cosine" \
# --ds_config "deepspeed/ds_config_zero2.json" \
# --model_max_length 512 \
# --lazy_preprocess True \
