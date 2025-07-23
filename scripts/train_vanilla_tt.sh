uv run train_vanilla_tt.py \
    --train.save-model-end \
    --train.logging-steps 1 \
    --train.epochs 1 \
    --train.warmup-steps 500 \
    --model.disentangled False \
    --model.with-pos \
    --model.num-layers 12 \
    --model.output-dir "saved_models" \
    --data.sample-distance 2 \
    --data.sample-limit 1000000 \
    --data.data-path "data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0" \

#torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601
