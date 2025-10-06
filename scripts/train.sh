uv run train.py \
    --train.save-model-end \
    --train.logging-steps 1 \
    --train.epochs 1 \
    --train.batch-size 128 \
    --train.warmup-steps 64 \
    --model.num-layers 12 \
    --model.head-dim 64 \
    --model.output-dir "models" \
    --data.max-len 512 \
    --data.n-samples 2000000 \
    --data.data-path "start_and_goal-2025-10-02T23:08:10.859631049+02:00/0" \

#torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601