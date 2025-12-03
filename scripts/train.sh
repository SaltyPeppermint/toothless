uv run train.py \
    --train.logging-steps 1 \
    --train.batch-size 128 \
    --train.warmup-steps 64 \
    --data.n-samples 2000000 \
    --data.data-path "data/start_and_goal-2025-10-02T23:08:10.859631049+02:00/0"
