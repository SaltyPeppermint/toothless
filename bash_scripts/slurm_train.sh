#!/bin/bash

#SBATCH -o log/%j.log                   # Output-File
#SBATCH -J egraph-ast-transformer-1	    # Job Name
#SBATCH --cpus-per-task=10	            # Give me 10 Cores per process plox
#SBATCH --gres=gpu:v100s:2	            # Give me 2 V100 plox
# #SBATCH --gres=gpu:a100:1	            ## Give me 1 A100 plox
#SBATCH --mem=250G                      # 250 GB

#Max Walltime:
#SBATCH --time=7-00:00:00 # Expected runtime

#Run on GPU-Node:
#SBATCH --partition=scioi_gpu
# #SBATCH --partition=gpu_short

#Job-Status via Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=heinimann@tu-berlin.de

# load singularity (and cuda) on the host module
# module load singularity/4.0.2
# module load nvidia/cuda/12.2

mkdir -p /tmp/toothless && rm -rf /tmp/toothless

mkdir -p /tmp/sample_cache && rm -rf /tmp/sample_cache

singularity exec --nv --bind /beegfs:/mnt /scratch/heinimann/container.sif \
    git clone https://github.com/SaltyPeppermint/toothless /tmp/toothless

singularity exec --nv --bind /beegfs:/mnt /scratch/heinimann/container.sif \
    /venv/bin/python3 train.py \
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
    --data.data-path "data/start_and_goal-2025-10-02T23:08:10.859631049+02:00/0"
