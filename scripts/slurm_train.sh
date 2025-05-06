#!/bin/bash

#SBATCH -o ~/log/out.%j.%N.out          # Output-File
#SBATCH -D ~/                           # Working Directory
#SBATCH -J egraph-ast-transformer-1	    # Job Name
#SBATCH --ntasks=1 		                # Number of processes
#SBATCH --cpus-per-task=10	            # Give me 10 Cores per process plox
#SBATCH --gres=gpu:v100s:2	            # Give me 2 V100 plox
##SBATCH --gres=gpu:a100:1	            ## Give me 1 A100 plox
#SBATCH --mem=128G                      # 128 GB

##Max Walltime:
#SBATCH --time=8:00:00 # Expected runtime

##Run on GPU-Node:
#SBATCH --partition=scioi_gpu
##SBATCH --partition=gpu_short

#Job-Status via Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=heinimann@tu-berlin.de

# load singularity module
module load singularity/4.0.2
module load nvidia/cuda/12.2

rm -rf /tmp/*
git clone https://github.com/SaltyPeppermint/toothless /tmp/toothless

singularity exec --nv --bind /beegfs:/mnt /scratch/heinimann/container.sif \
    /venv/bin/torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 /tmp/toothless/train_tree_transformer.py \
    --data-path "/mnt/scratch/heinimann/data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0" \
    --cache-dir "/mnt/scratch/heinimann/cache" \
    --output-dir "/mnt/home/users/h/heinimann/saved_models" \
    --log-dir "/mnt/home/users/h/heinimann/runs" \
    --save_model_end True \
    --logging_steps 1 \
    --num-layers 12 \
    --sample-distance 2 \
    --epochs 1 \
    --warmup-steps 500 \
    --data-limit 1000000

scp -r ~/runs/* /beegfs/home/users/h/heinimann/scratch/heinimann/
