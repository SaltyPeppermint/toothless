#!/bin/bash

#SBATCH -o ~/log/out.%j.%N.out          # Output-File
#SBATCH -D ~/                           # Working Directory
#SBATCH -J egraph-ast-transformer-1	    # Job Name
#SBATCH --ntasks=1 		                # Number of processes
#SBATCH --cpus-per-task=10	            # Give me 10 Cores per process plox
#SBATCH --gres=gpu:a100:1	            # Give me 1 A100 plox
#SBATCH --mem=32G                       # 32 GB

##Max Walltime:
#SBATCH --time=8:00:00 # Expected runtime

#Run on GPU-Node:
#SBATCH --partition=gpu_short

#Job-Status via Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=heinimann@tu-berlin.de

# load singularity module
module load singularity/4.0.2

##scp -r /scratch/heinimann/data /tmp

singularity exec --bind /beegfs:/mnt /scratch/heinimann/container.sif \
    torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 train_tree_transformer.py \
    --data-path "/mnt/scratch/heinimann/data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0" \
    --cache-dir "/tmp/cache" \
    --output-dir "~/saved_models" \
    --save_model_end True \
    --logging_steps 1 \
    --num-layers 12 \
    --sample-distance 2 \
    --epochs 1 \
    --warmup-steps 500 \
    --data-limit 1000000
