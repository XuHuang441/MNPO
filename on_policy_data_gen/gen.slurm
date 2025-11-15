#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p hai-lo
#SBATCH --gres=gpu:4
#SBATCH --job-name=mpo_infer
#SBATCH --output=output_infer.log
#SBATCH --mem=128G

for SEED in 13 21 42 79 100; do
    echo "Running decode with seed $SEED..."
    stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
        --data_dir /hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split2.jsonl \
        --model /hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_1_armo_inpo_iter1_20k \
        --seed $SEED \
        --output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter2_armo \
        --batch_size 96 \
        --cache_dir /hai/scratch/fangwu97/xu/cache \
        --num_gpu 4
done