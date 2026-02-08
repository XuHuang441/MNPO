#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin-lo
#SBATCH --gres=gpu:2
#SBATCH --job-name=mpo_infer
#SBATCH --output=logs/output_iathene.log
#SBATCH --error=logs/output_iathene.err
#SBATCH --mem=128G

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m test