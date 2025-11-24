#!/bin/bash

WORKDIR="/hai/scratch/fangwu97/xu/FastChat/fastchat/llm_judge"
cd "${WORKDIR}"

/hai/scratch/fangwu97/miniconda3/envs/mt/bin/python gen_judgment.py \
    --model-list olmo-2-0325-32b-instruct gemma-2-9b-it_mnpo_stage_2_armo_beta1_ratio0.8_eta0.005_weights0.75-0.25_3pl.jsonl gemma-2-9b-it_mnpo_stage_2_athene_beta1_ratio0.85_eta0.005_weights0.75-0.25.jsonl gemma-2-9b-it_mnpo_stage_2_skywork_beta3_ratio0.85_eta0.01_weights1-0_3pl_fixed.jsonl Llama-3.1-Tulu-3-70B-DPO.jsonl SmolLM3-3B.jsonl \
    --parallel 12 \
    --judge-model gpt-5-mini