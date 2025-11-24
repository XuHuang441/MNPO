
API_KEY=$OPENAI_API_KEY

WORKDIR="/hai/scratch/fangwu97/xu/FastChat/fastchat/llm_judge"
cd "${WORKDIR}"

/hai/scratch/fangwu97/miniconda3/envs/mt/bin/python gen_api_answer.py \
    --model "allenai/olmo-2-0325-32b-instruct" \
    --bench-name mt_bench \
    --parallel 12 \
    --max-tokens 4096 \
    --openai-api-base "https://openrouter.ai/api/v1" \
    --openai-api-key "$API_KEY"