#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=vllm_mt2
#SBATCH --output=logs/slurm_vllm_mt2_%j.log

set -euo pipefail

# ====== 1. 基本配置 ======
MODEL_NAME="/hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_skywork_beta3_ratio0.85_eta0.01_weights1-0_3pl_fixed"           # HF 上的模型名称 / 本地路径
MODEL_BASENAME="gemma-2-9b-it_mnpo_stage_2_skywork_beta3_ratio0.85_eta0.01_weights1-0_3pl_fixed"              # 对外暴露的名称 & eval 中使用的名称
PORT=8011                                   # vLLM HTTP 端口
LOG_DIR="/hai/scratch/fangwu97/xu/MNPO/logs"

mkdir -p "${LOG_DIR}"

# ====== 2. 启动 vLLM OpenAI API server（后台） ======
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting vLLM server..." \
  | tee -a "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

/hai/scratch/fangwu97/miniconda3/envs/evalscope/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --served-model-name "${MODEL_BASENAME}" \
  --trust-remote-code \
  --port "${PORT}" \
  > "${LOG_DIR}/vllm-${MODEL_BASENAME}-${PORT}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}" >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

# ====== 3. 轮询日志，等待 vLLM 启动完成 ======
VLLM_LOG="${LOG_DIR}/vllm-${MODEL_BASENAME}-${PORT}.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for vLLM to finish startup (checking ${VLLM_LOG})..." \
  >> "${LOG_DIR}/mt-${MODEL_BASENAME}.out"

while true; do
  if [ -f "$VLLM_LOG" ] && grep -q "Application startup complete." "$VLLM_LOG"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') vLLM startup confirmed." \
      >> "${LOG_DIR}/mt-${MODEL_BASENAME}.out"
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') Still waiting for vLLM startup (checking ${VLLM_LOG})..." \
    >> "${LOG_DIR}/mt-${MODEL_BASENAME}.out"
  sleep 10
done

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting eval..." \
  >> "${LOG_DIR}/mt-${MODEL_BASENAME}.out"

WORKDIR="/hai/scratch/fangwu97/xu/FastChat/fastchat/llm_judge"
cd "${WORKDIR}"

/hai/scratch/fangwu97/miniconda3/envs/mt/bin/python gen_api_answer.py \
    --model "${MODEL_BASENAME}" \
    --bench-name mt_bench \
    --parallel 12 \
    --max-tokens 4096 \
    --openai-api-base "http://127.0.0.1:${PORT}/v1" \
    --openai-api-key "EMPTY" \
  >> "${LOG_DIR}/mt-${MODEL_BASENAME}.out" 2>&1

EVAL_STATUS=$?

# ====== 5. 收尾：关掉 vLLM 进程 ======
echo "$(date '+%Y-%m-%d %H:%M:%S') Killing vLLM (PID=${VLLM_PID})..." \
  >> "${LOG_DIR}/mt-${MODEL_BASENAME}.out"
kill "${VLLM_PID}" || true

wait "${VLLM_PID}" || true

echo "$(date '+%Y-%m-%d %H:%M:%S') Done. Eval exit code: ${EVAL_STATUS}" \
  >> "${LOG_DIR}/mt-${MODEL_BASENAME}.out"

exit "${EVAL_STATUS}"
