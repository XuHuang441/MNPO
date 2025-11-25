#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=vllm_eval_new
#SBATCH --output=logs/slurm_vllm_eval_new_%j.log

set -euo pipefail

# ====== 1. 基本配置 ======
MODEL_NAME="/hai/scratch/fangwu97/xu/MNPO/outputs/gemma-2-9b-it_mnpo_stage_2_athene_beta1_ratio0.85_eta0.005_weights0.75-0.25"           # HF 上的模型名称 / 本地路径
MODEL_BASENAME="gemma-2-9b-it_mnpo_stage_2_athene_beta1_ratio0.85_eta0.005_weights0.75-0.25"              # 对外暴露的名称 & eval 中使用的名称
PORT=9025                                  # vLLM HTTP 端口
TASK_SCRIPT="/hai/scratch/fangwu97/xu/MNPO/evalscope/run_rule_based_task.py"
LOG_DIR="logs"
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
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

while true; do
  if [ -f "$VLLM_LOG" ] && grep -q "Application startup complete." "$VLLM_LOG"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') vLLM startup confirmed." \
      >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') Still waiting for vLLM startup (checking ${VLLM_LOG})..." \
    >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"
  sleep 10
done

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting eval..." \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

/hai/scratch/fangwu97/miniconda3/envs/evalscope/bin/python "${TASK_SCRIPT}" \
  --model-name "${MODEL_BASENAME}" \
  --port "${PORT}" \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out" 2>&1

EVAL_STATUS=$?

# ====== 5. 收尾：关掉 vLLM 进程 ======
echo "$(date '+%Y-%m-%d %H:%M:%S') Killing vLLM (PID=${VLLM_PID})..." \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"
kill "${VLLM_PID}" || true

wait "${VLLM_PID}" || true

echo "$(date '+%Y-%m-%d %H:%M:%S') Done. Eval exit code: ${EVAL_STATUS}" \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

exit "${EVAL_STATUS}"
