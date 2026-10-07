# shellcheck shell=bash
# LLM-Economist committee serving for the Model<->Data feasibility loop (HITL resolved by a
# three-member committee, --hitl-mode llm_economist).
#
# The committee shares no model family with the generating model (Qwen, GPU0) or the
# Tier-2 judges (Mistral, Gemma): OpenAI gpt-oss-20b, Microsoft Phi-4-mini (bf16), AllenAI
# Olmo-3.1-32B-Instruct (bf16). Layout on a 4 x A100-40GB node:
#
#   GPU1  phi-4-mini  :11435  (~8 GB bf16 weights; 0.40 of the card, 16k-token KV ~2 GiB)
#   GPU1  gpt-oss-20b :11436  (~13 GB weights; 0.50, started after phi-4-mini)
#   GPU2+3 olmo-3.1-32b-instruct :11437  (64 GB bf16, tensor-parallel 2)
#
# Phi-4 (14B) was tried first: on A100 every FP8 path (RedHatAI/phi-4-FP8-dynamic and vLLM
# online fp8, both Marlin weight-only) produced unusable output in vLLM 0.22.1, and bf16
# Phi-4 (~29 GB) needs a whole card, so Phi-4-mini is used.
#
# Source after _env.sh. Call `serve_committee`; on success it exports
# AEL_HITL_COMMITTEE and COMMITTEE_ENDPOINTS (to append to AEL_VLLM_ENDPOINTS).

serve_committee() {
  local jid="${SLURM_JOB_ID:-local}"
  echo "=== serving committee (gpt-oss / phi-4-mini / olmo-3.1) @ $(date) ==="

  VLOG_P="$AEL_LOG_DIR/vllm-phi4-${jid}.log"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="" vllm serve microsoft/Phi-4-mini-instruct \
    --port 11435 --served-model-name phi-4-mini \
    --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.40 \
    --enforce-eager > "$VLOG_P" 2>&1 &
  PID_P=$!
  wait_ready 11435 "$PID_P" "$VLOG_P" || return 1

  # second server on the same card: start only after phi-4-mini has claimed its share
  VLOG_O="$AEL_LOG_DIR/vllm-gptoss-${jid}.log"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="" vllm serve openai/gpt-oss-20b \
    --port 11436 --served-model-name gpt-oss-20b \
    --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.50 \
    --enforce-eager > "$VLOG_O" 2>&1 &
  PID_O=$!

  VLOG_L="$AEL_LOG_DIR/vllm-olmo-${jid}.log"
  CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH="" vllm serve allenai/Olmo-3.1-32B-Instruct \
    --port 11437 --served-model-name olmo-3.1-32b-instruct \
    --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --enforce-eager > "$VLOG_L" 2>&1 &
  PID_L=$!

  wait_ready 11436 "$PID_O" "$VLOG_O" || return 1
  wait_ready 11437 "$PID_L" "$VLOG_L" || return 1

  export AEL_HITL_COMMITTEE="vllm/gpt-oss-20b,vllm/phi-4-mini,vllm/olmo-3.1-32b-instruct"
  export COMMITTEE_ENDPOINTS="phi-4-mini=http://localhost:11435/v1/chat/completions,gpt-oss-20b=http://localhost:11436/v1/chat/completions,olmo-3.1-32b-instruct=http://localhost:11437/v1/chat/completions"
  echo "=== committee ready @ $(date) ==="
}
