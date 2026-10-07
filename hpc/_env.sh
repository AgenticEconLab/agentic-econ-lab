# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
#
# Shared environment for the AEL HPC job scripts (batch / smoke / eval / full_pipeline /
# committee_smoke). Sourced by each .sbatch right after its #SBATCH header. Every value
# below can be overridden by exporting it before `sbatch` (sbatch passes the submitting
# shell's environment to the job by default).
#
# Required:
#   AEL_SCRATCH   a writable directory on a filesystem with room for venvs, model weights,
#                 caches and run outputs (tens to hundreds of GB), e.g. /scratch/<project>/<user>
#
# NOTE: #SBATCH header lines cannot read environment variables (SLURM parses the header
# before the script runs). Account, partition and log-file paths are therefore set in each
# script's header or on the sbatch command line; see README.md, "Running on a GPU cluster".

# --- repository layout: derived from this file's location (hpc/ -> repository root) ---
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AEL_CODE_DIR="${AEL_CODE_DIR:-$(cd "$_ENV_DIR/.." && pwd)}"   # repository root (contains run_ael_pipeline.py)

# --- storage roots ---
: "${AEL_SCRATCH:?set AEL_SCRATCH to a writable scratch directory (see README.md, Running on a GPU cluster)}"
export AEL_SCRATCH
export AEL_LOG_DIR="${AEL_LOG_DIR:-$AEL_SCRATCH/slurm-logs}"        # vLLM / SearXNG server logs
export AEL_RUNS_DIR="${AEL_RUNS_DIR:-$AEL_SCRATCH/agentic-runs}"    # pipeline and evaluation outputs

# --- virtual environments: model serving / agents + evaluation / SearXNG ---
export AEL_VLLM_VENV="${AEL_VLLM_VENV:-$AEL_SCRATCH/tools/venv-vllm}"
export AEL_VENV="${AEL_VENV:-$AEL_SCRATCH/tools/venv-ael}"
export AEL_SEARXNG_VENV="${AEL_SEARXNG_VENV:-$AEL_SCRATCH/tools/venv-searxng}"

# --- Hugging Face + JIT caches (kept on scratch; home directories usually have a quota) ---
export HF_HOME="${HF_HOME:-$AEL_SCRATCH/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # weights must be downloaded before the job
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$AEL_SCRATCH/tools/cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$AEL_SCRATCH/tools/cache/triton}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$AEL_SCRATCH/tools/cache/flashinfer}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$AEL_SCRATCH/tools/playwright-browsers}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$AEL_LOG_DIR" 2>/dev/null || true

# --- vLLM serving invariants (CUDA 12.8 wheels without a system nvcc -> flashinfer JIT off) ---
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PATH="$AEL_VLLM_VENV/bin:$PATH"

# --- AEL client knobs (HTTP timeout; AEL_DISABLE_THINKING is per-model and set in each
#     script: =1 for Qwen3.x, =0 for the Mistral/Gemma judges) ---
export AEL_LLM_TIMEOUT="${AEL_LLM_TIMEOUT:-600}"

# Both venvs are self-contained; a site Python module on PYTHONPATH could shadow their
# dependencies. Callers run vLLM with PYTHONPATH="" and AEL with PYTHONPATH="$AEL_PYTHONPATH".
export AEL_PYTHONPATH="${AEL_PYTHONPATH:-$AEL_CODE_DIR}"

# --- shared helper: poll an OpenAI-compatible server until ready ---
#   wait_ready <port> <server_pid> <logfile> [max_tries=900]   (2s/try)
wait_ready() {
  local port="$1" pid="$2" log="$3" max="${4:-900}" i
  for i in $(seq 1 "$max"); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { echo "  vLLM ready after $((i*2))s"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "ERROR: vLLM died during load"; tail -40 "$log"; return 1; }
    sleep 2
  done
  echo "ERROR: vLLM never became ready"; tail -40 "$log"; return 1
}

