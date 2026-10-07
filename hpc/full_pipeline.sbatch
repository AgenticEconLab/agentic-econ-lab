#!/usr/bin/env bash
#SBATCH --job-name=ael-fullpipe
##SBATCH --account=<your-account>          # site-specific: remove one '#' and edit, or pass --account on the sbatch line
##SBATCH --partition=<your-gpu-partition>  # site-specific: remove one '#' and edit, or pass --partition on the sbatch line
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=380G
#SBATCH --time=07:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# One chained run of the full seven-team AEL pipeline (Ideation -> Literature -> Model ->
# Data -> Model<->Data feasibility loop -> Code -> Estimation -> Reporting) on one node with
# four A100 40GB GPUs:
#   GPU0     generating model Qwen3.6-27B-FP8 (served as qwen3.6-27b-fp8, port 11434)
#   GPU1-3   the three-member LLM-Economist committee that resolves the feasibility-loop
#            HITL review (see _committee.sh, ports 11435-11437)
# SearXNG runs on the same node (port 8899) for open web search.
# run_ael_pipeline.py is called with its defaults (default topic direction, --hitl-mode
# llm_economist, --data-mode open_source_api); CodeTeam is switched on below.
#
# Env knobs:
#   AEL_SCRATCH    (required, see _env.sh)
#   AEL_MODEL_DIR  generating-model weights: a local directory or a Hugging Face id
#                  (default Qwen/Qwen3.6-27B-FP8, read from HF_HOME with HF_HUB_OFFLINE=1)
#   RUN_LABEL      prefix of the run directory $AEL_RUNS_DIR/<RUN_LABEL>_<jobid>
#                  (default full_pipeline)
#
# Usage: AEL_SCRATCH=/path/to/scratch sbatch hpc/full_pipeline.sbatch
set -uo pipefail

_cmd="$(scontrol show job "${SLURM_JOB_ID:-_}" 2>/dev/null | sed -n 's/.*Command=\([^ ]*\).*/\1/p')"
HPC_DIR="$(cd "$(dirname "${_cmd:-${BASH_SOURCE[0]}}")" && pwd)"
source "$HPC_DIR/_env.sh"
source "$HPC_DIR/_committee.sh"   # committee model families differ from the generating model and the judges
cd "$AEL_CODE_DIR"

# Kernel-compile caches on node-local /tmp: many concurrent jobs writing JIT caches to a
# shared parallel filesystem can overload its metadata server.
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID}_triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${SLURM_JOB_ID}_inductor"
export VLLM_CACHE_ROOT="/tmp/${SLURM_JOB_ID}_vllm"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

QWEN_DIR="${AEL_MODEL_DIR:-Qwen/Qwen3.6-27B-FP8}"   # a pinned local copy of the weights is also accepted

export AEL_DISABLE_THINKING=1     # Qwen-only in practice: shared/llm.py's _thinking_kwargs
                                   # gates on "qwen" in the model name, so this is safe to
                                   # leave on globally -- Mistral/Gemma calls are unaffected.
export AEL_CODE_TEAM_ENABLED=1    # CodeTeam is off by default in run_ael_pipeline.py; the
                                   # paper's runs include it (seven teams).
export SEARXNG_BASE_URL="${SEARXNG_BASE_URL:-http://localhost:8899}"
SEARXNG_SETTINGS="$AEL_CODE_DIR/hpc/searxng-settings.yml"

echo "=================================================================="
echo "AEL FULL-PIPELINE (7-team, committee HITL) ${SLURM_JOB_ID:-local} on $(hostname) @ $(date)"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- SearXNG (open search + web-crawl discovery) --------------------------------------
echo "=== starting SearXNG (open search + web-crawl discovery) ==="
env -u PYTHONPATH SEARXNG_SETTINGS_PATH="$SEARXNG_SETTINGS" \
  "$AEL_SEARXNG_VENV/bin/python" -m searx.webapp > "$AEL_LOG_DIR/searxng-${SLURM_JOB_ID:-local}.log" 2>&1 &
SXPID=$!
for i in $(seq 1 30); do
  curl -sf "$SEARXNG_BASE_URL/search?q=test&format=json" >/dev/null 2>&1 && break
  kill -0 "$SXPID" 2>/dev/null || { echo "SEARXNG DIED DURING START"; tail -30 "$AEL_LOG_DIR/searxng-${SLURM_JOB_ID:-local}.log"; exit 1; }
  sleep 2
done

# --- serve the generating model (Qwen, GPU0) and the committee (GPU1-3) ---------------
# PATH puts $AEL_VLLM_VENV/bin first (set by _env.sh) for the bare `vllm` calls below; the
# agents' python is always called explicitly as "$AEL_VENV/bin/python".
echo "=== serving the generating model @ $(date) on $(hostname) ==="

VLOG_Q="$AEL_LOG_DIR/vllm-qwen-${SLURM_JOB_ID:-local}.log"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" vllm serve "$QWEN_DIR" \
  --port 11434 --served-model-name qwen3.6-27b-fp8 --reasoning-parser qwen3 \
  --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --enforce-eager > "$VLOG_Q" 2>&1 &
PID_Q=$!

serve_committee || exit 1
wait_ready 11434 "$PID_Q" "$VLOG_Q" || exit 1

export AEL_MODEL="vllm/qwen3.6-27b-fp8"
export AEL_VLLM_ENDPOINTS="qwen3.6-27b-fp8=http://localhost:11434/v1/chat/completions,${COMMITTEE_ENDPOINTS}"

# --- run the full seven-team chained pipeline (all CLI flags at their defaults) ---------
# RUN_LABEL names the run (e.g. demo, rep07). Replicate runs that execute concurrently must
# be submitted from separate copies of the code tree, because teams write working files
# into their own code directory.
RUN_DIR="$AEL_RUNS_DIR/${RUN_LABEL:-full_pipeline}_${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"
# record the code version: a FROZEN_COMMIT file at the tree root if present, else git HEAD
if [ -f "$AEL_CODE_DIR/FROZEN_COMMIT" ]; then cp "$AEL_CODE_DIR/FROZEN_COMMIT" "$RUN_DIR/FROZEN_COMMIT"
else git -C "$AEL_CODE_DIR" rev-parse HEAD > "$RUN_DIR/FROZEN_COMMIT" 2>/dev/null || rm -f "$RUN_DIR/FROZEN_COMMIT"; fi
echo "=== run FULL pipeline (built-in defaults, 7 teams) @ $(date) ==="
env -u PYTHONPATH PYTHONPATH="$AEL_CODE_DIR" PYDANTIC_DISABLE_PLUGINS=1 \
  "$AEL_VENV/bin/python" run_ael_pipeline.py --output-dir "$RUN_DIR"
RC=$?

# --- publish curated output view -------------------------------------------------------
echo "=== publish curated output view (output/) ==="
env -u PYTHONPATH PYTHONPATH="$AEL_CODE_DIR" "$AEL_VENV/bin/python" pipeline/link_outputs.py "$RUN_DIR"
LINK_RC=$?
# the pipeline's own status takes precedence; a failed publish step alone also fails the job
if [ "$RC" -eq 0 ] && [ "$LINK_RC" -ne 0 ]; then echo "link_outputs.py failed (exit $LINK_RC)"; RC=$LINK_RC; fi

echo "=== FULL-PIPELINE DONE rc=$RC @ $(date) -> $RUN_DIR ==="
exit $RC
