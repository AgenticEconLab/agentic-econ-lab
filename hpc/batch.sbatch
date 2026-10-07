#!/usr/bin/env bash
#SBATCH --job-name=ael-vllm-batch
##SBATCH --account=<your-account>          # site-specific: remove one '#' and edit, or pass --account on the sbatch line
##SBATCH --partition=<your-gpu-partition>  # site-specific: remove one '#' and edit, or pass --partition on the sbatch line
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Standalone per-team batch: K repeated runs of each team/mode configuration via
# evaluation.runners.multirun_orchestrator. One vLLM instance serves the whole job.
# Parametrize via env (override on the sbatch command line):
#   MODEL_REPO  (default Qwen/Qwen3.6-27B-FP8)
#   SERVED      (default qwen3.6-27b-fp8)        -> AEL_MODEL=vllm/$SERVED  (local vLLM)
#   TP          (default 1; use 2 for the 35B-A3B MoE, with --gres=gpu:a100:2)
#   RUNS        (default 3 = K)
#   TEAMS       (default "IdeationTeam LiteratureTeam ModelTeam DataTeam")
#   MODE        (default empty = all modes of each team; set one mode for per-config jobs)
#   RUN_TIMEOUT (default 7200 s per run)
#   SUBDIR      (default = $SERVED)      -> results in $AEL_RUNS_DIR/multirun_results/$SUBDIR
#   COMMIT      (default = git HEAD of the code tree; recorded in the provenance file)
#   EXTRA_SERVE (default "--reasoning-parser qwen3"; set "" for gpt-oss-20b -- an empty value
#                is honoured, only an unset EXTRA_SERVE takes the default)
# Exit status: non-zero if any team's multirun_orchestrator call failed (all teams still run).
#
# Example (primary):       sbatch hpc/batch.sbatch
# Example (35B-A3B MoE):   MODEL_REPO=Qwen/Qwen3.6-35B-A3B-FP8 SERVED=qwen3.6-35b-a3b-fp8 TP=2 \
#                            sbatch --gres=gpu:a100:2 hpc/batch.sbatch
# Example (gpt-oss-20b):   MODEL_REPO=openai/gpt-oss-20b SERVED=gpt-oss-20b EXTRA_SERVE="" \
#                            sbatch hpc/batch.sbatch
set -uo pipefail

# --- locate hpc/ (robust under the SLURM spool copy) and load shared env ---
_cmd="$(scontrol show job "${SLURM_JOB_ID:-_}" 2>/dev/null | sed -n 's/.*Command=\([^ ]*\).*/\1/p')"
HPC_DIR="$(cd "$(dirname "${_cmd:-${BASH_SOURCE[0]}}")" && pwd)"
source "$HPC_DIR/_env.sh"
cd "$AEL_CODE_DIR"

# AEL knobs: disable Qwen thinking (set =0 on the sbatch line for gpt-oss robustness,
# whose chat template may not accept enable_thinking).
export AEL_DISABLE_THINKING="${AEL_DISABLE_THINKING:-1}"
# SearXNG: open news search for IdeationTeam --auto (open default -> commercial fallback).
# The SearXNG venv is built once with hpc/setup_searxng.sh; optional.
export SEARXNG_BASE_URL="${SEARXNG_BASE_URL:-http://localhost:8899}"
SEARXNG_SETTINGS="$AEL_CODE_DIR/hpc/searxng-settings.yml"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-27B-FP8}"
SERVED="${SERVED:-qwen3.6-27b-fp8}"
TP="${TP:-1}"
RUNS="${RUNS:-3}"
TEAMS="${TEAMS:-IdeationTeam LiteratureTeam ModelTeam DataTeam}"
MODE="${MODE:-}"                       # if set, run only this single mode (per-config parallelism)
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"     # per-run wall cap (eager ModelTeam runs are slow)
SUBDIR="${SUBDIR:-$SERVED}"
_DEFAULT_EXTRA_SERVE='--reasoning-parser qwen3'
EXTRA_SERVE="${EXTRA_SERVE-$_DEFAULT_EXTRA_SERVE}"   # '-' not ':-': EXTRA_SERVE="" means no extra flags
export AEL_MODEL="vllm/$SERVED"        # provider 'vllm' -> local server on :11434
PORT=11434
RESULTS="$AEL_RUNS_DIR/multirun_results/$SUBDIR"
mkdir -p "$RESULTS"
# COMMIT may be pinned via env so every parallel per-config job of one batch records the
# same commit, regardless of when each job starts or whether HEAD moves afterward.
# Defaults to the current HEAD.
COMMIT="${COMMIT:-$(git -C "$AEL_CODE_DIR" rev-parse HEAD 2>/dev/null)}"

echo "=================================================================="
echo "AEL BATCH ${SLURM_JOB_ID:-local} on $(hostname) @ $(date)"
echo "repo=$MODEL_REPO served=$SERVED TP=$TP RUNS=$RUNS commit=$COMMIT"
echo "teams=$TEAMS  results=$RESULTS"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
VLOG="$AEL_LOG_DIR/vllm-${SLURM_JOB_ID:-local}.log"
# stamp provenance into a dedicated subfolder (keeps the results root = team dirs only)
mkdir -p "$RESULTS/_provenance"
cat > "$RESULTS/_provenance/provenance-${SLURM_JOB_ID:-local}.txt" <<EOF
model_repo=$MODEL_REPO
served_name=$SERVED
ael_model=$AEL_MODEL
tensor_parallel=$TP
k_runs=$RUNS
commit_sha=$COMMIT
vllm_version=0.22.1
node=$(hostname)
job=${SLURM_JOB_ID:-local}
vllm_log=$VLOG
date=$(date -Iseconds)
EOF

# --- start vLLM -------------------------------------------------------------
echo "### starting vLLM (TP=$TP) -> $VLOG ###"
PYTHONPATH="" vllm serve "$MODEL_REPO" \
  --port "$PORT" --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP" --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --enforce-eager $EXTRA_SERVE \
  > "$VLOG" 2>&1 &
SERVE_PID=$!

# --- start SearXNG (optional; open news discovery for IdeationTeam --auto) ----
SEARX_PID=""
if [ -x "$AEL_SEARXNG_VENV/bin/python" ] && [ -f "$SEARXNG_SETTINGS" ]; then
  SXLOG="$AEL_LOG_DIR/searxng-${SLURM_JOB_ID:-local}.log"
  echo "### starting SearXNG (open news search) -> $SXLOG ###"
  PYTHONPATH="" SEARXNG_SETTINGS_PATH="$SEARXNG_SETTINGS" \
    "$AEL_SEARXNG_VENV/bin/python" -m searx.webapp > "$SXLOG" 2>&1 &
  SEARX_PID=$!
else
  echo "### SearXNG unavailable (\$AEL_SEARXNG_VENV or settings missing) — IdeationTeam --auto falls back to commercial/static ###"
fi

stop_server() { kill "$SERVE_PID" 2>/dev/null; [ -n "$SEARX_PID" ] && kill "$SEARX_PID" 2>/dev/null; wait 2>/dev/null; }
trap stop_server EXIT

wait_ready "$PORT" "$SERVE_PID" "$VLOG" 900 || exit 1

echo "### ROUTING VERIFY ###"
PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" hpc/verify_routing.py || { echo "ROUTING VERIFY FAILED"; exit 1; }

# --- run configs (all modes, or a single MODE for per-config parallelism) ----
if [ -n "$MODE" ]; then MODE_ARGS="--mode $MODE"; else MODE_ARGS="--all-modes"; fi
RC=0; FAILED_TEAMS=""
for team in $TEAMS; do
  echo; echo "############### TEAM $team $MODE_ARGS @ $(date +%H:%M:%S) ###############"
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" -m evaluation.runners.multirun_orchestrator \
      --team "$team" $MODE_ARGS --runs "$RUNS" \
      --output-dir "$RESULTS" --timeout "$RUN_TIMEOUT"
  TEAM_RC=$?
  echo "### TEAM $team done exit=$TEAM_RC @ $(date +%H:%M:%S) ###"
  if [ "$TEAM_RC" -ne 0 ]; then RC=$TEAM_RC; FAILED_TEAMS="$FAILED_TEAMS $team"; fi
done

echo; echo "=== BATCH DONE @ $(date). results in $RESULTS ==="
find "$RESULTS" -name 'execution_log*.json' | wc -l
[ "$RC" -ne 0 ] && echo "=== FAILED TEAMS:$FAILED_TEAMS (exit $RC) ==="
exit "$RC"

