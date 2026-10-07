#!/usr/bin/env bash
#SBATCH --job-name=ael-eval
##SBATCH --account=<your-account>          # site-specific: remove one '#' and edit, or pass --account on the sbatch line
##SBATCH --partition=<your-gpu-partition>  # site-specific: remove one '#' and edit, or pass --partition on the sbatch line
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# AEL evaluation with a local open-weight judge served by vLLM (one GPU) — pick a mode:
#   sbatch eval.sbatch tier2         # Tier-2 LLM-as-reviewer over a multirun tree (RUN_TREE)
#   sbatch eval.sbatch reliability   # judge bait check (GOOD scored well above BAD, stable)
# Env knobs: JUDGE_REPO, SERVED, JUDGE_SERVE (judge vLLM flags; JUDGE_SERVE="" passes none),
#   RUN_TREE (tier2 only). The job exits with the evaluator's status.
# Default judge: Mistral-Small-3.2-24B FP8. Second judge used in the paper (Gemma-3-27B):
#   JUDGE_REPO=RedHatAI/gemma-3-27b-it-FP8-dynamic SERVED=gemma-3-27b-it-fp8 \
#   JUDGE_SERVE='--limit-mm-per-prompt {"image":0}' RUN_TREE=... sbatch hpc/eval.sbatch tier2
set -uo pipefail

# --- locate hpc/ (robust under the SLURM spool copy) and load shared env ---
_cmd="$(scontrol show job "${SLURM_JOB_ID:-_}" 2>/dev/null | sed -n 's/.*Command=\([^ ]*\).*/\1/p')"
HPC_DIR="$(cd "$(dirname "${_cmd:-${BASH_SOURCE[0]}}")" && pwd)"
source "$HPC_DIR/_env.sh"

MODE="${1:-tier2}"
cd "$AEL_CODE_DIR"

# AEL_DISABLE_THINKING is Qwen3.x-specific (sends chat_template_kwargs{enable_thinking}).
# Mistral (mistral-tokenizer) / Gemma judges reject chat_template kwargs -> HTTP 400. Keep OFF.
export AEL_DISABLE_THINKING=0

JUDGE_REPO="${JUDGE_REPO:-RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8}"
SERVED="${SERVED:-mistral-small-3.2-24b-fp8}"
# Keep the default in its OWN var — a literal '}' inside ${VAR:-default} closes the ${...}
# early (bash brace mis-parse), corrupting any JUDGE_SERVE override.
_DEFAULT_JUDGE_SERVE='--tokenizer-mode mistral --config-format mistral --load-format mistral --limit-mm-per-prompt {"image":0}'
# '-' (not ':-'): an explicitly empty JUDGE_SERVE="" is honoured; only an unset one takes the default.
JUDGE_SERVE="${JUDGE_SERVE-$_DEFAULT_JUDGE_SERVE}"
export AEL_MODEL="vllm/$SERVED"       # local vLLM; align LLMClient override with the judge
# Job-unique port: two 1-GPU judge jobs placed on the same node would otherwise both bind
# 11434, and requests for one model would reach the other's server (HTTP 404 "model not
# found"). VLLM_BASE_URL points the client at the chosen port.
PORT="${PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
export VLLM_BASE_URL="http://localhost:$PORT/v1/chat/completions"
VLOG="$AEL_LOG_DIR/vllm-eval-${SLURM_JOB_ID:-local}.log"

echo "=== eval [$MODE]: serve judge $JUDGE_REPO (served=$SERVED) on $(hostname) @ $(date) ==="
PYTHONPATH="" vllm serve "$JUDGE_REPO" --port "$PORT" --served-model-name "$SERVED" \
   --max-model-len 32768 --gpu-memory-utilization 0.90 --enforce-eager $JUDGE_SERVE > "$VLOG" 2>&1 &
VP=$!
trap 'kill $VP 2>/dev/null' EXIT
wait_ready "$PORT" "$VP" "$VLOG" 900 || exit 1

RC=0
case "$MODE" in
tier2)
  RUN_TREE="${RUN_TREE:?set RUN_TREE to a multirun tree (batch.sbatch results dir or <run_dir>/_eval_view)}"
  echo "=== run multirun tier-2 (judge=$AEL_MODEL) over $RUN_TREE ==="
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" evaluation/run_ael_evaluation.py \
     --multirun-dir "$RUN_TREE" --tier2 --judge-model "vllm/$SERVED"
  RC=$?
  echo "=== TIER-2 DONE rc=$RC @ $(date) -> $RUN_TREE/tier2_multirun_<judge>.json ==="
  ;;
reliability)
  echo "=== judge reliability bait check (vllm/$SERVED) ==="
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" -m evaluation.audits.judge_reliability "vllm/$SERVED"
  RC=$?
  echo "=== JUDGE-REL DONE (exit $RC) @ $(date) ==="
  ;;
*)
  echo "unknown mode '$MODE' — use: tier2 | reliability"; exit 2 ;;
esac
exit "$RC"
