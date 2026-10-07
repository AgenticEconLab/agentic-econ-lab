#!/usr/bin/env bash
#SBATCH --job-name=ael-smoke
##SBATCH --account=<your-account>          # site-specific: remove one '#' and edit, or pass --account on the sbatch line
##SBATCH --partition=<your-gpu-partition>  # site-specific: remove one '#' and edit, or pass --partition on the sbatch line
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# AEL pre-flight smokes — one script, pick a target:
#   sbatch smoke.sbatch serve      # vLLM brings a model up + one chat completion (default: Mistral judge)
#   sbatch smoke.sbatch pipeline   # egress probe -> Qwen -> routing verify -> 1-run Ideation(x2)+ModelTeam
#   sbatch smoke.sbatch model      # Qwen -> routing verify -> ModelTeam NoWcNoHITL only (fast)
#   sbatch smoke.sbatch data       # Qwen -> routing verify -> DataTeam ModeOpenSourceAPI only (fast)
#   sbatch smoke.sbatch code       # Qwen -> routing verify -> CodeTeam ModeNoWcWithHITL only (fast)
#   sbatch smoke.sbatch estimation # Qwen -> routing verify -> EstimationTeam ModeNoWcWithHITL only (fast)
#   sbatch smoke.sbatch reporting  # Qwen -> routing verify -> ReportingTeam ModeNoWcWithHITL only (fast)
#   sbatch smoke.sbatch speedup    # Qwen (CUDA-graphs vs --enforce-eager) + throughput benchmark  [EAGER=1]
#   sbatch smoke.sbatch ideation   # Qwen + SearXNG -> real fetch_trending_topics() open path
# Env knobs: MODEL_REPO, SERVED, EXTRA_SERVE (serve target; EXTRA_SERVE="" passes no extra
# flags), PORT (target-specific defaults below; the AEL client follows it via VLLM_BASE_URL).
# The job exits non-zero if the target's smoke command (or any of the pipeline target's runs) fails.
# The code / estimation / reporting targets read the outputs of a completed full-pipeline run:
#   AEL_SMOKE_REF_RUN=<run dir written by full_pipeline.sbatch>  (or the per-file variables
#   named in each target).
set -uo pipefail

# --- locate hpc/ (robust under the SLURM spool copy) and load shared env ---
_cmd="$(scontrol show job "${SLURM_JOB_ID:-_}" 2>/dev/null | sed -n 's/.*Command=\([^ ]*\).*/\1/p')"
HPC_DIR="$(cd "$(dirname "${_cmd:-${BASH_SOURCE[0]}}")" && pwd)"
source "$HPC_DIR/_env.sh"

TARGET="${1:-pipeline}"
cd "$AEL_CODE_DIR"
echo "=================================================================="
echo "AEL smoke [$TARGET] ${SLURM_JOB_ID:-local} on $(hostname) @ $(date)"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

RC=0   # exit status of the job: set by each target from its smoke command(s)
case "$TARGET" in

serve)
  # De-risk a new model's serving path: load it, confirm one chat completion, exit.
  MODEL_REPO="${MODEL_REPO:-RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8}"
  SERVED="${SERVED:-mistral-small-3.2-24b-fp8}"
  # Mistral-Small-3.2 is Pixtral/multimodal; --limit-mm-per-prompt {"image":0} skips image
  # profiling so vLLM serves it text-only. Keep the default in its OWN var — a literal '}'
  # inside ${VAR:-default} closes the ${...} early (bash brace mis-parse).
  _DEFAULT_SERVE='--tokenizer-mode mistral --config-format mistral --load-format mistral --limit-mm-per-prompt {"image":0}'
  # '-' (not ':-'): an explicitly empty EXTRA_SERVE="" is honoured (e.g. a non-Mistral model).
  EXTRA_SERVE="${EXTRA_SERVE-$_DEFAULT_SERVE}"
  PORT="${PORT:-11435}"   # off 11434 so a co-located run's vLLM doesn't collide
  VLOG="$AEL_LOG_DIR/vllm-smoke-${SLURM_JOB_ID:-local}.log"

  echo "=== serve smoke: $MODEL_REPO (served=$SERVED) ==="
  PYTHONPATH="" vllm serve "$MODEL_REPO" --port "$PORT" --served-model-name "$SERVED" \
     --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager $EXTRA_SERVE \
     > "$VLOG" 2>&1 &
  SERVE_PID=$!
  trap 'kill $SERVE_PID 2>/dev/null' EXIT
  wait_ready "$PORT" "$SERVE_PID" "$VLOG" 450 || exit 1

  echo "=== confirm served model is really $SERVED ==="
  curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c "import sys,json; print('served:', [m['id'] for m in json.load(sys.stdin).get('data',[])])"
  echo "=== test chat completion ==="
  RESP=$(curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: JUDGE-OK\"}],\"max_tokens\":16,\"temperature\":0}")
  echo "RAW: ${RESP:0:400}"
  echo "$RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin); c=d.get('choices')
if c: print('=== SMOKE PASS:', repr(c[0]['message']['content'])); sys.exit(0)
print('=== SMOKE FAIL: no choices ->', json.dumps(d)[:300]); sys.exit(1)
"
  RC=$?
  ;;

pipeline|model|data)
  # Full pipeline risk-smoke (pipeline), ModelTeam-only fast check (model),
  # or DataTeam ModeOpenSourceAPI-only fast check (data).
  export AEL_DISABLE_THINKING=1        # Qwen3.6: fast, no reasoning-trace timeouts
  MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-27B-FP8}"
  SERVED="${SERVED:-qwen3.6-27b-fp8}"
  export AEL_MODEL="vllm/$SERVED"       # local vLLM on $PORT (default :11434)
  PORT="${PORT:-11434}"
  # full chat-completions URL (the format shared/llm.py expects); follows a PORT override
  export VLLM_BASE_URL="http://localhost:$PORT/v1/chat/completions"
  SMOKE="$AEL_RUNS_DIR/b_smoke_$TARGET"; mkdir -p "$SMOKE"
  VLOG="$AEL_LOG_DIR/vllm-${SLURM_JOB_ID:-local}.log"

  if [ "$TARGET" = pipeline ]; then
    echo; echo "### EGRESS PROBE (compute node $(hostname)) ###"
    for url in https://huggingface.co https://api.openai.com https://export.arxiv.org \
               https://api.firecrawl.dev https://api.semanticscholar.org https://api.stlouisfed.org; do
      code=$(timeout 8 curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo "FAIL")
      echo "  egress $url -> $code"
    done
    GPU_UTIL=0.90; EAGER=--enforce-eager
  elif [ "$TARGET" = data ]; then
    # CUDA-graph capture (graphs on, 0.85) ran out of memory with this model on an
    # A100 40GB (torch.OutOfMemoryError in profile_cudagraph_memory, ~38.8 of 39.5 GiB
    # in use before the 1.53 GiB graph-capture allocation). Eager mode avoids that
    # overhead, as in the pipeline target.
    GPU_UTIL=0.90; EAGER=--enforce-eager
  else
    GPU_UTIL=0.85; EAGER=""            # model target: original smoke_model used graphs @ 0.85
  fi

  echo; echo "### starting vLLM -> $VLOG ###"
  PYTHONPATH="" vllm serve "$MODEL_REPO" --port "$PORT" --served-model-name "$SERVED" \
    --tensor-parallel-size 1 --max-model-len 16384 \
    --gpu-memory-utilization "$GPU_UTIL" --reasoning-parser qwen3 $EAGER \
    > "$VLOG" 2>&1 &
  SERVE_PID=$!
  trap 'kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null' EXIT
  wait_ready "$PORT" "$SERVE_PID" "$VLOG" 600 || exit 1

  echo; echo "### ROUTING VERIFY ###"
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" hpc/verify_routing.py || { echo "ROUTING VERIFY FAILED"; exit 1; }

  run_smoke() {  # $1=label $2=team $3=mode $4=extra-env $5=timeout
    echo; echo "############ SMOKE: $1 ($2/$3) @ $(date +%H:%M:%S) ############"
    env PYTHONPATH="$AEL_PYTHONPATH" $4 "$AEL_VENV/bin/python" -m evaluation.runners.multirun_orchestrator \
        --team "$2" --mode "$3" --runs 1 --output-dir "$SMOKE/$1" --timeout "$5"
    local rc=$?
    echo "smoke $1 exit=$rc -> $SMOKE/$1"
    [ "$rc" -ne 0 ] && RC=$rc   # keep going with the next run; the job still fails at the end
    return 0
  }
  if [ "$TARGET" = pipeline ]; then
    run_smoke ideation_nowc IdeationTeam ModeNoWcNoHITL "" 1800
    run_smoke ideation_hitl IdeationTeam ModeNoWcWithHITL "AUTO_HITL_MODE=true AUTO_HITL_TIMEOUT=30" 1800
    run_smoke modelteam_nowc ModelTeam ModeNoWcNoHITL "" 1800
  elif [ "$TARGET" = data ]; then
    run_smoke dataTeam_opensource DataTeam ModeOpenSourceAPI "" 1800
  else
    run_smoke modelteam_nowc ModelTeam ModeNoWcNoHITL "" 7200
  fi
  echo; echo "=== SMOKE [$TARGET] DONE @ $(date). results in $SMOKE ==="
  find "$SMOKE" -name 'execution_log*.json' | sort
  ;;

code)
  # CodeTeam ModeNoWcWithHITL only (fast). Runs 0-MasterOrchestrator.py from a scratch
  # copy (run artifacts are not written into the tracked mode directory), fed the
  # ModelTeam design/calibration outputs of a completed run. AEL_MODEL is exported before
  # anything runs so it overrides every agent's default model: if .env holds an
  # OPENAI_API_KEY as commercial fallback, that fallback would otherwise become the default.
  export AEL_DISABLE_THINKING=1
  MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-27B-FP8}"
  SERVED="${SERVED:-qwen3.6-27b-fp8}"
  export AEL_MODEL="vllm/$SERVED"
  PORT="${PORT:-11434}"
  export VLLM_BASE_URL="http://localhost:$PORT/v1/chat/completions"   # follows a PORT override
  VLOG="$AEL_LOG_DIR/vllm-${SLURM_JOB_ID:-local}.log"

  echo; echo "### starting vLLM (eager -- graphs OOM'd on this model/A100 combo, see 'data' target) -> $VLOG ###"
  PYTHONPATH="" vllm serve "$MODEL_REPO" --port "$PORT" --served-model-name "$SERVED" \
    --tensor-parallel-size 1 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --reasoning-parser qwen3 --enforce-eager \
    > "$VLOG" 2>&1 &
  SERVE_PID=$!
  trap 'kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null' EXIT
  wait_ready "$PORT" "$SERVE_PID" "$VLOG" 600 || exit 1

  echo; echo "### ROUTING VERIFY ###"
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" hpc/verify_routing.py || { echo "ROUTING VERIFY FAILED"; exit 1; }

  DESIGN_SRC="${AEL_CODE_MODEL_DESIGN:-${AEL_SMOKE_REF_RUN:?set AEL_SMOKE_REF_RUN or AEL_CODE_MODEL_DESIGN}/ModelTeam/model_design_output.json}"
  CALIB_SRC="${AEL_CODE_CALIBRATION:-${AEL_SMOKE_REF_RUN:?set AEL_SMOKE_REF_RUN or AEL_CODE_CALIBRATION}/ModelTeam/calibration_output.json}"
  RUN_DIR="$AEL_RUNS_DIR/b_smoke_code/ModeNoWcWithHITL_$(date +%Y%m%d%H%M%S)"
  mkdir -p "$RUN_DIR"
  cp "$AEL_CODE_DIR/CodeTeam/ael/ModeNoWcWithHITL/"*.py "$RUN_DIR/"

  echo; echo "### running CodeTeam ModeNoWcWithHITL (real ModelTeam inputs: $DESIGN_SRC) ###"
  cd "$RUN_DIR"
  AUTO_HITL_MODE=true \
  AEL_CODE_MODEL_DESIGN="$DESIGN_SRC" \
  AEL_CODE_CALIBRATION="$CALIB_SRC" \
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" 0-MasterOrchestrator.py
  RC=$?
  echo "CodeTeam smoke exit=$RC -> $RUN_DIR"
  echo; echo "=== SMOKE [code] DONE @ $(date). results in $RUN_DIR ==="
  ;;

estimation)
  # EstimationTeam ModeNoWcWithHITL only (fast). Runs 0-MasterOrchestrator.py from a
  # scratch copy, fed the ModelTeam + DataTeam outputs of a completed run. AEL_MODEL is set
  # before any stage module's load_dotenv() call, as in the 'code' target: load_dotenv()
  # re-reads .env and would otherwise restore a commercial OPENAI_API_KEY as the default.
  export AEL_DISABLE_THINKING=1
  MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-27B-FP8}"
  SERVED="${SERVED:-qwen3.6-27b-fp8}"
  export AEL_MODEL="vllm/$SERVED"
  PORT="${PORT:-11434}"
  export VLLM_BASE_URL="http://localhost:$PORT/v1/chat/completions"   # follows a PORT override
  VLOG="$AEL_LOG_DIR/vllm-${SLURM_JOB_ID:-local}.log"

  echo; echo "### starting vLLM (eager -- see 'data' target for why) -> $VLOG ###"
  PYTHONPATH="" vllm serve "$MODEL_REPO" --port "$PORT" --served-model-name "$SERVED" \
    --tensor-parallel-size 1 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --reasoning-parser qwen3 --enforce-eager \
    > "$VLOG" 2>&1 &
  SERVE_PID=$!
  trap 'kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null' EXIT
  wait_ready "$PORT" "$SERVE_PID" "$VLOG" 600 || exit 1

  echo; echo "### ROUTING VERIFY ###"
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" hpc/verify_routing.py || { echo "ROUTING VERIFY FAILED"; exit 1; }

  SPEC_SRC="${AEL_ESTIM_MODEL_SPEC:-${AEL_SMOKE_REF_RUN:?set AEL_SMOKE_REF_RUN or AEL_ESTIM_MODEL_SPEC}/ModelTeam/calibration_output.json}"
  DATA_SRC="${AEL_ESTIM_DATA:-${AEL_SMOKE_REF_RUN:?set AEL_SMOKE_REF_RUN or AEL_ESTIM_DATA}/DataTeam/api_source_output.json}"
  RUN_DIR="$AEL_RUNS_DIR/b_smoke_estimation/ModeNoWcWithHITL_$(date +%Y%m%d%H%M%S)"
  mkdir -p "$RUN_DIR"
  cp "$AEL_CODE_DIR/EstimationTeam/ael/ModeNoWcWithHITL/"*.py "$RUN_DIR/"

  echo; echo "### running EstimationTeam ModeNoWcWithHITL (real inputs: $SPEC_SRC / $DATA_SRC) ###"
  cd "$RUN_DIR"
  AUTO_HITL_MODE=true \
  AEL_ESTIM_MODEL_SPEC="$SPEC_SRC" \
  AEL_ESTIM_DATA="$DATA_SRC" \
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" 0-MasterOrchestrator.py
  RC=$?
  echo "EstimationTeam smoke exit=$RC -> $RUN_DIR"
  echo; echo "=== SMOKE [estimation] DONE @ $(date). results in $RUN_DIR ==="
  ;;

reporting)
  # ReportingTeam ModeNoWcWithHITL only (fast). Runs 0-MasterOrchestrator.py from a
  # scratch copy, fed the artifacts directory and EstimationTeam inference_output.json of a
  # completed run. AEL_MODEL is set first, as in the 'code' target.
  export AEL_DISABLE_THINKING=1
  MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-27B-FP8}"
  SERVED="${SERVED:-qwen3.6-27b-fp8}"
  export AEL_MODEL="vllm/$SERVED"
  PORT="${PORT:-11434}"
  export VLLM_BASE_URL="http://localhost:$PORT/v1/chat/completions"   # follows a PORT override
  VLOG="$AEL_LOG_DIR/vllm-${SLURM_JOB_ID:-local}.log"

  echo; echo "### starting vLLM (eager -- see 'data' target for why) -> $VLOG ###"
  PYTHONPATH="" vllm serve "$MODEL_REPO" --port "$PORT" --served-model-name "$SERVED" \
    --tensor-parallel-size 1 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --reasoning-parser qwen3 --enforce-eager \
    > "$VLOG" 2>&1 &
  SERVE_PID=$!
  trap 'kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null' EXIT
  wait_ready "$PORT" "$SERVE_PID" "$VLOG" 600 || exit 1

  echo; echo "### ROUTING VERIFY ###"
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" hpc/verify_routing.py || { echo "ROUTING VERIFY FAILED"; exit 1; }

  REF_RUN="${AEL_REPORT_REF_RUN:-${AEL_SMOKE_REF_RUN:-}}"
  if [ -z "$REF_RUN" ] && { [ -z "${AEL_REPORT_ESTIMATION:-}" ] || [ -z "${AEL_REPORT_ARTIFACTS:-}" ]; }; then
    echo "set AEL_SMOKE_REF_RUN (or AEL_REPORT_ESTIMATION and AEL_REPORT_ARTIFACTS)"; exit 2
  fi
  EST_SRC="${AEL_REPORT_ESTIMATION:-$REF_RUN/EstimationTeam/inference_output.json}"
  ARTIFACTS_SRC="${AEL_REPORT_ARTIFACTS:-$REF_RUN/artifacts}"
  RUN_DIR="$AEL_RUNS_DIR/b_smoke_reporting/ModeNoWcWithHITL_$(date +%Y%m%d%H%M%S)"
  mkdir -p "$RUN_DIR"
  cp "$AEL_CODE_DIR/ReportingTeam/ael/ModeNoWcWithHITL/"*.py "$RUN_DIR/"

  echo; echo "### running ReportingTeam ModeNoWcWithHITL (real inputs: $EST_SRC / $ARTIFACTS_SRC) ###"
  cd "$RUN_DIR"
  AUTO_HITL_MODE=true \
  AEL_REPORT_ESTIMATION="$EST_SRC" \
  AEL_REPORT_ARTIFACTS="$ARTIFACTS_SRC" \
  PYTHONPATH="$AEL_PYTHONPATH" "$AEL_VENV/bin/python" 0-MasterOrchestrator.py
  RC=$?
  echo "ReportingTeam smoke exit=$RC -> $RUN_DIR"
  echo; echo "=== SMOKE [reporting] DONE @ $(date). results in $RUN_DIR ==="
  ;;

speedup)
  # Measure throughput levers on Qwen3.6-27B-FP8 (does NOT touch the production path):
  #   CUDA-graphs vs --enforce-eager (EAGER=1 for the slow/stable baseline) + concurrent batching.
  MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-27B-FP8}"
  SERVED="${SERVED:-qwen3.6-27b-fp8}"
  PORT="${PORT:-11437}"
  EAGER="${EAGER:-0}"
  VLOG="$AEL_LOG_DIR/vllm-speedup-${SLURM_JOB_ID:-local}.log"
  EAGER_FLAG=""; [ "$EAGER" = 1 ] && EAGER_FLAG="--enforce-eager"

  echo "=== speedup: EAGER=$EAGER flag='$EAGER_FLAG' + prefix-caching ==="
  PYTHONPATH="" vllm serve "$MODEL_REPO" --port "$PORT" --served-model-name "$SERVED" \
     --max-model-len 16384 --gpu-memory-utilization 0.90 $EAGER_FLAG \
     --enable-prefix-caching --reasoning-parser qwen3 \
     > "$VLOG" 2>&1 &
  SERVE_PID=$!
  trap 'kill $SERVE_PID 2>/dev/null' EXIT
  wait_ready "$PORT" "$SERVE_PID" "$VLOG" 600 || exit 1

  python3 - "$PORT" "$SERVED" <<'PY'
import sys, json, time, urllib.request, concurrent.futures as cf
port, served = sys.argv[1], sys.argv[2]
URL=f"http://127.0.0.1:{port}/v1/chat/completions"
PROMPT="Write a detailed 400-word explanation of how monetary policy affects inflation and output."
def gen(_=None, n=256):
    body=json.dumps({"model":served,"messages":[{"role":"user","content":PROMPT}],
                     "max_tokens":n,"temperature":0,
                     "chat_template_kwargs":{"enable_thinking":False}}).encode()
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"}),timeout=300)
    d=json.load(r); dt=time.time()-t0
    return d.get("usage",{}).get("completion_tokens",0), dt
gen()                       # warmup (captures graphs / fills prefix cache)
ct,dt=gen()
single=ct/dt if dt else 0
print(f"SINGLE: {ct} tok in {dt:.1f}s -> {single:.1f} tok/s  (eager baseline on one A100 40GB: ~15)")
for N in (4, 8):
    t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        res=list(ex.map(gen, range(N)))
    wall=time.time()-t0; tot=sum(c for c,_ in res)
    print(f"CONCURRENT x{N}: {tot} tok in {wall:.1f}s wall -> {tot/wall:.1f} tok/s AGGREGATE  (~{(tot/wall)/single:.1f}x single)")
print("=> adopt eager-off if SINGLE >> 15 and it started; exploit concurrency if AGGREGATE scales with N")
PY
  RC=$?
  echo "=== SPEEDUP DONE rc=$RC ==="
  ;;

ideation)
  # End-to-end open ideation news path: serve Qwen + SearXNG, run real fetch_trending_topics().
  export AEL_DISABLE_THINKING=1
  export AEL_MODEL="vllm/qwen3.6-27b-fp8"
  export SEARXNG_BASE_URL="${SEARXNG_BASE_URL:-http://localhost:8899}"
  PORT="${PORT:-11434}"
  export VLLM_BASE_URL="http://localhost:$PORT/v1/chat/completions"   # follows a PORT override
  VLOG="$AEL_LOG_DIR/vllm-idea-${SLURM_JOB_ID:-local}.log"
  SXLOG="$AEL_LOG_DIR/searxng-idea-${SLURM_JOB_ID:-local}.log"

  echo "=== serving Qwen + SearXNG ==="
  PYTHONPATH="" vllm serve Qwen/Qwen3.6-27B-FP8 --port "$PORT" \
     --served-model-name qwen3.6-27b-fp8 --max-model-len 16384 --gpu-memory-utilization 0.90 \
     --enforce-eager --reasoning-parser qwen3 > "$VLOG" 2>&1 &
  VP=$!
  PYTHONPATH="" SEARXNG_SETTINGS_PATH="$AEL_CODE_DIR/hpc/searxng-settings.yml" \
     "$AEL_SEARXNG_VENV/bin/python" -m searx.webapp > "$SXLOG" 2>&1 &
  SP=$!
  trap 'kill $VP $SP 2>/dev/null' EXIT
  wait_ready "$PORT" "$VP" "$VLOG" 900 || exit 1
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:8899/ >/dev/null 2>&1 && { echo "SearXNG ready ${i}s"; break; }; sleep 2; done

  echo "=== run the REAL fetch_trending_topics (open path) ==="
  PYTHONPATH="$AEL_CODE_DIR" "$AEL_VENV/bin/python" - <<'PY'
import importlib.util, os, sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.environ["PWD"], ".env"))  # commercial keys = fallback only
d=os.path.join(os.environ["PWD"], "IdeationTeam/ael/ModeWithWcNoHITL")
os.chdir(d)
spec=importlib.util.spec_from_file_location("ideation_master", os.path.join(d,"0-MasterOrchestrator.py"))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
topic=m.fetch_trending_topics()
fallback=m._get_fallback_topics()
print("TRENDING TOPIC:", repr(topic))
print("IS-FALLBACK:", topic.strip()==fallback.strip())
ok = bool(topic) and topic.strip() != fallback.strip()
print("IDEATION-AUTO-PASS" if ok else "IDEATION-AUTO-FELL-BACK")
sys.exit(0 if ok else 1)
PY
  RC=$?
  echo "=== DONE rc=$RC ==="
  ;;

*)
  echo "unknown target '$TARGET' — use: serve | pipeline | model | data | code | estimation | reporting | speedup | ideation"; exit 2 ;;
esac
exit "$RC"
