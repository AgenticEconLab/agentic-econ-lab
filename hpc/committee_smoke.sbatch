#!/usr/bin/env bash
#SBATCH --job-name=ael-committee-smoke
##SBATCH --account=<your-account>          # site-specific: remove one '#' and edit, or pass --account on the sbatch line
##SBATCH --partition=<your-gpu-partition>  # site-specific: remove one '#' and edit, or pass --partition on the sbatch line
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=380G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Committee smoke test: serve the LLM-Economist committee in the same layout as
# full_pipeline.sbatch (hpc/_committee.sh; GPU0 left free for the generating model) and
# check that every member casts a vote and that a free-text review comes back.
#
# Usage: AEL_SCRATCH=/path/to/scratch sbatch hpc/committee_smoke.sbatch
set -uo pipefail

_cmd="$(scontrol show job "${SLURM_JOB_ID:-_}" 2>/dev/null | sed -n 's/.*Command=\([^ ]*\).*/\1/p')"
HPC_DIR="$(cd "$(dirname "${_cmd:-${BASH_SOURCE[0]}}")" && pwd)"
source "$HPC_DIR/_env.sh"
source "$HPC_DIR/_committee.sh"
cd "$AEL_CODE_DIR"

# kernel-compile caches on node-local /tmp (see full_pipeline.sbatch)
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID}_triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${SLURM_JOB_ID}_inductor"
export VLLM_CACHE_ROOT="/tmp/${SLURM_JOB_ID}_vllm"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

echo "=== AEL COMMITTEE SMOKE ${SLURM_JOB_ID:-local} on $(hostname) @ $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

serve_committee || exit 1
export AEL_VLLM_ENDPOINTS="$COMMITTEE_ENDPOINTS"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

env -u PYTHONPATH PYTHONPATH="$AEL_CODE_DIR" PYDANTIC_DISABLE_PLUGINS=1 "$AEL_VENV/bin/python" - <<'PY'
import sys, time
from shared.llm_economist import LLMEconomistCommittee, get_committee_models

members = get_committee_models()
print("committee:", members)
c = LLMEconomistCommittee(members=members, log_path="")

t0 = time.time()
d = c.decide(
    "The ModelTeam proposes calibrating the discount factor beta = 0.99 at quarterly "
    "frequency, matching a 4% annual real interest rate. Approve this calibration? (yes/no)",
    default="yes", input_type="yes_no")
print(f"vote: approve={d.approve} votes {d.approve_votes}-{d.object_votes} "
      f"agreement={d.agreement:.2f} ({time.time()-t0:.0f}s)")
abstained = [v for v in d.ballot if v.abstained]
for v in d.ballot:
    print("  ", v)
t0 = time.time()
f = c.decide("Review this research question and suggest one improvement: How does "
             "monetary policy affect inequality?", default="", input_type="feedback")
print(f"feedback ({time.time()-t0:.0f}s): {str(f.answer)[:400]}")

ok = not abstained and len(d.ballot) == 3 and bool(str(f.answer).strip())
print("COMMITTEE SMOKE", "PASSED" if ok else "FAILED",
      f"(abstained: {[v.model for v in abstained] if abstained else 'none'})")
sys.exit(0 if ok else 1)
PY
RC=$?
echo "=== COMMITTEE SMOKE DONE rc=$RC @ $(date) ==="
exit $RC
