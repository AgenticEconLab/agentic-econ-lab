#!/usr/bin/env bash
# One-time setup: install SearXNG into its own venv (no Docker), start it once and check that
# its JSON API returns results. Run on a login node with outbound internet access:
#   AEL_SCRATCH=/path/to/scratch bash hpc/setup_searxng.sh
# Does not source _env.sh, so that the vLLM venv on PATH cannot replace the system
# `python3` used for `python3 -m venv` below. The venv lands at the path _env.sh expects
# ($AEL_SEARXNG_VENV, default $AEL_SCRATCH/tools/venv-searxng).
set -uo pipefail
: "${AEL_SCRATCH:?set AEL_SCRATCH to a writable scratch directory}"
BASE="$AEL_SCRATCH/tools"
mkdir -p "$BASE"
SRC=$BASE/searxng-src
VENV="${AEL_SEARXNG_VENV:-$BASE/venv-searxng}"
PORT=8899
cd "$BASE"

echo "=== 1) clone searxng ==="
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 https://github.com/searxng/searxng.git "$SRC" 2>&1 | tail -3
else echo "(already cloned)"; fi

echo "=== 2) venv + install (requirements first, then --no-build-isolation) ==="
[ -d "$VENV" ] || env -u PYTHONPATH python3 -m venv "$VENV"
env -u PYTHONPATH "$VENV/bin/pip" install -q --upgrade pip setuptools wheel 2>&1 | tail -1
# setup.py imports searx/__init__ -> msgspec at build time; install runtime deps first
env -u PYTHONPATH "$VENV/bin/pip" install -q -r "$SRC/requirements.txt" 2>&1 | tail -3
env -u PYTHONPATH "$VENV/bin/pip" install -q -e "$SRC" --no-build-isolation 2>&1 | tail -4
"$VENV/bin/python" -c "import searx; print('searx import OK')" || { echo "searx still not importable"; exit 1; }

echo "=== 3) minimal settings (JSON API on, limiter/redis off) ==="
cat > "$BASE/searxng-settings.yml" <<'YML'
use_default_settings: true
general:
  debug: false
server:
  secret_key: "searxng-hpc-open-news-key-change-me"
  limiter: false
  image_proxy: false
  port: 8899
  bind_address: "127.0.0.1"
search:
  formats:
    - html
    - json
YML

echo "=== 4) launch webapp ==="
export SEARXNG_SETTINGS_PATH="$BASE/searxng-settings.yml"
env -u PYTHONPATH SEARXNG_SETTINGS_PATH="$BASE/searxng-settings.yml" \
   "$VENV/bin/python" -m searx.webapp > "$BASE/searxng-run.log" 2>&1 &
SXPID=$!
trap "kill $SXPID 2>/dev/null" EXIT

ready=0
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 || curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    ready=1; echo "READY after ${i}s"; break; fi
  kill -0 "$SXPID" 2>/dev/null || { echo "SEARXNG DIED DURING START"; tail -30 "$BASE/searxng-run.log"; exit 1; }
  sleep 2
done
[ "$ready" = 1 ] || { echo "NEVER READY"; tail -30 "$BASE/searxng-run.log"; exit 1; }

echo "=== 5) query JSON API: economic news ==="
curl -s "http://127.0.0.1:${PORT}/search?q=economic+research+trends+2026&format=json&categories=news" \
 | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception as e: print('JSON parse failed:',e); sys.exit(1)
r=d.get('results',[])
print('results:',len(r),'| engines used:', d.get('search_engines') or [e for e in d.get('engines',[])][:6])
for x in r[:6]: print('  -',(x.get('title') or '')[:70],'|',(x.get('url') or '')[:60])
print('SEARXNG-JSON-OK' if r else 'SEARXNG-RETURNED-ZERO (search engines may block the cluster IP)')
"
echo "=== SEARXNG TEST DONE ==="
