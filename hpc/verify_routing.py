#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""One-call routing verification: prove AEL -> local vLLM (NOT OpenAI).

Run under the AEL venv ($AEL_VENV) with PYTHONPATH=<repository root> and AEL_MODEL set.
Deliberately leaves OPENAI_API_KEY unset: a successful completion can ONLY have
come from a local OpenAI-compatible endpoint (Layer 1 vllm/ or Layer 2 ollama/). The endpoint
checked is the one shared/llm.py will call: for vllm/, AEL_VLLM_ENDPOINTS[<served name>], else
VLLM_BASE_URL, else the default http://localhost:11434/v1/chat/completions.
"""
import os
import socket
import sys
from urllib.parse import urlparse

os.environ.pop("OPENAI_API_KEY", None)  # make the proof airtight

from shared.llm import LLMClient, _vllm_endpoint  # noqa: E402
from shared.provider_map import detect_provider, OPENAI_COMPATIBLE_PROVIDERS  # noqa: E402

model = os.environ.get("AEL_MODEL", "vllm/qwen3.6-27b-fp8")
provider = detect_provider(model)
base_url = OPENAI_COMPATIBLE_PROVIDERS.get(provider, {}).get("base_url", "<none>")
if provider == "vllm":
    served = model[len("vllm/"):] if model.startswith("vllm/") else model
    base_url = _vllm_endpoint(served) or os.environ.get("VLLM_BASE_URL", base_url)
print(f"[verify] AEL_MODEL={model}  provider={provider}  base_url={base_url}", flush=True)
assert provider in ("vllm", "ollama"), f"expected a local provider (vllm/ollama), got {provider!r}"
host = urlparse(base_url).hostname or ""
local_hosts = {"localhost", "127.0.0.1", "::1", socket.gethostname(), socket.getfqdn()}
assert host in local_hosts, f"base_url does not point at a server on this node: {base_url}"

client = LLMClient(model=model, temperature=0)
reply = client.invoke([
    {"role": "user", "content": "Reply with exactly the single word: PONG"},
])
print(f"[verify] raw reply={reply!r}", flush=True)
assert "PONG" in reply.upper(), f"unexpected reply: {reply!r}"
print("[verify] OK — AEL routed to local vLLM and got a completion.", flush=True)
sys.exit(0)
