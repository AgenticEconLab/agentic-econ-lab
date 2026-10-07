# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Link the estimation to the CodeTeam's executable model.

The CodeTeam derives one steady-state module per formal model. The estimation stage works
on the run's lead model (the first calibrated model in the specification). This module

* selects the lead model's executable module,
* summarizes it for the Estimator prompt (module variables with their names, the
  validation verdict, and the solved steady state when there is one), and
* records, deterministically, which part of the module the estimation used: the module
  variables the specification declares for its series (``VariableSpec.model_variable``),
  whether each declared symbol exists in the module, and its steady-state value when the
  module solved one.

Nothing here changes the estimate or the verdict.
"""

import re
from typing import Any, Dict, List, Optional

_TITLE_PREFIX = re.compile(r"^\s*(formal|calibrated)\s+model\s*:\s*", re.IGNORECASE)


def _norm_title(title: Any) -> str:
    return _TITLE_PREFIX.sub("", str(title or "")).strip().lower()


def _base_symbol(symbol: str) -> str:
    """'c_{i,t}' -> 'c', 'R_t' -> 'R', '\\pi_t' -> 'pi' (module variables drop subscripts)."""
    s = str(symbol or "").strip().lstrip("\\")
    s = re.split(r"_\{|_t\b|_\{t|\^|\(", s)[0]
    return s.strip()


def lead_model_title(model_spec: Dict) -> str:
    models = (model_spec or {}).get("calibrated_models") or []
    if models and isinstance(models[0], dict):
        return str(models[0].get("model_title") or "")
    return ""


def select_module(executable_model: Optional[Dict], model_spec: Optional[Dict],
                  model_design: Optional[Dict] = None) -> Optional[Dict]:
    """Return {index, generation, validation, variable_names} for the lead model, or None."""
    if not isinstance(executable_model, dict):
        return None
    gens = ((executable_model.get("generation") or {}).get("results")) or []
    vals = ((executable_model.get("validation") or {}).get("results")) or []
    if not gens:
        return None
    lead = _norm_title(lead_model_title(model_spec or {}))
    idx = next((i for i, g in enumerate(gens)
                if lead and _norm_title(g.get("model_title")) == lead), 0)
    gen = gens[idx]
    val = vals[idx] if idx < len(vals) else {}

    # human-readable names for module variables, from the matching formal model
    names: Dict[str, str] = {}
    formal = ((model_design or {}).get("formal_models")) or []
    fm = next((f for f in formal if isinstance(f, dict)
               and _norm_title(f.get("model_title")) == _norm_title(gen.get("model_title"))),
              None)
    module_vars = set(gen.get("variables") or [])
    for v in (fm or {}).get("variables") or []:
        if not isinstance(v, dict):
            continue
        base = _base_symbol(v.get("variable_symbol", ""))
        if base in module_vars and base not in names:
            names[base] = str(v.get("variable_name") or "")
    return {"index": idx, "generation": gen, "validation": val or {}, "variable_names": names}


def describe_for_prompt(module: Optional[Dict], limit: int = 1200) -> str:
    """Prompt block for the Estimator; empty when the CodeTeam did not run."""
    if not module:
        return ""
    gen, val = module["generation"], module["validation"]
    ss = val.get("steady_state") or {}
    names = module["variable_names"]
    var_lines = []
    for sym in gen.get("variables") or []:
        if sym.startswith("sym_"):
            continue                    # emitted names for compound expressions
        label = names.get(sym, "")
        value = f", steady state {ss[sym]:.4g}" if sym in ss else ""
        var_lines.append(f"{sym}" + (f" ({label})" if label else "") + value)
    text = (f"Module for '{gen.get('model_title', '')}': generation `{gen.get('verdict')}`, "
            f"validation `{val.get('verdict', 'n/a')}`; "
            + ("steady state solved" if ss else "no solved steady state")
            + ".\nModule variables: " + "; ".join(var_lines))
    return text[:limit]


def record_use(module: Optional[Dict], spec: Any, *, code_team_ran: bool) -> Dict[str, Any]:
    """Deterministic record of which part of the executable model the estimation used."""
    if module is None:
        return {
            "available": False,
            "reason": ("the CodeTeam produced no module for the lead model" if code_team_ran
                       else "the CodeTeam did not run (optional node disabled)"),
        }
    gen, val = module["generation"], module["validation"]
    module_vars = set(gen.get("variables") or [])
    ss = val.get("steady_state") or {}
    declared: Dict[str, str] = {}
    if spec is not None:
        for role in [spec.dependent, *spec.regressors]:
            sym = getattr(role, "model_variable", None)
            if sym:
                declared[role.name] = sym
    verified = {r: s for r, s in declared.items() if s in module_vars}
    unknown = {r: s for r, s in declared.items() if s not in module_vars}
    notes: List[str] = []
    if spec is None:
        notes.append("no specification was estimated")
    elif not declared:
        notes.append("the specification links no series to a module variable")
    if unknown:
        notes.append("declared symbols absent from the module: "
                     + ", ".join(f"{r} -> {s}" for r, s in unknown.items()))
    if not ss:
        notes.append("the module has no solved steady state, so no model-implied "
                     "levels are available for the linked variables")
    return {
        "available": True,
        "module_index": module["index"],
        "model_title": gen.get("model_title", ""),
        "generation_verdict": gen.get("verdict"),
        "validation_verdict": val.get("verdict"),
        "module_variables_shown": sorted(v for v in module_vars if not v.startswith("sym_")),
        "linked_variables": verified,
        "unknown_symbols": unknown,
        "steady_state_values": {s: ss[s] for s in verified.values() if s in ss},
        "notes": notes,
    }
