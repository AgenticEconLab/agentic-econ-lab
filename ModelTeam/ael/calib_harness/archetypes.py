# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Hand-coded, unit-tested economic archetypes for the deterministic calibration harness.

Two public entry points (both pure-Python, **zero LLM calls**):

    map_param_roles(parameters)        -> List[ParamRole]   # CONFIRM canonical roles (never guess)
    detect_moments(parsed_eqs, roles)  -> List[Moment]      # signature-propose + bounded-confirm

Design contract:

* **CONFIRM, never guess.** A parameter is given a canonical role ONLY when BOTH its
  ``typical_range`` AND its name/description agree. Anything ambiguous resolves to ``role=None``
  (fail-closed). This is what stops the classic role trap: in real Model 0 the symbol ``alpha`` is a
  **LABOR-share** parameter on ``[0.3, 0.7]`` ("...output with respect to human labor input..."),
  NOT capital share — so the ``capital_output_ratio`` / ``labor_share = 1 - alpha`` moments that
  assume alpha = capital share MUST NOT fire on it.

* **Signature-propose + bounded-confirm.** A moment is proposed from a cheap STRUCTURAL signature
  (e.g. a CES exponent ``(rho-1)/rho``, a Cobb-Douglas ``K^alpha``, a capital law of motion
  ``(1-delta)*K``, a discount Euler ``beta*E_t``) and then confirmed by the *presence of the
  relevant param roles*. It is NEVER required that the model's whole equation be algebraically
  equal to a textbook one (the equations are messy, AI-augmented, often non-algebraic).

* **Closed forms carry frequency metadata + an explicit ``annualize()``.** The infamous 9.40 -> 2.35
  capital/output gap was a quarterly-vs-annual *unit* error, not a modelling failure: K/Y built from
  a quarterly beta is a quarterly stock/flow ratio and must be divided by periods-per-year to compare
  to an annual target.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Tuple

from .types import Moment, ParamRole

try:  # parser is a sibling module; only its dataclass type is referenced.
    from .parser import ParsedEquation
except Exception:  # pragma: no cover - parser import is part of the locked contract
    ParsedEquation = object  # type: ignore


# ---------------------------------------------------------------------------------------------
# 1. CLOSED-FORM ARCHETYPES  (pure functions, directly unit-testable against textbook values)
# ---------------------------------------------------------------------------------------------
# All RBC/NK steady-state identities below take a *capital* share alpha (NOT labor share) and a
# quarterly beta/delta unless stated. Keep them tiny and dependency-free so the tests are airtight.

def capital_output_ratio(alpha: float, beta: float, delta: float) -> float:
    """Steady-state K/Y in a Cobb-Douglas RBC/NK core: K/Y = alpha / (1/beta - 1 + delta).

    With a quarterly beta/delta this is a *quarterly* stock/flow ratio (annualize -> divide by 4).
    """
    return alpha / (1.0 / beta - 1.0 + delta)


def investment_output_ratio(alpha: float, beta: float, delta: float) -> float:
    """Steady-state I/Y = delta * K/Y. A flow/flow ratio -> frequency-invariant (no annualization)."""
    return delta * capital_output_ratio(alpha, beta, delta)


def real_interest_rate(beta: float) -> float:
    """Steady-state net real rate r* = 1/beta - 1 (per-period; quarterly beta -> quarterly r*)."""
    return 1.0 / beta - 1.0


def labor_share_from_capital(alpha_capital: float) -> float:
    """Cobb-Douglas labor share = 1 - capital_share. (Only valid when alpha really IS capital share.)"""
    return 1.0 - alpha_capital


# Quarterly MPC of the UNCONSTRAINED (permanent-income) household — the standard HANK/TANK
# auxiliary constant (~r/(1+r); Kaplan & Violante 2014 use ≈0.05 quarterly). Fixed, documented,
# and NOT estimated: the archetype identifies only the hand-to-mouth share.
MPC_UNCONSTRAINED_Q = 0.05


def aggregate_mpc(htm_share: float, mpc_unconstrained: float = MPC_UNCONSTRAINED_Q) -> float:
    """Aggregate quarterly MPC in a two-agent (TANK/HANK-lite) economy.

    Hand-to-mouth households consume transfers one-for-one (MPC=1); the unconstrained consume
    ``mpc_unconstrained``: MPC = λ·1 + (1−λ)·mpc_u. Monotone in λ, so the cited quarterly MPC
    target (Johnson-Parker-Souleles 2006; Kaplan-Violante 2014) just-identifies λ.
    """
    return htm_share * 1.0 + (1.0 - htm_share) * mpc_unconstrained


def annualize(moment_key: str, value: float, periods_per_year: int = 4) -> float:
    """Convert a per-period (quarterly) closed-form moment to an annual basis to match annual targets.

    * capital_output_ratio: stock / (quarterly flow) -> divide by periods_per_year.
    * real_interest_rate:   per-period net rate       -> (1+r)^periods_per_year - 1 (compounded).
    * everything else (I/Y, labor_share, capital_share, substitution_elasticity): frequency-invariant.
    """
    if moment_key == "capital_output_ratio":
        return value / periods_per_year
    if moment_key == "real_interest_rate":
        return (1.0 + value) ** periods_per_year - 1.0
    return value


# ---------------------------------------------------------------------------------------------
# 2. PARAM ROLE-MAPPER  (CONFIRM canonical roles from typical_range AND name+description)
# ---------------------------------------------------------------------------------------------

_CANON_STRIP = re.compile(r"[\\${}\s]")


def canonical_symbol(raw: str) -> str:
    """Strip latex/braces/whitespace but KEEP qualitative subscripts (delta_K, eta_L, phi_pi)."""
    return _CANON_STRIP.sub("", str(raw or ""))


def _base_family(sym: str) -> str:
    """Greek/letter family for role gating: 'delta_K'->'delta', 'eta_L'->'eta', 'phi_pi'->'phi'."""
    return canonical_symbol(sym).split("_")[0]


def _parse_range(raw) -> Optional[Tuple[float, float]]:
    """'[0.3, 0.7]' / [0.3,0.7] / '[1.0]' -> (lo, hi). Returns None when unparseable."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        nums = [float(x) for x in raw if isinstance(x, (int, float))]
    else:
        try:
            val = ast.literal_eval(str(raw))
        except Exception:
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", str(raw))]
        else:
            if isinstance(val, (list, tuple)):
                nums = [float(x) for x in val]
            elif isinstance(val, (int, float)):
                nums = [float(val)]
            else:
                nums = []
    if not nums:
        return None
    return (min(nums), max(nums))


def _resolve_role(symbol: str, text: str, rng: Optional[Tuple[float, float]], name: str = "") -> Optional[str]:
    """The CONFIRM core: return a canonical role only when range AND text agree, else None."""
    base = _base_family(symbol)
    lo, hi = (rng if rng else (None, None))
    has_range = rng is not None
    t = text  # already lowercased name + description

    # --- hand-to-mouth share (HANK/TANK; family-independent: lambda, chi, omega, mu, ...) --------
    # The keywords are unambiguous (unlike alpha/beta), so no symbol-family gate — but the range
    # must be a plausible population share.
    htm_household = any(k in t for k in ("hand-to-mouth", "hand to mouth", "liquidity-constrained",
                                         "liquidity constrained", "constrained household",
                                         "constrained agent", "non-ricardian", "keynesian household"))
    # "Rule of thumb" names both household consumption rules (Campbell-Mankiw) and firm price
    # setting (Gali-Gertler backward-looking price setters). It indicates the hand-to-mouth share
    # only with household/consumption context and no firm/pricing context: a backward-looking
    # price-setter share mapped to the MPC target would calibrate a pricing parameter to a
    # consumption moment.
    rot = any(k in t for k in ("rule-of-thumb", "rule of thumb"))
    household_ctx = any(k in t for k in ("household", "consumer", "consumption", "consume"))
    pricing_ctx = any(k in t for k in ("firm", "price", "pricing", "inflation", "wage setter",
                                       "wage-setter", "producer"))
    htm = htm_household or (rot and household_ctx and not pricing_ctx)
    # The parameter itself must BE the share: the share word in its name, and not a borrowing-
    # constraint parameter whose description merely mentions constrained households (a
    # collateral coefficient must not be point-calibrated to the MPC target).
    nm = (name or "").lower()
    credit_param = any(k in nm for k in ("collateral", "loan-to-value", "loan to value", "ltv",
                                         "haircut", "leverage", "margin", "pledge", "borrowing limit",
                                         "debt limit"))
    if htm and not credit_param and any(k in (nm or t) for k in ("share", "fraction", "proportion", "mass")):
        if has_range and lo >= 0.0 and hi <= 1.0:
            return "hand_to_mouth_share"
        return None

    # --- share parameters (conventionally 'alpha'); capital vs labor decided by the text ---------
    if base == "alpha":
        # a matching-function or bargaining elasticity is not a factor share, whatever else the
        # description mentions (a matching elasticity must not take the labor-share target)
        if any(k in t for k in ("matching", "vacanc", "bargain", "job finding", "job-finding")):
            return None
        cap = "capital" in t
        lab = "labor" in t
        descriptive = any(k in t for k in ("share", "exponent", "elasticity", "contribution", "weight"))
        # capital_share: text says capital (and NOT labor), range a plausible capital share.
        if cap and not lab and descriptive and has_range and lo >= 0.05 and hi <= 0.5:
            return "capital_share"
        # labor_share: text says labor (and NOT capital), range a plausible labor share.
        if lab and not cap and descriptive and has_range and lo >= 0.2 and hi <= 0.8:
            return "labor_share"
        return None

    # --- discount factor beta in [~0.9, ~1.0) with time-preference language ----------------------
    if base == "beta":
        wants = any(k in t for k in ("discount", "time preference", "subjective", "weight placed on future"))
        if wants and has_range and lo >= 0.9 and hi <= 0.9999:
            return "discount"
        return None  # e.g. real Model: beta = "Opacity Sensitivity" [0.5,2.0] -> NOT discount

    # --- capital depreciation delta: SMALL rate (<=0.05) + depreciation language -----------------
    if base == "delta":
        depr = any(k in t for k in ("depreciat", "obsolesc", "depletion")) or (
            "capital" in t and "rate" in t
        )
        if depr and has_range and hi <= 0.05:
            return "depreciation"
        return None  # knowledge/skill/AI-capital "depreciation" with hi>0.05 stays unresolved

    # --- Calvo stickiness theta/xi in [0,1] with price-stickiness language -----------------------
    if base in ("theta", "xi"):
        sticky = any(
            k in t for k in ("calvo", "price adjustment", "stickiness", "re-optimize",
                             "cannot re-optimize", "sticky")
        ) or ("price" in t and "probability" in t)
        if sticky and has_range and lo >= 0.0 and hi <= 1.0:
            return "calvo"
        return None  # e.g. real Model 0: theta_t = "Technology Cost Deflator" [0.9,1.0] -> NOT calvo

    # --- production elasticity of substitution rho/sigma (EXCLUDE intertemporal sigma) -----------
    if base in ("rho", "sigma"):
        subst = "substitut" in t
        intertemporal = "intertemporal" in t or "risk aversion" in t or "smooth consumption" in t
        persistence = "persist" in t or "autoregress" in t
        if subst and not intertemporal and not persistence:
            return "substitution_elasticity"
        return None

    return None


def map_param_roles(parameters: List[dict]) -> List[ParamRole]:
    """Map each calibrated parameter to a CONFIRMED canonical role (unresolved -> role=None).

    Reads ``parameter_symbol``, ``parameter_name``, ``description`` and ``typical_range`` from each
    parameter dict (real model_design_output.json layout). ``value`` is read only if the dict already
    carries a calibrated point value (keys ``value`` / ``calibrated_value`` / ``point_value``);
    otherwise it stays ``None`` -- we NEVER invent one (range midpoints are not calibrated values).
    """
    roles: List[ParamRole] = []
    for p in parameters or []:
        raw_sym = p.get("parameter_symbol") or p.get("symbol") or ""
        sym = canonical_symbol(raw_sym)
        name = str(p.get("parameter_name") or p.get("name") or "")
        desc = str(p.get("description") or "")
        rng = _parse_range(p.get("typical_range") or p.get("range"))
        text = (name + " " + desc).lower()
        role = _resolve_role(sym, text, rng, name)

        value = None
        for k in ("value", "calibrated_value", "point_value"):
            v = p.get(k)
            if isinstance(v, (int, float)):
                value = float(v)
                break

        roles.append(ParamRole(symbol=sym, role=role, value=value, range=rng, description=desc))
    return roles


# ---------------------------------------------------------------------------------------------
# 3. STRUCTURAL SIGNATURES  (cheap regex over raw equation text -- evidence, not computation)
# ---------------------------------------------------------------------------------------------
# Signatures read the RAW equation_plain (even for equations the parser REFUSED), because they are
# only structural evidence; the moment VALUE always comes from the closed form above, never from the
# (possibly non-algebraic) equation itself.

_CES_EXPONENT = re.compile(r"\(\s*(rho|sigma)\s*-\s*1\s*\)\s*/\s*\(?\s*(rho|sigma)\b")
_COBB_DOUGLAS = re.compile(r"\^\s*\(?\s*alpha\b")            # X^alpha  (NOT (1-alpha))
_CAPITAL_LOM = re.compile(r"\(\s*1\s*-\s*delta\w*\s*\)\s*\*?\s*K")  # (1-delta)*K
_DISCOUNT_EULER = re.compile(r"\b[Bb]eta\s*\*\s*E_t")        # beta * E_t[...]


def _raw_texts(parsed_eqs) -> List[str]:
    out = []
    for pe in parsed_eqs or []:
        raw = getattr(pe, "raw", None)
        if raw is None and isinstance(pe, dict):
            raw = pe.get("equation_plain")
        out.append(str(raw or ""))
    return out


def _signatures(parsed_eqs) -> Dict[str, bool]:
    texts = _raw_texts(parsed_eqs)
    ces = any(_CES_EXPONENT.search(t) for t in texts)
    cd = any(_COBB_DOUGLAS.search(t) for t in texts)
    lom = any(_CAPITAL_LOM.search(t) for t in texts)
    euler = any(_DISCOUNT_EULER.search(t) for t in texts)
    return {
        "ces": ces,
        "cobb_douglas": cd,
        "production": ces or cd,
        "capital_lom": lom,
        "discount_euler": euler,
        "rbc_nk": lom or euler,   # capital dynamics OR a discount Euler/NKPC => RBC/NK family
    }


# ---------------------------------------------------------------------------------------------
# 4. MOMENT DETECTION  (signature-propose + bounded-confirm)
# ---------------------------------------------------------------------------------------------

def _value_fn_ky(a: str, b: str, d: str):
    return lambda p, a=a, b=b, d=d: capital_output_ratio(p[a], p[b], p[d])


def _value_fn_iy(a: str, b: str, d: str):
    return lambda p, a=a, b=b, d=d: investment_output_ratio(p[a], p[b], p[d])


def _value_fn_r(b: str):
    return lambda p, b=b: real_interest_rate(p[b])


def _value_fn_identity(s: str):
    return lambda p, s=s: float(p[s])


def _value_fn_one_minus(s: str):
    return lambda p, s=s: 1.0 - float(p[s])


def model_period(formal_model: Dict, roles: Optional[List["ParamRole"]] = None) -> str:
    """The model's time period: 'quarterly', 'monthly' or 'annual'.

    Read from the model's own wording; failing that, from a discount factor's declared range
    (an annual beta sits near 0.96, a quarterly one near 0.99). The r* and K/Y moments used
    to assume quarterly, so an annual model's beta was pushed to its 0.99 bound to hit a 2%
    annual r*."""
    import json as _json
    try:
        text = _json.dumps({k: formal_model.get(k) for k in (
            "model_title", "model_summary", "dynamics", "equilibrium", "model_notation",
            "solution_method", "variables", "parameters")}, default=str).lower()
    except Exception:
        text = ""
    counts = {"quarterly": len(re.findall(r"\bquarter", text)),
              "monthly": len(re.findall(r"\bmonth", text)),
              "annual": len(re.findall(r"\bannual|\byearly|\bper year|\beach year", text))}
    best = max(counts, key=counts.get)
    if counts[best] > 0 and list(counts.values()).count(counts[best]) == 1:
        return best
    for pr in roles or []:
        if pr.role == "discount" and pr.range:
            mid = (pr.range[0] + pr.range[1]) / 2.0
            return "quarterly" if mid >= 0.985 else "annual"
    return "quarterly"


def detect_moments(parsed_eqs, roles: List[ParamRole], period: str = "quarterly") -> List[Moment]:
    """Propose archetype moments from structural signatures, confirm with present param roles.

    Each returned :class:`Moment` carries a ``value_fn`` taking ``{symbol: float}``, its
    ``free_params`` (>=2 => genuinely scorable; ==1 => a pinning identity), a ``frequency`` tag and
    the ``pins`` symbol for just-identifying moments. No external targets / no values are touched
    here -- this is purely the recipe layer.
    """
    sig = _signatures(parsed_eqs)

    by_role: Dict[str, List[str]] = {}
    for pr in roles or []:
        if pr.role:
            by_role.setdefault(pr.role, []).append(pr.symbol)

    def has(role: str) -> bool:
        return role in by_role

    def sym(role: str) -> str:
        return by_role[role][0]  # v1: take the first confirmed symbol for a role

    moments: List[Moment] = []

    # --- RBC/NK core: capital_output_ratio + investment_output_ratio (over-identifying) ----------
    if sig["rbc_nk"] and has("capital_share") and has("discount") and has("depreciation"):
        a, b, d = sym("capital_share"), sym("discount"), sym("depreciation")
        moments.append(Moment(
            moment_key="capital_output_ratio", value_fn=_value_fn_ky(a, b, d),
            free_params={a, b, d}, frequency=period, pins=None, archetype="rbc_nk_core",
        ))
        moments.append(Moment(
            moment_key="investment_output_ratio", value_fn=_value_fn_iy(a, b, d),
            free_params={a, b, d}, frequency=period, pins=None, archetype="rbc_nk_core",
        ))

    # --- real_interest_rate r* = 1/beta - 1 (pins beta) -----------------------------------------
    if sig["rbc_nk"] and has("discount"):
        b = sym("discount")
        moments.append(Moment(
            moment_key="real_interest_rate", value_fn=_value_fn_r(b),
            free_params={b}, frequency=period, pins=b, archetype="rbc_nk_core",
        ))

    # --- CES/CD production: labor_share, capital_share, substitution_elasticity ------------------
    if sig["production"]:
        if has("capital_share"):
            a = sym("capital_share")
            # labor_share = 1 - alpha   (alpha confirmed as CAPITAL share)
            moments.append(Moment(
                moment_key="labor_share", value_fn=_value_fn_one_minus(a),
                free_params={a}, frequency="annual", pins=a, archetype="ces_cd_production",
            ))
            # capital_share = alpha
            moments.append(Moment(
                moment_key="capital_share", value_fn=_value_fn_identity(a),
                free_params={a}, frequency="annual", pins=a, archetype="ces_cd_production",
            ))
        elif has("labor_share"):
            # alpha is DIRECTLY the labor share here (the real-Model-0 role trap) => labor_share = alpha
            a = sym("labor_share")
            moments.append(Moment(
                moment_key="labor_share", value_fn=_value_fn_identity(a),
                free_params={a}, frequency="annual", pins=a, archetype="ces_cd_production",
            ))

        if sig["ces"] and has("substitution_elasticity"):
            s = sym("substitution_elasticity")
            moments.append(Moment(
                moment_key="substitution_elasticity", value_fn=_value_fn_identity(s),
                free_params={s}, frequency="annual", pins=s, archetype="ces_cd_production",
            ))

    # --- HANK/TANK hand-to-mouth block: aggregate quarterly MPC pins the HtM share ---------------
    # The role keywords ("hand-to-mouth share", "rule-of-thumb fraction", ...) ARE the structural
    # evidence here — they are unambiguous in a way alpha/beta are not — so the confirmed role is
    # sufficient to propose the moment (no separate equation signature required).
    if has("hand_to_mouth_share"):
        lam = sym("hand_to_mouth_share")
        moments.append(Moment(
            moment_key="mpc", value_fn=(lambda p, lam=lam: aggregate_mpc(float(p[lam]))),
            free_params={lam}, frequency="quarterly", pins=lam, archetype="hank_htm",
        ))

    return moments
