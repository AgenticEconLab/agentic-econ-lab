# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic equation parser for the calibration harness.

Turns a model's ``equation_plain`` string into a sympy expression so the harness can compute
model-implied moments WITHOUT asking an LLM to generate code. Two hard rules keep this honest:

1. **Refuse, don't guess.** Anything non-algebraic (optimization ``Maximize``, integrals, sums,
   expectations ``E_t[...]``, indicators/piecewise, inequalities, distributions, differentials
   ``dW``/``d(x)/dt``, difference operators ``Delta p``) is REFUSED — the model is marked
   uncalibratable on that equation rather than silently mis-parsed.
2. **Strip indices, keep identities.** Only INDEX subscripts collapse to the steady-state
   symbol: time (``_t``, ``_{t-1}``, ``_t+1``), agent (``_i``, ``_j``, ``_k``, ``_ij``) and
   compound index lists (``_{i,t}``). Every other subscript, accent or superscript label
   IDENTIFIES the symbol and is kept: ``phi_pi`` and ``phi_y`` stay distinct, ``\\hat{y}_t`` and
   ``y_hat_t`` both become ``y_hat``, ``\\epsilon_{A,t}`` becomes ``epsilon_A``, ``r^*`` becomes
   ``r_star``, ``\\pi_t^{public}`` becomes ``pi_public``. (Stripping every
   subscript would merge ``phi_pi``/``phi_y`` into one ``phi`` and ``rho_A``/``sigma_A`` into the
   model's discount rate and CRRA.) Two distinct declared symbols that still map to one name
   (a parameter ``sigma`` and a variable ``sigma_{i,t}``) make the equation REFUSED.

The symbol table is built ONLY from the equation's declared ``variables_used``/``parameters_used``,
so sympy never auto-invents wrong symbols or treats a name as a function. Names that collide with
Python/sympy builtins (``lambda``, ``I``, ``E``, ``O``, ``S``, ``N``, ``Q``) are aliased.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    T as _T,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
)

# Constructs that are NOT closed-form algebra → refuse (mark uncalibratable, never mis-parse).
_REFUSE = re.compile(
    r"(Maximize|Minimize|\bmax_|\bmin_|argm(ax|in)|integral|∫|\\int|sum_|Σ|\\sum|prod_|∏"
    r"|\bif\b|\belse\b|\bd/dt|∂|partial|\bE_t|E_\{|\\mathbb\{E\}|\\mathbb|\\Phi|\bPhi\(|Prob|\\sim"
    r"|<=|>=|≤|≥|\bmax\(|\bmin\(|\bI\()"
)

# Time derivatives / differentials written as operators ('d(theta)/dt', 'dK/dt', '\dot{x}',
# '\frac{dx}{dt}' parsed as products d*theta/(d*t)). Token-level differentials ('dW', 'dx_i,t',
# 'd_epsilon') need the declared symbol table and are caught during symbol resolution.
_REFUSE_DIFFERENTIAL = re.compile(
    r"(\\dot\{|\\ddot\{|\\frac\{\s*d[^{}]*\}\{\s*d\s*t\s*\}|\bd\s*\([^()]*\)\s*/\s*d"
    r"|/\s*dt\b|\\nabla|∇)"
)

# Builtin / single-letter symbols that sympy treats specially → alias to a safe distinct symbol.
_RESERVED = {"lambda", "I", "E", "O", "S", "N", "Q", "C", "beta", "gamma", "zeta"}

# parse_expr's "all" transformations minus split_symbols: an undeclared multi-letter name must
# stay ONE symbol, never be split into a product of letters ('dW' -> d*W, 'SII' -> S*I*I).
_TRANSFORMS = tuple(
    f for i, group in enumerate(_T) for f in (
        (implicit_multiplication, implicit_application, function_exponentiation)
        if i == 5 else group)
)

# Subscript components that are INDICES (collapse in steady state). Everything else identifies.
_INDEX_COMPONENT = re.compile(r"(?:[ijk]{1,3}|[ijk]{0,3}t(?:[+-](?:\d+|[a-z]))*)")
_TIME_COMPONENT = re.compile(r"[ijk]{0,3}t(?:[+-](?:\d+|[a-z]))*")
_ACCENTS = ("hat", "bar", "tilde", "star")
_SUPERSCRIPT_INDEX = ("i", "j")                 # y_t^i: agent index written as a superscript
_MATH_FUNCTIONS = {"log", "ln", "exp", "sqrt", "abs", "sin", "cos", "tan", "Var", "Cov", "E"}
# natural-language words in an equation ('average of delta_i samples over S iterations') are
# prose, not algebra; implicit multiplication would turn them into a product of symbols
_PROSE_WORDS = {"of", "over", "for", "the", "and", "with", "where", "is", "are", "from", "given",
                "to", "by", "all", "each", "when", "follows", "average", "if", "otherwise",
                "such", "that", "then", "subject", "across", "per", "in", "as", "at"}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# One identifier with its chain of subscripts / superscripts (applied after LaTeX cleanup).
_UNBRACED_COMP = r"[A-Za-z0-9]+(?:[+-](?:\d+|[a-z])(?![A-Za-z0-9]))*"
_SUB = r"(?:\{[^{}]*\}|" + _UNBRACED_COMP + r"(?:," + _UNBRACED_COMP + r")*)"
# an unbraced superscript is a (possibly subscripted) name or a number: K_t^alpha_prod
_SUP = r"(?:\{[^{}]*\}|[A-Za-z][A-Za-z0-9]*(?:_" + _SUB + r")*|\d+(?:\.\d+)?)"
_TOKEN = re.compile(r"(?<![\w.])([A-Za-z][A-Za-z0-9]*)((?:_" + _SUB + r"|\^" + _SUP + r")*)")
_CHAIN_ITEM = re.compile(r"_(" + _SUB + r")|\^(" + _SUP + r")")


class _Refusal(Exception):
    def __init__(self, status: str):
        super().__init__(status)
        self.status = status


@dataclass
class ParsedEquation:
    equation_id: str
    status: str                      # "ok" | "refused_nonalgebraic" | "no_equation" | "parse_fail:<Err>"
    lhs_symbol: Optional[str] = None
    expr: Optional[sp.Expr] = None   # sympy RHS expression (steady-state form)
    free_symbols: Optional[List[str]] = None
    raw: str = ""
    # A compound LHS ('P*c + a', 'ln(y)') is an EXPRESSION, not a
    # variable — atomizing it into a pseudo-symbol would make y and ln(y) independent unknowns.
    # When lhs_symbol is not a plain identifier this carries the parsed LHS expression.
    lhs_expr: Optional[sp.Expr] = None
    # Canonical names of the declared symbols, and the names that carried a time index
    # in this equation (CodeTeam sets exogenous innovations to zero in the steady state).
    declared_parameters: List[str] = field(default_factory=list)
    declared_variables: List[str] = field(default_factory=list)
    time_indexed: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# LaTeX cleanup + symbol canonicalization (shared by declared names and equation text)
# ---------------------------------------------------------------------------------------------

def _latex_clean(s: str, merge_delta: bool = True) -> str:
    """Map LaTeX decorations onto the plain-text conventions the resolver understands."""
    s = re.sub(r"\\left|\\right|\\,|\\!|\\;|\\quad", "", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("Δ", "\\Delta ")
    accents = {"hat": "hat", "widehat": "hat", "bar": "bar", "overline": "bar",
               "tilde": "tilde", "widetilde": "tilde"}
    s = re.sub(r"\\(widehat|hat|overline|bar|widetilde|tilde)\{\s*\\?([A-Za-z][A-Za-z0-9]*)\s*\}",
               lambda m: f"{m.group(2)}_{accents[m.group(1)]}", s)
    s = re.sub(r"\\(?:mathbf|boldsymbol|mathrm|mathit|textrm|text|operatorname)\{([^{}]*)\}",
               r"\1", s)
    s = re.sub(r"\^\{?\s*(?:\*|\\ast)\s*\}?", "_star", s)     # r^* / r^{*} -> r_star
    s = re.sub(r"\\([A-Za-z])", r"\1", s)
    s = s.replace("\\", "")
    # function-of-time notation is a time index: 'r_b(t)' / 'K(t-1)' -> r_b_{t} / K_{t-1}
    s = re.sub(r"(?<![\w.])([A-Za-z][A-Za-z0-9]*)((?:_\{[^{}]*\}|_[A-Za-z0-9]+|\^\{[^{}]*\}"
               r"|\^[A-Za-z0-9]+)*)\(\s*(t(?:\s*[+-]\s*\d+)?)\s*\)",
               lambda m: (m.group(0) if m.group(1) in _MATH_FUNCTIONS and not m.group(2)
                          else m.group(1) + m.group(2) + "_{" + m.group(3).replace(" ", "") + "}"),
               s)
    if merge_delta:
        # a difference operator written as a prefix: 'Delta K_i' -> one token 'Delta_K_i' that
        # the resolver keeps only when the model DECLARES that symbol (else: refused)
        s = re.sub(r"(?<![\w.])Delta\s+(?=[A-Za-z])", "Delta_", s)
    return s


def _compose(base: str, parts: List[str]) -> str:
    """Base + identifying parts; accents (hat/bar/tilde/star) go last so 'sigma_hat_beta' and
    'sigma_beta_hat' are one symbol."""
    plain = [p for p in parts if p not in _ACCENTS]
    acc = [p for p in parts if p in _ACCENTS]
    return "_".join([base] + plain + acc)


def _unbrace(body: str) -> str:
    return (body[1:-1] if body.startswith("{") and body.endswith("}") else body).strip()


def _split_components(body: str) -> List[str]:
    return [c.strip() for c in body.strip("{}").split(",") if c.strip() != ""]


def _is_index(comp: str) -> bool:
    return bool(_INDEX_COMPONENT.fullmatch(comp))


def _subscript_parts(body: str) -> Tuple[List[str], bool, List[str]]:
    """(identifying parts, time_indexed, trailing text to re-emit) for one subscript group."""
    parts: List[str] = []
    timed = False
    trailing: List[str] = []
    braced = body.startswith("{")
    for comp in _split_components(body):
        if _is_index(comp):
            timed = timed or bool(_TIME_COMPONENT.fullmatch(comp))
            continue
        stem = re.match(r"[A-Za-z0-9]+", comp)
        if stem is None:
            raise _Refusal("refused_unparsed_subscript")
        rest = comp[stem.end():]
        if rest:
            if braced:
                # '_{t - Delta t}' or similar arithmetic inside a subscript: not a symbol name
                raise _Refusal("refused_unparsed_subscript")
            # unbraced 'alpha_K-1' is alpha_K minus 1, not a lagged index
            trailing.append(rest)
        parts.append(stem.group(0))
    return parts, timed, trailing


def canonical_name(name: str) -> str:
    """Canonical steady-state symbol name of a DECLARED symbol ('' when it is not one symbol).

    Index subscripts are dropped; identifying subscripts, accents and superscript labels are
    kept ('\\phi_{\\pi}' -> 'phi_pi', '\\hat{y}_t' -> 'y_hat', '\\pi_t^{public}' -> 'pi_public',
    'r^*' -> 'r_star'); a numeric superscript is a power on the symbol ('\\sigma^2' -> 'sigma').
    """
    s = _latex_clean(str(name or "")).strip()
    m = _TOKEN.fullmatch(s)
    if m is None:
        return ""
    base, chain = m.group(1), m.group(2) or ""
    parts: List[str] = []
    try:
        for it in _CHAIN_ITEM.finditer(chain):
            sub, sup = it.group(1), it.group(2)
            if sub is not None:
                p, _timed, trailing = _subscript_parts(sub)
                if trailing:
                    return ""
                parts.extend(p)
            else:
                body = _unbrace(sup)
                if _NUMBER.fullmatch(body) or body in _SUPERSCRIPT_INDEX:
                    continue
                label = _sup_name(body)
                if label:
                    parts.append(label)
                    continue
                return ""
    except _Refusal:
        return ""
    return _compose(base, parts)


def _sup_name(body: str) -> str:
    """Canonical name of a superscript body that is a (subscripted) name, else ''."""
    if not re.match(r"[A-Za-z]", body):
        return ""
    m = _TOKEN.fullmatch(body)
    return canonical_name(body) if m is not None else ""


def _canonical_base(name: str) -> str:
    """Back-compat alias (CodeTeam): canonical declared-symbol name, see ``canonical_name``."""
    return canonical_name(name)


def _safe_symbol(base: str) -> Optional[sp.Symbol]:
    if not base or not base.isidentifier():
        return None
    # sympy.Symbol("lambda") is fine as a symbol but parse_expr would choke on the keyword;
    # alias reserved names so parse_expr maps the token to our symbol via local_dict.
    return sp.Symbol(base)


def build_symbol_table(variables_used, parameters_used):
    """Symbol table from DECLARED symbols only (so sympy never invents wrong free symbols)."""
    table = {}
    for nm in list(parameters_used or []) + list(variables_used or []):
        base = canonical_name(nm)
        sym = _safe_symbol(base)
        if sym is not None:
            table[base] = sym
            if base == "lambda":
                # normalize_rhs rewrites the keyword token to 'lambda_'; map that alias to the
                # REAL Symbol('lambda') so free_symbols / value lookups stay 'lambda'.
                table["lambda_"] = sp.Symbol("lambda")
    return table


def _declared_collision(variables_used, parameters_used) -> Optional[str]:
    """Two DISTINCT declared symbols mapping to one canonical name (a parameter 'sigma'
    and a variable 'sigma_{i,t}') cannot both be honoured — return the colliding name."""
    def raw_key(nm):
        return re.sub(r"[\\{}\s]", "", str(nm))
    groups: Dict[str, Dict[str, Set[str]]] = {}
    for kind, names in (("p", parameters_used or []), ("v", variables_used or [])):
        for nm in names:
            c = canonical_name(nm)
            if c:
                groups.setdefault(c, {"p": set(), "v": set()})[kind].add(raw_key(nm))
    for c, g in sorted(groups.items()):
        if g["p"] and g["v"] and (g["p"] | g["v"]) != (g["p"] & g["v"]):
            return c
        if len(g["p"]) > 1:
            return c
    return None


class _Resolver:
    """Rewrites every identifier token of an equation side to its canonical symbol name."""

    def __init__(self, known: Set[str]):
        self.known = known
        self.time_indexed: Set[str] = set()

    def rewrite(self, text: str) -> str:
        return _TOKEN.sub(self._token, text)

    def _token(self, m) -> str:
        base, chain = m.group(1), m.group(2) or ""
        parts: List[str] = []
        timed = False
        trailing: List[str] = []
        sups: List[Tuple[str, bool]] = []
        for it in _CHAIN_ITEM.finditer(chain):
            sub, sup = it.group(1), it.group(2)
            if sub is not None:
                p, t, tr = _subscript_parts(sub)
                parts.extend(p)
                timed = timed or t
                trailing.extend(tr)
            else:
                sups.append((_unbrace(sup), sup.startswith("{")))
        exponents: List[str] = []
        for body, braced in sups:
            label = _sup_name(body)
            if body in _SUPERSCRIPT_INDEX and body not in self.known:
                continue                                       # y_t^i: agent index
            if label:
                if _compose(base, parts + [label]) in self.known:
                    parts.append(label)                        # declared label: pi^{public}
                elif label in self.known:
                    exponents.append(label)                    # declared exponent: N^{alpha}
                else:
                    # 'pi_t^f', 'W^MP', 'x_t^{home}': an undeclared superscript name is a label,
                    # not a power, and the composite 'x_home' is not declared. Dropping it would make
                    # 'x_t^{home} - x_t^{foreign}' parse as 0: refuse instead.
                    raise _Refusal(f"refused_superscript_label:{label}")
            elif _NUMBER.fullmatch(body) or braced:
                exponents.append(self.rewrite(body))           # ^2, ^{1-alpha}
            else:
                raise _Refusal(f"refused_superscript_label:{body}")
        name = _compose(base, parts)
        if name not in self.known:
            self._check_operator(base, parts, name)
        if timed:
            self.time_indexed.add(name)
        out = name + "".join(f"**({e})" for e in exponents)
        return out + "".join(trailing)

    def _check_operator(self, base: str, parts: List[str], name: str) -> None:
        if base == "Delta" and "Delta" not in self.known:
            raise _Refusal("refused_difference_operator")
        # differentials: 'dW', 'dx_i,t', 'dt', 'd_epsilon_i,t' on a declared symbol
        if base == "dt":
            raise _Refusal("refused_differential")
        if base.startswith("d") and len(base) > 1 and (
                base[1:] in self.known or _compose(base[1:], parts) in self.known):
            raise _Refusal("refused_differential")
        if base == "d" and parts and _compose(parts[0], parts[1:]) in self.known:
            raise _Refusal("refused_differential")


def normalize_rhs(rhs: str, known=(), resolver: Optional[_Resolver] = None) -> str:
    """Normalize one equation side to a sympy-parseable string of canonical symbol names."""
    known = set(known or ())
    resolver = resolver or _Resolver(known)
    # a DECLARED scalar Delta multiplies; otherwise "Delta x" is an operator or a declared name
    s = _latex_clean(rhs.strip(), merge_delta="Delta" not in known)
    if re.search(r"(?<![\w.+\-])1_", s):
        raise _Refusal("refused_nonalgebraic")            # indicator 1_{...}: piecewise
    if "e" not in known:
        # an undeclared 'e' raised to a power is the exponential function
        s = re.sub(r"(?<![\w.])e\s*\^\s*\{([^{}]*)\}", r"exp(\1)", s)
        s = re.sub(r"(?<![\w.])e\s*\^\s*\(", "exp(", s)
    s = resolver.rewrite(s)
    if re.search(r"(?<![A-Za-z0-9])_|_\{", s):
        # a subscript no identifier could absorb ('a_{RL, t - Delta t_{latency}}' with nested
        # braces): stripping it would silently change which symbol is meant
        raise _Refusal("refused_unparsed_subscript")
    words = set(re.findall(r"(?<![\w.])([A-Za-z]+)(?![\w(])", s))
    if (words & _PROSE_WORDS) - known:
        raise _Refusal("refused_prose")
    # ^{...} left after a non-identifier (')^{1-alpha}'): a declared symbol or anything with
    # digits/operators is an exponent; a bare undeclared word is a superscript label.
    # Turning the label into '**{pri}' raises a TypeError;
    # erasing it changes which quantity is meant, so the equation is refused.

    def _sup(m):
        body = m.group(1).strip()
        if re.fullmatch(r"[A-Za-z]+", body) and body not in known:
            raise _Refusal(f"refused_superscript_label:{body}")
        return "**(" + resolver.rewrite(body) + ")"
    s = re.sub(r"\^\{([^}]*)\}", _sup, s)
    s = s.replace("^", "**").replace("[", "(").replace("]", ")")
    s = re.sub(r"\bln\b", "log", s)
    # 'lambda' is a Python keyword — parse_expr raises TokenError on the bare token, and λ is
    # ubiquitous in economics (HtM share, Calvo, arrival rates). Alias it for parsing; the
    # symbol is renamed back post-parse so downstream sees 'lambda' (see parse_equation).
    s = re.sub(r"\blambda\b", "lambda_", s)
    return s


_SYMPY_FUNCTION_NAMES = {"log", "exp", "sqrt", "sin", "cos", "tan", "Abs", "abs", "sign",
                         "Min", "Max", "floor", "ceiling"}


def _parse(s: str, table) -> sp.Expr:
    # an undeclared name that sympy binds to a function/class ('zeta', 'beta', 'gamma', 'S',
    # 'N', 'O', 'Q') and is not applied to an argument is a plain symbol: '+ zeta' raised a
    # TypeError (Function + Expr) instead of parsing the economist's shock zeta_t
    local = dict(table)
    # an UNDECLARED name applied to an argument ('Cov(theta)', 'u(c)', 'sigmoid(x)') is an
    # unspecified function; implicit multiplication would silently read it as Cov*theta
    for name in re.findall(r"(?<![\w.])([A-Za-z]\w*)\(", s):
        if name not in local and name not in _SYMPY_FUNCTION_NAMES:
            raise _Refusal("refused_undefined_function")
    for name in set(re.findall(r"(?<![\w.])([A-Za-z]\w*)(?!\w)(?!\s*\()", s)):
        if name in local or name in _SYMPY_FUNCTION_NAMES:
            continue
        obj = getattr(sp, name, None)
        if obj is not None and (callable(obj) or name in ("O", "Q", "S", "N")):
            local[name] = sp.Symbol(name)
    return sp.parse_expr(s, local_dict=local, transformations=_TRANSFORMS, evaluate=True)


def parse_equation(eq: dict) -> ParsedEquation:
    """Parse one equation dict (keys: equation_id, equation_plain, variables_used, parameters_used)."""
    eid = str(eq.get("equation_id", "?"))
    plain = eq.get("equation_plain") or ""
    vu = eq.get("variables_used") or []
    pu = eq.get("parameters_used") or []
    if isinstance(vu, str):
        try: vu = eval(vu)  # the artifacts store these as a stringified list
        except Exception: vu = []
    if isinstance(pu, str):
        try: pu = eval(pu)
        except Exception: pu = []

    if "=" not in plain:
        return ParsedEquation(eid, "no_equation", raw=plain)
    if _REFUSE.search(plain):
        return ParsedEquation(eid, "refused_nonalgebraic", raw=plain)
    if _REFUSE_DIFFERENTIAL.search(plain):
        return ParsedEquation(eid, "refused_differential", raw=plain)

    lhs, rhs = plain.split("=", 1)
    # A chained equality ('A = B = C') leaves a second literal '='
    # inside rhs. parse_expr's "all" transformations include convert_equals_signs, which
    # then silently turns rhs into an Eq(...) object instead of a plain expression --
    # generate.py's `lhs - p.expr` crashes uncaught (Mul - Equality) the moment such a
    # module is derived, failing the whole run rather than refusing this one equation.
    # Refuse honestly instead of parsing a relation where an expression is required.
    if "=" in rhs:
        return ParsedEquation(eid, "refused_chained_equality", raw=plain)
    collision = _declared_collision(vu, pu)
    if collision:
        return ParsedEquation(eid, f"refused_symbol_collision:{collision}", raw=plain)
    table = build_symbol_table(vu, pu)
    known = set(table.keys())
    resolver = _Resolver(known)
    try:
        s = normalize_rhs(rhs, known, resolver)
        lhs_s = normalize_rhs(lhs, known, resolver)
    except _Refusal as r:
        return ParsedEquation(eid, r.status, raw=plain)
    # An UNDECLARED bare 'E' or 'I' is not a symbol to sympy — it is
    # Euler's constant / the imaginary unit. An econ equation using either token without
    # declaring it almost certainly means an expectation operator (E) or investment (I)
    # that the subscript-stripper exposed; silently computing with 2.718... or sqrt(-1)
    # fabricates semantics, so refuse honestly instead.
    for reserved in ("E", "I"):
        if reserved not in table and re.search(rf"(?<![\w.]){reserved}(?!\w)", s + " " + lhs_s):
            return ParsedEquation(eid, f"refused_undeclared_reserved_symbol:{reserved}", raw=plain)
    try:
        expr = _parse(s, table)
    except _Refusal as r:
        return ParsedEquation(eid, r.status, raw=plain)
    except Exception as e:  # never raise into the stage — refuse honestly
        return ParsedEquation(eid, f"parse_fail:{type(e).__name__}", raw=plain)
    # parse_expr can return a tuple/relational/non-Expr for a system or comparison — those are not
    # a scalar closed-form moment source, so refuse honestly rather than crash downstream.
    # NOTE: a Relational (Eq/Ne/Lt/...) has a perfectly normal free_symbols (not None), so
    # the check below alone does NOT catch it: an rhs-side 'convert_equals_signs' hit
    # turns expr into an Eq, which would crash uncaught downstream (Mul - Equality)
    # instead of refusing here. Reject explicitly.
    if isinstance(expr, sp.core.relational.Relational):
        return ParsedEquation(eid, "parse_fail:RelationalExpr", raw=plain)
    fs = getattr(expr, "free_symbols", None)
    if fs is None:
        return ParsedEquation(eid, "parse_fail:NonScalarExpr", raw=plain)
    # an undefined function ('u(c)', 'f(x, a)') is not closed-form algebra: its value is unknown
    if expr.atoms(sp.core.function.AppliedUndef):
        return ParsedEquation(eid, "refused_undefined_function", raw=plain)
    # rename the parse-time keyword alias back so downstream sees the economist's 'lambda'
    if any(str(x) == "lambda_" for x in fs):
        expr = expr.subs(sp.Symbol("lambda_"), sp.Symbol("lambda"))
        fs = expr.free_symbols
    lhs_base = lhs_s.strip()
    if lhs_base == "lambda_":
        lhs_base = "lambda"
    lhs_expr = None
    if lhs_base and not lhs_base.isidentifier():
        # compound LHS: parse it with the same table/normalizer as the RHS; a LHS that
        # will not parse makes the whole equation unusable — refuse honestly.
        try:
            lhs_expr = _parse(lhs_s, table)
        except _Refusal as r:
            return ParsedEquation(eid, r.status, raw=plain)
        except Exception:
            return ParsedEquation(eid, "parse_fail:LHS", raw=plain)
        if isinstance(lhs_expr, sp.core.relational.Relational):
            return ParsedEquation(eid, "parse_fail:LHSRelationalExpr", raw=plain)
        lfs = getattr(lhs_expr, "free_symbols", None)
        if lfs is None:
            return ParsedEquation(eid, "parse_fail:LHSNonScalar", raw=plain)
        if lhs_expr.atoms(sp.core.function.AppliedUndef):
            return ParsedEquation(eid, "refused_undefined_function", raw=plain)
        if any(str(x) == "lambda_" for x in lfs):
            lhs_expr = lhs_expr.subs(sp.Symbol("lambda_"), sp.Symbol("lambda"))
            lfs = lhs_expr.free_symbols
        fs = set(fs) | set(lfs)
    return ParsedEquation(
        eid, "ok",
        lhs_symbol=lhs_base,
        expr=expr,
        free_symbols=sorted(str(x) for x in fs),
        raw=plain,
        lhs_expr=lhs_expr,
        declared_parameters=sorted({c for c in (canonical_name(n) for n in pu) if c}),
        declared_variables=sorted({c for c in (canonical_name(n) for n in vu) if c}),
        time_indexed=sorted(resolver.time_indexed),
    )
