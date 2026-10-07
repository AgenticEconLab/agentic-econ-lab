# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Statistical aggregation and significance testing for multi-run evaluation.

Aggregates evaluation scores across repeated runs to produce:
- Mean, std, confidence intervals per dimension
- Factorial effects (Firecrawl, HITL, interaction) with standard errors
- Non-parametric significance tests (Mann-Whitney U)
- LaTeX summary tables

Usage:
    from evaluation.analysis.statistical_aggregator import StatisticalAggregator

    agg = StatisticalAggregator()
    result = agg.aggregate_runs(run_results)
    effects = agg.compute_factorial_effects(scores_by_config)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DimensionStats:
    """Aggregated statistics for one dimension across multiple runs."""
    dimension: str
    n: int
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    min_val: float
    max_val: float
    values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "n": self.n,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "min": round(self.min_val, 4),
            "max": round(self.max_val, 4),
        }


@dataclass
class AggregatedScores:
    """Full aggregation result across all dimensions."""
    team: str
    mode: str
    n_runs: int
    dimensions: Dict[str, DimensionStats] = field(default_factory=dict)
    overall: Optional[DimensionStats] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "mode": self.mode,
            "n_runs": self.n_runs,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "overall": self.overall.to_dict() if self.overall else None,
        }


@dataclass
class FactorialEffect:
    """A single factorial effect estimate."""
    name: str
    effect: float
    se: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "effect": round(self.effect, 4),
            "se": round(self.se, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
        }


@dataclass
class FactorialEffects:
    """Factorial effect decomposition for a dimension."""
    dimension: str
    firecrawl: FactorialEffect
    hitl: FactorialEffect
    interaction: FactorialEffect

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "firecrawl": self.firecrawl.to_dict(),
            "hitl": self.hitl.to_dict(),
            "interaction": self.interaction.to_dict(),
        }


@dataclass
class TestResult:
    """Result of a significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    n_a: int = 0
    n_b: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "significant": self.significant,
            "n_a": self.n_a,
            "n_b": self.n_b,
        }
        if self.effect_size is not None:
            d["effect_size"] = round(self.effect_size, 4)
        return d


# ---------------------------------------------------------------------------
# t-distribution critical values (two-tailed, alpha=0.05)
# For small samples. Indexed by degrees of freedom (1-30, then 40, 60, 120, inf).
# ---------------------------------------------------------------------------

_T_CRITICAL_005 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t_critical(df: int) -> float:
    """Look up two-tailed t critical value for alpha=0.05."""
    if df in _T_CRITICAL_005:
        return _T_CRITICAL_005[df]
    # Find closest df in table
    keys = sorted(_T_CRITICAL_005.keys())
    for k in keys:
        if k >= df:
            return _T_CRITICAL_005[k]
    return 1.96  # z-approximation for large df


# ---------------------------------------------------------------------------
# Mann-Whitney U (pure Python, no scipy dependency)
# ---------------------------------------------------------------------------

def _mann_whitney_u(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Compute Mann-Whitney U statistic and approximate p-value.

    Uses normal approximation for p-value (valid for n >= 8).
    Returns (U, p_value).
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0

    # Combine and rank
    combined = [(v, 0, i) for i, v in enumerate(a)] + [
        (v, 1, i) for i, v in enumerate(b)
    ]
    combined.sort(key=lambda x: x[0])

    # Assign ranks with ties
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum ranks for group a
    rank_sum_a = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)

    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    # Normal approximation for p-value
    mu = n_a * n_b / 2.0
    sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)

    if sigma == 0:
        return u, 1.0

    z = abs(u - mu) / sigma

    # Two-tailed p-value using error function approximation
    p = 2.0 * (1.0 - _norm_cdf(z))

    return u, p


def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF using Abramowitz & Stegun."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# StatisticalAggregator
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "reliability", "correctness", "soundness",
    "efficiency", "scalability", "robustness",
    "transparency", "traceability", "reproducibility",
    "innovation_potential",
    "decision_quality", "economic_rigor",
]


class StatisticalAggregator:
    """
    Aggregates evaluation scores across multiple runs.

    Reads per-run evaluation results (either as dicts or from multirun
    manifest directories) and computes descriptive statistics, confidence
    intervals, factorial effects, and significance tests.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: Confidence level for CIs (default 0.95 → 95% CI).
        """
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def aggregate_runs(
        self,
        run_results: List[Dict[str, Any]],
        team: str = "",
        mode: str = "",
    ) -> AggregatedScores:
        """
        Aggregate dimension scores across multiple run results.

        Each element of run_results should be a dict with dimension keys
        mapping to float scores (0-1). This matches the "combined" or
        "tier1" dictionaries produced by run_ael_evaluation.py.

        Args:
            run_results: List of per-run score dicts ({dimension: score})
            team: Team name (for labelling)
            mode: Mode name (for labelling)

        Returns:
            AggregatedScores with per-dimension statistics
        """
        n = len(run_results)
        if n == 0:
            return AggregatedScores(team=team, mode=mode, n_runs=0)

        agg = AggregatedScores(team=team, mode=mode, n_runs=n)

        for dim in DIMENSIONS:
            values = [r.get(dim, 0.0) for r in run_results]
            agg.dimensions[dim] = self._compute_stats(dim, values)

        # Overall = mean of all dimension means
        overall_values = [
            sum(r.get(dim, 0.0) for dim in DIMENSIONS) / len(DIMENSIONS)
            for r in run_results
        ]
        agg.overall = self._compute_stats("overall", overall_values)

        return agg

    def _compute_stats(self, name: str, values: List[float]) -> DimensionStats:
        """Compute descriptive statistics and CI for a list of values."""
        n = len(values)
        if n == 0:
            return DimensionStats(
                dimension=name, n=0, mean=0, std=0,
                ci_lower=0, ci_upper=0, min_val=0, max_val=0,
            )

        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0

        # Confidence interval
        if n > 1:
            se = std / math.sqrt(n)
            t_crit = _t_critical(n - 1)
            margin = t_crit * se
            ci_lower = mean - margin
            ci_upper = mean + margin
        else:
            ci_lower = mean
            ci_upper = mean

        return DimensionStats(
            dimension=name,
            n=n,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            min_val=min(values),
            max_val=max(values),
            values=values,
        )

    # ------------------------------------------------------------------
    # Factorial effects (2x2 design: Firecrawl x HITL)
    # ------------------------------------------------------------------

    def compute_factorial_effects(
        self,
        scores_by_config: Dict[str, List[Dict[str, float]]],
    ) -> Dict[str, FactorialEffects]:
        """
        Compute 2x2 factorial effects for each dimension.

        Expects scores_by_config keyed by mode name, where modes follow the
        pattern Mode{With/No}Fc{With/No}HITL.  DataTeam modes are excluded.

        Args:
            scores_by_config: {mode_name: [run_scores_dict, ...]}

        Returns:
            {dimension: FactorialEffects} for each dimension.
        """
        # Map modes to factorial cells
        fc_present = {}     # {mode: bool}
        hitl_present = {}   # {mode: bool}
        for mode in scores_by_config:
            if "WithFc" in mode:
                fc_present[mode] = True
            elif "NoFc" in mode:
                fc_present[mode] = False
            else:
                continue  # skip DataTeam modes
            hitl_present[mode] = "WithHITL" in mode

        # Need all 4 cells
        factorial_modes = [m for m in scores_by_config if m in fc_present]
        if len(factorial_modes) < 2:
            return {}

        results = {}
        for dim in DIMENSIONS:
            # Collect cell means
            fc_yes = []   # scores when Firecrawl is present
            fc_no = []    # scores when Firecrawl is absent
            hitl_yes = [] # scores when HITL is present
            hitl_no = []  # scores when HITL is absent

            # For interaction: (Fc=1,HITL=1), (Fc=1,HITL=0), (Fc=0,HITL=1), (Fc=0,HITL=0)
            cells = {(True, True): [], (True, False): [],
                     (False, True): [], (False, False): []}

            for mode in factorial_modes:
                fc = fc_present[mode]
                hitl = hitl_present[mode]
                for run_scores in scores_by_config[mode]:
                    val = run_scores.get(dim, 0.0)
                    if fc:
                        fc_yes.append(val)
                    else:
                        fc_no.append(val)
                    if hitl:
                        hitl_yes.append(val)
                    else:
                        hitl_no.append(val)
                    cells[(fc, hitl)].append(val)

            fc_effect = self._compute_effect("Firecrawl", fc_yes, fc_no)
            hitl_effect = self._compute_effect("HITL", hitl_yes, hitl_no)
            interaction = self._compute_interaction(cells)

            results[dim] = FactorialEffects(
                dimension=dim,
                firecrawl=fc_effect,
                hitl=hitl_effect,
                interaction=interaction,
            )

        return results

    def _compute_effect(
        self, name: str, group_high: List[float], group_low: List[float]
    ) -> FactorialEffect:
        """Compute main effect as difference of means ± SE."""
        n_h, n_l = len(group_high), len(group_low)
        mean_h = sum(group_high) / n_h if n_h > 0 else 0.0
        mean_l = sum(group_low) / n_l if n_l > 0 else 0.0
        effect = mean_h - mean_l

        # Pooled SE
        if n_h > 1 and n_l > 1:
            var_h = sum((v - mean_h) ** 2 for v in group_high) / (n_h - 1)
            var_l = sum((v - mean_l) ** 2 for v in group_low) / (n_l - 1)
            se = math.sqrt(var_h / n_h + var_l / n_l)
        else:
            se = 0.0

        t_crit = _t_critical(max(1, n_h + n_l - 2))
        margin = t_crit * se

        return FactorialEffect(
            name=name,
            effect=effect,
            se=se,
            ci_lower=effect - margin,
            ci_upper=effect + margin,
        )

    def _compute_interaction(
        self, cells: Dict[Tuple[bool, bool], List[float]]
    ) -> FactorialEffect:
        """Compute interaction effect from 2x2 cell means."""
        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        m11 = _mean(cells[(True, True)])
        m10 = _mean(cells[(True, False)])
        m01 = _mean(cells[(False, True)])
        m00 = _mean(cells[(False, False)])

        # Interaction = (m11 - m10) - (m01 - m00)
        interaction = (m11 - m10) - (m01 - m00)

        # Approximate SE using all observations
        all_vals = []
        for v_list in cells.values():
            all_vals.extend(v_list)
        n = len(all_vals)
        if n > 1:
            grand_mean = sum(all_vals) / n
            mse = sum((v - grand_mean) ** 2 for v in all_vals) / (n - 1)
            # SE of interaction in balanced 2x2 design
            n_per_cell = n / 4.0
            se = math.sqrt(4.0 * mse / max(1, n_per_cell)) if n_per_cell > 0 else 0.0
        else:
            se = 0.0

        t_crit = _t_critical(max(1, n - 4))
        margin = t_crit * se

        return FactorialEffect(
            name="Fc x HITL Interaction",
            effect=interaction,
            se=se,
            ci_lower=interaction - margin,
            ci_upper=interaction + margin,
        )

    # ------------------------------------------------------------------
    # Significance testing
    # ------------------------------------------------------------------

    def significance_test(
        self,
        group_a: List[float],
        group_b: List[float],
        alpha: float = 0.05,
    ) -> TestResult:
        """
        Non-parametric significance test (Mann-Whitney U).

        Safe for small sample sizes. Uses normal approximation for p-value.

        Args:
            group_a: Scores from group A
            group_b: Scores from group B
            alpha: Significance level

        Returns:
            TestResult with statistic, p-value, and significance flag
        """
        u, p = _mann_whitney_u(group_a, group_b)

        n_a, n_b = len(group_a), len(group_b)
        # Effect size: r = Z / sqrt(N)
        if n_a + n_b > 0:
            mu = n_a * n_b / 2.0
            sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)
            z = abs(u - mu) / sigma if sigma > 0 else 0
            effect_size = z / math.sqrt(n_a + n_b)
        else:
            effect_size = 0.0

        return TestResult(
            test_name="Mann-Whitney U",
            statistic=u,
            p_value=p,
            significant=p < alpha,
            effect_size=effect_size,
            n_a=n_a,
            n_b=n_b,
        )

    # ------------------------------------------------------------------
    # LaTeX table generation
    # ------------------------------------------------------------------

    def generate_summary_table(
        self,
        aggregated_by_config: Dict[str, AggregatedScores],
    ) -> str:
        """
        Generate a LaTeX table with mean ± CI for each config × dimension.

        Args:
            aggregated_by_config: {config_key: AggregatedScores}

        Returns:
            LaTeX table string
        """
        dim_headers = ["Rel", "Cor", "Snd", "Eff", "Scl", "Rob", "Trn", "Trc", "Rep", "Inn"]

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Multi-run evaluation scores (mean $\pm$ 95\% CI)}")
        lines.append(r"\label{tab:multirun-scores}")
        lines.append(r"\footnotesize")
        col_spec = "l" + "c" * len(DIMENSIONS)
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")

        # Header row
        header = "Configuration"
        for h in dim_headers:
            header += f" & {h}"
        header += r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        # Data rows
        for config_key, agg in aggregated_by_config.items():
            short_key = config_key.replace("_Mode", " ").replace("Team", "")
            row = short_key
            for dim in DIMENSIONS:
                stats = agg.dimensions.get(dim)
                if stats and stats.n > 1:
                    ci_half = (stats.ci_upper - stats.ci_lower) / 2
                    row += f" & ${stats.mean:.2f} \\pm {ci_half:.2f}$"
                elif stats:
                    row += f" & ${stats.mean:.2f}$"
                else:
                    row += " & --"
            row += r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_effects_table(
        self,
        effects_by_dim: Dict[str, FactorialEffects],
    ) -> str:
        """
        Generate a LaTeX table for factorial effects.

        Args:
            effects_by_dim: {dimension: FactorialEffects}

        Returns:
            LaTeX table string
        """
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Factorial effects of Firecrawl and HITL features (effect $\pm$ SE)}")
        lines.append(r"\label{tab:factorial-effects}")
        lines.append(r"\footnotesize")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Dimension & Firecrawl Effect & HITL Effect & Interaction \\")
        lines.append(r"\midrule")

        dim_short = {
            "reliability": "Reliability", "correctness": "Correctness",
            "soundness": "Soundness", "efficiency": "Efficiency",
            "scalability": "Scalability", "robustness": "Robustness",
            "transparency": "Transparency", "traceability": "Traceability",
            "reproducibility": "Reproducibility", "innovation_potential": "Innovation",
        }

        for dim in DIMENSIONS:
            if dim not in effects_by_dim:
                continue
            fx = effects_by_dim[dim]
            name = dim_short.get(dim, dim)
            fc = f"${fx.firecrawl.effect:+.3f} \\pm {fx.firecrawl.se:.3f}$"
            ht = f"${fx.hitl.effect:+.3f} \\pm {fx.hitl.se:.3f}$"
            ix = f"${fx.interaction.effect:+.3f} \\pm {fx.interaction.se:.3f}$"
            lines.append(f"{name} & {fc} & {ht} & {ix} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_from_multirun_dir(
        self,
        multirun_dir: Path,
        score_key: str = "combined",
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Load per-run scores from a multirun output directory.

        Expects directory structure:
            multirun_dir/
                {Team}/{Mode}/
                    run_001/execution_log.json
                    run_002/execution_log.json
                    ...
                    multirun_manifest.json

        For each run, calculates tier1 metrics from the execution log.

        Args:
            multirun_dir: Path to multirun output root
            score_key: Which score set to load (default "combined")

        Returns:
            {config_key: [scores_dict_per_run, ...]}
        """
        # Import tier1 calculator from the evaluation script
        import importlib.util
        eval_script = Path(__file__).resolve().parent.parent / "run_ael_evaluation.py"
        spec = importlib.util.spec_from_file_location("run_ael_eval", eval_script)
        eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod)
        calc_tier1 = eval_mod.calculate_tier1_metrics

        results = {}
        multirun_dir = Path(multirun_dir)

        for team_dir in sorted(multirun_dir.iterdir()):
            if not team_dir.is_dir():
                continue
            team = team_dir.name

            for mode_dir in sorted(team_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                mode = mode_dir.name
                config_key = f"{team}_{mode}"

                run_scores = []
                for run_dir in sorted(mode_dir.iterdir()):
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue
                    log_path = run_dir / "execution_log.json"
                    if log_path.exists():
                        with open(log_path, "r", encoding="utf-8") as f:
                            log = json.load(f)
                        scores = calc_tier1(log)
                        run_scores.append(scores)

                if run_scores:
                    results[config_key] = run_scores

        return results

    def aggregate_multirun_dir(
        self,
        multirun_dir: Path,
    ) -> Tuple[Dict[str, AggregatedScores], Dict[str, FactorialEffects]]:
        """
        Full aggregation pipeline from a multirun output directory.

        Returns:
            (aggregated_by_config, factorial_effects_by_dim)
        """
        scores_by_config = self.load_from_multirun_dir(multirun_dir)

        aggregated = {}
        for config_key, run_scores in scores_by_config.items():
            parts = config_key.split("_", 1)
            team = parts[0] if len(parts) > 0 else ""
            mode = parts[1] if len(parts) > 1 else ""
            aggregated[config_key] = self.aggregate_runs(run_scores, team=team, mode=mode)

        # Factorial effects (only for teams with 2x2 design)
        # Group by team, then compute per-team
        teams_modes = {}
        for config_key, run_scores in scores_by_config.items():
            parts = config_key.split("_", 1)
            team = parts[0]
            mode = parts[1] if len(parts) > 1 else ""
            if team not in teams_modes:
                teams_modes[team] = {}
            teams_modes[team][mode] = run_scores

        all_effects = {}
        for team, mode_scores in teams_modes.items():
            effects = self.compute_factorial_effects(mode_scores)
            for dim, fx in effects.items():
                key = f"{team}_{dim}"
                all_effects[key] = fx

        return aggregated, all_effects
