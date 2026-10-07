# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Visualization tools for workflow evaluation.

Generates charts and plots for evaluation results:
- Radar charts for dimension comparison
- Bar charts for mode comparison
- Timeline visualizations for execution analysis
- Heatmaps for cross-team analysis
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

from ..schemas.metrics import EvaluationResult
from ..core.dimensions import DIMENSIONS, DIMENSION_CLUSTERS
from .dimension_analyzer import RadarChartData, DimensionAnalysis
from .statistical_aggregator import AggregatedScores, DimensionStats, FactorialEffects, FactorialEffect


class EvaluationVisualizer:
    """
    Visualizer for evaluation results.

    Generates matplotlib-based visualizations or returns data structures
    for external plotting.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Default output directory for plots
        """
        self.output_dir = output_dir or Path("plots")
        self._matplotlib_available = self._check_matplotlib()

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            return True
        except ImportError:
            return False

    def plot_radar_chart(
        self,
        radar_data: RadarChartData,
        output_path: Optional[Path] = None,
        title: str = "Dimension Comparison",
        figsize: Tuple[int, int] = (10, 8),
    ) -> Optional[Path]:
        """
        Generate radar chart comparing workflows across dimensions.

        Args:
            radar_data: RadarChartData structure
            output_path: Output file path (if None, just displays)
            title: Chart title
            figsize: Figure size

        Returns:
            Path to saved file if output_path provided, None otherwise
        """
        if not self._matplotlib_available:
            print("matplotlib not available. Use get_radar_chart_data() for raw data.")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        # Setup
        n_dims = len(radar_data.dimensions)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # Plot each series
        colors = plt.cm.Set2(np.linspace(0, 1, len(radar_data.series)))

        for (series_name, scores), color in zip(radar_data.series.items(), colors):
            values = scores + scores[:1]  # Close the polygon
            ax.plot(angles, values, 'o-', linewidth=2, label=series_name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_data.dimension_names, size=9)

        # Set y-axis
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)

        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title(title, size=14, y=1.08)

        # Save or show
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def get_radar_chart_data(
        self,
        radar_data: RadarChartData,
    ) -> Dict[str, Any]:
        """
        Get radar chart data in a format suitable for other plotting libraries.

        Args:
            radar_data: RadarChartData structure

        Returns:
            Dictionary with chart data
        """
        n_dims = len(radar_data.dimensions)
        angles = [i * 2 * math.pi / n_dims for i in range(n_dims)]
        angles.append(angles[0])  # Close

        return {
            "dimensions": radar_data.dimensions,
            "dimension_names": radar_data.dimension_names,
            "angles": angles,
            "series": {
                name: scores + [scores[0]]
                for name, scores in radar_data.series.items()
            },
        }

    def plot_dimension_bars(
        self,
        results_by_mode: Dict[str, List[EvaluationResult]],
        output_path: Optional[Path] = None,
        title: str = "Dimension Scores by Mode",
        figsize: Tuple[int, int] = (14, 8),
    ) -> Optional[Path]:
        """
        Generate grouped bar chart for dimension comparison.

        Args:
            results_by_mode: Dictionary mapping mode name to results
            output_path: Output file path
            title: Chart title
            figsize: Figure size

        Returns:
            Path to saved file if output_path provided
        """
        if not self._matplotlib_available:
            print("matplotlib not available.")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        # Calculate average scores
        modes = list(results_by_mode.keys())
        dimensions = list(DIMENSIONS.keys())
        dim_names = [DIMENSIONS[d].name for d in dimensions]

        scores_matrix = []
        for mode in modes:
            mode_scores = []
            for dim_id in dimensions:
                dim_scores = []
                for result in results_by_mode[mode]:
                    for ds in result.dimension_scores:
                        if ds.dimension == dim_id:
                            dim_scores.append(ds.score)
                mode_scores.append(sum(dim_scores) / len(dim_scores) if dim_scores else 0)
            scores_matrix.append(mode_scores)

        # Plot
        x = np.arange(len(dimensions))
        width = 0.8 / len(modes)

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))
        for i, (mode, scores, color) in enumerate(zip(modes, scores_matrix, colors)):
            offset = (i - len(modes) / 2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=mode, color=color)

        ax.set_xlabel('Dimension')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def plot_cluster_comparison(
        self,
        cluster_scores: Dict[str, Dict[str, float]],
        output_path: Optional[Path] = None,
        title: str = "Cluster Scores Comparison",
        figsize: Tuple[int, int] = (10, 6),
    ) -> Optional[Path]:
        """
        Generate bar chart comparing cluster scores across teams/modes.

        Args:
            cluster_scores: Dict of {series_name: {cluster: score}}
            output_path: Output file path
            title: Chart title
            figsize: Figure size

        Returns:
            Path to saved file if output_path provided
        """
        if not self._matplotlib_available:
            print("matplotlib not available.")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        clusters = list(DIMENSION_CLUSTERS.keys())
        series_names = list(cluster_scores.keys())

        x = np.arange(len(clusters))
        width = 0.8 / len(series_names)

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set2(np.linspace(0, 1, len(series_names)))
        for i, (series_name, color) in enumerate(zip(series_names, colors)):
            scores = [cluster_scores[series_name].get(c, 0) for c in clusters]
            offset = (i - len(series_names) / 2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=series_name, color=color)

        ax.set_xlabel('Cluster')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(clusters)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def plot_heatmap(
        self,
        data: Dict[str, Dict[str, float]],
        row_labels: List[str],
        col_labels: List[str],
        output_path: Optional[Path] = None,
        title: str = "Evaluation Heatmap",
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "RdYlGn",
    ) -> Optional[Path]:
        """
        Generate heatmap for cross-comparison.

        Args:
            data: Nested dict {row: {col: value}}
            row_labels: Labels for rows
            col_labels: Labels for columns
            output_path: Output file path
            title: Chart title
            figsize: Figure size
            cmap: Colormap name

        Returns:
            Path to saved file if output_path provided
        """
        if not self._matplotlib_available:
            print("matplotlib not available.")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        # Build matrix
        matrix = np.zeros((len(row_labels), len(col_labels)))
        for i, row in enumerate(row_labels):
            for j, col in enumerate(col_labels):
                matrix[i, j] = data.get(row, {}).get(col, 0)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Labels
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = matrix[i, j]
                text_color = "white" if value < 0.3 or value > 0.7 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Score")

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def plot_execution_timeline(
        self,
        execution_times: Dict[str, List[float]],
        output_path: Optional[Path] = None,
        title: str = "Execution Time Distribution",
        figsize: Tuple[int, int] = (10, 6),
    ) -> Optional[Path]:
        """
        Generate box plot for execution time comparison.

        Args:
            execution_times: Dict of {mode: [times]}
            output_path: Output file path
            title: Chart title
            figsize: Figure size

        Returns:
            Path to saved file if output_path provided
        """
        if not self._matplotlib_available:
            print("matplotlib not available.")
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        modes = list(execution_times.keys())
        data = [execution_times[m] for m in modes]

        bp = ax.boxplot(data, labels=modes, patch_artist=True)

        colors = plt.cm.Set2(range(len(modes)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Mode')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def generate_pgfplots_radar(
        self,
        radar_data: RadarChartData,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate PGFPlots code for radar chart (for LaTeX documents).

        Args:
            radar_data: RadarChartData structure
            output_path: Optional path to save the LaTeX code

        Returns:
            LaTeX/PGFPlots code string
        """
        n_dims = len(radar_data.dimensions)

        lines = []
        lines.append(r"\begin{tikzpicture}")
        lines.append(r"\begin{polaraxis}[")
        lines.append(f"    xtick={{0, {', '.join([str(i * 360 // n_dims) for i in range(n_dims)])}}},")
        lines.append(f"    xticklabels={{{', '.join(radar_data.dimension_names)}}},")
        lines.append(r"    ymin=0, ymax=1,")
        lines.append(r"    ytick={0.2, 0.4, 0.6, 0.8, 1.0},")
        lines.append(r"    legend style={at={(1.3,1)}, anchor=north west},")
        lines.append(r"]")

        colors = ["blue", "red", "green", "orange", "purple", "cyan"]
        for i, (series_name, scores) in enumerate(radar_data.series.items()):
            color = colors[i % len(colors)]
            # Convert to angles
            coords = []
            for j, score in enumerate(scores):
                angle = j * 360 / n_dims
                coords.append(f"({angle},{score})")
            coords.append(coords[0])  # Close

            lines.append(f"\\addplot[{color}, thick, mark=*] coordinates {{{' '.join(coords)}}};")
            lines.append(f"\\addlegendentry{{{series_name}}}")

        lines.append(r"\end{polaraxis}")
        lines.append(r"\end{tikzpicture}")

        latex_code = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(latex_code)

        return latex_code

    # ------------------------------------------------------------------
    # WP7: Multi-run visualization methods
    # ------------------------------------------------------------------

    def plot_confidence_intervals(
        self,
        aggregated_scores: Dict[str, AggregatedScores],
        output_path: Optional[Path] = None,
        title: str = "Dimension Scores with Confidence Intervals",
        figsize: Tuple[int, int] = (14, 8),
    ) -> Optional[Path]:
        """
        Generate error bar chart showing mean ± CI per dimension per team/mode.

        Args:
            aggregated_scores: Dict mapping label (e.g., "IdeationTeam_ModeNoWcNoHITL")
                to AggregatedScores from StatisticalAggregator.
            output_path: Output file path (if None, displays interactively).
            title: Chart title.
            figsize: Figure size.

        Returns:
            Path to saved file if output_path provided, None otherwise.
        """
        if not self._matplotlib_available:
            print("matplotlib not available. Cannot generate confidence interval plot.")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        # Collect all dimensions across all configs
        all_dims = set()
        for agg in aggregated_scores.values():
            all_dims.update(agg.dimensions.keys())
        dim_ids = sorted(all_dims)
        dim_names = [DIMENSIONS[d].name if d in DIMENSIONS else d for d in dim_ids]

        labels = list(aggregated_scores.keys())
        n_labels = len(labels)
        n_dims = len(dim_ids)

        if n_dims == 0 or n_labels == 0:
            return None

        x = np.arange(n_dims)
        width = 0.8 / n_labels

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set2(np.linspace(0, 1, n_labels))

        for i, (label, color) in enumerate(zip(labels, colors)):
            agg = aggregated_scores[label]
            means = []
            ci_lows = []
            ci_highs = []

            for dim_id in dim_ids:
                if dim_id in agg.dimensions:
                    ds = agg.dimensions[dim_id]
                    means.append(ds.mean)
                    ci_lows.append(ds.mean - ds.ci_lower)
                    ci_highs.append(ds.ci_upper - ds.mean)
                else:
                    means.append(0.0)
                    ci_lows.append(0.0)
                    ci_highs.append(0.0)

            offset = (i - n_labels / 2 + 0.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=[ci_lows, ci_highs],
                capsize=3,
                label=label,
                color=color,
                alpha=0.85,
                error_kw={"elinewidth": 1.2},
            )

        ax.set_xlabel("Dimension")
        ax.set_ylabel("Score (mean ± 95% CI)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names, rotation=45, ha="right")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def plot_factorial_effects(
        self,
        effects: Dict[str, FactorialEffects],
        output_path: Optional[Path] = None,
        title: str = "Factorial Effects: Firecrawl × HITL",
        figsize: Tuple[int, int] = (10, 8),
    ) -> Optional[Path]:
        """
        Generate forest plot of Fc/HITL/interaction effects ± SE.

        Args:
            effects: Dict mapping dimension_id to FactorialEffects
                from StatisticalAggregator.compute_factorial_effects().
            output_path: Output file path.
            title: Chart title.
            figsize: Figure size.

        Returns:
            Path to saved file if output_path provided, None otherwise.
        """
        if not self._matplotlib_available:
            print("matplotlib not available. Cannot generate factorial effects plot.")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        if not effects:
            return None

        # Build data: for each dimension, show Fc, HITL, and interaction effects
        dim_ids = sorted(effects.keys())
        effect_names = ["Firecrawl", "HITL", "Interaction"]
        n_effects = len(effect_names)
        n_dims = len(dim_ids)

        fig, ax = plt.subplots(figsize=figsize)

        y_positions = []
        y_labels = []
        colors_list = []
        effect_colors = {"Firecrawl": "#1f77b4", "HITL": "#ff7f0e", "Interaction": "#2ca02c"}

        pos = 0
        for dim_id in dim_ids:
            fe = effects[dim_id]
            dim_name = DIMENSIONS[dim_id].name if dim_id in DIMENSIONS else dim_id

            for eff_name, eff_obj in [
                ("Firecrawl", fe.firecrawl),
                ("HITL", fe.hitl),
                ("Interaction", fe.interaction),
            ]:
                ci_err = [[eff_obj.effect - eff_obj.ci_lower],
                          [eff_obj.ci_upper - eff_obj.effect]]
                ax.errorbar(
                    eff_obj.effect,
                    pos,
                    xerr=ci_err,
                    fmt="o",
                    color=effect_colors[eff_name],
                    markersize=6,
                    capsize=4,
                    elinewidth=1.5,
                )
                y_positions.append(pos)
                y_labels.append(f"{dim_name} — {eff_name}")
                pos += 1

            pos += 0.5  # gap between dimensions

        # Zero-effect reference line
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Effect Size")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=effect_colors[n], markersize=8, label=n)
            for n in effect_names
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def export_latex_tables(
        self,
        scores: Dict[str, Dict[str, float]],
        caption: str = "Evaluation Scores by Configuration",
        label: str = "tab:eval-scores",
    ) -> str:
        """
        Auto-generate LaTeX table markup from dimension scores.

        Args:
            scores: Dict mapping configuration label to {dimension_id: score}.
                e.g., {"IdeationTeam_ModeNoWcNoHITL": {"correctness": 0.85, ...}}
            caption: LaTeX table caption.
            label: LaTeX table label.

        Returns:
            LaTeX table string.
        """
        if not scores:
            return ""

        # Collect all dimensions
        all_dims = set()
        for s in scores.values():
            all_dims.update(s.keys())
        dim_ids = sorted(all_dims)
        dim_names = [DIMENSIONS[d].name if d in DIMENSIONS else d for d in dim_ids]

        configs = sorted(scores.keys())

        # Build table
        n_cols = len(dim_ids) + 1  # config + dimensions
        col_spec = "l" + "c" * len(dim_ids)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            "Configuration & " + " & ".join(dim_names) + r" \\",
            r"\midrule",
        ]

        for config in configs:
            row_scores = [f"{scores[config].get(d, 0.0):.3f}" for d in dim_ids]
            lines.append(f"{config} & " + " & ".join(row_scores) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def export_ci_latex_table(
        self,
        aggregated_scores: Dict[str, AggregatedScores],
        caption: str = "Evaluation Scores with 95\\% Confidence Intervals",
        label: str = "tab:eval-ci",
    ) -> str:
        """
        Generate LaTeX table with mean ± CI for each dimension.

        Args:
            aggregated_scores: Dict mapping label to AggregatedScores.
            caption: Table caption.
            label: Table label.

        Returns:
            LaTeX table string.
        """
        if not aggregated_scores:
            return ""

        # Collect all dimensions
        all_dims = set()
        for agg in aggregated_scores.values():
            all_dims.update(agg.dimensions.keys())
        dim_ids = sorted(all_dims)
        dim_names = [DIMENSIONS[d].name if d in DIMENSIONS else d for d in dim_ids]

        configs = sorted(aggregated_scores.keys())

        col_spec = "l" + "c" * len(dim_ids)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\footnotesize",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            "Configuration & " + " & ".join(dim_names) + r" \\",
            r"\midrule",
        ]

        for config in configs:
            agg = aggregated_scores[config]
            cells = []
            for d in dim_ids:
                if d in agg.dimensions:
                    ds = agg.dimensions[d]
                    cells.append(
                        f"${ds.mean:.2f}"
                        f"\\pm{(ds.ci_upper - ds.ci_lower) / 2:.2f}$"
                    )
                else:
                    cells.append("---")
            lines.append(f"{config} & " + " & ".join(cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def generate_all_plots(
        self,
        radar_data: RadarChartData,
        results_by_mode: Dict[str, List[EvaluationResult]],
        cluster_scores: Dict[str, Dict[str, float]],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Generate all standard visualization plots.

        Args:
            radar_data: Data for radar chart
            results_by_mode: Results grouped by mode
            cluster_scores: Cluster scores by series
            output_dir: Output directory

        Returns:
            Dictionary mapping plot name to output path
        """
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Radar chart
        path = self.plot_radar_chart(
            radar_data,
            output_path=output_dir / "radar_dimensions.png",
            title="Dimension Comparison",
        )
        if path:
            outputs["radar_dimensions"] = path

        # Dimension bars
        path = self.plot_dimension_bars(
            results_by_mode,
            output_path=output_dir / "dimension_bars.png",
            title="Dimension Scores by Mode",
        )
        if path:
            outputs["dimension_bars"] = path

        # Cluster comparison
        path = self.plot_cluster_comparison(
            cluster_scores,
            output_path=output_dir / "cluster_comparison.png",
            title="Cluster Scores Comparison",
        )
        if path:
            outputs["cluster_comparison"] = path

        # PGFPlots for LaTeX
        latex_code = self.generate_pgfplots_radar(
            radar_data,
            output_path=output_dir / "radar_pgfplots.tex",
        )
        outputs["radar_pgfplots"] = output_dir / "radar_pgfplots.tex"

        return outputs
