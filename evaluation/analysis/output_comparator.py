# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Cross-run output comparison for reliability measurement.

Compares workflow outputs across multiple runs to measure true output
consistency — going beyond structural completion to assess whether
the same workflow produces similar results when run repeatedly.

Usage:
    from evaluation.analysis.output_comparator import OutputComparator

    comp = OutputComparator()
    score = comp.compute_reliability_score(run_outputs)
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

# Common English stop words to exclude from term extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "this", "that", "these", "those", "i", "we", "you", "he",
    "she", "they", "me", "us", "him", "her", "them", "my", "our", "your",
    "his", "their", "not", "no", "nor", "so", "if", "then", "than",
    "as", "about", "up", "out", "into", "over", "after", "before",
    "between", "under", "above", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "only", "also", "very",
    "just", "because", "through", "during", "while", "where", "when",
    "how", "what", "which", "who", "whom", "why",
})

_WORD_PATTERN = re.compile(r"[a-z][a-z0-9_-]{2,}", re.IGNORECASE)


def _extract_terms(text: str, min_freq: int = 1) -> Set[str]:
    """Extract meaningful terms from text, excluding stop words."""
    words = _WORD_PATTERN.findall(text.lower())
    terms = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    if min_freq > 1:
        counts = Counter(terms)
        return {t for t, c in counts.items() if c >= min_freq}
    return set(terms)


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _pairwise_mean(values: List[float], func) -> float:
    """Compute mean of pairwise function application."""
    n = len(values)
    if n < 2:
        return 1.0  # single item is perfectly consistent with itself
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += func(values[i], values[j])
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# OutputComparator
# ---------------------------------------------------------------------------

class OutputComparator:
    """
    Compare workflow outputs across multiple runs for reliability measurement.

    Provides three comparison methods:
    1. Text similarity (Jaccard on key terms)
    2. JSON structural overlap (shared keys and value similarity)
    3. Item count stability (1 - CV)

    Combined into a weighted reliability score.
    """

    def __init__(
        self,
        text_weight: float = 0.4,
        structural_weight: float = 0.3,
        count_weight: float = 0.3,
    ):
        """
        Args:
            text_weight: Weight for text similarity in composite score
            structural_weight: Weight for JSON structural overlap
            count_weight: Weight for item count stability
        """
        assert abs(text_weight + structural_weight + count_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.text_weight = text_weight
        self.structural_weight = structural_weight
        self.count_weight = count_weight

    # ------------------------------------------------------------------
    # Text comparison
    # ------------------------------------------------------------------

    def compare_text_outputs(self, outputs: List[str]) -> float:
        """
        Compare text outputs across runs using Jaccard similarity of key terms.

        Computes pairwise Jaccard similarity of extracted term sets and
        returns the mean.

        Args:
            outputs: List of text outputs (one per run)

        Returns:
            Mean pairwise Jaccard similarity (0-1)
        """
        if len(outputs) < 2:
            return 1.0

        term_sets = [_extract_terms(text) for text in outputs]

        # Pairwise Jaccard
        similarities = []
        for i in range(len(term_sets)):
            for j in range(i + 1, len(term_sets)):
                similarities.append(_jaccard_similarity(term_sets[i], term_sets[j]))

        return sum(similarities) / len(similarities) if similarities else 0.0

    # ------------------------------------------------------------------
    # JSON structural comparison
    # ------------------------------------------------------------------

    def compare_json_outputs(self, outputs: List[Dict[str, Any]]) -> float:
        """
        Compare JSON outputs for structural overlap.

        Measures two aspects:
        1. Key overlap — do the same keys appear across runs?
        2. Value similarity — for shared keys, how similar are the values?

        Args:
            outputs: List of JSON dict outputs (one per run)

        Returns:
            Structural similarity score (0-1)
        """
        if len(outputs) < 2:
            return 1.0

        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = self._json_pair_similarity(outputs[i], outputs[j])
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _json_pair_similarity(self, a: Dict, b: Dict) -> float:
        """Compute structural similarity between two JSON dicts."""
        keys_a = set(self._flatten_keys(a))
        keys_b = set(self._flatten_keys(b))

        # Key overlap (Jaccard)
        key_sim = _jaccard_similarity(keys_a, keys_b)

        # Value similarity for shared keys
        shared_keys = keys_a & keys_b
        if not shared_keys:
            return key_sim * 0.5  # only key-level similarity

        value_matches = 0
        for key in shared_keys:
            val_a = self._get_nested(a, key)
            val_b = self._get_nested(b, key)
            if self._values_similar(val_a, val_b):
                value_matches += 1

        value_sim = value_matches / len(shared_keys)

        # Combined: 50% key overlap + 50% value similarity
        return 0.5 * key_sim + 0.5 * value_sim

    def _flatten_keys(self, d: Dict, prefix: str = "") -> List[str]:
        """Flatten nested dict keys into dot-notation paths."""
        keys = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.append(full_key)
            if isinstance(v, dict):
                keys.extend(self._flatten_keys(v, full_key))
        return keys

    def _get_nested(self, d: Dict, dotted_key: str) -> Any:
        """Get value from nested dict using dot-notation key."""
        parts = dotted_key.split(".")
        current = d
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _values_similar(self, a: Any, b: Any) -> bool:
        """Check if two values are similar enough to count as a match."""
        if a is None or b is None:
            return a is b

        if type(a) != type(b):
            return False

        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # Numeric: within 20% relative tolerance
            if a == 0 and b == 0:
                return True
            max_val = max(abs(a), abs(b))
            return abs(a - b) / max_val < 0.2 if max_val > 0 else True

        if isinstance(a, str):
            # String: Jaccard of terms
            terms_a = _extract_terms(a)
            terms_b = _extract_terms(b)
            return _jaccard_similarity(terms_a, terms_b) > 0.5

        if isinstance(a, list):
            # Lists: similar length
            if len(a) == 0 and len(b) == 0:
                return True
            max_len = max(len(a), len(b))
            return abs(len(a) - len(b)) / max_len < 0.3 if max_len > 0 else True

        if isinstance(a, dict):
            # Dicts: key overlap
            keys_a = set(a.keys())
            keys_b = set(b.keys())
            return _jaccard_similarity(keys_a, keys_b) > 0.5

        return a == b

    # ------------------------------------------------------------------
    # Item count comparison
    # ------------------------------------------------------------------

    def compare_item_counts(self, counts: List[int]) -> float:
        """
        Compare item counts across runs using coefficient of variation.

        Returns 1.0 - CV, clamped to [0, 1]. Lower CV means more stable
        output volumes across runs.

        Args:
            counts: List of item counts (one per run)

        Returns:
            Count stability score (0-1), where 1.0 = perfectly stable
        """
        if len(counts) < 2:
            return 1.0

        counts_f = [float(c) for c in counts]
        mean = sum(counts_f) / len(counts_f)
        if mean == 0:
            # All zeros = perfectly consistent
            return 1.0

        variance = sum((c - mean) ** 2 for c in counts_f) / len(counts_f)
        cv = math.sqrt(variance) / mean

        return max(0.0, min(1.0, 1.0 - cv))

    # ------------------------------------------------------------------
    # Composite reliability score
    # ------------------------------------------------------------------

    def compute_reliability_score(
        self,
        run_results: List[Dict[str, Any]],
        text_key: Optional[str] = None,
        json_key: Optional[str] = None,
        count_key: str = "total_items",
    ) -> float:
        """
        Compute composite reliability score from multiple run results.

        Each run_result should be a dict that may contain:
        - Text outputs (for Jaccard comparison)
        - JSON structure (for structural overlap)
        - Item counts (for count stability)

        The function automatically extracts what's available and computes
        a weighted score using only the available comparison types.

        Args:
            run_results: List of per-run result dicts. Expected structure:
                {
                    "text_output": "...",      # or any string field
                    "json_output": {...},       # or the dict itself
                    "total_items": 42,          # or "item_count"
                    "summary": {"total_items": 42},
                }
            text_key: Key for text output in each result (auto-detected if None)
            json_key: Key for JSON output (auto-detected if None)
            count_key: Key for item count (default "total_items")

        Returns:
            Weighted reliability score (0-1)
        """
        if len(run_results) < 2:
            return 1.0

        scores = {}
        weights = {}

        # --- Text comparison ---
        texts = self._extract_texts(run_results, text_key)
        if len(texts) >= 2:
            scores["text"] = self.compare_text_outputs(texts)
            weights["text"] = self.text_weight

        # --- JSON structural comparison ---
        jsons = self._extract_jsons(run_results, json_key)
        if len(jsons) >= 2:
            scores["structural"] = self.compare_json_outputs(jsons)
            weights["structural"] = self.structural_weight

        # --- Item count comparison ---
        counts = self._extract_counts(run_results, count_key)
        if len(counts) >= 2:
            scores["count"] = self.compare_item_counts(counts)
            weights["count"] = self.count_weight

        if not scores:
            return 0.0

        # Re-normalize weights to sum to 1.0 based on available components
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        composite = sum(
            scores[k] * (weights[k] / total_weight)
            for k in scores
        )

        return max(0.0, min(1.0, composite))

    def compute_reliability_breakdown(
        self,
        run_results: List[Dict[str, Any]],
        text_key: Optional[str] = None,
        json_key: Optional[str] = None,
        count_key: str = "total_items",
    ) -> Dict[str, Any]:
        """
        Compute reliability with detailed breakdown of each component.

        Returns:
            Dict with composite score and per-component scores.
        """
        if len(run_results) < 2:
            return {"composite": 1.0, "n_runs": len(run_results), "components": {}}

        components = {}

        texts = self._extract_texts(run_results, text_key)
        if len(texts) >= 2:
            components["text_similarity"] = {
                "score": self.compare_text_outputs(texts),
                "weight": self.text_weight,
                "n_samples": len(texts),
            }

        jsons = self._extract_jsons(run_results, json_key)
        if len(jsons) >= 2:
            components["structural_overlap"] = {
                "score": self.compare_json_outputs(jsons),
                "weight": self.structural_weight,
                "n_samples": len(jsons),
            }

        counts = self._extract_counts(run_results, count_key)
        if len(counts) >= 2:
            mean_count = sum(counts) / len(counts)
            components["count_stability"] = {
                "score": self.compare_item_counts(counts),
                "weight": self.count_weight,
                "n_samples": len(counts),
                "counts": counts,
                "mean": mean_count,
            }

        # Composite
        total_weight = sum(c["weight"] for c in components.values())
        composite = sum(
            c["score"] * (c["weight"] / total_weight)
            for c in components.values()
        ) if total_weight > 0 else 0.0

        return {
            "composite": max(0.0, min(1.0, composite)),
            "n_runs": len(run_results),
            "components": components,
        }

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_texts(
        self, run_results: List[Dict], key: Optional[str] = None
    ) -> List[str]:
        """Extract text outputs from run results."""
        texts = []
        # Try specified key, then common text field names
        text_keys = [key] if key else [
            "text_output", "output", "review", "literature_review",
            "synthesis", "final_output", "result",
        ]

        for result in run_results:
            for tk in text_keys:
                if tk and tk in result and isinstance(result[tk], str) and len(result[tk]) > 20:
                    texts.append(result[tk])
                    break
            else:
                # Fall back: concatenate all string values > 50 chars
                long_strings = [
                    v for v in result.values()
                    if isinstance(v, str) and len(v) > 50
                ]
                if long_strings:
                    texts.append(" ".join(long_strings))

        return texts

    def _extract_jsons(
        self, run_results: List[Dict], key: Optional[str] = None
    ) -> List[Dict]:
        """Extract JSON/dict outputs from run results."""
        jsons = []
        json_keys = [key] if key else [
            "json_output", "data", "results", "output",
            "summary", "stages",
        ]

        for result in run_results:
            for jk in json_keys:
                if jk and jk in result and isinstance(result[jk], dict):
                    jsons.append(result[jk])
                    break
            else:
                # Use the result dict itself (excluding meta fields)
                filtered = {
                    k: v for k, v in result.items()
                    if k not in {"run_id", "run_number", "total_runs", "timestamp"}
                    and isinstance(v, (dict, list, int, float, str))
                }
                if filtered:
                    jsons.append(filtered)

        return jsons

    def _extract_counts(
        self, run_results: List[Dict], key: str = "total_items"
    ) -> List[int]:
        """Extract item counts from run results."""
        counts = []
        count_keys = [key, "total_items", "item_count", "n_items", "count"]

        for result in run_results:
            for ck in count_keys:
                if ck in result and isinstance(result[ck], (int, float)):
                    counts.append(int(result[ck]))
                    break
            else:
                # Try nested: summary.total_items
                summary = result.get("summary", {})
                if isinstance(summary, dict):
                    for ck in count_keys:
                        if ck in summary and isinstance(summary[ck], (int, float)):
                            counts.append(int(summary[ck]))
                            break

        return counts

    # ------------------------------------------------------------------
    # File-based comparison (reads output files from run directories)
    # ------------------------------------------------------------------

    def compare_run_directories(
        self,
        run_dirs: List[Path],
    ) -> Dict[str, Any]:
        """
        Compare outputs across run directories from WP1 multirun structure.

        Reads execution_log.json and any text/JSON output files from each
        run_NNN/ directory.

        Args:
            run_dirs: List of paths to run directories

        Returns:
            Reliability breakdown dict
        """
        run_results = []
        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            result = {}

            # Read execution log
            log_path = run_dir / "execution_log.json"
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
                result["summary"] = log_data.get("summary", {})
                result["total_items"] = log_data.get("summary", {}).get("total_items", 0)
                result["stages"] = log_data.get("stages", [])

            # Read text output files
            text_parts = []
            for txt_file in sorted(run_dir.glob("*.txt")):
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    text_parts.append(f.read())
            if text_parts:
                result["text_output"] = "\n".join(text_parts)

            # Read JSON output files (excluding execution_log.json)
            for json_file in sorted(run_dir.glob("*.json")):
                if json_file.name == "execution_log.json":
                    continue
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        result["json_output"] = json.load(f)
                        break  # use first non-log JSON
                    except json.JSONDecodeError:
                        continue

            if result:
                run_results.append(result)

        return self.compute_reliability_breakdown(run_results)
