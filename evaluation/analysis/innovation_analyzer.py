# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Innovation metrics analyzer using embedding-based novelty measurement.

Provides three complementary innovation metrics:
1. Embedding Novelty — cosine distance from outputs to reference corpus
2. Cross-Domain Score — breadth of economic subfields referenced
3. Diversity Index — pairwise distance entropy across generated ideas

Uses OpenAI text-embedding-3-small for embedding computation.
Falls back to keyword-based heuristics when embeddings are unavailable.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Economic subdomain taxonomy for cross-domain scoring
# ---------------------------------------------------------------------------

ECONOMIC_SUBFIELDS: Dict[str, List[str]] = {
    "macroeconomics": [
        "gdp", "inflation", "monetary policy", "fiscal policy", "business cycle",
        "aggregate demand", "aggregate supply", "central bank", "interest rate",
        "output gap", "unemployment rate", "recession", "economic growth",
        "quantitative easing", "phillips curve",
    ],
    "microeconomics": [
        "consumer choice", "utility", "demand curve", "supply curve",
        "market equilibrium", "price elasticity", "marginal cost",
        "marginal utility", "indifference curve", "budget constraint",
        "perfect competition", "monopoly", "oligopoly",
    ],
    "econometrics": [
        "regression", "instrumental variable", "panel data", "time series",
        "causal inference", "difference-in-differences", "ols", "iv estimation",
        "gmm", "maximum likelihood", "bayesian estimation", "var model",
        "cointegration", "granger causality", "heteroskedasticity",
    ],
    "behavioral_economics": [
        "bounded rationality", "prospect theory", "nudge", "heuristic",
        "cognitive bias", "loss aversion", "framing effect", "mental accounting",
        "anchoring", "overconfidence", "behavioral finance", "default effect",
    ],
    "development_economics": [
        "poverty", "inequality", "foreign aid", "microfinance",
        "human capital", "institutions", "developing countries",
        "economic development", "structural transformation", "migration",
        "rct", "randomized controlled trial",
    ],
    "international_economics": [
        "trade", "tariff", "exchange rate", "balance of payments",
        "comparative advantage", "globalization", "fdi", "foreign direct investment",
        "trade policy", "currency", "trade deficit", "protectionism",
    ],
    "labor_economics": [
        "wage", "labor market", "human capital", "education",
        "skill premium", "minimum wage", "labor supply", "labor demand",
        "unemployment", "job search", "occupational choice", "gig economy",
    ],
    "public_economics": [
        "taxation", "public goods", "externality", "welfare",
        "social insurance", "redistribution", "government spending",
        "public finance", "optimal taxation", "pigouvian tax",
    ],
    "financial_economics": [
        "asset pricing", "portfolio", "risk premium", "capm",
        "efficient market", "option pricing", "derivatives",
        "credit risk", "financial crisis", "banking", "stock market",
        "bond market", "volatility", "hedge fund",
    ],
    "environmental_economics": [
        "carbon tax", "emissions trading", "climate change", "sustainability",
        "natural resources", "pollution", "green economy",
        "renewable energy", "environmental regulation", "externalities",
    ],
    "health_economics": [
        "healthcare", "health insurance", "pharmaceutical",
        "mortality", "morbidity", "health expenditure", "pandemic",
        "epidemiology", "public health", "mental health",
    ],
    "industrial_organization": [
        "market structure", "antitrust", "merger", "entry barrier",
        "network effects", "platform", "regulation", "market power",
        "price discrimination", "vertical integration",
    ],
    "computational_economics": [
        "agent-based model", "simulation", "machine learning",
        "artificial intelligence", "natural language processing",
        "deep learning", "neural network", "algorithm",
        "computational method", "numerical method",
    ],
    "urban_economics": [
        "housing", "real estate", "urban planning", "agglomeration",
        "spatial economics", "commuting", "gentrification", "land use",
        "city", "metropolitan",
    ],
}


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization with basic cleanup."""
    return re.findall(r"[a-z][a-z'-]+", text.lower())


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InnovationAnalyzer:
    """
    Embedding-based innovation metrics analyzer.

    Computes novelty, cross-domain integration, and diversity scores
    for workflow outputs. Uses OpenAI embeddings when available,
    falls back to keyword heuristics otherwise.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        subfields: Optional[Dict[str, List[str]]] = None,
        collector: "Optional[Any]" = None,
    ):
        """
        Initialize InnovationAnalyzer.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Embedding model name.
            subfields: Custom economic subfield taxonomy. Uses default if None.
            collector: Optional MetricsCollector for observability tracking.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.subfields = subfields or ECONOMIC_SUBFIELDS
        self.collector = collector

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts via OpenAI API (httpx-based).

        Args:
            texts: Texts to embed (truncated to 8191 tokens each).

        Returns:
            List of embedding vectors.
        """
        import time
        import httpx

        truncated = [t[:30000] for t in texts]  # ~8k tokens approx
        url = "https://api.openai.com/v1/embeddings"

        start = time.time()
        error_msg = None
        try:
            resp = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": truncated,
                    "model": self.model,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.time() - start

            # Record metrics if collector is available
            if self.collector is not None:
                from shared.observability import LLMCallRecord, estimate_cost

                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens)
                cost = estimate_cost(self.model, prompt_tokens, 0)

                record = LLMCallRecord(
                    agent="InnovationAnalyzer",
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                    latency_seconds=latency,
                )
                self.collector.record_llm_call(record)

            return [item["embedding"] for item in data["data"]]

        except Exception as e:
            latency = time.time() - start
            if self.collector is not None:
                from shared.observability import LLMCallRecord

                record = LLMCallRecord(
                    agent="InnovationAnalyzer",
                    model=self.model,
                    latency_seconds=latency,
                    error=str(e),
                )
                self.collector.record_llm_call(record)
            raise

    # ------------------------------------------------------------------
    # Core innovation metrics
    # ------------------------------------------------------------------

    def compute_embedding_novelty(
        self,
        outputs: List[str],
        reference_corpus: List[str],
    ) -> float:
        """
        Compute novelty as average cosine distance from outputs to reference corpus.

        Higher scores indicate outputs that are more distant (novel) relative
        to existing literature. Score range: 0.0 (identical) to 1.0 (maximally distant).

        Args:
            outputs: Generated workflow output texts.
            reference_corpus: Reference literature texts for comparison.

        Returns:
            Novelty score in [0, 1].
        """
        if not outputs or not reference_corpus:
            return 0.0

        try:
            output_embeddings = self._get_embeddings(outputs)
            ref_embeddings = self._get_embeddings(reference_corpus)
        except Exception:
            # Fall back to keyword-based novelty
            return self._keyword_novelty(outputs, reference_corpus)

        # For each output, find max similarity to any reference
        # Novelty = 1 - avg(max_similarity)
        max_similarities = []
        for out_emb in output_embeddings:
            sims = [_cosine_similarity(out_emb, ref_emb) for ref_emb in ref_embeddings]
            max_similarities.append(max(sims) if sims else 0.0)

        avg_max_sim = sum(max_similarities) / len(max_similarities)
        return max(0.0, min(1.0, 1.0 - avg_max_sim))

    def _keyword_novelty(
        self,
        outputs: List[str],
        reference_corpus: List[str],
    ) -> float:
        """Keyword-based fallback for novelty when embeddings unavailable."""
        output_terms = set()
        for text in outputs:
            output_terms.update(_tokenize(text))

        ref_terms = set()
        for text in reference_corpus:
            ref_terms.update(_tokenize(text))

        if not output_terms:
            return 0.0
        if not ref_terms:
            return 1.0

        # Novel terms = output terms not in reference
        novel = output_terms - ref_terms
        return len(novel) / len(output_terms)

    def compute_cross_domain_score(
        self,
        outputs: List[str],
        domain_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> float:
        """
        Compute cross-domain integration score.

        Measures how many distinct economic subfields are meaningfully
        referenced in the outputs. Score normalized to [0, 1].

        Args:
            outputs: Generated workflow output texts.
            domain_keywords: Custom domain taxonomy. Uses self.subfields if None.

        Returns:
            Cross-domain score in [0, 1].
        """
        domains = domain_keywords or self.subfields
        if not outputs or not domains:
            return 0.0

        combined_text = " ".join(outputs).lower()
        tokens = set(_tokenize(combined_text))

        # Count subfields with at least 2 keyword matches (to reduce noise)
        matched_fields = []
        for field_name, keywords in domains.items():
            matches = sum(1 for kw in keywords if kw in combined_text)
            if matches >= 2:
                matched_fields.append(field_name)

        # Normalize: score of 1.0 if >= 5 distinct subfields referenced
        # (most research touches 1-3 subfields; 5+ indicates strong cross-domain work)
        max_expected = 5.0
        score = min(len(matched_fields) / max_expected, 1.0)

        return score

    def compute_diversity_index(
        self,
        outputs: List[str],
    ) -> float:
        """
        Compute diversity index from pairwise distances across outputs.

        Measures how diverse the generated ideas are from each other.
        High diversity means outputs explore different conceptual spaces.
        Score range: 0.0 (all identical) to 1.0 (maximally diverse).

        Args:
            outputs: Generated workflow output texts.

        Returns:
            Diversity index in [0, 1].
        """
        if len(outputs) < 2:
            return 0.0

        try:
            embeddings = self._get_embeddings(outputs)
        except Exception:
            return self._keyword_diversity(outputs)

        # Compute pairwise cosine distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                distances.append(1.0 - sim)

        if not distances:
            return 0.0

        # Average pairwise distance as diversity measure
        avg_distance = sum(distances) / len(distances)
        return max(0.0, min(1.0, avg_distance))

    def _keyword_diversity(self, outputs: List[str]) -> float:
        """Keyword-based fallback for diversity when embeddings unavailable."""
        if len(outputs) < 2:
            return 0.0

        token_sets = [set(_tokenize(text)) for text in outputs]

        # Pairwise Jaccard distance
        distances = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                union = token_sets[i] | token_sets[j]
                intersection = token_sets[i] & token_sets[j]
                if union:
                    jaccard = len(intersection) / len(union)
                    distances.append(1.0 - jaccard)
                else:
                    distances.append(0.0)

        return sum(distances) / len(distances) if distances else 0.0

    # ------------------------------------------------------------------
    # Composite innovation score
    # ------------------------------------------------------------------

    def compute_composite_innovation(
        self,
        outputs: List[str],
        reference_corpus: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute composite innovation score from all three sub-metrics.

        Args:
            outputs: Generated workflow output texts.
            reference_corpus: Reference texts for novelty comparison.
                If None, novelty is excluded from composite.
            weights: Custom weights for sub-metrics. Keys: 'novelty',
                'cross_domain', 'diversity'. Default: equal weights.

        Returns:
            Dictionary with individual scores and composite:
            {
                'novelty': float,
                'cross_domain': float,
                'diversity': float,
                'composite': float,
                'n_outputs': int,
                'n_references': int,
                'subfields_matched': int,
            }
        """
        default_weights = {
            "novelty": 0.4,
            "cross_domain": 0.3,
            "diversity": 0.3,
        }
        w = weights or default_weights

        # Compute sub-metrics
        novelty = 0.0
        has_novelty = False
        if reference_corpus:
            novelty = self.compute_embedding_novelty(outputs, reference_corpus)
            has_novelty = True

        cross_domain = self.compute_cross_domain_score(outputs)
        diversity = self.compute_diversity_index(outputs)

        # Count matched subfields for reporting
        combined_text = " ".join(outputs).lower()
        matched_count = sum(
            1 for keywords in self.subfields.values()
            if sum(1 for kw in keywords if kw in combined_text) >= 2
        )

        # Composite: re-normalize weights if novelty excluded
        if has_novelty:
            total_w = w.get("novelty", 0.4) + w.get("cross_domain", 0.3) + w.get("diversity", 0.3)
            composite = (
                w.get("novelty", 0.4) * novelty
                + w.get("cross_domain", 0.3) * cross_domain
                + w.get("diversity", 0.3) * diversity
            ) / total_w
        else:
            total_w = w.get("cross_domain", 0.3) + w.get("diversity", 0.3)
            composite = (
                w.get("cross_domain", 0.3) * cross_domain
                + w.get("diversity", 0.3) * diversity
            ) / total_w if total_w > 0 else 0.0

        return {
            "novelty": novelty,
            "cross_domain": cross_domain,
            "diversity": diversity,
            "composite": max(0.0, min(1.0, composite)),
            "n_outputs": len(outputs),
            "n_references": len(reference_corpus) if reference_corpus else 0,
            "subfields_matched": matched_count,
        }

    # ------------------------------------------------------------------
    # File-based analysis helpers
    # ------------------------------------------------------------------

    def analyze_output_directory(
        self,
        output_dir: Path,
        reference_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Analyze innovation from output files in a directory.

        Reads all .json and .txt files from output_dir as outputs,
        and optionally from reference_dir as reference corpus.

        Args:
            output_dir: Directory containing workflow outputs.
            reference_dir: Directory containing reference literature.

        Returns:
            Composite innovation result dictionary.
        """
        outputs = self._load_texts_from_dir(output_dir)
        reference_corpus = None
        if reference_dir and reference_dir.exists():
            reference_corpus = self._load_texts_from_dir(reference_dir)

        return self.compute_composite_innovation(outputs, reference_corpus)

    def _load_texts_from_dir(self, directory: Path) -> List[str]:
        """Load text content from .json and .txt files in a directory."""
        texts = []
        if not directory.exists():
            return texts

        for fpath in sorted(directory.iterdir()):
            if fpath.suffix == ".json":
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    texts.append(json.dumps(data, indent=2))
                except (json.JSONDecodeError, IOError):
                    pass
            elif fpath.suffix in (".txt", ".md"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                except IOError:
                    pass

        return texts
