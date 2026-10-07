# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Deterministic ancestry check for LLM steps that may only REFORMULATE given questions.

The Contextualizer and Finalizer of the IdeationTeam Integration stage receive a numbered
list of questions and must return reformulations of those questions, not new ones. Without
a link back to the input (their output used to be parsed, sorted and truncated), a wholly new question carrying enough seed-topic vocabulary passed
the topic check and became rank one.

Each input question carries a stable source id (``Q1`` .. ``Qn``) in the prompt and each
output must cite one in ``source_id``. :func:`validate_ancestry` then decides, per output:

  * the cited id must parse to an input question and must not already be claimed by a
    higher-ranked output; otherwise the output is dropped;
  * the output text must be a reformulation of the cited source:
    :func:`reformulation_similarity` >= ``REFORMULATION_MIN_SIMILARITY``; otherwise the
    output is replaced by its source question unchanged;
  * dropped outputs free their slot; the slot is filled with an input question no output
    cited (in input order), carried forward unchanged.

The result never has more entries than inputs, and each entry records ``source_id`` and a
status: ``accepted`` (text equal to the source after normalisation), ``reformulated``
(changed text that passed the similarity threshold) or ``carried_forward`` (the source
text replaced a rejected output or filled a dropped slot).

Similarity measure (generic, no vocabulary list): the Dice coefficient
2|S n C| / (|S| + |C|) over the content words of source S and candidate C (as computed by
:func:`shared.selection_aggregation.content_words`: stop words removed, words stemmed to
five characters), after removing the seed topic's own content words when a topic is
given. Removing the topic words means two questions do not count as related merely for
sharing the seed topic's vocabulary, which every question in the run is asked to carry.
If removing them leaves either side empty, the full word sets are used.

Threshold 0.3: on recorded v0.7.1 runs (demo + 25 replicates, same seed topic),
comparing each final question with the best-matching question that entered the
Integration stage (three rounds of reformulation apart), 0.3 accepts 65% of them, while a
final question taken from a DIFFERENT run, compared with the best of this run's six inputs,
passes in 16% of cases (it would pass less often against the single source it cites). A
single reformulation step changes less text than three rounds, so per-step acceptance is
higher than the end-to-end figure.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shared.selection_aggregation import content_words

REFORMULATION_MIN_SIMILARITY = 0.3

ACCEPTED = "accepted"
REFORMULATED = "reformulated"
CARRIED_FORWARD = "carried_forward"

_ID_RE = re.compile(r"^\s*(?:q(?:uestion)?\s*[#:.\-]?\s*)?(\d+)\s*$", re.IGNORECASE)


def source_label(index: int) -> str:
    """Stable prompt id for the 0-based input ``index`` (``Q1`` for index 0)."""
    return f"Q{index + 1}"


def parse_source_id(raw: Any, n_sources: int) -> Optional[int]:
    """0-based index for a cited id (``"Q3"``, ``"q3"``, ``"Question 3"``, ``"3"``, ``3``),
    or None when it is missing, malformed or outside 1..n_sources."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        if float(raw) != int(raw):
            return None
        num = int(raw)
    else:
        m = _ID_RE.match(str(raw))
        if not m:
            return None
        num = int(m.group(1))
    return num - 1 if 1 <= num <= n_sources else None


def _norm_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def reformulation_similarity(source: str, candidate: str, topic: Optional[str] = None) -> float:
    """Dice coefficient of content words, seed-topic words removed (see module docstring)."""
    if _norm_text(source) and _norm_text(source) == _norm_text(candidate):
        return 1.0
    s_all, c_all = set(content_words(source)), set(content_words(candidate))
    drop = set(content_words(topic)) if topic else set()
    s, c = s_all - drop, c_all - drop
    if not s or not c:
        s, c = s_all, c_all
    if not s or not c:
        return 0.0
    return round(2 * len(s & c) / (len(s) + len(c)), 3)


def validate_ancestry(
    sources: Sequence[str],
    outputs: Sequence[Tuple[str, Any]],
    topic: Optional[str] = None,
    slots: Optional[int] = None,
    threshold: float = REFORMULATION_MIN_SIMILARITY,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Decide which LLM outputs descend from which input question.

    ``sources``: input question texts (id ``Q{i+1}`` for position i).
    ``outputs``: ``(text, cited_source_id)`` in priority order (earlier claims an id first).
    ``slots``: number of entries to return; default ``min(len(outputs), len(sources))``,
    always capped at ``len(sources)``.

    Returns ``(entries, rejected)``. Each entry: ``source_index``, ``source_id``,
    ``output_index`` (None when the source fills a slot), ``status``, ``similarity``,
    ``text`` (the text to keep — always the source question's own text) and
    ``proposed_wording`` (the LLM's wording when it differed, recorded but not published). ``rejected`` lists outputs whose text was not kept, with
    ``output_index``, ``text``, ``cited``, ``reason``.
    """
    n = len(sources)
    if slots is None:
        slots = len(outputs)
    slots = max(0, min(slots, n))
    entries: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    used = set()
    for j, (text, cited) in enumerate(outputs):
        if len(entries) >= slots:
            rejected.append({"output_index": j, "text": text, "cited": cited,
                             "reason": "beyond the number of questions allowed"})
            continue
        i = parse_source_id(cited, n)
        if i is None:
            rejected.append({"output_index": j, "text": text, "cited": cited,
                             "reason": "no valid source id"})
            continue
        if i in used:
            rejected.append({"output_index": j, "text": text, "cited": cited,
                             "reason": f"source id {source_label(i)} already used"})
            continue
        used.add(i)
        sim = reformulation_similarity(sources[i], text, topic)
        entry = {"source_index": i, "source_id": source_label(i), "output_index": j,
                 "similarity": sim}
        if sim >= threshold:
            # verbatim = identical up to whitespace; _norm_text also drops signs and punctuation,
            # so "-2%" vs "+2%" would count as unchanged
            same = " ".join(str(text).split()) == " ".join(str(sources[i]).split())
            # Word overlap cannot tell a rewording from a different question
            # (consumption in France -> bankruptcy in Germany scored 0.5). The published text
            # is always the source question; the LLM's wording is kept only as a proposal.
            entry.update(status=ACCEPTED if same else REFORMULATED, text=sources[i],
                         proposed_wording=(None if same else text))
        else:
            entry.update(status=CARRIED_FORWARD, text=sources[i])
            rejected.append({"output_index": j, "text": text, "cited": source_label(i),
                             "reason": f"similarity {sim} to {source_label(i)} below {threshold}"})
        entries.append(entry)
    for i in range(n):
        if len(entries) >= slots:
            break
        if i in used:
            continue
        used.add(i)
        entries.append({"source_index": i, "source_id": source_label(i), "output_index": None,
                        "similarity": None, "status": CARRIED_FORWARD, "text": sources[i]})
    return entries, rejected
