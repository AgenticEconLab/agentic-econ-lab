# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Deterministic aggregation of committee answers to SELECTION checkpoints.

A selection checkpoint asks the reviewer to pick items from a numbered list
("questions to discard (comma-separated)", "pairs of questions to merge",
"relevant paper titles OR numbers"). A free-text committee chair was observed
returning the UNION of the members' picks: members wanted to discard '2', '2,3,4,5' and
'3,5', the chair returned '2,3,4,5'; merge answers came back with overlapping pairs and
an index 6 on a 5-item list. This module replaces the chair's free-text consolidation for
such prompts with a majority rule:

  * an item (or pair) is kept only if a strict majority of the responding members chose
    it (2 of 3);
  * numeric indices outside 1..N are dropped when N is known;
  * merge pairs are de-duplicated, self-pairs are dropped, and the kept pairs never share
    an index (greedy by support, then by index order).

Nothing here is topic-specific; it operates on the answer format only.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PAIRS_RE = re.compile(r"\bpairs?\b", re.IGNORECASE)
_ITEMS_RE = re.compile(
    r"\bnumbers?\b|\bto (?:discard|prioriti[sz]e|merge|keep|drop|remove)\b"
    r"|\b(?:discard|prioriti[sz]e)\b.*comma[- ]separated",
    re.IGNORECASE,
)
_PAIR_TOKEN_RE = re.compile(r"(\d+)\s*[,&/-]\s*(\d+)")
_INT_RE = re.compile(r"^\d+$")


def selection_kind(prompt: str) -> Optional[str]:
    """'pairs' for pair-selection prompts, 'items' for item-selection prompts, else None."""
    p = prompt or ""
    if _PAIRS_RE.search(p) and re.search(r"merge|combine|pair", p, re.IGNORECASE):
        return "pairs"
    if _ITEMS_RE.search(p):
        return "items"
    return None


def count_listed_items(context: Optional[Dict[str, Any]]) -> Optional[int]:
    """Number of items under review: total length of the list values in ``context``.

    Stages that show several lists in one numbered preview (e.g. concepts followed by
    questions) number them as one flat list, so the total is the valid index range.
    Returns None when the context carries no list."""
    if not isinstance(context, dict):
        return None
    total, seen = 0, False
    for v in context.values():
        if isinstance(v, list):
            total += len(v)
            seen = True
    return total if seen else None


def _norm(tok: str) -> str:
    return re.sub(r"\s+", " ", tok.strip().strip("'\"").strip()).lower()


def parse_items(text: str) -> List[str]:
    """Split an answer into normalized item tokens (numbers stay as digit strings)."""
    out: List[str] = []
    for raw in re.split(r"[,;\n]", text or ""):
        t = _norm(raw)
        t = t.rstrip(".")
        if not t:
            continue
        # "2 5" (space separated numbers) -> two items
        parts = t.split()
        if len(parts) > 1 and all(_INT_RE.match(x) for x in parts):
            out.extend(str(int(x)) for x in parts)
        elif _INT_RE.match(t):
            out.append(str(int(t)))
        else:
            out.append(t)
    return list(OrderedDict.fromkeys(out))


def parse_pairs(text: str) -> List[Tuple[int, int]]:
    """Parse 'a,b c,d' into sorted (a, b) tuples (self-pairs dropped, duplicates removed)."""
    out: List[Tuple[int, int]] = []
    for a, b in _PAIR_TOKEN_RE.findall(text or ""):
        i, j = int(a), int(b)
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair not in out:
            out.append(pair)
    return out


def majority_threshold(n_respondents: int) -> int:
    """Strict majority of the members that responded (2 of 3; 1 of 1)."""
    return n_respondents // 2 + 1 if n_respondents > 0 else 1


def _in_range(i: int, n_items: Optional[int]) -> bool:
    return i >= 1 and (n_items is None or i <= n_items)


def aggregate_items(answers: Sequence[str], n_items: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """Majority-aggregate item selections. Returns (answer_string, details)."""
    threshold = majority_threshold(len(answers))
    support: Counter = Counter()
    first_seen: Dict[str, int] = {}
    rejected: List[str] = []
    for ans in answers:
        for tok in parse_items(ans):
            if _INT_RE.match(tok) and not _in_range(int(tok), n_items):
                rejected.append(tok)
                continue
            support[tok] += 1
            first_seen.setdefault(tok, len(first_seen))
    kept = [t for t in support if support[t] >= threshold]
    nums = sorted((t for t in kept if _INT_RE.match(t)), key=int)
    words = sorted((t for t in kept if not _INT_RE.match(t)), key=lambda t: first_seen[t])
    answer = ",".join(nums + words)
    return answer, {
        "rule": f"majority ({threshold} of {len(answers)})",
        "support": dict(support),
        "rejected_out_of_range": sorted(set(rejected)),
        "n_items": n_items,
    }


def aggregate_pairs(answers: Sequence[str], n_items: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """Majority-aggregate merge pairs; kept pairs are valid and never share an index."""
    threshold = majority_threshold(len(answers))
    support: Counter = Counter()
    rejected: List[str] = []
    for ans in answers:
        for i, j in parse_pairs(ans):
            if not (_in_range(i, n_items) and _in_range(j, n_items)):
                rejected.append(f"{i},{j}")
                continue
            support[(i, j)] += 1
    candidates = sorted((p for p in support if support[p] >= threshold),
                        key=lambda p: (-support[p], p))
    used: set = set()
    kept: List[Tuple[int, int]] = []
    for i, j in candidates:
        if i in used or j in used:
            continue
        kept.append((i, j))
        used.update((i, j))
    kept.sort()
    answer = " ".join(f"{i},{j}" for i, j in kept)
    return answer, {
        "rule": f"majority ({threshold} of {len(answers)}), non-overlapping",
        "support": {f"{i},{j}": c for (i, j), c in support.items()},
        "rejected_out_of_range": sorted(set(rejected)),
        "n_items": n_items,
    }


def aggregate_selection(kind: str, answers: Iterable[str],
                        n_items: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """Dispatch on ``kind`` ('pairs' | 'items')."""
    answers = [a or "" for a in answers]
    if kind == "pairs":
        return aggregate_pairs(answers, n_items)
    return aggregate_items(answers, n_items)


# --------------------------------------------------------------------------- #
# Applying selections to a ranked list (shared by stages that take merge/discard)
# --------------------------------------------------------------------------- #

def plan_selection(
    n: int,
    discard: Iterable[int],
    merge: Iterable[Tuple[int, int]],
    min_keep: int = 3,
) -> Dict[str, Any]:
    """Resolve 0-based discard indices and merge pairs against ONE original numbering.

    * indices outside 0..n-1 are dropped;
    * discards never leave fewer than ``min(min_keep, n)`` survivors — when they would,
      the highest-ranked (lowest index) discards are restored first;
    * merges apply only between two surviving indices, and no index is merged twice.

    Returns {"keep": [...], "discard": [...], "merge": [(i, j), ...], "restored": [...],
    "dropped_merges": [...]}; ``keep`` lists surviving indices in original order, with a
    merged pair's second index folded into its first (so it does not appear in keep)."""
    floor = min(max(min_keep, 0), n)
    disc = sorted({int(i) for i in discard if 0 <= int(i) < n})
    restored: List[int] = []
    while n - len(disc) < floor and disc:
        restored.append(disc.pop(0))
    alive = [i for i in range(n) if i not in disc]
    merges: List[Tuple[int, int]] = []
    dropped: List[Tuple[int, int]] = []
    used: set = set()
    for pair in merge:
        try:
            a, b = int(pair[0]), int(pair[1])
        except Exception:
            continue
        i, j = min(a, b), max(a, b)
        if i == j or i not in alive or j not in alive or i in used or j in used:
            dropped.append((a, b))
            continue
        merges.append((i, j))
        used.update((i, j))
    # each merge removes one question; merges never take the set below the floor either
    while merges and len(alive) - len(merges) < floor:
        dropped.append(merges.pop())
    absorbed = {j for _, j in merges}
    keep = [i for i in alive if i not in absorbed]
    return {"keep": keep, "discard": disc, "merge": merges,
            "restored": restored, "dropped_merges": dropped}


# --------------------------------------------------------------------------- #
# Topic adherence (deterministic, generic)
# --------------------------------------------------------------------------- #

_STOP = {
    "the", "and", "for", "with", "from", "into", "onto", "that", "this", "these", "those",
    "about", "under", "over", "between", "among", "within", "without", "their", "there",
    "what", "which", "when", "where", "while", "whose", "does", "doing", "done", "have",
    "has", "how", "why", "are", "was", "were", "been", "being", "its", "they", "them",
    "than", "then", "also", "such", "using", "use", "used", "via", "economic", "economics",
    "economy", "research", "study", "studies", "analysis", "effect", "effects", "impact",
    "impacts", "role", "evidence", "approach", "new", "based", "toward", "towards",
    "upstream", "topic", "metadata", "unavailable",
}


def content_words(text: str) -> List[str]:
    """Lower-cased content words (>=3 letters, not stop words), crudely stemmed to 5 chars
    so 'inflation'/'inflationary' and 'tariff'/'tariffs' match."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or "")
    out = []
    for w in words:
        w = w.lower().strip("-")
        if w in _STOP or len(w) < 3:
            continue
        out.append(w[:5])
    return list(OrderedDict.fromkeys(out))


def topic_overlap(topic: str, text: str) -> Optional[float]:
    """Share of the topic's content words that occur in ``text`` (None if the topic has
    no content words, e.g. an empty or placeholder topic)."""
    tw = content_words(topic)
    if not tw:
        return None
    present = set(content_words(text))
    return round(sum(1 for w in tw if w in present) / len(tw), 3)


def is_on_topic(topic: str, text: str, min_share: float = 0.2) -> Optional[bool]:
    """True when at least one topic content word, and at least ``min_share`` of them,
    occurs in ``text``; None when the topic has no content words."""
    share = topic_overlap(topic, text)
    if share is None:
        return None
    return share > 0 and share >= min_share
