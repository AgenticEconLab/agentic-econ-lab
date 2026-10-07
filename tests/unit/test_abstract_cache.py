# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for the generic, optional local abstract cache (shared/tools/abstract_cache.py).

The repo ships no data: the cache is a no-op unless ABSTRACT_CACHE points at a user CSV.
"""
from shared.tools import abstract_cache as ac


def _write(tmp_path, text):
    p = tmp_path / "abs.csv"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_lookup_and_doi_normalization(tmp_path, monkeypatch):
    p = _write(tmp_path, "DOI,Abstract\n10.1/x,Hello world\n")
    monkeypatch.setenv("ABSTRACT_CACHE", p)
    ac._load_cache.cache_clear()
    assert ac.cache_size() == 1
    assert ac.abstract_from_cache("10.1/x") == "Hello world"
    assert ac.abstract_from_cache("https://doi.org/10.1/X") == "Hello world"  # prefix + case
    assert ac.abstract_from_cache("doi:10.1/x") == "Hello world"
    assert ac.abstract_from_cache("10.1/missing") == ""


def test_noop_when_unset(monkeypatch):
    monkeypatch.delenv("ABSTRACT_CACHE", raising=False)
    ac._load_cache.cache_clear()
    assert ac.cache_size() == 0
    assert ac.abstract_from_cache("10.1/x") == ""


def test_semicolon_delimiter_and_column_autodetect(tmp_path, monkeypatch):
    p = _write(tmp_path, "doi;title;abstract\n10.2/y;A title;Sci abstract\n")
    monkeypatch.setenv("ABSTRACT_CACHE", p)
    ac._load_cache.cache_clear()
    assert ac.abstract_from_cache("10.2/y") == "Sci abstract"


def test_column_override_env(tmp_path, monkeypatch):
    p = _write(tmp_path, "id,summary\n10.3/z,Overridden cols\n")
    monkeypatch.setenv("ABSTRACT_CACHE", p)
    monkeypatch.setenv("ABSTRACT_CACHE_DOI_COL", "id")
    monkeypatch.setenv("ABSTRACT_CACHE_ABS_COL", "summary")
    ac._load_cache.cache_clear()
    assert ac.abstract_from_cache("10.3/z") == "Overridden cols"
    monkeypatch.delenv("ABSTRACT_CACHE_DOI_COL", raising=False)
    monkeypatch.delenv("ABSTRACT_CACHE_ABS_COL", raising=False)
    ac._load_cache.cache_clear()


def test_extract_doi():
    assert ac.extract_doi("https://doi.org/10.1/AbC") == "10.1/AbC"
    assert ac.extract_doi("10.1016/j.x") == "10.1016/j.x"
    assert ac.extract_doi("https://semanticscholar.org/paper/abc") == ""
    assert ac.extract_doi("") == ""


class _Item:
    def __init__(self, abstract="", url=""):
        self.abstract = abstract; self.url = url


def test_has_usable_abstract_and_coverage(monkeypatch):
    # no fill configured -> the ⚠ "configure something" reminder
    monkeypatch.delenv("ABSTRACT_CACHE", raising=False)
    monkeypatch.delenv("ELSEVIER_API_KEY", raising=False)
    ac._load_cache.cache_clear()
    assert ac.has_usable_abstract(_Item(abstract="x" * 60)) is True
    assert ac.has_usable_abstract(_Item(abstract="No abstract available")) is False
    assert ac.has_usable_abstract(_Item(abstract="short")) is False
    cov = ac.abstract_coverage(10, 7)
    assert cov["dropped_no_abstract"] == 3 and cov["reminder"].startswith("⚠")
