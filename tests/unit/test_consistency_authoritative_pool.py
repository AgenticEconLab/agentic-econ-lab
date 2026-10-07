# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The number-consistency pool formerly accepted numbers from arbitrary
strings - a fabricated 73.42 in LLM-written literature prose verified "The estimated effect
is 73.42" - and every integer up to 12 was exempt, so "Our effect is 9 percent" was
consistent with no artifacts at all. The pool is now built from authoritative values only,
and small integers are exempt only in enumeration/ordinal/count positions by syntax."""

from ReportingTeam.ael.report_harness import artifact_number_pool, check_consistency


def test_number_in_literature_prose_does_not_verify():
    arts = {"literature_review": {"synthesis": "Prior work finds an effect of 73.42."}}
    r = check_consistency("The estimated effect is 73.42.", arts)
    assert r.verdict == "has_unverified" and r.unverified[0].value == 73.42


def test_number_in_spec_rationale_does_not_verify():
    arts = {"estimation_results": {"spec": {"rationale": "expect a coefficient near 0.537"}}}
    assert check_consistency("The coefficient is 0.537.", arts).verdict == "has_unverified"


def test_small_integer_percent_is_never_exempt():
    r = check_consistency("Our effect is 9 percent.", {})
    assert r.verdict == "has_unverified" and r.skipped_small_ints == 0
    assert check_consistency("Our effect is 9%.", {}).verdict == "has_unverified"
    assert check_consistency("Our effect is 9.", {}).verdict == "has_unverified"


def test_authoritative_values_verify():
    arts = {"estimation_results": {"coef": 0.128473, "n_obs": 61, "share": "0.42",
                                   "sample_start": "1960-01-01", "pct": "12.5%"},
            "derived": {"code_generation.refused_total": {"value": 17, "derivation": "sum"}}}
    pool = artifact_number_pool(arts.values())
    assert {0.128473, 61.0, 0.42, 1960.0, 12.5, 17.0} <= pool
    text = ("The coefficient is 0.13 on 61 observations from 1960 onwards; the share is "
            "0.42 and 17 equations were refused; 12.5% of the sample.")
    r = check_consistency(text, arts)
    assert r.verdict == "consistent", [u.context for u in r.unverified]


def test_enumeration_contexts_are_exempt():
    # counts of findings ("2 of 4 hypotheses supported") are claims and need provenance;
    # structural counts and list/section markers stay exempt
    text = ("## 3. Results\n"
            "1. first item\n"
            "| 2 | proposal | approved |\n"
            "See Section 4 and round 2; we fit 3 models; "
            "an AR(1) process; the 95% CI is reported.")
    r = check_consistency(text, {})
    assert r.verdict == "consistent", [u.context for u in r.unverified]
    assert r.skipped_small_ints == r.total_numbers


def test_confidence_level_needs_ci_context():
    assert check_consistency("A 95% CI excludes zero.", {}).verdict == "consistent"
    assert check_consistency("Output rose 95% over the decade.", {}).verdict == "has_unverified"


def test_urls_and_source_titles_are_identifiers():
    arts = {"literature_batch": {"literature_items": [
        {"title": "Herding during the Covid-19 pandemic", "doi": "10.1007/s11403-019-00268-z"}]},
            "data_source": {"retrieved_data": [{"series_id": "SP500", "source_title": "S&P 500"}]}}
    text = ("- Smith (2021). Herding during the Covid-19 pandemic. "
            "https://doi.org/10.1007/s11403-019-00268-z\n| `SP500` (S&P 500) | FRED |")
    r = check_consistency(text, arts)
    assert r.verdict == "has_unverified"            # 2021 has no provenance here
    assert [u.value for u in r.unverified] == [2021.0]
    # an LLM-written title is not a source title
    arts2 = {"model_specification": {"model_title": "A 7.25 percent wedge model"}}
    assert check_consistency("The wedge is 7.25.", arts2).verdict == "has_unverified"


def test_round2_generated_titles_and_claim_counts_need_provenance():
    from ReportingTeam.ael.report_harness.consistency import check_consistency
    lit = {"literature_review": {"title": "A review with effect 73.42"}}
    assert check_consistency("The estimated effect is 73.42.", [lit]).verdict != "consistent"
    for txt in ("The coefficient is 9 million dollars.", "We find 9 significant effects.",
                "The estimate (9) is significant."):
        assert check_consistency(txt, []).verdict != "consistent", txt
    for txt in ("The run produced 3 models.", "An AR(1) term is included.", "(1) First item"):
        assert check_consistency(txt, []).verdict == "consistent", txt
    batch = {"literature_items": [{"title": "Covid-19 and the S&P 500"}]}
    assert check_consistency("Covid-19 and the S&P 500 matter.", [batch]).verdict == "consistent"


def test_round3_template_labels_and_robustness_counts_verify():
    from ReportingTeam.ael.report_harness.consistency import check_consistency
    from ReportingTeam.ael.report_harness.assemble import derived_numbers
    assert check_consistency("verified by the Stage-3 number-consistency check.", []).verdict == "consistent"
    est = {"inference": {"robustness": [{"variant_estimate": 0.1}, {"variant_estimate": 0.2},
                                        {"variant_estimate": None}]}}
    d = derived_numbers({}, est)
    pool = {"derived": {k: v["value"] for k, v in d.items()}}
    txt = "Robustness: 2 estimable variant(s); 1 variant(s) could not be estimated."
    assert check_consistency(txt, [pool]).verdict == "consistent"


def test_v072_checker_misses():
    from ReportingTeam.ael.report_harness.consistency import check_consistency
    est = {"outcome": {"coefficients": [{"estimate": -0.0747}, {"estimate": -0.0567},
                                         {"estimate": 122.2832}]}}
    for txt in ("the real interest rate (−0.075) is small.",
                "the interaction term (coefficient –0.057, p = 0.242)",
                "the Output Gap Proxy shows an estimate of 122."):
        r = check_consistency(txt, [est, {"p": 0.242}])
        assert r.verdict == "consistent", (txt, [u.value for u in r.unverified])
    spec = {"parameters": [{"typical_range": "[1.2, 1.8]"}]}
    assert check_consistency("stopped at a bound of [1.2, 1.8]", [spec]).verdict == "consistent"
    assert check_consistency("covering 1960–2024", [{"y": [1960, 2024]}]).verdict == "consistent"
    # 122 must still not verify a value that rounds elsewhere
    assert check_consistency("an estimate of 122.", [{"x": 121.4}]).verdict != "consistent"


def test_v073_review_regressions():
    from ReportingTeam.ael.report_harness.consistency import check_consistency
    assert check_consistency("The estimate is 234.", [{"n": "1,234"}]).verdict != "consistent"
    assert check_consistency("We use 1,234 observations.", [{"n": "1,234"}]).verdict == "consistent"
    assert check_consistency("The estimate is 73.42.", [{"rationale": "[73.42, 81.23]"}]).verdict != "consistent"
    assert check_consistency("bounds [1.2, 1.8]", [{"typical_range": "[1.2, 1.8]"}]).verdict == "consistent"
    for txt in ("covering 1960 –2024", "covering 1960 – 2024", "covering 1960–2024"):
        assert check_consistency(txt, [{"y": [1960, 2024]}]).verdict == "consistent", txt
    assert check_consistency("a rate of (−0.5)", [{"x": -0.5}]).verdict == "consistent"


def test_v073_dash_after_word_is_a_sign_after_number_a_range():
    from ReportingTeam.ael.report_harness.consistency import check_consistency
    assert check_consistency("coefficient –0.057", [{"x": -0.0567}]).verdict == "consistent"
    assert check_consistency("model A–5 is used", [{"x": 5}]).verdict == "consistent"


def test_v073_followup_range_field_scalar_and_true_minus():
    from ReportingTeam.ael.report_harness.consistency import check_consistency, _extract
    assert check_consistency("n = 1,234", [{"range": "1,234"}]).verdict == "consistent"
    assert check_consistency("value 234", [{"range": "1,234"}]).verdict != "consistent"
    assert [x[0] for x in _extract("Estimates: 0.12 −0.057")] == [0.12, -0.057]
    assert [x[0] for x in _extract("covering 1960 –2024")] == [1960.0, 2024.0]


def test_negative_small_int_is_never_exempt():
    from ReportingTeam.ael.report_harness.consistency import check_consistency
    assert check_consistency("We fit −3 models.", []).verdict != "consistent"
