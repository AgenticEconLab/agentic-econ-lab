# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""DataCleaningStage / QualityAssuranceStage fixes found by auditing the v0.7.1 runs.

- previews were zipped by POSITION (the demo's "2026-04-01" row carried 2021 World Bank
  GDP); num_observations was the max over series; an LLM scored quality from metadata
  (EMVMACROBUS previewed 0,0,0 scored 78); "validated" meant only that a script ran.
- checkpoints 4 and 5 voted on a blank prompt with no context; objections were ignored."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_DIR = Path(__file__).resolve().parent.parent.parent / "DataTeam" / "ael" / "ModeOpenSourceAPI"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s2():
    return _load("cleaning_stage_v071", "2-DataCleaningStage.py")


@pytest.fixture(scope="module")
def s3():
    return _load("qa_stage_v071", "3-QualityAssuranceStage.py")


def _monthly(start, end_year, end_month, f=lambda i: 1.0 + i * 0.1):
    out, i, y, m = [], 0, start, 1
    while (y, m) <= (end_year, end_month):
        out.append({"date": f"{y}-{m:02d}-01", "value": f(i)})
        i, m = i + 1, m + 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _annual(y0, y1, f=lambda y: 1000.0 + y):
    return [{"date": str(y), "value": f(y)} for y in range(y0, y1 + 1)]


def _rd(sid, obs, source="FRED", simulated=False, freq="monthly"):
    return {"series_id": sid, "series_name": sid, "source_name": source,
            "retrieval_date": "2026-10-01", "num_observations": len(obs),
            "start_date": obs[0]["date"], "end_date": obs[-1]["date"], "frequency": freq,
            "data_preview": [{"date": o["date"], sid: o["value"]} for o in obs[-5:]],
            "quality_notes": "", "retrieval_status": "Simulated" if simulated else "Success",
            "data_simulated": simulated, "title_check": "match"}


def _source(tmp_path, full):
    obs_file = tmp_path / "api_source_output_observations.json"
    obs_file.write_text(json.dumps({sid: obs for sid, (_, obs) in full.items()}))
    return {"research_question": "RQ",
            "retrieved_data": [_rd(sid, obs, source=src) for sid, (src, obs) in full.items()],
            "selected_series": [], "metadata": {"observations_file": str(obs_file)}}


def _cleaner(s2):
    orch = s2.DataCleaningOrchestrator.__new__(s2.DataCleaningOrchestrator)
    va = s2.DataValidationAgent.__new__(s2.DataValidationAgent)
    va.agent_name, va.narrate = "DataValidationAgent", False
    orch.validation_agent = va
    orch.alignment_agent = s2.TemporalAlignmentAgent("sk-test")
    orch.integration_agent = s2.MultiSourceIntegrationAgent("sk-test")
    orch.transformation_agent = SimpleNamespace(transform_data=lambda d, q: d)
    orch.cleaning_output, orch.merged = None, None
    return orch


# --------------------------------------------------------------------------------------------
# alignment by date, real counts, deterministic quality
# --------------------------------------------------------------------------------------------

def test_merge_aligns_by_date_not_position(s2, tmp_path):
    fed = _monthly(2015, 2026, 4)                 # monthly, ends 2026-04
    gdp = _annual(2000, 2021)                     # annual, ends 2021
    src = _source(tmp_path, {"FEDFUNDS": ("FRED", fed), "NY.GDP.MKTP.CD": ("World Bank", gdp)})
    src["retrieved_data"][1]["frequency"] = "annual"
    out = _cleaner(s2).run_cleaning_pipeline(src, enable_hitl=False)
    ds = out.integrated_datasets[0]
    assert ds.frequency == "annual"
    rows = {r["date"]: r for r in ds.data_preview}
    # the last complete row is 2021 — a 2026 date can never carry the 2021 GDP value
    assert max(rows) == "2021-01-01" and rows["2021-01-01"]["NY.GDP.MKTP.CD"] == 1000.0 + 2021
    # FRED 2021 annual value = mean of the 12 monthly values of 2021 (documented rule)
    months_2021 = [o["value"] for o in fed if o["date"].startswith("2021")]
    assert rows["2021-01-01"]["FEDFUNDS"] == pytest.approx(sum(months_2021) / 12)
    assert "PERIOD MEAN" in ds.alignment["aggregation"]


def test_partial_trailing_period_is_not_averaged(s2, tmp_path):
    # Jan-Apr 2026 is not "the 2026 value" of a monthly series aligned to annual
    src = _source(tmp_path, {"FEDFUNDS": ("FRED", _monthly(2015, 2026, 4)),
                             "NY.GDP.MKTP.CD": ("World Bank", _annual(2000, 2026))})
    out = _cleaner(s2).run_cleaning_pipeline(src, enable_hitl=False)
    al = out.aligned_datasets
    fed = next(a for a in al if a.series_id == "FEDFUNDS")
    assert fed.end_date == "2025-01-01" and "partial periods" in fed.conversion_method


def test_num_observations_is_complete_cases_not_max(s2, tmp_path):
    src = _source(tmp_path, {"A": ("FRED", _monthly(2000, 2020, 12)),        # 252 obs
                             "B": ("FRED", _monthly(2015, 2020, 12))})       # 72 obs
    out = _cleaner(s2).run_cleaning_pipeline(src, enable_hitl=False)
    ds = out.integrated_datasets[0]
    assert ds.num_observations == 72 and ds.alignment["outer_rows"] == 252
    assert ds.time_period == "2015-01-01 to 2020-12-01"


def test_constant_series_fails_deterministically(s2, tmp_path):
    # a v0.7.1 run: EMVMACROBUS previewed 0,0,0 and an LLM scored it 78
    src = _source(tmp_path, {"EMVMACROBUS": ("FRED", _monthly(2000, 2020, 12, f=lambda i: 0.0)),
                             "UNRATE": ("FRED", _monthly(2000, 2020, 12))})
    out = _cleaner(s2).run_cleaning_pipeline(src, enable_hitl=False)
    qa = {a.series_id: a for a in out.quality_assessments}
    assert qa["EMVMACROBUS"].verdict == "fail" and qa["EMVMACROBUS"].overall_score <= 40
    assert any("constant" in i for i in qa["EMVMACROBUS"].actionable_issues)
    assert qa["UNRATE"].verdict == "pass"
    ds = out.integrated_datasets[0]
    assert "EMVMACROBUS" in ds.alignment["excluded"] and "EMVMACROBUS" not in ds.variables
    assert ds.alignment["dataset_verdict"] == "fail"


def test_frozen_tail_is_reported(s2):
    # a v0.7.1 run: EMVMACROBUS — real series, but its last months are all 0
    obs = _monthly(2000, 2020, 12, f=lambda i: 0.0 if i >= 240 else 1.0 + (i % 5))
    va = s2.DataValidationAgent.__new__(s2.DataValidationAgent)
    va.narrate = False
    a = va._assess_series_quality(s2.RetrievedData(**_rd("EMV", obs)), obs)
    assert a.verdict == "pass"
    assert any("last 12 observations identical" in i for i in a.actionable_issues)
    assert a.overall_score < 100


@pytest.mark.parametrize("obs,issue", [
    (_monthly(2000, 2000, 10), "only 10 observation"),
    ([{"date": "2000-01-01", "value": 1.0}] * 2 + _monthly(2001, 2005, 12), "duplicate date"),
    ([{"date": f"20{i:02d}", "value": None} for i in range(30)], "no numeric observations"),
])
def test_blocking_checks(s2, obs, issue):
    va = s2.DataValidationAgent.__new__(s2.DataValidationAgent)
    va.narrate = False
    data = s2.RetrievedData(**_rd("X", obs if obs[0]["value"] is not None else
                                  [{"date": "2000", "value": 0.0}]))
    a = va._assess_series_quality(data, obs)
    assert a.verdict == "fail" and any(issue in i for i in a.actionable_issues)


def test_simulated_series_scores_zero_and_is_excluded(s2, tmp_path):
    src = _source(tmp_path, {"UNRATE": ("FRED", _monthly(2000, 2020, 12))})
    sim = _rd("FAKE", _monthly(2019, 2023, 12), simulated=True)
    src["retrieved_data"].append(sim)
    out = _cleaner(s2).run_cleaning_pipeline(src, enable_hitl=False)
    qa = {a.series_id: a for a in out.quality_assessments}
    assert qa["FAKE"].overall_score == 0.0 and qa["FAKE"].verdict == "fail"
    assert "FAKE" in out.integrated_datasets[0].alignment["excluded"]


def test_preview_only_fallback_is_stated(s2):
    src = {"research_question": "RQ",
           "retrieved_data": [_rd("UNRATE", _monthly(2000, 2020, 12))], "metadata": {}}
    out = _cleaner(s2).run_cleaning_pipeline(src, enable_hitl=False)
    al = out.integrated_datasets[0].alignment
    assert al["preview_only"] == ["UNRATE"] and al["dataset_verdict"] == "fail"


def test_merged_csv_written(s2, tmp_path):
    src = _source(tmp_path, {"UNRATE": ("FRED", _monthly(2000, 2020, 12))})
    orch = _cleaner(s2)
    orch.run_cleaning_pipeline(src, enable_hitl=False)
    target = tmp_path / "api_cleaning_output.json"
    orch.save_cleaning_output(str(target))
    saved = json.loads(target.read_text())
    csv = Path(saved["integrated_datasets"][0]["alignment"]["merged_file"])
    assert csv.exists() and len(csv.read_text().splitlines()) == 1 + 252


# --------------------------------------------------------------------------------------------
# QA validation fails on the deterministic checks
# --------------------------------------------------------------------------------------------

def _qa(s3, monkeypatch):
    orch = s3.QualityAssuranceOrchestrator.__new__(s3.QualityAssuranceOrchestrator)
    orch.documentation_agent = SimpleNamespace(llm=SimpleNamespace(
        invoke=lambda msgs: "import json\nprint(json.dumps({'validated': True}))"))

    class _Sandbox:
        def __init__(self, **k):
            pass

        @staticmethod
        def clean_llm_output(x):
            return x

        def execute(self, code):
            return SimpleNamespace(success=True, stdout="{}", stderr="", error=None,
                                   execution_time_sec=0.1)
    monkeypatch.setattr(s3, "CodeSandbox", _Sandbox)
    return orch


def _dataset(s3, verdict, issues=()):
    return s3.IntegratedDataset(
        dataset_id="D", dataset_name="D", source_apis=["FRED"], num_variables=2,
        num_observations=100, time_period="", frequency="monthly", merge_strategy="",
        variables=["date", "X"], data_preview=[{"date": "2020-01-01", "X": 1.0}], provenance={},
        alignment={"dataset_verdict": verdict, "dataset_issues": list(issues)})


def test_sandbox_validation_fails_on_failed_checks(s3, monkeypatch):
    orch = _qa(s3, monkeypatch)
    bad = orch._validate_data_with_sandbox(_dataset(s3, "fail", ["X: constant series"]), [])
    assert bad["sandbox_executed"] and not bad["validated"]
    assert bad["deterministic_issues"] == ["X: constant series"]
    good = orch._validate_data_with_sandbox(_dataset(s3, "pass"), [])
    assert good["validated"]


def test_failed_checks_cap_certification(s3):
    agent = s3.FinalValidationAgent.__new__(s3.FinalValidationAgent)
    agent.agent_name = "FinalValidationAgent"
    score, cert = agent.validate_dataset(_dataset(s3, "fail", ["x"]),
                                         [SimpleNamespace(overall_score=95.0)])
    assert cert == "Bronze" and score <= 60


# --------------------------------------------------------------------------------------------
# checkpoints 4 and 5
# --------------------------------------------------------------------------------------------

def test_alignment_checkpoint_asks_real_question_and_records_objection(s2, tmp_path, monkeypatch):
    import DataTeam.ael.hitl as hitl
    asked = []

    def fake_input(prompt, default="", context=None, **k):
        asked.append((prompt, context))
        return "no - annual aggregation discards the monthly policy dynamics"
    monkeypatch.setattr(hitl, "auto_input", fake_input)
    src = _source(tmp_path, {"UNRATE": ("FRED", _monthly(2000, 2020, 12))})
    out = _cleaner(s2).run_cleaning_pipeline(src, enable_hitl=True)
    prompt, ctx = asked[0]
    assert "Alignment Review" in prompt and "UNRATE" in prompt and "monthly" in prompt
    assert ctx["alignment"]["complete_case_rows"] == 252
    assert out.metadata["hitl_objections"][0]["checkpoint"] == "Alignment Review"
    assert "did not approve" in out.metadata["limitations"][0]


def test_final_approval_checkpoint_carries_the_package(s3, monkeypatch):
    import DataTeam.ael.hitl as hitl
    asked = []
    monkeypatch.setattr(hitl, "auto_input",
                        lambda p, default="", context=None, **k: asked.append((p, context)) or "no")
    ds = _dataset(s3, "fail", ["X: constant series"])
    codebook = s3.DataCodebook(title="t", version="1", created_date="d", variables=[],
                               summary_statistics={}, data_sources=[], methodology="",
                               limitations=[], usage_notes="")
    report = s3.DataReport(title="t", research_question="RQ", executive_summary="",
                           api_sources_section="", series_retrieval_section="",
                           quality_assessment_section="", temporal_alignment_section="",
                           integration_section="", transformations_section="", citations=[])
    doc = s3.DocumentedDataset(
        dataset_id="D", dataset_name="D", integrated_dataset=ds, codebook=codebook,
        report=report, quality_score=50.0, certification_level="Bronze", citation="",
        documentation={"sandbox_validation": {"deterministic_verdict": "fail",
                                              "deterministic_issues": ["X: constant series"]}})
    res = s3.checkpoint5_final_approval(doc, "Does X move Y?")
    prompt, ctx = asked[0]
    assert "Does X move Y?" in prompt and "constant series" in prompt
    assert ctx["deterministic_verdict"] == "fail" and not res.approved
    assert "did not approve" in res.limitation()


def test_hitl_answer_classification():
    from DataTeam.ael.hitl import is_approval
    assert is_approval("approved") and is_approval("") and is_approval("Yes.")
    assert not is_approval("no") and not is_approval("retry")
    assert not is_approval("approved but the sample is too short")
