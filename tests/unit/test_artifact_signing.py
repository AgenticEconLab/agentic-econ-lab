# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Artifacts handed between teams are HMAC-signed
when a team registers them and verified when the next team reads them. A modified artifact
fails the reading team instead of flowing downstream.
"""

import json
import os

import pytest

from pipeline.artifact_store import ArtifactIntegrityError, ArtifactStore
from pipeline.pipeline_config import PipelineConfig, StageConfig
from pipeline.research_pipeline import (
    ResearchPipelineOrchestrator,
    _TEAM_RUNNERS,
    register_team_runner,
)


@pytest.fixture(autouse=True)
def clean_runners():
    _TEAM_RUNNERS.clear()
    yield
    _TEAM_RUNNERS.clear()


def test_registered_artifact_is_signed_and_verified():
    store = ArtifactStore(signing_key="k")
    store.register("research_questions", {"questions": ["q1"]}, producer="IdeationTeam")
    art = store.get_artifact("research_questions")
    assert len(art.signature) == 64 and art.signed_at
    assert store.get("research_questions") == {"questions": ["q1"]}
    assert store.verified_reads == 1 and store.integrity_failures == []


def test_caller_mutations_do_not_reach_the_stored_artifact():
    """Deep copies on register and get: a producer or consumer editing its own dict in place
    neither changes the stored artifact nor trips the signature check."""
    store = ArtifactStore(signing_key="k")
    data = {"questions": ["q1"]}
    store.register("research_questions", data, producer="IdeationTeam")
    data["questions"].append("added by producer after hand-off")
    read = store.get("research_questions")
    read["questions"].append("added by consumer")
    assert store.get("research_questions") == {"questions": ["q1"]}
    assert store.integrity_failures == []


def test_modified_artifact_is_rejected():
    store = ArtifactStore(signing_key="k")
    store.register("research_questions", {"questions": ["q1"]}, producer="IdeationTeam")
    store.get_artifact("research_questions").data["questions"] = ["injected"]
    with pytest.raises(ArtifactIntegrityError):
        store.get("research_questions")
    assert store.integrity_failures == ["research_questions"]


def test_artifact_from_another_key_is_rejected(tmp_path):
    ArtifactStore(base_dir=str(tmp_path), signing_key="run-a").register(
        "research_questions", {"questions": ["q1"]}, producer="IdeationTeam")
    other = ArtifactStore(base_dir=str(tmp_path), signing_key="run-b")
    other.load_from_disk("research_questions")
    with pytest.raises(ArtifactIntegrityError):
        other.get("research_questions")

    same = ArtifactStore(base_dir=str(tmp_path), signing_key="run-a")
    same.load_from_disk("research_questions")
    assert same.get("research_questions") == {"questions": ["q1"]}


def test_signing_key_from_environment(monkeypatch):
    monkeypatch.setenv("AEL_PIPELINE_SIGNING_KEY", "shared")
    a = ArtifactStore()
    a.register("x", {"v": 1}, producer="T")
    monkeypatch.delenv("AEL_PIPELINE_SIGNING_KEY")
    b = ArtifactStore(signing_key="shared")
    b._artifacts["x"] = a.get_artifact("x")
    assert b.get("x") == {"v": 1}


def _config(tmp_path):
    return PipelineConfig(
        name="signing",
        stages=[
            StageConfig(team="IdeationTeam", mode="ModeNoWcNoHITL",
                        inputs=["research_topic"], outputs=["research_questions"]),
            StageConfig(team="LiteratureTeam", mode="ModeNoWcNoHITL",
                        inputs=["research_questions"], outputs=["literature_review"],
                        depends_on=["IdeationTeam"]),
        ],
        output_dir=str(tmp_path),
    )


def _ideation(upstream_artifacts, mode, output_dir, collector=None):
    return {"research_questions": {"questions": ["How does AI affect GDP?"]}}


def _literature(upstream_artifacts, mode, output_dir, collector=None):
    return {"literature_review": {"source": upstream_artifacts.get("research_questions")}}


def test_pipeline_records_verified_reads_in_manifest(tmp_path):
    register_team_runner("IdeationTeam", _ideation)
    register_team_runner("LiteratureTeam", _literature)
    result = ResearchPipelineOrchestrator(_config(tmp_path)).run("AI")
    assert result.success
    with open(os.path.join(str(tmp_path), "pipeline_manifest.json")) as f:
        sigs = json.load(f)["artifact_signatures"]
    assert sigs["verified_reads"] >= 2 and sigs["integrity_failures"] == []


def test_tampered_hand_off_fails_the_reading_team(tmp_path):
    register_team_runner("IdeationTeam", _ideation)
    literature_ran = []
    register_team_runner("LiteratureTeam",
                         lambda *a, **k: literature_ran.append(1) or _literature(*a, **k))
    pipeline = ResearchPipelineOrchestrator(_config(tmp_path))

    original_register = pipeline.artifact_store.register

    def register_then_tamper(name, data, *args, **kwargs):
        art = original_register(name, data, *args, **kwargs)
        if name == "research_questions":
            pipeline.artifact_store.get_artifact(name).data["questions"] = ["injected"]
        return art

    pipeline.artifact_store.register = register_then_tamper
    result = pipeline.run("AI")

    assert not result.success
    assert "LiteratureTeam" in result.teams_failed
    assert literature_ran == []
    with open(os.path.join(str(tmp_path), "pipeline_manifest.json")) as f:
        assert json.load(f)["artifact_signatures"]["integrity_failures"] == ["research_questions"]
