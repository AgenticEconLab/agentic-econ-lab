# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""--multirun-dir on a directory without run data exits with code 1."""
import sys

import pytest

from evaluation import run_ael_evaluation as rae


def test_multirun_without_run_data_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_ael_evaluation", "--multirun-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        rae.main()
    assert exc.value.code == 1
