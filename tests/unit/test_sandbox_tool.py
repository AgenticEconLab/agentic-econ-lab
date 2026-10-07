# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for CodeSandbox — secure code execution environment."""

import pytest
from shared.tools.sandbox_tool import (
    CodeSandbox,
    ExecutionResult,
    SafetyCheck,
    ALLOWED_PACKAGES,
    BLOCKED_MODULES,
    BLOCKED_BUILTINS,
)


# ---------------------------------------------------------------------------
# Safety check tests
# ---------------------------------------------------------------------------

class TestSafetyCheck:
    def test_safe_code_passes(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("import numpy as np\nprint(np.mean([1,2,3]))")
        assert result.safe is True
        assert result.violations == []

    def test_blocked_module_os(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("import os\nos.system('ls')")
        assert result.safe is False
        assert any("os" in v for v in result.violations)

    def test_blocked_module_subprocess(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("import subprocess")
        assert result.safe is False

    def test_blocked_module_socket(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("import socket")
        assert result.safe is False

    def test_blocked_from_import(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("from os.path import join")
        assert result.safe is False
        assert any("os" in v for v in result.violations)

    def test_blocked_builtin_exec(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("exec('print(1)')")
        assert result.safe is False
        assert any("exec" in v for v in result.violations)

    def test_blocked_builtin_eval(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("x = eval('1+1')")
        assert result.safe is False

    def test_blocked_builtin_open(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("f = open('/etc/passwd')")
        assert result.safe is False

    def test_blocked_dunder_import(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("__import__('os')")
        assert result.safe is False

    def test_syntax_error_caught(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("def foo(:\n  pass")
        assert result.safe is False
        assert "Syntax error" in result.reason

    def test_allowed_packages_accepted(self):
        sandbox = CodeSandbox()
        code = "import numpy\nimport pandas\nimport scipy\nimport math\nimport json"
        result = sandbox.check_safety(code)
        assert result.safe is True

    def test_disallowed_unknown_package(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("import some_unknown_pkg")
        assert result.safe is False
        assert any("Disallowed" in v for v in result.violations)

    def test_blocked_attribute_system(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("something.system('cmd')")
        assert result.safe is False

    def test_multiple_violations_counted(self):
        sandbox = CodeSandbox()
        result = sandbox.check_safety("import os\nimport sys\nexec('x')")
        assert result.safe is False
        assert len(result.violations) == 3

    def test_custom_allowed_packages(self):
        sandbox = CodeSandbox(allowed_packages={"numpy", "custom_pkg"})
        result = sandbox.check_safety("import custom_pkg")
        assert result.safe is True
        result2 = sandbox.check_safety("import pandas")
        assert result2.safe is False


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------

class TestCodeExecution:
    def test_simple_print(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("print('hello world')")
        assert result.success is True
        assert "hello world" in result.stdout

    def test_math_computation(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("print(2 + 3)")
        assert result.success is True
        assert "5" in result.stdout

    def test_numpy_computation(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("import numpy as np\nprint(np.mean([1,2,3,4,5]))")
        assert result.success is True
        assert "3.0" in result.stdout

    def test_unsafe_code_blocked(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("import os\nos.system('echo hacked')")
        assert result.success is False
        assert "Unsafe code" in result.error

    def test_runtime_error_caught(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("x = 1 / 0")
        assert result.success is False
        assert result.return_code != 0

    def test_timeout_enforcement(self):
        sandbox = CodeSandbox(timeout_sec=2)
        # Use a busy loop instead of time.sleep (time is not in allowed packages)
        result = sandbox.execute("while True: pass")
        assert result.success is False
        assert "timed out" in result.error

    def test_execution_time_tracked(self):
        sandbox = CodeSandbox()
        result = sandbox.execute("print('fast')")
        assert result.execution_time_sec > 0
        assert result.execution_time_sec < 30

    def test_result_to_dict(self):
        result = ExecutionResult(
            success=True, stdout="ok", execution_time_sec=1.234, return_code=0
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["stdout"] == "ok"
        assert d["execution_time_sec"] == 1.234


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestSandboxConstants:
    def test_numpy_in_allowed(self):
        assert "numpy" in ALLOWED_PACKAGES

    def test_pandas_in_allowed(self):
        assert "pandas" in ALLOWED_PACKAGES

    def test_os_in_blocked(self):
        assert "os" in BLOCKED_MODULES

    def test_subprocess_in_blocked(self):
        assert "subprocess" in BLOCKED_MODULES

    def test_exec_in_blocked_builtins(self):
        assert "exec" in BLOCKED_BUILTINS

    def test_open_in_blocked_builtins(self):
        assert "open" in BLOCKED_BUILTINS
