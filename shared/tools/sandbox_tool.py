# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Code Sandbox — Secure execution environment for agent-generated code.

Runs Python code in a subprocess with safety checks.
Used by ModelTeam (calibration validation) and DataTeam (data transforms).

Security:
- CPU time limit: 60s default (enforced via subprocess timeout)
- Memory limit: 512 MB default (NOT enforced on Windows — see _run_subprocess)
- Minimal environment: only PATH/TEMP/SYSTEMROOT passed to subprocess (no API keys)
- Whitelist of allowed packages
- Static AST safety checks before execution

Usage:
    from shared.tools.sandbox_tool import CodeSandbox, ExecutionResult

    sandbox = CodeSandbox()
    result = sandbox.execute("import numpy as np; print(np.mean([1,2,3]))")
    if result.success:
        print(result.stdout)
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class SafetyCheck:
    """Result of a code safety analysis."""

    safe: bool
    reason: str = ""
    violations: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    execution_time_sec: float = 0.0
    return_code: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "execution_time_sec": round(self.execution_time_sec, 3),
            "return_code": self.return_code,
        }


# Packages that are safe for economics computation
ALLOWED_PACKAGES: Set[str] = {
    "numpy", "np",
    "scipy",
    "pandas", "pd",
    "statsmodels", "sm",
    "matplotlib", "plt",
    "sklearn",
    "sympy",
    "math",
    "statistics",
    "decimal",
    "fractions",
    "json",
    "csv",
    "collections",
    "itertools",
    "functools",
    "operator",
    "dataclasses",
    "typing",
    "re",
    "datetime",
    "copy",
    "io",
    "textwrap",
    "pprint",
}

# Dangerous modules/builtins that should never be imported
BLOCKED_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "ctypes", "multiprocessing", "threading",
    "importlib", "code", "compile", "compileall",
    "signal", "resource", "pty", "fcntl",
    "pickle", "shelve", "marshal",
    "builtins", "__builtin__",
    "webbrowser", "antigravity",
}

# Dangerous builtin functions
BLOCKED_BUILTINS: Set[str] = {
    "exec", "eval", "compile", "__import__",
    "open", "input", "breakpoint",
    "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    "exit", "quit",
}


class CodeSandbox:
    """
    Secure code execution environment for agent-generated Python code.

    Executes code in a subprocess with resource limits. Performs static
    analysis before execution to catch dangerous patterns.
    """

    def __init__(
        self,
        timeout_sec: int = 60,
        memory_mb: int = 512,
        allowed_packages: Optional[Set[str]] = None,
    ):
        """
        Args:
            timeout_sec: Maximum execution time in seconds.
            memory_mb: Maximum memory in MB.
            allowed_packages: Override the default allowed package set.
        """
        self.timeout_sec = timeout_sec
        self.memory_mb = memory_mb
        self.allowed_packages = allowed_packages or ALLOWED_PACKAGES

    def execute(
        self,
        code: str,
        timeout_sec: Optional[int] = None,
        memory_mb: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute Python code in a sandboxed subprocess.

        Args:
            code: Python source code to execute.
            timeout_sec: Override timeout for this execution.
            memory_mb: Override memory limit for this execution.

        Returns:
            ExecutionResult with stdout, stderr, and success status.
        """
        timeout = timeout_sec or self.timeout_sec
        mem = memory_mb or self.memory_mb

        # Static safety check
        safety = self.check_safety(code)
        if not safety.safe:
            return ExecutionResult(
                success=False,
                error=f"Unsafe code: {safety.reason}",
                stderr="\n".join(safety.violations),
            )

        # Write code to temp file
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name
        except OSError as e:
            return ExecutionResult(success=False, error=f"Failed to write temp file: {e}")

        try:
            return self._run_subprocess(temp_path, timeout, mem)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def check_safety(self, code: str) -> SafetyCheck:
        """
        Static analysis of code safety.

        Checks:
        1. Valid Python syntax
        2. No blocked module imports
        3. No blocked builtin calls
        4. No dangerous patterns (file I/O, network, exec)
        """
        violations: List[str] = []

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return SafetyCheck(safe=False, reason=f"Syntax error: {e}")

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in BLOCKED_MODULES:
                        violations.append(f"Blocked import: {alias.name}")
                    elif module not in self.allowed_packages:
                        violations.append(f"Disallowed package: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in BLOCKED_MODULES:
                        violations.append(f"Blocked import: from {node.module}")
                    elif module not in self.allowed_packages:
                        violations.append(f"Disallowed package: from {node.module}")

            # Check dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_BUILTINS:
                        violations.append(f"Blocked builtin: {node.func.id}()")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "exec", "spawn"):
                        violations.append(f"Blocked call: .{node.func.attr}()")

        if violations:
            return SafetyCheck(
                safe=False,
                reason=f"{len(violations)} safety violation(s)",
                violations=violations,
            )

        return SafetyCheck(safe=True)

    @staticmethod
    def clean_llm_output(code: str) -> str:
        """Strip markdown code fences and language tags from LLM output.

        Handles patterns like:
            ```python\\n...code...\\n```
            ```\\n...code...\\n```
        """
        code = code.strip()
        code = re.sub(r"^```[a-zA-Z]*\n", "", code)
        code = re.sub(r"\n```$", "", code)
        return code.strip()

    def _run_subprocess(
        self, script_path: str, timeout: int, memory_mb: int
    ) -> ExecutionResult:
        """Execute the script in a subprocess.

        Note: ``memory_mb`` is not enforced on Windows (no RLIMIT_AS).
        On Linux it could be enforced via ``resource.setrlimit`` in a
        ``preexec_fn``, but that is not yet implemented.  The timeout
        IS enforced via ``subprocess.run(timeout=...)``.
        """
        start = time.time()

        # Build command — use same Python interpreter
        cmd = [sys.executable, script_path]

        # Minimal environment — prevent API key leakage. Keep only PATH-like and
        # loader vars (no secrets): LD_LIBRARY_PATH is required so a venv interpreter
        # linked against a relocatable libpython (e.g. EasyBuild/HPC Python modules)
        # can load its own shared library; without it the subprocess exits 127 with
        # "libpython3.x.so: cannot open shared object file".
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "TEMP": os.environ.get("TEMP", tempfile.gettempdir()),
            "TMP": os.environ.get("TMP", tempfile.gettempdir()),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "USERPROFILE": os.environ.get("USERPROFILE", ""),
        }

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=tempfile.gettempdir(),
            )
            elapsed = time.time() - start

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout[:50000],  # Truncate large outputs
                stderr=proc.stderr[:10000],
                execution_time_sec=elapsed,
                return_code=proc.returncode,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                execution_time_sec=elapsed,
                return_code=-1,
            )
        except OSError as e:
            elapsed = time.time() - start
            return ExecutionResult(
                success=False,
                error=f"Subprocess error: {e}",
                execution_time_sec=elapsed,
                return_code=-1,
            )
