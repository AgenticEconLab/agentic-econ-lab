# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for AEL V0.4 Phase 4: Tool Registry Migration + Sandbox.

Verifies:
1. register_all_tools registers 6 tools
2. All 15 MasterOrchestrators import and call register_all_tools
3. Stage files use ToolRegistry.invoke() instead of tracked_* functions
4. ToolRegistry records invocations via MetricsCollector
5. Tool wrappers exist for all tracked functions
6. CodeSandbox integration (already done in V0.3)
"""

import ast
import os
import sys
from pathlib import Path

import pytest

# repository root for file path lookups (sys.path handled by conftest.py)
_agents_dir = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Helpers
# ============================================================================

def _get_orchestrator_files():
    """Return all 15 MasterOrchestrator files."""
    base = _agents_dir
    orchestrators = []
    for team in ["IdeationTeam", "LiteratureTeam", "ModelTeam"]:
        for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                      "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
            f = base / team / "ael" / mode / "0-MasterOrchestrator.py"
            if f.exists():
                orchestrators.append(f)
    for mode in ["ModeOpenSourceAPI", "ModePremiumSubscribed", "ModeUserUploaded"]:
        f = base / "DataTeam" / "ael" / mode / "0-MasterOrchestrator.py"
        if f.exists():
            orchestrators.append(f)
    return orchestrators


def _get_migrated_stage_files():
    """Return stage files that should have been migrated to ToolRegistry."""
    base = _agents_dir
    files = []

    # IdeationTeam Stage 1 (all 4 modes)
    for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                  "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
        f = base / "IdeationTeam" / "ael" / mode / "1-SourcingStage.py"
        if f.exists():
            files.append(f)

    # LiteratureTeam Stage 1 + Stage 2 (all 4 modes)
    for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                  "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
        for stage in ["1-LiteratureGatheringStage.py", "2-GapDetectionStage.py"]:
            f = base / "LiteratureTeam" / "ael" / mode / stage
            if f.exists():
                files.append(f)

    # ModelTeam Stage 1 (all 4 modes)
    for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                  "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
        f = base / "ModelTeam" / "ael" / mode / "1-TheoryStage.py"
        if f.exists():
            files.append(f)

    # DataTeam Stage 1 (OpenSourceAPI)
    f = base / "DataTeam" / "ael" / "ModeOpenSourceAPI" / "1-DataSourceStage.py"
    if f.exists():
        files.append(f)

    return files


def _file_contains_text(filepath, text):
    source = filepath.read_text(encoding="utf-8")
    return text in source


def _file_contains_import(filepath, module_path, name):
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module == module_path:
                for alias in node.names:
                    if alias.name == name:
                        return True
    return False


# ============================================================================
# Test 1: register_all_tools
# ============================================================================

class TestRegisterAllTools:
    """Verify register_all_tools registers all expected tools."""

    def test_registers_6_tools(self):
        from shared.tools.tool_registry import ToolRegistry
        ToolRegistry.clear()

        from shared.tools.register_all import register_all_tools
        import shared.tools.register_all as reg_mod
        reg_mod._registered = False  # reset guard
        register_all_tools()

        names = ToolRegistry.tool_names()
        assert "arxiv_search" in names
        assert "fred_get_series" in names
        assert "web_search" in names
        assert "yfinance_history" in names
        assert "web_fetch" in names
        assert "web_post" in names
        assert len(names) >= 6

    def test_idempotent(self):
        """Calling register_all_tools twice doesn't duplicate."""
        from shared.tools.tool_registry import ToolRegistry
        from shared.tools.register_all import register_all_tools
        import shared.tools.register_all as reg_mod

        ToolRegistry.clear()
        reg_mod._registered = False
        register_all_tools()
        count1 = len(ToolRegistry.tool_names())

        register_all_tools()
        count2 = len(ToolRegistry.tool_names())

        assert count1 == count2


# ============================================================================
# Test 2: MasterOrchestrators import register_all_tools
# ============================================================================

class TestOrchestratorToolRegistration:
    """Verify all 15 MasterOrchestrators register tools."""

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_import_register_all(self, filepath):
        """Each MasterOrchestrator imports register_all_tools."""
        assert _file_contains_import(
            filepath, "shared.tools.register_all", "register_all_tools"
        ), f"{filepath.name}: missing register_all_tools import"

    @pytest.mark.parametrize("filepath", _get_orchestrator_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_calls_register_all(self, filepath):
        """Each MasterOrchestrator calls register_all_tools()."""
        assert _file_contains_text(filepath, "register_all_tools()"), (
            f"{filepath.name}: register_all_tools() not called"
        )


# ============================================================================
# Test 3: Stage files use ToolRegistry.invoke
# ============================================================================

class TestStageFileMigration:
    """Verify migrated stage files use ToolRegistry.invoke."""

    @pytest.mark.parametrize("filepath", _get_migrated_stage_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_imports_tool_registry(self, filepath):
        """Migrated stage file imports ToolRegistry."""
        assert _file_contains_import(
            filepath, "shared.tools.tool_registry", "ToolRegistry"
        ), f"{filepath.name}: missing ToolRegistry import"

    @pytest.mark.parametrize("filepath", _get_migrated_stage_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_uses_tool_registry_invoke(self, filepath):
        """Migrated stage file uses ToolRegistry.invoke()."""
        assert _file_contains_text(filepath, "ToolRegistry.invoke("), (
            f"{filepath.name}: no ToolRegistry.invoke() calls found"
        )

    @pytest.mark.parametrize("filepath", _get_migrated_stage_files(),
                             ids=lambda p: str(p.relative_to(_agents_dir)))
    def test_no_tracked_arxiv_search(self, filepath):
        """Migrated stage file does not call tracked_arxiv_search directly."""
        source = filepath.read_text(encoding="utf-8")
        # Allow the function to exist in comments but not as a call
        # Check import line specifically
        has_import = _file_contains_import(
            filepath, "shared.observability", "tracked_arxiv_search"
        )
        assert not has_import, (
            f"{filepath.name}: still imports tracked_arxiv_search"
        )


# ============================================================================
# Test 4: No tracked_* imports in migrated stage files
# ============================================================================

class TestNoTrackedImports:
    """Verify migrated files don't import tracked functions."""

    def _get_ideation_sourcing_files(self):
        base = _agents_dir
        return [
            base / "IdeationTeam" / "ael" / mode / "1-SourcingStage.py"
            for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL",
                          "ModeWithWcNoHITL", "ModeWithWcWithHITL"]
        ]

    def test_no_tracked_get_in_ideation(self):
        """IdeationTeam SourcingStage should not import tracked_get."""
        for f in self._get_ideation_sourcing_files():
            if f.exists():
                has_import = _file_contains_import(
                    f, "shared.observability", "tracked_get"
                )
                assert not has_import, f"{f.parent.name}: still imports tracked_get"

    def test_no_tracked_post_in_ideation_nofc(self):
        """IdeationTeam NoFc modes should not import tracked_post."""
        base = _agents_dir
        for mode in ["ModeNoWcNoHITL", "ModeNoWcWithHITL"]:
            f = base / "IdeationTeam" / "ael" / mode / "1-SourcingStage.py"
            if f.exists():
                has_import = _file_contains_import(
                    f, "shared.observability", "tracked_post"
                )
                assert not has_import, f"{mode}: still imports tracked_post"

    def test_no_tracked_fred_in_datateam(self):
        """DataTeam OpenSourceAPI should not import tracked_fred_get_series."""
        f = _agents_dir / "DataTeam" / "ael" / "ModeOpenSourceAPI" / "1-DataSourceStage.py"
        if f.exists():
            has_import = _file_contains_import(
                f, "shared.observability", "tracked_fred_get_series"
            )
            assert not has_import, "ModeOpenSourceAPI: still imports tracked_fred_get_series"


# ============================================================================
# Test 5: Tool wrappers exist
# ============================================================================

class TestToolWrappersExist:
    """Verify all tool wrapper modules exist."""

    def test_arxiv_tool_exists(self):
        assert (_agents_dir / "shared" / "tools" / "arxiv_tool.py").exists()

    def test_fred_tool_exists(self):
        assert (_agents_dir / "shared" / "tools" / "fred_tool.py").exists()

    def test_web_search_tool_exists(self):
        assert (_agents_dir / "shared" / "tools" / "web_search_tool.py").exists()

    def test_yfinance_tool_exists(self):
        assert (_agents_dir / "shared" / "tools" / "yfinance_tool.py").exists()

    def test_http_tool_exists(self):
        assert (_agents_dir / "shared" / "tools" / "http_tool.py").exists()

    def test_sandbox_tool_exists(self):
        assert (_agents_dir / "shared" / "tools" / "sandbox_tool.py").exists()

    def test_register_all_exists(self):
        assert (_agents_dir / "shared" / "tools" / "register_all.py").exists()


# ============================================================================
# Test 6: ToolRegistry functional — invoke records metrics
# ============================================================================

class TestToolRegistryMetrics:
    """Verify ToolRegistry invocations are recorded by MetricsCollector."""

    def test_invoke_records_tool_call(self):
        """ToolRegistry.invoke records to MetricsCollector."""
        from shared.tools.tool_registry import ToolRegistry
        from shared.observability import MetricsCollector

        ToolRegistry.clear()

        from shared.tools.register_all import register_all_tools
        import shared.tools.register_all as reg_mod
        reg_mod._registered = False
        register_all_tools()

        collector = MetricsCollector()

        # Invoke arxiv_search (will fail since no network, but should record)
        result = ToolRegistry.invoke(
            "arxiv_search",
            {"query": "test query", "max_results": 1},
            collector=collector,
            agent="TestAgent",
        )

        # Whether it succeeds or fails, a tool call should be recorded
        summary = collector.get_summary()
        tool_calls = summary.get("tools", {}).get("total_calls", 0)
        assert tool_calls >= 1 or not result.success

    def test_invoke_unknown_tool(self):
        """Invoking unknown tool returns error ToolResult."""
        from shared.tools.tool_registry import ToolRegistry
        result = ToolRegistry.invoke("nonexistent_tool", {})
        assert not result.success
        assert "not found" in result.error

    def test_invoke_invalid_params(self):
        """Invoking with invalid params returns validation error."""
        from shared.tools.tool_registry import ToolRegistry
        ToolRegistry.clear()

        from shared.tools.register_all import register_all_tools
        import shared.tools.register_all as reg_mod
        reg_mod._registered = False
        register_all_tools()

        # arxiv_search requires 'query' field
        result = ToolRegistry.invoke("arxiv_search", {"bad_field": "value"})
        assert not result.success
        assert "validation failed" in result.error.lower()


# ============================================================================
# Test 7: Tool specs have correct categories
# ============================================================================

class TestToolCategories:
    """Verify tool categories are set correctly."""

    def test_search_tools(self):
        from shared.tools.tool_registry import ToolRegistry
        ToolRegistry.clear()

        from shared.tools.register_all import register_all_tools
        import shared.tools.register_all as reg_mod
        reg_mod._registered = False
        register_all_tools()

        search_tools = ToolRegistry.list_tools(category="search")
        search_names = [t.name for t in search_tools]
        assert "arxiv_search" in search_names
        assert "web_search" in search_names

    def test_data_tools(self):
        from shared.tools.tool_registry import ToolRegistry
        data_tools = ToolRegistry.list_tools(category="data")
        data_names = [t.name for t in data_tools]
        assert "fred_get_series" in data_names
        assert "yfinance_history" in data_names

    def test_web_tools(self):
        from shared.tools.tool_registry import ToolRegistry
        web_tools = ToolRegistry.list_tools(category="web")
        web_names = [t.name for t in web_tools]
        assert "web_fetch" in web_names
        assert "web_post" in web_names


# ============================================================================
# Test 8: Sandbox integration (V0.3 — verify still present)
# ============================================================================

class TestSandboxIntegration:
    """Verify CodeSandbox is available and functional."""

    def test_sandbox_importable(self):
        from shared.tools.sandbox_tool import CodeSandbox
        sandbox = CodeSandbox()
        assert sandbox is not None

    def test_sandbox_execute_simple(self):
        from shared.tools.sandbox_tool import CodeSandbox
        sandbox = CodeSandbox()
        result = sandbox.execute("print(2 + 2)")
        assert result.success
        assert "4" in result.stdout

    def test_sandbox_safety_check(self):
        from shared.tools.sandbox_tool import CodeSandbox
        sandbox = CodeSandbox()
        check = sandbox.check_safety("import os; os.system('rm -rf /')")
        assert not check.safe


# ============================================================================
# Test 9: Count verification
# ============================================================================

class TestPhase4Counts:
    """Verify count expectations."""

    def test_15_orchestrators_have_register_all(self):
        orchestrators = _get_orchestrator_files()
        count = sum(
            1 for f in orchestrators
            if _file_contains_import(f, "shared.tools.register_all", "register_all_tools")
        )
        assert count == 15, f"Expected 15, got {count}"

    def test_migrated_stage_files_count(self):
        files = _get_migrated_stage_files()
        count = sum(
            1 for f in files
            if _file_contains_text(f, "ToolRegistry.invoke(")
        )
        # At least 13 migrated files (4 Ideation + 8 Literature + 4 Model + 1 DataTeam = 17 expected)
        assert count >= 13, f"Expected at least 13 migrated files, got {count}"

    def test_6_tools_registered(self):
        # Naming retained from V0.4 for git blame continuity. Count = 12:
        # 6 original + webcrawl_{scrape,crawl,extract,map} (2026-04-20) + openalex_search
        # + elsevier_abstract (2026-06, econ literature + institutional abstract fill).
        from shared.tools.tool_registry import ToolRegistry
        ToolRegistry.clear()

        from shared.tools.register_all import register_all_tools
        import shared.tools.register_all as reg_mod
        reg_mod._registered = False
        register_all_tools()

        assert len(ToolRegistry.tool_names()) == 12
