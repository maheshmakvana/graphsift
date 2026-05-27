"""Tests for the auto-fix suggestions engine (graphsift.auto_fix)."""

import pytest

from graphsift import (
    ContextBuilder,
    ContextConfig,
    DependencyGraph,
    FixReport,
    FixSeverity,
    FixSuggestion,
    PythonParser,
    detect_language,
    estimate_tokens,
)
from graphsift.auto_fix import FixSuggester


# ---------------------------------------------------------------------------
# Helper source fixtures
# ---------------------------------------------------------------------------

SOURCE_WITH_UNUSED_IMPORTS = '''"""Module with some unused imports."""
from typing import Optional, List
import os
import unused_thing

def used_function() -> str:
    """Do something."""
    return os.getcwd()
'''

SOURCE_MISSING_TYPES = '''"""Module with missing type annotations."""

def process_items(items, count, verbose):
    """Process items with no type hints."""
    results = []
    for item in items:
        results.append(item)
    return results

def typed_function(name: str, age: int) -> str:
    """This one has full annotations."""
    return f"{name} is {age}"
'''

SOURCE_LONG_FUNCTION = '''"""Module with a long function."""
def long_function():
    """A very long function."""
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
    g = 7
    h = 8
    i = 9
    j = 10
    k = 11
    l = 12
    m = 13
    n = 14
    o = 15
    p = 16
    q = 17
    r = 18
    s = 19
    t = 20
    u = 21
    v = 22
    w = 23
    x = 24
    y = 25
    z = 26
    aa = 27
    bb = 28
    cc = 29
    dd = 30
    ee = 31
    ff = 32
    gg = 33
    hh = 34
    ii = 35
    jj = 36
    kk = 37
    ll = 38
    mm = 39
    nn = 40
    oo = 41
    pp = 42
    qq = 43
    rr = 44
    ss = 45
    tt = 46
    uu = 47
    vv = 48
    ww = 49
    xx = 50
    yy = 51
    zz = 52
    result = a + b
    return result
'''

SOURCE_LONG_PARAMS = '''"""Module with too many parameters."""

def save_report(title, author, date, content, format, path,
                overwrite, compress, encrypt, notify, tags):
    """Save a report with too many options."""
    pass
'''


# ---------------------------------------------------------------------------
# Import checker tests
# ---------------------------------------------------------------------------


class TestImportChecker:
    """Tests for suggest_import_fixes."""

    def test_finds_unused_imports(self):
        """Import checker should detect symbols imported but never referenced."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/example.py", SOURCE_WITH_UNUSED_IMPORTS)
        graph.add_file(fn)

        source_map = {"src/example.py": SOURCE_WITH_UNUSED_IMPORTS}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_import_fixes()

        # unused_thing should be flagged
        unused = [f for f in fixes if "unused_thing" in f.description]
        assert len(unused) >= 1, (
            f"Expected unused_thing to be flagged, got: {[f.title for f in fixes]}"
        )

    def test_does_not_flag_used_imports(self):
        """Import checker should NOT flag imports that are actually used."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/example.py", SOURCE_WITH_UNUSED_IMPORTS)
        graph.add_file(fn)

        source_map = {"src/example.py": SOURCE_WITH_UNUSED_IMPORTS}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_import_fixes()

        # os is used (os.getcwd()), so it should not be flagged
        os_flags = [f for f in fixes if f.file_path == "src/example.py" and "os" in f.title]
        assert len(os_flags) == 0, (
            f"Used import 'os' should not be flagged: {[f.title for f in os_flags]}"
        )

    def test_no_false_positives_on_clean_file(self):
        """A file with all imports used should produce no import warnings."""
        source = '''"""Clean module."""
import os
import sys

def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}"

def main():
    """Main entry."""
    print(greet(sys.argv[1]))
    return os.getcwd()
'''
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("clean.py", source)
        graph.add_file(fn)

        source_map = {"clean.py": source}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_import_fixes()

        import_fixes = [f for f in fixes if f.file_path == "clean.py"]
        assert len(import_fixes) == 0, (
            f"Clean file should have no import warnings: {[f.title for f in import_fixes]}"
        )


# ---------------------------------------------------------------------------
# Type checker tests
# ---------------------------------------------------------------------------


class TestTypeChecker:
    """Tests for suggest_type_fixes."""

    def test_finds_missing_return_types(self):
        """Type checker should detect functions with no return type annotation."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/example.py", SOURCE_MISSING_TYPES)
        graph.add_file(fn)

        source_map = {"src/example.py": SOURCE_MISSING_TYPES}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_type_fixes()

        # process_items has no return type
        ret_fixes = [f for f in fixes if "return type" in f.title.lower() and "process_items" in f.title]
        assert len(ret_fixes) >= 1, (
            f"Expected return type warning for process_items: {[f.title for f in fixes]}"
        )

    def test_finds_missing_param_types(self):
        """Type checker should detect parameters with no type annotations."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/example.py", SOURCE_MISSING_TYPES)
        graph.add_file(fn)

        source_map = {"src/example.py": SOURCE_MISSING_TYPES}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_type_fixes()

        # process_items has items, count, verbose params with no types
        param_fixes = [f for f in fixes if "parameter type" in f.title.lower()]
        assert len(param_fixes) >= 1, (
            f"Expected param type warnings: {[f.title for f in fixes]}"
        )

    def test_does_not_flag_typed_function(self):
        """Type checker should NOT flag functions with full annotations."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/example.py", SOURCE_MISSING_TYPES)
        graph.add_file(fn)

        source_map = {"src/example.py": SOURCE_MISSING_TYPES}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_type_fixes()

        # typed_function has full annotations
        typed_fixes = [f for f in fixes if "typed_function" in f.title]
        assert len(typed_fixes) == 0, (
            f"typed_function should not have type warnings: {[f.title for f in typed_fixes]}"
        )

    def test_auto_fixable_flag(self):
        """Type suggestions should be marked auto_fixable."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/example.py", SOURCE_MISSING_TYPES)
        graph.add_file(fn)

        source_map = {"src/example.py": SOURCE_MISSING_TYPES}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_type_fixes()

        for fix in fixes:
            assert fix.auto_fixable, (
                f"Type fix should be auto_fixable: {fix.title}"
            )
            assert fix.confidence > 0.8, (
                f"Type fix confidence should be >0.8: {fix.title} ({fix.confidence})"
            )


# ---------------------------------------------------------------------------
# Structure checker tests
# ---------------------------------------------------------------------------


class TestStructureChecker:
    """Tests for suggest_structure_fixes."""

    def test_finds_long_parameter_list(self):
        """Structure checker should detect functions with >5 params."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/report.py", SOURCE_LONG_PARAMS)
        graph.add_file(fn)

        source_map = {"src/report.py": SOURCE_LONG_PARAMS}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_structure_fixes()

        param_fixes = [f for f in fixes if "parameter" in f.title.lower()]
        assert len(param_fixes) >= 1, (
            f"Expected param list warning for save_report: {[f.title for f in fixes]}"
        )

    def test_finds_long_function(self):
        """Structure checker should detect functions >50 lines."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/long_func.py", SOURCE_LONG_FUNCTION)
        graph.add_file(fn)

        source_map = {"src/long_func.py": SOURCE_LONG_FUNCTION}
        suggester = FixSuggester(graph, source_map=source_map)
        fixes = suggester.suggest_structure_fixes()

        long_fixes = [f for f in fixes if "long function" in f.title.lower()]
        assert len(long_fixes) >= 1, (
            f"Expected long function warning: {[f.title for f in fixes]}"
        )


# ---------------------------------------------------------------------------
# FixReport serialization tests
# ---------------------------------------------------------------------------


class TestFixReport:
    """Tests for FixReport construction and serialization."""

    def test_empty_report(self):
        """An empty FixReport should have zero totals."""
        report = FixReport(
            suggestions=[],
            total_issues=0,
            by_severity={},
            by_category={},
            summary="No issues found.",
        )
        assert report.total_issues == 0
        assert len(report.suggestions) == 0
        assert report.summary == "No issues found."

    def test_report_with_suggestions(self):
        """FixReport should correctly aggregate counts."""
        suggestions = [
            FixSuggestion(
                suggestion_id="1",
                file_path="src/a.py",
                line_start=10,
                severity=FixSeverity.WARNING,
                category="import",
                title="Unused import: foo",
                description="foo is imported but never used",
                confidence=0.7,
            ),
            FixSuggestion(
                suggestion_id="2",
                file_path="src/b.py",
                line_start=20,
                severity=FixSeverity.ERROR,
                category="cycle",
                title="Dependency cycle",
                description="Cycle detected",
                confidence=0.5,
            ),
        ]
        report = FixReport(
            suggestions=suggestions,
            total_issues=2,
            by_severity={"warning": 1, "error": 1},
            by_category={"import": 1, "cycle": 1},
            summary="Found 2 issue(s): cycle=1, import=1",
        )
        assert report.total_issues == 2
        assert report.by_severity["warning"] == 1
        assert report.by_severity["error"] == 1
        assert report.by_category["import"] == 1
        assert report.by_category["cycle"] == 1

    def test_report_json_serialization(self):
        """FixReport should serialize to JSON correctly."""
        suggestion = FixSuggestion(
            suggestion_id="test-1",
            file_path="src/a.py",
            line_start=10,
            severity=FixSeverity.WARNING,
            category="import",
            title="Unused import: bar",
            description="bar is imported but never used",
            confidence=0.7,
        )
        report = FixReport(
            suggestions=[suggestion],
            total_issues=1,
            by_severity={"warning": 1},
            by_category={"import": 1},
            summary="Found 1 issue(s): import=1",
        )
        data = report.model_dump()
        assert data["total_issues"] == 1
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["suggestion_id"] == "test-1"
        assert data["suggestions"][0]["severity"] == "warning"
        assert data["suggestions"][0]["category"] == "import"

        json_str = report.model_dump_json()
        assert '"suggestion_id":"test-1"' in json_str
        assert '"total_issues":1' in json_str


# ---------------------------------------------------------------------------
# Confidence scoring tests
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    """Tests for fix suggestion confidence thresholds."""

    def test_confidence_in_range(self):
        """All fix suggestions should have confidence in [0, 1]."""
        graph = DependencyGraph()
        parser = PythonParser()

        # Combine sources to get various fix types
        source_map = {
            "src/imports.py": SOURCE_WITH_UNUSED_IMPORTS,
            "src/types.py": SOURCE_MISSING_TYPES,
            "src/params.py": SOURCE_LONG_PARAMS,
        }
        for path, src in source_map.items():
            fn = parser.parse_file(path, src)
            graph.add_file(fn)

        suggester = FixSuggester(graph, source_map=source_map)
        report = suggester.analyze()

        for s in report.suggestions:
            assert 0.0 <= s.confidence <= 1.0, (
                f"Confidence out of range: {s.confidence} for {s.title}"
            )

    def test_auto_fixable_high_confidence_only(self):
        """Only suggestions with confidence >0.8 should be auto_fixable."""
        graph = DependencyGraph()
        parser = PythonParser()

        source_map = {
            "src/imports.py": SOURCE_WITH_UNUSED_IMPORTS,
            "src/types.py": SOURCE_MISSING_TYPES,
            "src/params.py": SOURCE_LONG_PARAMS,
        }
        for path, src in source_map.items():
            fn = parser.parse_file(path, src)
            graph.add_file(fn)

        suggester = FixSuggester(graph, source_map=source_map)
        report = suggester.analyze()

        for s in report.suggestions:
            if s.auto_fixable:
                assert s.confidence > 0.8, (
                    f"Auto-fixable suggestion must have confidence >0.8: "
                    f"{s.title} ({s.confidence})"
                )


# ---------------------------------------------------------------------------
# Analyze integration test
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Tests for the full analyze() pipeline."""

    def test_analyze_returns_report(self):
        """analyze() should return a FixReport."""
        graph = DependencyGraph()
        parser = PythonParser()

        source_map = {
            "src/imports.py": SOURCE_WITH_UNUSED_IMPORTS,
            "src/types.py": SOURCE_MISSING_TYPES,
            "src/params.py": SOURCE_LONG_PARAMS,
        }
        for path, src in source_map.items():
            fn = parser.parse_file(path, src)
            graph.add_file(fn)

        suggester = FixSuggester(graph, source_map=source_map)
        report = suggester.analyze()

        assert isinstance(report, FixReport)
        assert report.total_issues >= 1
        assert len(report.suggestions) == report.total_issues
        assert "by_severity" in FixReport.model_fields
        assert "by_category" in FixReport.model_fields

    def test_analyze_changed_files_filter(self):
        """analyze() should filter by changed_files when provided."""
        graph = DependencyGraph()
        parser = PythonParser()

        source_map = {
            "src/imports.py": SOURCE_WITH_UNUSED_IMPORTS,
            "src/types.py": SOURCE_MISSING_TYPES,
        }
        for path, src in source_map.items():
            fn = parser.parse_file(path, src)
            graph.add_file(fn)

        suggester = FixSuggester(graph, source_map=source_map)
        report = suggester.analyze(changed_files=["src/types.py"])

        for s in report.suggestions:
            assert s.file_path == "src/types.py", (
                f"Filtered report should only contain src/types.py: "
                f"got {s.file_path}"
            )

    def test_analyze_deduplication(self):
        """analyze() should not produce duplicate suggestions."""
        graph = DependencyGraph()
        parser = PythonParser()

        source_map = {"src/types.py": SOURCE_MISSING_TYPES}
        fn = parser.parse_file("src/types.py", SOURCE_MISSING_TYPES)
        graph.add_file(fn)

        suggester = FixSuggester(graph, source_map=source_map)
        report = suggester.analyze()

        ids = [s.suggestion_id for s in report.suggestions]
        assert len(ids) == len(set(ids)), (
            f"Duplicate suggestion IDs found: {ids}"
        )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_graph(self):
        """FixSuggester should handle an empty graph gracefully."""
        graph = DependencyGraph()
        suggester = FixSuggester(graph)
        report = suggester.analyze()

        assert isinstance(report, FixReport)
        assert report.total_issues == 0
        assert report.suggestions == []

    def test_dead_code_with_empty_graph(self):
        """suggest_dead_code_removal should handle empty graph."""
        graph = DependencyGraph()
        suggester = FixSuggester(graph)
        fixes = suggester.suggest_dead_code_removal()

        assert fixes == []

    def test_cycle_with_empty_graph(self):
        """suggest_cycle_fixes should handle empty graph."""
        graph = DependencyGraph()
        suggester = FixSuggester(graph)
        fixes = suggester.suggest_cycle_fixes()

        assert fixes == []

    def test_import_check_without_source_map(self):
        """Import checker should not crash when source_map is empty."""
        graph = DependencyGraph()
        parser = PythonParser()
        fn = parser.parse_file("src/example.py", SOURCE_WITH_UNUSED_IMPORTS)
        graph.add_file(fn)

        suggester = FixSuggester(graph)  # no source_map
        fixes = suggester.suggest_import_fixes()

        # Without source_map, import checking is limited
        assert isinstance(fixes, list)

    def test_fix_suggestion_roundtrip(self):
        """FixSuggestion should round-trip through dict serialization."""
        original = FixSuggestion(
            suggestion_id="rt-1",
            file_path="src/test.py",
            line_start=42,
            line_end=50,
            severity=FixSeverity.ERROR,
            category="dead_code",
            title="Unused function: old_helper",
            description="This function has no callers.",
            suggested_change="Remove the function old_helper",
            confidence=0.9,
            auto_fixable=True,
        )
        data = original.model_dump()
        restored = FixSuggestion(**data)

        assert original.suggestion_id == restored.suggestion_id
        assert original.file_path == restored.file_path
        assert original.severity == restored.severity
        assert original.confidence == restored.confidence
        assert original.auto_fixable == restored.auto_fixable
        assert original.suggested_change == restored.suggested_change
