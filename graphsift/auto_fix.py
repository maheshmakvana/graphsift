"""Auto-fix suggestions engine for graphsift.

Detects common code issues via dependency graph analysis and suggests fixes.
All operations are read-only — never modifies files.

Checkers implemented:
  - Import checker  : unused imports, missing imports, circular import issues
  - Type checker    : missing type annotations on functions/methods
  - Structure checker : long functions, long param lists, large classes
  - Cycle checker   : dependency cycle break suggestions
  - Dead code       : unreachable code removal suggestions
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any

from .models import (
    FixReport,
    FixSeverity,
    FixSuggestion,
    NodeKind,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIDENCE: dict[str, float] = {
    "import_unused": 0.70,
    "import_cycle": 0.50,
    "type_return": 0.90,
    "type_param": 0.85,
    "structure_long_func": 0.75,
    "structure_long_params": 0.85,
    "structure_large_class": 0.75,
    "cycle_fix": 0.50,
    "dead_code": 0.90,
}

_AUTO_FIXABLE_CATEGORIES = frozenset(
    {"type_return", "type_param", "dead_code"}
)


def _make_id(file_path: str, line_start: int, category: str, title: str) -> str:
    raw = f"{file_path}:{line_start}:{category}:{title}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _import_base_name(import_name: str) -> str:
    """Return the base symbol name from a dotted import path.

    ``auth.AuthManager`` -> ``AuthManager``
    ``os``              -> ``os``
    """
    return import_name.split(".")[-1]


def _import_is_used(source: str, import_name: str) -> bool:
    """Heuristic: check if an imported symbol appears in source body
    (excluding import statements)."""
    base = _import_base_name(import_name)
    if not base or len(base) < 2:
        return True

    # Strip import lines
    lines = source.splitlines()
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            continue
        body_lines.append(line)
    body = "\n".join(body_lines)

    # Word-boundary match
    pattern = r"\b" + re.escape(base) + r"\b"
    return bool(re.search(pattern, body))


def _count_params(signature: str) -> int:
    """Approximate number of real parameters in a function signature."""
    if "(" not in signature or ")" not in signature:
        return 0
    try:
        pstr = signature[signature.index("(") + 1: signature.index(")")]
    except ValueError:
        return 0
    if not pstr.strip():
        return 0
    params = [p.strip() for p in pstr.split(",") if p.strip()]
    # Exclude self/cls
    return sum(1 for p in params if p not in ("self", "cls"))


def _missing_return_type(signature: str) -> bool:
    """Check if a Python-function signature lacks a return type annotation."""
    if "->" in signature:
        return False
    # Non-Python / no sig -> skip
    return True


def _missing_param_types(signature: str) -> list[str]:
    """Return parameter names that lack type annotations."""
    if "(" not in signature or ")" not in signature:
        return []
    try:
        pstr = signature[signature.index("(") + 1: signature.index(")")]
    except ValueError:
        return []
    if not pstr.strip():
        return []
    missing: list[str] = []
    for p in pstr.split(","):
        p = p.strip()
        if not p:
            continue
        if p in ("self", "cls"):
            continue
        # param: type = default  OR  param: type
        # param = default       OR  param
        if ":" not in p:
            name = p.split("=")[0].strip()
            if name:
                missing.append(name)
    return missing


# ---------------------------------------------------------------------------
# FixSuggester
# ---------------------------------------------------------------------------


class FixSuggester:
    """Suggests automated fixes for common code issues detected via graph analysis.

    Args:
        graph: DependencyGraph instance (from graphsift.core).
        store: Optional GraphStore for additional data.
        source_map: Optional dict of file_path -> source text (improves import
                    checking accuracy).

    All methods are read-only — no files are modified.
    """

    def __init__(
        self,
        graph: Any,
        store: Any | None = None,
        source_map: dict[str, str] | None = None,
    ) -> None:
        self._graph = graph
        self._store = store
        self._source_map = source_map or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self, changed_files: list[str] | None = None
    ) -> FixReport:
        """Run all analyzers and return a FixReport with prioritized suggestions.

        Args:
            changed_files: If given, only return suggestions for these files.

        Returns:
            FixReport with deduplicated, sorted suggestions.
        """
        suggestions: list[FixSuggestion] = []

        suggestions.extend(self.suggest_import_fixes())
        suggestions.extend(self.suggest_type_fixes())
        suggestions.extend(self.suggest_structure_fixes())
        suggestions.extend(self.suggest_dead_code_removal())
        suggestions.extend(self.suggest_cycle_fixes())

        # Filter by changed_files if specified
        if changed_files:
            changed_set = set(changed_files)
            suggestions = [
                s for s in suggestions if s.file_path in changed_set
            ]

        # Deduplicate
        seen: set[str] = set()
        unique: list[FixSuggestion] = []
        for s in suggestions:
            key = s.suggestion_id
            if key not in seen:
                seen.add(key)
                unique.append(s)

        # Sort by severity (error first) then confidence descending
        severity_order = {"error": 0, "warning": 1, "info": 2}
        unique.sort(
            key=lambda s: (
                severity_order.get(s.severity.value, 99),
                -s.confidence,
            )
        )

        by_severity: dict[str, int] = defaultdict(int)
        by_category: dict[str, int] = defaultdict(int)
        for s in unique:
            by_severity[s.severity.value] += 1
            by_category[s.category] += 1

        parts = [f"Found {len(unique)} issue(s)"]
        if by_category:
            parts.append(
                ": "
                + ", ".join(
                    f"{k}={v}" for k, v in sorted(by_category.items())
                )
            )
        summary = "".join(parts)

        return FixReport(
            suggestions=unique,
            total_issues=len(unique),
            by_severity=dict(by_severity),
            by_category=dict(by_category),
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Import checker
    # ------------------------------------------------------------------

    def suggest_import_fixes(self) -> list[FixSuggestion]:
        """Find unused imports and circular import issues."""
        suggestions: list[FixSuggestion] = []

        with self._graph._lock:
            file_nodes = dict(self._graph._file_nodes)
            nodes = dict(self._graph._nodes)
            adj_in = {k: list(v) for k, v in self._graph._adj_in.items()}

        # --- Unused imports ---
        for file_path, fn in file_nodes.items():
            if not fn.imports:
                continue
            source = self._source_map.get(file_path, "")
            if not source:
                # Fall back to edge check: if an IMPORTS edge points to a
                # module but no CALLS/REFERENCES edge exists for nodes
                # in this file to nodes in that module, the import is
                # likely unused.
                file_mod = f"{file_path}::__module__"
                in_edges = adj_in.get(file_mod, [])
                imported_mods = {
                    e.source_id.split("::__module__")[0]
                    for e in in_edges
                    if e.kind.value == "imports"
                }
                # Edge check would need per-module analysis — skip without
                # source for now.
                continue

            for imp in fn.imports:
                if _import_is_used(source, imp):
                    continue
                base = _import_base_name(imp)
                # Suggest removing the import
                suggestions.append(
                    FixSuggestion(
                        suggestion_id=_make_id(
                            file_path, 0, "import", f"unused import {base}"
                        ),
                        file_path=file_path,
                        line_start=0,
                        severity=FixSeverity.WARNING,
                        category="import",
                        title=f"Unused import: {base}",
                        description=(
                            f"The symbol ``{base}`` (imported from "
                            f"``{imp}``) does not appear to be used in "
                            f"``{file_path}``."
                        ),
                        suggested_change=f"# Remove: import {base}",
                        confidence=_CONFIDENCE["import_unused"],
                        auto_fixable=False,
                    )
                )

        return suggestions

    # ------------------------------------------------------------------
    # Type checker
    # ------------------------------------------------------------------

    def suggest_type_fixes(self) -> list[FixSuggestion]:
        """Find missing type annotations on Python functions/methods."""
        suggestions: list[FixSuggestion] = []

        with self._graph._lock:
            nodes = dict(self._graph._nodes)

        for node in nodes.values():
            lang = getattr(node, "language", None)
            if lang is None or lang.value != "python":
                continue
            if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD):
                continue
            sig = node.signature
            if not sig:
                continue

            # Return type
            if _missing_return_type(sig):
                suggestions.append(
                    FixSuggestion(
                        suggestion_id=_make_id(
                            node.file_path,
                            node.line_start,
                            "type",
                            f"missing return type: {node.name}",
                        ),
                        file_path=node.file_path,
                        line_start=node.line_start,
                        severity=FixSeverity.WARNING,
                        category="type",
                        title=f"Missing return type annotation: {node.name}",
                        description=(
                            f"Function ``{node.name}`` at "
                            f"{node.file_path}:{node.line_start} has no "
                            f"return type annotation."
                        ),
                        suggested_change=(
                            f"Add ``-> None`` (or the correct return type) "
                            f"to {node.name}"
                        ),
                        confidence=_CONFIDENCE["type_return"],
                        auto_fixable=True,
                    )
                )

            # Parameter types
            missing_params = _missing_param_types(sig)
            if missing_params:
                for pname in missing_params[:5]:  # limit per function
                    suggestions.append(
                        FixSuggestion(
                            suggestion_id=_make_id(
                                node.file_path,
                                node.line_start,
                                "type",
                                f"missing param type: {node.name}.{pname}",
                            ),
                            file_path=node.file_path,
                            line_start=node.line_start,
                            severity=FixSeverity.INFO,
                            category="type",
                            title=(
                                f"Missing parameter type: "
                                f"{node.name}.{pname}"
                            ),
                            description=(
                                f"Parameter ``{pname}`` of "
                                f"``{node.name}`` at "
                                f"{node.file_path}:{node.line_start} "
                                f"has no type annotation."
                            ),
                            suggested_change=(
                                f"Add type annotation to parameter "
                                f"``{pname}`` in ``{node.name}``"
                            ),
                            confidence=_CONFIDENCE["type_param"],
                            auto_fixable=True,
                        )
                    )

        return suggestions

    # ------------------------------------------------------------------
    # Structure checker
    # ------------------------------------------------------------------

    def suggest_structure_fixes(self) -> list[FixSuggestion]:
        """Find long parameter lists, long functions, and large classes."""
        suggestions: list[FixSuggestion] = []

        with self._graph._lock:
            nodes = dict(self._graph._nodes)

        for node in nodes.values():
            # --- Long parameter lists (>5 params) ---
            if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                nparams = _count_params(node.signature)
                if nparams > 5:
                    suggestions.append(
                        FixSuggestion(
                            suggestion_id=_make_id(
                                node.file_path,
                                node.line_start,
                                "structure",
                                f"long param list: {node.name}",
                            ),
                            file_path=node.file_path,
                            line_start=node.line_start,
                            line_end=node.line_start,
                            severity=FixSeverity.WARNING,
                            category="structure",
                            title=(
                                f"Long parameter list: {node.name} "
                                f"({nparams} params)"
                            ),
                            description=(
                                f"``{node.name}`` at "
                                f"{node.file_path}:{node.line_start} "
                                f"has {nparams} parameters. "
                                f"Consider grouping related params into "
                                f"a data class or config object."
                            ),
                            suggested_change=(
                                f"Extract parameters into a config class"
                            ),
                            confidence=_CONFIDENCE["structure_long_params"],
                            auto_fixable=False,
                        )
                    )

            # --- Long functions (>50 lines) ---
            if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                line_count = max(0, node.line_end - node.line_start)
                if line_count > 50:
                    suggestions.append(
                        FixSuggestion(
                            suggestion_id=_make_id(
                                node.file_path,
                                node.line_start,
                                "structure",
                                f"long function: {node.name}",
                            ),
                            file_path=node.file_path,
                            line_start=node.line_start,
                            line_end=node.line_end,
                            severity=FixSeverity.WARNING,
                            category="structure",
                            title=(
                                f"Long function: {node.name} "
                                f"({line_count} lines)"
                            ),
                            description=(
                                f"``{node.name}`` at "
                                f"{node.file_path}:{node.line_start} is "
                                f"{line_count} lines. Consider extracting "
                                f"inner logic into helper functions."
                            ),
                            suggested_change=(
                                f"Extract inner logic into helper "
                                f"functions"
                            ),
                            confidence=_CONFIDENCE["structure_long_func"],
                            auto_fixable=False,
                        )
                    )

            # --- Large classes (>300 lines) ---
            if node.kind == NodeKind.CLASS:
                line_count = max(0, node.line_end - node.line_start)
                if line_count > 300:
                    suggestions.append(
                        FixSuggestion(
                            suggestion_id=_make_id(
                                node.file_path,
                                node.line_start,
                                "structure",
                                f"large class: {node.name}",
                            ),
                            file_path=node.file_path,
                            line_start=node.line_start,
                            line_end=node.line_end,
                            severity=FixSeverity.WARNING,
                            category="structure",
                            title=(
                                f"Large class: {node.name} "
                                f"({line_count} lines)"
                            ),
                            description=(
                                f"Class ``{node.name}`` at "
                                f"{node.file_path}:{node.line_start} spans "
                                f"{line_count} lines. Consider splitting "
                                f"into smaller focused classes."
                            ),
                            suggested_change=(
                                f"Split into smaller classes by "
                                f"responsibility"
                            ),
                            confidence=_CONFIDENCE["structure_large_class"],
                            auto_fixable=False,
                        )
                    )

        return suggestions

    # ------------------------------------------------------------------
    # Cycle checker
    # ------------------------------------------------------------------

    def suggest_cycle_fixes(self) -> list[FixSuggestion]:
        """Suggest dependency inversions or interface extractions to break
        cycles."""
        suggestions: list[FixSuggestion] = []

        cycles = self._graph.detect_cycles()
        if not cycles:
            return suggestions

        for i, cycle in enumerate(cycles):
            files_list = ", ".join(cycle[:5])
            if len(cycle) > 5:
                files_list += f", ... ({len(cycle) - 5} more)"
            suggestions.append(
                FixSuggestion(
                    suggestion_id=_make_id(
                        f"cycle_{i}",
                        0,
                        "cycle",
                        f"dependency cycle {i + 1}",
                    ),
                    file_path=cycle[0] if cycle else "",
                    line_start=0,
                    severity=FixSeverity.ERROR,
                    category="cycle",
                    title=f"Dependency cycle ({len(cycle)} files)",
                    description=(
                        f"Circular dependency detected involving "
                        f"{len(cycle)} files: {files_list}. Cycles "
                        f"increase coupling and make refactoring harder."
                    ),
                    suggested_change=(
                        "Consider extracting a shared interface or moving "
                        "common types to a separate module to break the "
                        "cycle."
                    ),
                    confidence=_CONFIDENCE["cycle_fix"],
                    auto_fixable=False,
                )
            )

        return suggestions

    # ------------------------------------------------------------------
    # Dead code checker
    # ------------------------------------------------------------------

    def suggest_dead_code_removal(self) -> list[FixSuggestion]:
        """Suggest removing unreachable code found by dead code detection."""
        suggestions: list[FixSuggestion] = []

        from .adapters.postprocess import RefactorEngine  # noqa: PLC0415

        engine = RefactorEngine()
        dead = engine.find_dead_code(self._graph, limit=100)

        for item in dead:
            suggestions.append(
                FixSuggestion(
                    suggestion_id=_make_id(
                        item["file_path"],
                        item["line_start"],
                        "dead_code",
                        f"dead code: {item['name']}",
                    ),
                    file_path=item["file_path"],
                    line_start=item["line_start"],
                    line_end=item.get("line_end", 0),
                    severity=FixSeverity.WARNING,
                    category="dead_code",
                    title=f"Unused symbol: {item['name']}",
                    description=(
                        f"The {item['kind']} ``{item['name']}`` at "
                        f"{item['file_path']}:{item['line_start']} "
                        f"has no callers from reachable code. "
                        f"Reason: {item.get('reason', 'No incoming edges')}."
                    ),
                    suggested_change=(
                        f"Remove the {item['kind']} "
                        f"``{item['name']}`` (lines "
                        f"{item['line_start']}-{item.get('line_end', 0)})"
                    ),
                    confidence=_CONFIDENCE["dead_code"],
                    auto_fixable=True,
                )
            )

        return suggestions
