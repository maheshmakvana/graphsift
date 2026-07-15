"""Pure domain logic for graphsift — zero I/O, zero side effects.

Architecture:
  ASTParser        — language-specific symbol/import extraction (pure)
  DependencyGraph  — in-memory directed graph of symbols and edges
  RelevanceRanker  — multi-signal scoring of files given a diff
  ContextSelector  — token-budget-aware file selection + rendering
  ContextBuilder   — orchestrates the full pipeline
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import re
import sys
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from graphsift.adapters.storage import GraphStore

from .exceptions import (
    BudgetExceededError,
    GraphError,
    LanguageNotSupportedError,
    ParseError,
    ValidationError,
)
from .models import (
    ContextConfig,
    ContextResult,
    DepthTier,
    DiffSpec,
    EdgeKind,
    FileNode,
    GraphEdge,
    GraphNode,
    IndexStats,
    Language,
    NodeKind,
    OutputMode,
    ScoredFile,
    SourceConfidence,
    TierLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXT_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".cxx": Language.CPP,
    ".cc": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".sh": Language.BASH,
    ".bash": Language.BASH,
    ".zsh": Language.BASH,
    ".tf": Language.HCL,
    ".tfvars": Language.HCL,
    ".hcl": Language.HCL,
}


def detect_language(path: str) -> Language:
    """Detect language from file extension.

    Helm charts are detected by the presence of ``templates/`` in the path
    for ``.yaml``/``.yml`` files, or by ``Chart.yaml`` filename.

    Args:
        path: File path.

    Returns:
        Language enum value.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    mapped = _EXT_MAP.get(suffix)
    if mapped is not None:
        return mapped
    # Helm chart detection: templates/*.yaml or Chart.yaml
    if suffix in (".yaml", ".yml"):
        parts = p.parts
        if "templates" in parts or p.name in ("Chart.yaml", "values.yaml"):
            return Language.HELM
    return Language.UNKNOWN


def estimate_tokens(text: str) -> int:
    """Fast token estimate (4 chars per token heuristic)."""
    return max(1, len(text) // 4)


def _build_diff_hash(diff_spec: DiffSpec) -> str:
    """Deterministic SHA-256 hash of diff spec for cache keying.

    Key = sorted(changed_files) + query + commit_message.
    This ensures the same code change + question always hits the same cache.
    """
    raw = json.dumps(
        {
            "changed_files": sorted(diff_spec.changed_files),
            "query": diff_spec.query,
            "commit_message": diff_spec.commit_message,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Protocol: LanguageParser
# ---------------------------------------------------------------------------


@runtime_checkable
class LanguageParser(Protocol):
    """Structural protocol for language-specific AST parsers.

    No inheritance required — any object with these methods qualifies.
    """

    def parse_file(self, path: str, source: str) -> FileNode:
        """Extract symbols and imports from source text.

        Args:
            path: File path (for node IDs).
            source: Full source code text.

        Returns:
            FileNode with extracted symbols, imports, dynamic_imports.
        """
        ...

    def extract_signatures(self, source: str) -> str:
        """Return signatures-only view of source (no bodies).

        Args:
            source: Full source code.

        Returns:
            Condensed string with function/class signatures only.
        """
        ...


# ---------------------------------------------------------------------------
# Python AST parser (pure, no subprocess)
# ---------------------------------------------------------------------------


class PythonParser:
    """Pure-Python AST parser for Python source files.

    Extracts: functions, classes, methods, imports, decorators,
    async functions, dynamic imports (importlib.import_module, __import__).

    Fixes over code-review-graph:
    - Decorator nodes and DECORATES edges (not just ignored)
    - Dynamic import detection via regex + AST call inspection
    - Async function flag
    - Signature extraction without bodies
    """

    # Dynamic import patterns
    _DYN_PATTERNS = [
        re.compile(r'importlib\.import_module\(["\']([^"\']+)["\']\)'),
        re.compile(r'__import__\(["\']([^"\']+)["\']\)'),
        re.compile(r'importlib\.util\.spec_from_file_location\([^,]+,\s*["\']([^"\']+)["\']\)'),
        re.compile(r'plugin\s*=\s*__import__\(["\']([^"\']+)["\']\)'),
    ]

    def parse_file(self, path: str, source: str) -> FileNode:
        """Parse a Python source file into a FileNode.

        Args:
            path: File path.
            source: Python source text.

        Returns:
            FileNode with all extracted symbols.

        Raises:
            ParseError: If source cannot be parsed.
        """
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            raise ParseError(f"Python syntax error in {path}: {exc}") from exc

        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []

        # Module node
        module_id = f"{path}::__module__"
        symbols.append(GraphNode(
            node_id=module_id,
            file_path=path,
            kind=NodeKind.MODULE,
            name=Path(path).stem,
            qualified_name=Path(path).stem,
            line_start=1,
            line_end=len(source.splitlines()),
            language=Language.PYTHON,
            source_confidence=SourceConfidence.EXTRACTED,
        ))

        self._walk(tree, path, "", symbols, imports)

        # Dynamic import detection via regex
        for pat in self._DYN_PATTERNS:
            for m in pat.finditer(source):
                mod = m.group(1)
                if mod not in dynamic_imports:
                    dynamic_imports.append(mod)

        sha = hashlib.sha256(source.encode(errors="replace")).hexdigest()
        return FileNode(
            path=path,
            language=Language.PYTHON,
            size_bytes=len(source.encode(errors="replace")),
            line_count=len(source.splitlines()),
            sha256=sha,
            symbols=symbols,
            imports=imports,
            dynamic_imports=dynamic_imports,
            token_estimate=estimate_tokens(source),
        )

    def _walk(
        self,
        node: ast.AST,
        path: str,
        parent_qual: str,
        symbols: list[GraphNode],
        imports: list[str],
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{parent_qual}.{child.name}" if parent_qual else child.name
                decs = [self._dec_name(d) for d in child.decorator_list]
                sig = self._build_signature(child)
                symbols.append(GraphNode(
                    node_id=f"{path}::{qual}",
                    file_path=path,
                    kind=NodeKind.METHOD if "." in qual else NodeKind.FUNCTION,
                    name=child.name,
                    qualified_name=qual,
                    line_start=child.lineno,
                    line_end=getattr(child, "end_lineno", child.lineno),
                    language=Language.PYTHON,
                    signature=sig,
                    decorators=decs,
                    is_async=isinstance(child, ast.AsyncFunctionDef),
                    source_confidence=SourceConfidence.EXTRACTED,
                ))
                self._walk(child, path, qual, symbols, imports)

            elif isinstance(child, ast.ClassDef):
                qual = f"{parent_qual}.{child.name}" if parent_qual else child.name
                decs = [self._dec_name(d) for d in child.decorator_list]
                bases = [self._node_name(b) for b in child.bases]
                symbols.append(GraphNode(
                    node_id=f"{path}::{qual}",
                    file_path=path,
                    kind=NodeKind.CLASS,
                    name=child.name,
                    qualified_name=qual,
                    line_start=child.lineno,
                    line_end=getattr(child, "end_lineno", child.lineno),
                    language=Language.PYTHON,
                    decorators=decs,
                    metadata={"bases": bases},
                    source_confidence=SourceConfidence.EXTRACTED,
                ))
                self._walk(child, path, qual, symbols, imports)

            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name not in imports:
                        imports.append(alias.name)

            elif isinstance(child, ast.ImportFrom):
                mod = child.module or ""
                if mod and mod not in imports:
                    imports.append(mod)
                # Also capture: from module import name
                for alias in child.names:
                    full = f"{mod}.{alias.name}" if mod else alias.name
                    if full not in imports:
                        imports.append(full)

    @staticmethod
    def _dec_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{PythonParser._dec_name(node.value)}.{node.attr}"
        if isinstance(node, ast.Call):
            return PythonParser._dec_name(node.func)
        return ""

    @staticmethod
    def _node_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{PythonParser._node_name(node.value)}.{node.attr}"
        return ""

    @staticmethod
    def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = []
        fn_args = node.args
        all_args = fn_args.args + fn_args.posonlyargs + fn_args.kwonlyargs
        for arg in all_args:
            ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
            args.append(f"{arg.arg}{ann}")
        if fn_args.vararg:
            args.append(f"*{fn_args.vararg.arg}")
        if fn_args.kwarg:
            args.append(f"**{fn_args.kwarg.arg}")
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)}){ret}"

    def extract_signatures(self, source: str) -> str:
        """Return signatures-only view (no bodies).

        Args:
            source: Python source text.

        Returns:
            String with only function/class signatures and docstrings.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source[:500]

        lines: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                lines.append(f"class {node.name}({', '.join(self._node_name(b) for b in node.bases)}):")
                doc = ast.get_docstring(node)
                if doc:
                    lines.append(f'    """{doc[:120]}"""')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.append(f"    {self._build_signature(node)}")
                doc = ast.get_docstring(node)
                if doc:
                    lines.append(f'        """{doc[:80]}"""')
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generic regex-based parser (JS/TS/Go/Rust/etc)
# ---------------------------------------------------------------------------


class GenericParser:
    """Regex-based parser for non-Python languages.

    Extracts: function definitions, class definitions, import statements,
    dynamic require/import patterns.

    No external tree-sitter dependency required.
    """

    _PATTERNS: dict[Language, dict[str, re.Pattern[str]]] = {
        Language.JAVASCRIPT: {
            "function": re.compile(
                r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
            ),
            "arrow": re.compile(
                r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>"
            ),
            "class": re.compile(r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?"),
            "import": re.compile(r'import\s+.*?\s+from\s+["\']([^"\']+)["\']'),
            "require": re.compile(r'require\(["\']([^"\']+)["\']\)'),
            "dynamic": re.compile(r'import\(["\']([^"\']+)["\']\)'),
        },
        Language.TYPESCRIPT: {
            "function": re.compile(
                r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)"
            ),
            "arrow": re.compile(
                r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>"
            ),
            "class": re.compile(r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?"),
            "import": re.compile(r'import\s+.*?\s+from\s+["\']([^"\']+)["\']'),
            "require": re.compile(r'require\(["\']([^"\']+)["\']\)'),
            "dynamic": re.compile(r'import\(["\']([^"\']+)["\']\)'),
        },
        Language.GO: {
            # Plain function: func FuncName(...)
            "function": re.compile(r"^func\s+(\w+)\s*\(", re.MULTILINE),
            # Receiver method: func (r *Type) MethodName(...) — captures Type.MethodName
            "method": re.compile(r"^func\s+\(\w+\s+\*?(\w+)\)\s+(\w+)\s*\(", re.MULTILINE),
            # Struct type definition
            "class": re.compile(r"^type\s+(\w+)\s+struct\s*\{", re.MULTILINE),
            # Interface type definition
            "interface": re.compile(r"^type\s+(\w+)\s+interface\s*\{", re.MULTILINE),
            "import": re.compile(r'"([^"]+)"'),
            "dynamic": re.compile(r'plugin\.Open\(["\']([^"\']+)["\']\)'),
        },
        Language.RUST: {
            "function": re.compile(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(([^)]*)\)"),
            "import": re.compile(r"use\s+([\w:]+)"),
            "dynamic": re.compile(r'libloading::Library::new\(["\']([^"\']+)["\']\)'),
        },
    }

    def parse_file(self, path: str, source: str) -> FileNode:
        """Parse a generic source file.

        Args:
            path: File path.
            source: Source text.

        Returns:
            FileNode with extracted symbols.
        """
        lang = detect_language(path)
        pats = self._PATTERNS.get(lang, {})
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []

        module_id = f"{path}::__module__"
        symbols.append(GraphNode(
            node_id=module_id,
            file_path=path,
            kind=NodeKind.MODULE,
            name=Path(path).stem,
            qualified_name=Path(path).stem,
            language=lang,
            source_confidence=SourceConfidence.INFERRED,
        ))

        for key, pat in pats.items():
            for m in pat.finditer(source):
                if key == "import" or key == "require":
                    name = m.group(1)
                    if name not in imports:
                        imports.append(name)
                elif key == "dynamic":
                    name = m.group(1)
                    if name not in dynamic_imports:
                        dynamic_imports.append(name)
                elif key == "method":
                    # Go receiver method: group(1)=Type, group(2)=MethodName
                    type_name = m.group(1)
                    method_name = m.group(2)
                    qual = f"{type_name}.{method_name}"
                    sig = m.group(0)[:120]
                    line = source[: m.start()].count("\n") + 1
                    symbols.append(GraphNode(
                        node_id=f"{path}::{qual}",
                        file_path=path,
                        kind=NodeKind.METHOD,
                        name=method_name,
                        qualified_name=qual,
                        line_start=line,
                        language=lang,
                        signature=sig,
                        metadata={"receiver_type": type_name},
                        source_confidence=SourceConfidence.INFERRED,
                    ))
                elif key == "interface":
                    name = m.group(1)
                    sig = m.group(0)[:120]
                    line = source[: m.start()].count("\n") + 1
                    symbols.append(GraphNode(
                        node_id=f"{path}::{name}",
                        file_path=path,
                        kind=NodeKind.CLASS,
                        name=name,
                        qualified_name=name,
                        line_start=line,
                        language=lang,
                        signature=sig,
                        metadata={"is_interface": True},
                        source_confidence=SourceConfidence.INFERRED,
                    ))
                else:
                    name = m.group(1)
                    kind = NodeKind.CLASS if key == "class" else NodeKind.FUNCTION
                    sig = m.group(0)[:120]
                    line = source[: m.start()].count("\n") + 1
                    symbols.append(GraphNode(
                        node_id=f"{path}::{name}",
                        file_path=path,
                        kind=kind,
                        name=name,
                        qualified_name=name,
                        line_start=line,
                        language=lang,
                        signature=sig,
                        is_async="async" in sig,
                        source_confidence=SourceConfidence.INFERRED,
                    ))

        sha = hashlib.sha256(source.encode(errors="replace")).hexdigest()
        return FileNode(
            path=path,
            language=lang,
            size_bytes=len(source.encode(errors="replace")),
            line_count=len(source.splitlines()),
            sha256=sha,
            symbols=symbols,
            imports=imports,
            dynamic_imports=dynamic_imports,
            token_estimate=estimate_tokens(source),
        )

    def extract_signatures(self, source: str) -> str:
        """Return first 60 lines — signature approximation for generic files."""
        lines = source.splitlines()[:60]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BashParser — shell script parser (.sh, .bash, .zsh)
# ---------------------------------------------------------------------------


class BashParser:
    """Regex-based parser for Bash/Shell scripts.

    Extracts: function definitions, sourced files (. / source), variable
    assignments, and dynamic eval/exec patterns.

    Fixes code-review-graph gap: shell scripts were completely unindexed,
    meaning infra/deploy scripts were invisible to context selection.

    Args:
        None
    """

    _PATTERNS = {
        # function name() { or function name {
        "function": re.compile(r"^(?:function\s+)?(\w+)\s*\(\s*\)\s*\{", re.MULTILINE),
        # source ./file or . ./file
        "source": re.compile(r"^(?:source|\.)\s+([\w./\-]+)", re.MULTILINE),
        # export VAR= or VAR=
        "variable": re.compile(r"^(?:export\s+)?([A-Z_][A-Z0-9_]{2,})\s*=", re.MULTILINE),
        # eval or $(command) dynamic exec
        "dynamic": re.compile(r"\beval\s+[\"'`]([^\"'`]+)[\"'`]", re.MULTILINE),
    }

    def parse_file(self, path: str, source: str) -> FileNode:
        """Parse a shell script.

        Args:
            path: File path.
            source: Shell script source text.

        Returns:
            FileNode with extracted symbols.
        """
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []

        module_id = f"{path}::__module__"
        symbols.append(GraphNode(
            node_id=module_id,
            file_path=path,
            kind=NodeKind.MODULE,
            name=Path(path).stem,
            qualified_name=Path(path).stem,
            language=Language.BASH,
            source_confidence=SourceConfidence.INFERRED,
        ))

        for key, pat in self._PATTERNS.items():
            for m in pat.finditer(source):
                name = m.group(1)
                line = source[: m.start()].count("\n") + 1
                if key == "source":
                    if name not in imports:
                        imports.append(name)
                elif key == "dynamic":
                    if name not in dynamic_imports:
                        dynamic_imports.append(name[:80])
                elif key == "variable":
                    symbols.append(GraphNode(
                        node_id=f"{path}::{name}",
                        file_path=path,
                        kind=NodeKind.VARIABLE,
                        name=name,
                        qualified_name=name,
                        line_start=line,
                        language=Language.BASH,
                        source_confidence=SourceConfidence.INFERRED,
                    ))
                else:
                    symbols.append(GraphNode(
                        node_id=f"{path}::{name}",
                        file_path=path,
                        kind=NodeKind.FUNCTION,
                        name=name,
                        qualified_name=name,
                        line_start=line,
                        language=Language.BASH,
                        signature=f"function {name}()",
                        source_confidence=SourceConfidence.INFERRED,
                    ))

        sha = hashlib.sha256(source.encode(errors="replace")).hexdigest()
        return FileNode(
            path=path,
            language=Language.BASH,
            size_bytes=len(source.encode(errors="replace")),
            line_count=len(source.splitlines()),
            sha256=sha,
            symbols=symbols,
            imports=imports,
            dynamic_imports=dynamic_imports,
            token_estimate=estimate_tokens(source),
        )

    def extract_signatures(self, source: str) -> str:
        """Return function definitions only."""
        lines = []
        for m in self._PATTERNS["function"].finditer(source):
            lines.append(f"function {m.group(1)}()")
        return "\n".join(lines) if lines else source[:200]


# ---------------------------------------------------------------------------
# HCLParser — Terraform / OpenTofu parser (.tf, .tfvars, .hcl)
# ---------------------------------------------------------------------------


class HCLParser:
    """Regex-based parser for HCL (HashiCorp Configuration Language).

    Extracts: resource blocks, data blocks, module calls, variable
    declarations, output blocks, and locals.

    Fixes code-review-graph gap: Terraform/HCL files were not indexed,
    meaning infra code changes were invisible to context selection.
    (Requested in code-review-graph issue #199.)

    Args:
        None
    """

    _PATTERNS = {
        # resource "aws_s3_bucket" "my_bucket" {
        "resource": re.compile(
            r'^resource\s+"([^"]+)"\s+"([^"]+)"\s*\{', re.MULTILINE
        ),
        # data "aws_ami" "ubuntu" {
        "data": re.compile(
            r'^data\s+"([^"]+)"\s+"([^"]+)"\s*\{', re.MULTILINE
        ),
        # module "vpc" {
        "module": re.compile(r'^module\s+"([^"]+)"\s*\{', re.MULTILINE),
        # variable "instance_type" {
        "variable": re.compile(r'^variable\s+"([^"]+)"\s*\{', re.MULTILINE),
        # output "bucket_arn" {
        "output": re.compile(r'^output\s+"([^"]+)"\s*\{', re.MULTILINE),
        # source = "..." inside module blocks (treated as import)
        "source": re.compile(r'^\s*source\s*=\s*"([^"]+)"', re.MULTILINE),
    }

    def parse_file(self, path: str, source: str) -> FileNode:
        """Parse a Terraform/HCL file.

        Args:
            path: File path.
            source: HCL source text.

        Returns:
            FileNode with extracted symbols.
        """
        symbols: list[GraphNode] = []
        imports: list[str] = []

        module_id = f"{path}::__module__"
        symbols.append(GraphNode(
            node_id=module_id,
            file_path=path,
            kind=NodeKind.MODULE,
            name=Path(path).stem,
            qualified_name=Path(path).stem,
            language=Language.HCL,
            source_confidence=SourceConfidence.INFERRED,
        ))

        for key, pat in self._PATTERNS.items():
            for m in pat.finditer(source):
                line = source[: m.start()].count("\n") + 1
                if key == "source":
                    src = m.group(1)
                    if src not in imports:
                        imports.append(src)
                elif key in ("resource", "data"):
                    # name = "type.label"
                    resource_type = m.group(1)
                    label = m.group(2)
                    qual = f"{resource_type}.{label}"
                    symbols.append(GraphNode(
                        node_id=f"{path}::{qual}",
                        file_path=path,
                        kind=NodeKind.CLASS,
                        name=label,
                        qualified_name=qual,
                        line_start=line,
                        language=Language.HCL,
                        signature=m.group(0)[:120],
                        metadata={"hcl_block": key, "resource_type": resource_type},
                        source_confidence=SourceConfidence.INFERRED,
                    ))
                elif key == "variable":
                    name = m.group(1)
                    symbols.append(GraphNode(
                        node_id=f"{path}::var.{name}",
                        file_path=path,
                        kind=NodeKind.VARIABLE,
                        name=name,
                        qualified_name=f"var.{name}",
                        line_start=line,
                        language=Language.HCL,
                        metadata={"hcl_block": "variable"},
                        source_confidence=SourceConfidence.INFERRED,
                    ))
                else:
                    name = m.group(1)
                    kind = NodeKind.FUNCTION if key == "output" else NodeKind.MODULE
                    symbols.append(GraphNode(
                        node_id=f"{path}::{key}.{name}",
                        file_path=path,
                        kind=kind,
                        name=name,
                        qualified_name=f"{key}.{name}",
                        line_start=line,
                        language=Language.HCL,
                        signature=m.group(0)[:120],
                        metadata={"hcl_block": key},
                        source_confidence=SourceConfidence.INFERRED,
                    ))

        sha = hashlib.sha256(source.encode(errors="replace")).hexdigest()
        return FileNode(
            path=path,
            language=Language.HCL,
            size_bytes=len(source.encode(errors="replace")),
            line_count=len(source.splitlines()),
            sha256=sha,
            symbols=symbols,
            imports=imports,
            dynamic_imports=[],
            token_estimate=estimate_tokens(source),
        )

    def extract_signatures(self, source: str) -> str:
        """Return resource/variable block headers only."""
        lines = []
        for key in ("resource", "data", "module", "variable", "output"):
            for m in self._PATTERNS[key].finditer(source):
                lines.append(m.group(0).rstrip("{").strip())
        return "\n".join(lines) if lines else source[:200]


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------


_PARSER_REGISTRY: dict[Language, LanguageParser] = {
    Language.PYTHON: PythonParser(),
    Language.JAVASCRIPT: GenericParser(),
    Language.TYPESCRIPT: GenericParser(),
    Language.GO: GenericParser(),
    Language.RUST: GenericParser(),
    Language.JAVA: GenericParser(),
    Language.CPP: GenericParser(),
    Language.C: GenericParser(),
    Language.RUBY: GenericParser(),
    Language.PHP: GenericParser(),
    Language.BASH: BashParser(),
    Language.HCL: HCLParser(),
    Language.HELM: GenericParser(),  # Helm = YAML+Go template; generic parser covers basics
}


def get_parser(language: Language) -> LanguageParser:
    """Get the parser for a language.

    By default this returns the regex-based ``GenericParser`` (or
    ``PythonParser`` / ``BashParser`` / ``HCLParser`` for those languages).
    For AST-accurate parsing across 11 languages, register the optional
    ``TreeSitterParser``::

        from graphsift.parsers import TreeSitterParser, register_tree_sitter_parsers

        # Option A: register for all available grammars
        register_tree_sitter_parsers()

        # Option B: single-language opt-in
        from graphsift import register_parser
        register_parser(Language.PYTHON, TreeSitterParser())

    Args:
        language: Target language.

    Returns:
        LanguageParser implementation.

    Raises:
        LanguageNotSupportedError: If no parser is registered.
    """
    if language not in _PARSER_REGISTRY:
        raise LanguageNotSupportedError(f"No parser registered for {language.value}.")
    return _PARSER_REGISTRY[language]


def register_parser(language: Language, parser: LanguageParser) -> None:
    """Register a custom parser for a language.

    Allows callers to inject tree-sitter or other parsers without
    modifying library internals.

    Args:
        language: Language to register for.
        parser: Parser implementation (must satisfy LanguageParser protocol).
    """
    _PARSER_REGISTRY[language] = parser


# ---------------------------------------------------------------------------
# DependencyGraph — in-memory directed graph
# ---------------------------------------------------------------------------


class DependencyGraph:
    """Thread-safe in-memory directed dependency graph.

    Fixes over code-review-graph:
    - Ranked traversal (BFS with score decay by depth)
    - Multi-source BFS (union of multiple changed files)
    - Decorator edges (DECORATES kind)
    - Dynamic import edges (DYNAMIC_IMPORT kind)
    - Configurable depth cap (no infinite traversal hangs)

    Args:
        decay: Score multiplier per hop (0.7 = 30% decay each level).
        max_depth: Hard cap on BFS depth.
    """

    __slots__ = ("_nodes", "_edges", "_adj_out", "_adj_in", "_file_nodes", "_lock", "_decay", "_max_depth")

    def __init__(self, decay: float = 0.7, max_depth: int = 4) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._adj_out: dict[str, list[GraphEdge]] = defaultdict(list)
        self._adj_in: dict[str, list[GraphEdge]] = defaultdict(list)
        self._file_nodes: dict[str, FileNode] = {}
        self._lock = threading.RLock()
        self._decay = decay
        self._max_depth = max_depth

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"DependencyGraph(nodes={len(self._nodes)}, "
                f"edges={len(self._edges)}, "
                f"files={len(self._file_nodes)})"
            )

    def add_file(self, file_node: FileNode) -> None:
        """Add a parsed file and all its symbols to the graph.

        Args:
            file_node: Parsed FileNode.
        """
        with self._lock:
            self._file_nodes[file_node.path] = file_node
            for sym in file_node.symbols:
                self._nodes[sym.node_id] = sym

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a dependency edge.

        Args:
            edge: GraphEdge to add.
        """
        with self._lock:
            self._edges.append(edge)
            self._adj_out[edge.source_id].append(edge)
            self._adj_in[edge.target_id].append(edge)

    def build_import_edges(self) -> int:
        """Build IMPORTS and DYNAMIC_IMPORT edges from file import lists.

        Resolves import strings to file paths in the graph.
        Returns number of edges created.
        """
        created = 0
        with self._lock:
            path_index = self._build_path_index()

            for file_node in self._file_nodes.values():
                src_module_id = f"{file_node.path}::__module__"

                for imp in file_node.imports:
                    targets = self._resolve_import(imp, path_index)
                    for tgt_path in targets:
                        tgt_id = f"{tgt_path}::__module__"
                        edge = GraphEdge(
                            source_id=src_module_id,
                            target_id=tgt_id,
                            kind=EdgeKind.IMPORTS,
                        )
                        self._edges.append(edge)
                        self._adj_out[src_module_id].append(edge)
                        self._adj_in[tgt_id].append(edge)
                        created += 1

                for dyn in file_node.dynamic_imports:
                    targets = self._resolve_import(dyn, path_index)
                    for tgt_path in targets:
                        tgt_id = f"{tgt_path}::__module__"
                        edge = GraphEdge(
                            source_id=src_module_id,
                            target_id=tgt_id,
                            kind=EdgeKind.DYNAMIC_IMPORT,
                            weight=0.6,  # lower weight — dynamic = uncertain
                        )
                        self._edges.append(edge)
                        self._adj_out[src_module_id].append(edge)
                        self._adj_in[tgt_id].append(edge)
                        created += 1

        return created

    def build_inheritance_edges(self) -> int:
        """Build INHERITS edges from class base lists. Returns edge count."""
        created = 0
        with self._lock:
            name_index: dict[str, str] = {}
            for node in self._nodes.values():
                name_index[node.name] = node.node_id
                name_index[node.qualified_name] = node.node_id

            for node in self._nodes.values():
                if node.kind != NodeKind.CLASS:
                    continue
                for base in node.metadata.get("bases", []):
                    if base in name_index:
                        edge = GraphEdge(
                            source_id=node.node_id,
                            target_id=name_index[base],
                            kind=EdgeKind.INHERITS,
                            weight=1.5,  # inheritance = strong coupling
                        )
                        self._edges.append(edge)
                        self._adj_out[node.node_id].append(edge)
                        self._adj_in[name_index[base]].append(edge)
                        created += 1

        return created

    def build_decorator_edges(self) -> int:
        """Build DECORATES edges. Returns edge count.

        Fixes code-review-graph gap: decorator calls are tracked as edges
        so callers of decorated functions are properly propagated.
        """
        created = 0
        with self._lock:
            name_index: dict[str, str] = {}
            for node in self._nodes.values():
                name_index[node.name] = node.node_id
                name_index[node.qualified_name] = node.node_id

            for node in self._nodes.values():
                for dec_name in node.decorators:
                    base = dec_name.split(".")[0]
                    if base in name_index:
                        edge = GraphEdge(
                            source_id=node.node_id,
                            target_id=name_index[base],
                            kind=EdgeKind.DECORATES,
                            weight=0.8,
                        )
                        self._edges.append(edge)
                        self._adj_out[node.node_id].append(edge)
                        self._adj_in[name_index[base]].append(edge)
                        created += 1

        return created

    def ranked_neighbors(
        self,
        seed_paths: list[str],
        include_dynamic: bool = True,
    ) -> dict[str, tuple[float, int, list[str]]]:
        """BFS from seed files, scoring each reachable file by relevance.

        Improvements over code-review-graph's binary blast-radius:
        - Score decays by depth (depth 1 = 1.0, depth 2 = 0.7, depth 3 = 0.49...)
        - Edge weights modulate the score (inheritance stronger than dynamic imports)
        - Returns score, depth, and reasons per file

        Args:
            seed_paths: Changed file paths (multi-file diff supported).
            include_dynamic: Whether to traverse DYNAMIC_IMPORT edges.

        Returns:
            Dict mapping file_path → (score, depth, reasons).
        """
        with self._lock:
            excluded_kinds = set()
            if not include_dynamic:
                excluded_kinds.add(EdgeKind.DYNAMIC_IMPORT)

            # file_path → (score, depth, reasons)
            scores: dict[str, tuple[float, int, list[str]]] = {}

            # Seed files get score 1.0, depth 0
            for p in seed_paths:
                if p in self._file_nodes:
                    scores[p] = (1.0, 0, ["directly changed"])

            # BFS over module-level nodes
            queue: deque[tuple[str, float, int]] = deque()
            visited: set[str] = set()

            for p in seed_paths:
                seed_id = f"{p}::__module__"
                if seed_id in self._nodes:
                    queue.append((seed_id, 1.0, 0))
                    visited.add(seed_id)

            while queue:
                node_id, score, depth = queue.popleft()
                if depth >= self._max_depth:
                    continue

                # Traverse outgoing edges (who does this file depend on)
                for edge in self._adj_out.get(node_id, []):
                    if edge.kind in excluded_kinds:
                        continue
                    self._update_score(
                        edge.target_id, score, depth, edge, scores, queue, visited,
                        direction="depends_on"
                    )

                # Traverse incoming edges (who depends on this file — callers)
                for edge in self._adj_in.get(node_id, []):
                    if edge.kind in excluded_kinds:
                        continue
                    self._update_score(
                        edge.source_id, score, depth, edge, scores, queue, visited,
                        direction="caller"
                    )

            return scores

    def _update_score(
        self,
        neighbor_id: str,
        parent_score: float,
        depth: int,
        edge: GraphEdge,
        scores: dict[str, tuple[float, int, list[str]]],
        queue: deque[tuple[str, float, int]],
        visited: set[str],
        direction: str,
    ) -> None:
        if neighbor_id not in self._nodes:
            return
        neighbor_node = self._nodes[neighbor_id]
        neighbor_file = neighbor_node.file_path
        new_score = parent_score * self._decay * edge.weight
        new_depth = depth + 1
        reason = f"{direction} via {edge.kind.value} (depth {new_depth})"

        existing = scores.get(neighbor_file)
        if existing is None or new_score > existing[0]:
            reasons = list(existing[2]) if existing else []
            reasons.append(reason)
            scores[neighbor_file] = (new_score, new_depth, reasons)

        if neighbor_id not in visited:
            visited.add(neighbor_id)
            queue.append((neighbor_id, new_score, new_depth))

    def get_file(self, path: str) -> FileNode | None:
        """Retrieve an indexed FileNode by path."""
        with self._lock:
            return self._file_nodes.get(path)

    def all_files(self) -> list[FileNode]:
        """Return all indexed FileNodes."""
        with self._lock:
            return list(self._file_nodes.values())

    def stats(self) -> dict[str, int]:
        """Return graph statistics."""
        with self._lock:
            return {
                "nodes": len(self._nodes),
                "edges": len(self._edges),
                "files": len(self._file_nodes),
            }

    def _build_path_index(self) -> dict[str, str]:
        """Build module-name → file-path index for import resolution."""
        idx: dict[str, str] = {}
        for path in self._file_nodes:
            p = Path(path)
            # e.g. src/foo/bar.py → foo.bar, bar
            parts = list(p.with_suffix("").parts)
            idx[p.stem] = path
            idx[".".join(parts[-3:])] = path
            idx[".".join(parts[-2:])] = path
            idx[".".join(parts)] = path
        return idx

    @staticmethod
    def _resolve_import(imp: str, path_index: dict[str, str]) -> list[str]:
        """Resolve an import string to file paths."""
        results = []
        # Exact match
        if imp in path_index:
            results.append(path_index[imp])
        # Prefix match (e.g. "mypackage.module" → "mypackage/module.py")
        for key, path in path_index.items():
            if key.startswith(imp) or imp.startswith(key):
                if path not in results:
                    results.append(path)
        return results[:3]  # cap false-positive explosion

    def detect_cycles(self) -> list[list[str]]:
        """Find all dependency cycles using Tarjan's strongly-connected components.

        Returns a list of cycles, each a list of file paths in the cycle.
        Only returns cycles of length >= 2 (self-loops excluded).
        """
        # Build adjacency list: file_path -> set of dependent file_paths
        adj: dict[str, set[str]] = defaultdict(set)
        all_files: set[str] = set()

        for edge in self._edges:
            src_file = self._resolve_file(edge.source_id)
            tgt_file = self._resolve_file(edge.target_id)
            if src_file and tgt_file and src_file != tgt_file:
                adj[src_file].add(tgt_file)
                all_files.add(src_file)
                all_files.add(tgt_file)

        # Also add files with no edges
        for node in self._nodes.values():
            all_files.add(node.file_path)

        # Tarjan's SCC
        index_counter = [0]
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        on_stack: dict[str, bool] = defaultdict(bool)
        stack: list[str] = []
        cycles: list[list[str]] = []

        def strongconnect(v: str) -> None:
            indices[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in adj.get(v, set()):
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                scc: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) >= 2:
                    cycles.append(scc)

        for f in list(all_files):
            if f not in indices:
                strongconnect(f)

        return cycles

    def _resolve_file(self, node_id: str) -> str | None:
        """Resolve a node_id back to its file_path."""
        for node in self._nodes.values():
            if node.node_id == node_id:
                return node.file_path
        # node_id might already be a file path
        return node_id if "/" in node_id or "\\" in node_id else None

    def find_dead_code(
        self,
        entry_points: list[str] | None = None,
        kind: str | None = None,
    ) -> list[dict]:
        """Find potentially dead code via BFS reachability from entry points.

        Args:
            entry_points: Known entry-point files (main, app factory, CLI entry).
                          If None, auto-detects files with __main__ or main() patterns.
            kind: Filter to 'function', 'class', 'method', or None for all.

        Returns:
            List of dicts with node_id, file_path, name, kind, line_start, line_end, reason.
        """
        if entry_points is None:
            entry_points = self._detect_entry_points()

        # Build reachable set via BFS across edges
        reachable: set[str] = set()
        queue: deque[str] = deque()

        # Seed with entry-point files
        entry_files: set[str] = set()
        for ep in entry_points:
            for node in self._nodes.values():
                if node.file_path == ep or ep in node.file_path:
                    entry_files.add(node.file_path)
                    if node.node_id not in reachable:
                        reachable.add(node.node_id)
                        queue.append(node.node_id)

        # BFS traversal
        # Build adjacency: source_node_id -> [target_node_ids]
        forward: dict[str, list[str]] = defaultdict(list)
        for edge in self._edges:
            forward[edge.source_id].append(edge.target_id)

        while queue:
            current = queue.popleft()
            for target in forward.get(current, []):
                if target not in reachable:
                    reachable.add(target)
                    queue.append(target)

        # Collect unreachable nodes
        dead: list[dict] = []
        for node in self._nodes.values():
            if node.node_id not in reachable:
                if node.file_path in entry_files:
                    continue  # entry points are reachable by definition
                if kind and node.kind.value != kind:
                    continue
                dead.append({
                    "node_id": node.node_id,
                    "file_path": node.file_path,
                    "name": node.name,
                    "kind": node.kind.value,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                    "reason": "No reachable path from any entry point",
                })

        return dead

    def _detect_entry_points(self) -> list[str]:
        """Auto-detect entry-point files in the graph.

        Detects entry points for:
          - **Python**: ``__main__.py``, ``main()`` functions, Flask/FastAPI
            route decorators, Click command groups, Django URL configs,
            ``pyproject.toml`` script references, ``app`` / ``application``
            WSGI/ASGI objects.
          - **Next.js** (JS/TS): Files under ``pages/``, ``app/**/page.tsx``,
            ``layout.tsx``, ``loading.tsx``, ``error.tsx``, ``route.ts``,
            ``middleware.ts``, ``next.config.*``.
          - **React / general JS/TS**: ``index.tsx`` / ``index.js`` (app
            entry), ``main.tsx`` / ``main.js`` (Vite/CRA entry),
            ``export default`` components at module root, ``App.tsx``.
          - **General**: ``main.go``, ``main.rs``, ``index.js``,
            ``Main.java``, and any file with the most incoming edges (hub
            fallback).
        """
        entry_files: list[str] = []
        seen: set[str] = set()

        # --- Reusable helper ---
        def _add(fp: str) -> None:
            if fp not in seen:
                entry_files.append(fp)
                seen.add(fp)

        for node in self._nodes.values():
            fp = node.file_path
            if fp in seen:
                continue

            # ----------------------------------------------------------
            # Python entry points
            # ----------------------------------------------------------
            if fp.endswith("__main__.py") or fp.endswith("main.py"):
                _add(fp)
                continue

            # Flask / FastAPI style: @app.route(...) or @router.get(...)
            if node.kind == NodeKind.FUNCTION and node.decorators:
                for deco in node.decorators:
                    if any(
                        pat in deco
                        for pat in (".route(", ".get(", ".post(", ".put(",
                                    ".patch(", ".delete(", ".options(")
                    ):
                        _add(fp)
                        break

            # Python Click command groups (@click.group, @click.command)
            if node.kind == NodeKind.FUNCTION and node.decorators:
                for deco in node.decorators:
                    if "click.group" in deco or "click.command" in deco:
                        _add(fp)
                        break

            # WSGI/ASGI application objects
            if node.name in ("app", "application") and node.kind in (
                NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.VARIABLE
            ):
                _add(fp)
                continue

            # Django: urls.py with urlpatterns
            if fp.endswith("urls.py"):
                _add(fp)
                continue

            # ----------------------------------------------------------
            # Next.js / React / JS / TS entry points
            # ----------------------------------------------------------
            ext = Path(fp).suffix.lower()
            is_js_like = ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")

            if is_js_like:
                # Next.js App Router: app/**/page.tsx, layout.tsx, loading.tsx,
                # error.tsx, not-found.tsx, route.ts (API routes)
                if re.search(r"(?:^|[/\\])app[/\\]", fp):
                    basename = Path(fp).stem
                    if basename in (
                        "page", "layout", "loading", "error",
                        "not-found", "route", "global-error",
                        "template", "default",
                    ):
                        _add(fp)
                        continue

                # Next.js Pages Router: pages/**/*.tsx
                if re.search(r"(?:^|[/\\])pages[/\\]", fp):
                    _add(fp)
                    continue

                # Next.js middleware / config / instrumentation
                if Path(fp).name in (
                    "middleware.ts", "middleware.js",
                    "next.config.ts", "next.config.js", "next.config.mjs",
                    "instrumentation.ts", "instrumentation.js",
                    "sitemap.ts", "sitemap.js",
                ):
                    _add(fp)
                    continue

                # Vite / CRA / general: main.tsx, main.jsx, index.tsx,
                # index.js, App.tsx, App.jsx
                basename = Path(fp).stem
                if basename in ("main", "index", "App", "app") and ext in (
                    ".tsx", ".jsx", ".ts", ".js"
                ):
                    _add(fp)
                    continue

                # JS/TS exported default component (function returning JSX)
                if node.kind == NodeKind.FUNCTION and "default" in getattr(
                    node, "signature", ""
                ):
                    _add(fp)
                    continue

            # ----------------------------------------------------------
            # Other language entry points
            # ----------------------------------------------------------
            if fp.endswith("main.go"):
                _add(fp)
                continue

            if fp.endswith("main.rs"):
                _add(fp)
                continue

            if fp.endswith("Main.java"):
                _add(fp)
                continue

        # --- Fallback: files with most incoming edges (hub nodes) ---
        if not entry_files:
            incoming: dict[str, int] = defaultdict(int)
            for edge in self._edges:
                tgt_file = self._resolve_file(edge.target_id)
                if tgt_file:
                    incoming[tgt_file] += 1
            if incoming:
                top = sorted(incoming.items(), key=lambda x: x[1], reverse=True)[:3]
                for f, _ in top:
                    _add(f)

        return entry_files


# ---------------------------------------------------------------------------
# RelevanceRanker — multi-signal scoring
# ---------------------------------------------------------------------------


class RelevanceRanker:
    """Multi-signal relevance ranker for file selection.

    Signals used (fixes code-review-graph's binary include/exclude):
    1. Graph distance score (from DependencyGraph.ranked_neighbors)
    2. BM25-style keyword overlap with query/commit message
    3. Test file bonus (tests covering changed code are always relevant)
    4. Decorator proximity bonus (decorators of changed functions)
    5. Dynamic import penalty (uncertain deps get lower weight)
    6. File size penalty (huge files score lower unless directly changed)
    7. God-node penalty (highly-connected utility files penalized to avoid
       crowding out domain-specific business logic)

    Args:
        bm25_weight: Weight for BM25 keyword signal (0–1).
        graph_weight: Weight for graph distance signal (0–1).
        god_node_penalty: Centrality penalty strength (0-1, default 0.3).
    """

    # Generic utility file patterns — God-node candidates
    _UTILITY_PATTERNS = frozenset({
        "utils", "util", "helpers", "helper", "common", "config",
        "settings", "constants", "const", "base", "logger", "logging",
        "decorators", "exceptions", "middleware", "types",
    })

    def __init__(self, bm25_weight: float = 0.3, graph_weight: float = 0.7, god_node_penalty: float = 0.3) -> None:
        self._bm25_w = bm25_weight
        self._graph_w = graph_weight
        self._god_node_penalty = god_node_penalty
        self._centrality_penalty: dict[str, float] = {}

    def __repr__(self) -> str:
        return f"RelevanceRanker(bm25={self._bm25_w}, graph={self._graph_w}, god_node={self._god_node_penalty})"

    @staticmethod
    def _is_generic_utility(path: str) -> bool:
        """Check if a file path matches generic utility patterns (God-node)."""
        p = Path(path)
        stem = p.stem.lower()
        # Direct filename match
        if stem in RelevanceRanker._UTILITY_PATTERNS:
            return True
        # Parent directory match (e.g. src/utils/auth.py)
        for part in p.parts:
            if part.lower() in RelevanceRanker._UTILITY_PATTERNS:
                return True
        return False

    def _compute_centrality_penalty(
        self,
        graph_scores: dict[str, tuple[float, int, list[str]]],
    ) -> dict[str, float]:
        """Compute in-degree centrality penalty for each file.

        Files imported by many others get a higher penalty to prevent
        utility-heavy files from dominating the context window.

        Returns dict of file_path -> centrality_penalty (0.0 to 1.0).
        """
        if not graph_scores:
            return {}
        max_score = max((s[0] for s in graph_scores.values()), default=0.0)
        if max_score == 0.0:
            return {}
        return {
            path: score / max_score
            for path, (score, _, _) in graph_scores.items()
            if score > 0
        }

    def rank(
        self,
        diff_spec: DiffSpec,
        graph_scores: dict[str, tuple[float, int, list[str]]],
        all_files: list[FileNode],
        config: ContextConfig,
    ) -> list[ScoredFile]:
        """Rank all files by relevance to the diff.

        Args:
            diff_spec: The diff specification.
            graph_scores: Output of DependencyGraph.ranked_neighbors.
            all_files: All indexed FileNodes.
            config: Context configuration.

        Returns:
            List of ScoredFile sorted by score descending.
        """
        query_tokens = self._tokenize(
            diff_spec.diff_text + " " + diff_spec.commit_message + " " + diff_spec.query
        )
        changed_set = set(diff_spec.changed_files)

        # Pre-compute God-node centrality penalty
        self._centrality_penalty = self._compute_centrality_penalty(graph_scores)

        scored: list[ScoredFile] = []
        for fnode in all_files:
            path = fnode.path

            # Graph score
            g_score, depth, reasons = graph_scores.get(path, (0.0, 99, []))

            # Changed files always get 1.0
            if path in changed_set:
                g_score = 1.0
                depth = 0
                reasons = ["directly changed"]

            if g_score < config.min_score and path not in changed_set:
                continue

            # BM25-style keyword signal
            bm25 = self._bm25_score(fnode, query_tokens)

            # Combined score
            combined = self._graph_w * g_score + self._bm25_w * bm25

            # Test file bonus
            if self._is_test(path):
                if not config.include_tests:
                    continue
                if g_score > 0:
                    combined = min(1.0, combined + 0.15)
                    reasons.append("test coverage bonus")

            # Dynamic import penalty
            if any("dynamic_import" in r for r in reasons):
                combined *= 0.8

            # Size penalty (files > 1000 lines)
            if fnode.line_count > 1000 and path not in changed_set:
                combined *= 0.85
                reasons.append("size penalty")

            # ── God-node centrality penalty ──────────────────────────
            # Highly-connected utility files penalized to preserve context
            # for domain-specific business logic.
            if path not in changed_set:
                centrality_pen = self._centrality_penalty.get(path, 0.0)
                utility_pen = 0.2 if self._is_generic_utility(path) else 0.0
                if centrality_pen > 0 or utility_pen > 0:
                    combined *= (1.0 - centrality_pen * self._god_node_penalty) * (1.0 - utility_pen)
                    if utility_pen > 0:
                        reasons.append("god-node utility penalty")
                    if centrality_pen > 0.3:
                        reasons.append(f"centrality penalty ({centrality_pen:.2f})")

            combined = min(1.0, max(0.0, combined))

            # Determine output mode with 3-tier HOT/WARM/COLD routing
            # DepthTier overrides thresholds for different development phases
            if config.depth_tier == DepthTier.PLANNING:
                hot_threshold = 0.95  # Only absolute core gets full source
                warm_threshold = 0.4
                default_mode = OutputMode.SIGNATURES
                reasons.append("planning-tier")
            elif config.depth_tier == DepthTier.EXPLORATION:
                hot_threshold = 0.85
                warm_threshold = 0.2
                default_mode = OutputMode.SIGNATURES
                reasons.append("exploration-tier")
            else:  # EXECUTION
                hot_threshold = getattr(config, 'hot_threshold', 0.8)
                warm_threshold = getattr(config, 'warm_threshold', 0.25)
                default_mode = OutputMode.SMART

            if config.output_mode == OutputMode.SMART:
                if combined >= hot_threshold:
                    mode = OutputMode.FULL
                    reasons.append(f"HOT(tier={combined:.2f})")
                elif combined >= warm_threshold:
                    mode = OutputMode.SIGNATURES
                    reasons.append(f"WARM(tier={combined:.2f})")
                else:
                    mode = default_mode
                    reasons.append(f"COLD(tier={combined:.2f})")
            else:
                mode = config.output_mode

            # Compute file-level confidence from symbol-level granularity
            file_confidence = SourceConfidence.EXTRACTED
            for sym in fnode.symbols:
                if sym.source_confidence == SourceConfidence.INFERRED:
                    file_confidence = SourceConfidence.INFERRED
                    break

            scored.append(ScoredFile(
                file_node=fnode,
                score=round(combined, 4),
                rank=0,
                reasons=reasons,
                depth=depth,
                output_mode=mode,
                source_confidence=file_confidence,
            ))

        # Sort by score descending, assign ranks
        scored.sort(key=lambda s: s.score, reverse=True)
        ranked = []
        for i, sf in enumerate(scored):
            ranked.append(ScoredFile(
                file_node=sf.file_node,
                score=sf.score,
                rank=i + 1,
                reasons=sf.reasons,
                depth=sf.depth,
                output_mode=sf.output_mode,
                source_confidence=sf.source_confidence,
            ))

        return ranked

    @staticmethod
    def _tokenize(text: str) -> dict[str, int]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        freq: dict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        return dict(freq)

    @staticmethod
    def _bm25_score(fnode: FileNode, query_tokens: dict[str, int]) -> float:
        if not query_tokens:
            return 0.0
        # Use symbol names and import names as document terms
        doc_terms: list[str] = []
        for sym in fnode.symbols:
            doc_terms.extend(sym.name.lower().split("_"))
        doc_terms.extend(Path(fnode.path).stem.lower().split("_"))
        doc_freq: dict[str, int] = defaultdict(int)
        for t in doc_terms:
            doc_freq[t] += 1

        k1, b = 1.5, 0.75
        avg_dl = 20.0
        dl = len(doc_terms)
        score = 0.0
        for term, qf in query_tokens.items():
            if term not in doc_freq:
                continue
            tf = doc_freq[term]
            idf = math.log(1 + 1.0 / (0.5 + 0.5))  # simplified — no corpus IDF
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm

        return min(1.0, score / max(len(query_tokens), 1))

    @staticmethod
    def _is_test(path: str) -> bool:
        p = path.lower()
        return (
            "/test" in p
            or "/tests/" in p
            or "\\test" in p
            or "\\tests\\" in p
            or Path(path).name.startswith("test_")
            or Path(path).name.endswith("_test.py")
            or Path(path).name.endswith(".test.ts")
            or Path(path).name.endswith(".spec.ts")
            or Path(path).name.endswith(".test.js")
        )


# ---------------------------------------------------------------------------
# ContextSelector — token-budget-aware file selection + rendering
# ---------------------------------------------------------------------------


class ContextSelector:
    """Select and render files within a token budget.

    Key upgrade over code-review-graph:
    - Fills token budget greedily from highest-ranked files
    - Low-score files rendered as signatures-only (10x smaller)
    - Integrates tokenpruner for compression (when available)
    - Never exceeds token_budget hard limit

    Args:
        config: ContextConfig controlling budget and modes.
    """

    def __init__(self, config: ContextConfig | None = None, dedup_window: int = 64) -> None:
        self._config = config or ContextConfig()
        self._pruner = self._load_pruner()
        self._last_breakpoints = 0
        self._dedup_window = dedup_window
        # Diff-aware trimming state
        self._current_diff_spec: DiffSpec | None = None
        self._trim_stats: dict[str, dict] = {}

    def __repr__(self) -> str:
        return f"ContextSelector(budget={self._config.token_budget:,})"

    def _load_pruner(self) -> object | None:
        """Lazy-load tokenpruner if available."""
        try:
            from tokenpruner import PruningConfig, PruningStrategy, TextPruner  # noqa: PLC0415
            return TextPruner(PruningConfig(
                strategy=PruningStrategy.COMPOSITE,
                target_ratio=self._config.compression_ratio,
            ))
        except ImportError:
            logger.debug("tokenpruner not available — compression disabled")
            return None

    # ------------------------------------------------------------------
    # Entropy-based deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _simhash(source: str, window: int = 64) -> str:
        """Compute a SimHash-style fingerprint for similarity comparison.

        For each sliding window of ``window`` characters, compute an MD5
        hash and treat it as a 64-bit integer. The median of all window
        hashes is returned as the fingerprint. The median gives position-
        insensitive similarity detection — two near-identical files will
        have similar fingerprints regardless of where differences occur.

        Args:
            source: Source text to fingerprint.
            window: Sliding-window width in characters.

        Returns:
            16-character hex string (64-bit fingerprint).
        """
        if not source:
            return "0" * 16
        if len(source) < window:
            return hashlib.md5(source.encode()).hexdigest()[:16]

        hashes: list[int] = []
        for i in range(len(source) - window + 1):
            chunk = source[i:i + window]
            h = hashlib.md5(chunk.encode()).hexdigest()
            hashes.append(int(h[:16], 16))  # first 64 bits

        hashes.sort()
        median = hashes[len(hashes) // 2]
        return format(median, "016x")

    @staticmethod
    def _hamming_distance(fp1: str, fp2: str) -> int:
        """Compute the bit-level Hamming distance between two hex fingerprints.

        Args:
            fp1: First fingerprint (16-char hex string).
            fp2: Second fingerprint (16-char hex string).

        Returns:
            Number of differing bits.
        """
        v1 = int(fp1, 16)
        v2 = int(fp2, 16)
        xor = v1 ^ v2
        return xor.bit_count()

    def _is_duplicate(self, source: str, seen_fingerprints: set[str]) -> bool:
        """Check if *source* is a near-duplicate of already-seen content.

        Computes a SimHash fingerprint and compares it against all
        previously seen fingerprints. If the Hamming distance to *any*
        seen fingerprint is less than 3 bits (out of 64), the source
        is considered a duplicate (>85% similar).

        The fingerprint is **only** added to *seen_fingerprints* when
        this method returns ``False`` (i.e. the content is novel).

        Args:
            source: Source text to check.
            seen_fingerprints: Set of already-seen hex fingerprints.

        Returns:
            ``True`` if the source is a near-duplicate.
        """
        fingerprint = self._simhash(source, self._dedup_window)
        for seen in seen_fingerprints:
            if self._hamming_distance(fingerprint, seen) < 3:
                return True
        seen_fingerprints.add(fingerprint)
        return False

    # ------------------------------------------------------------------
    # Diff-aware context trimming
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_diff_hunks(
        diff_text: str,
    ) -> dict[str, list[tuple[tuple[int, int], set[int], set[int]]]]:
        """Parse unified diff format into per-file changed line ranges.

        Parses ``@@ -a,b +c,d @@`` hunk headers and ``+++ b/path`` file
        markers to produce a mapping of file_path → list of
        ``((hunk_start, hunk_end), {new_lines}, {old_lines})``.

        *new_lines* are ``+`` addition line numbers in the *new* file.
        *old_lines* are ``-`` removal line numbers in the *old* file.

        Args:
            diff_text: Raw unified diff text (git diff format).

        Returns:
            Dict mapping file path to list of
            ``((hunk_start, hunk_end), new_additions, old_removals)``.
            All line numbers are 1-based.
        """
        if not diff_text:
            return {}

        result: dict[str, list[tuple[tuple[int, int], set[int], set[int]]]] = {}
        current_file: str | None = None

        lines = diff_text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("+++ "):
                raw = line[4:]
                if raw.startswith(("b/", "a/")):
                    raw = raw[2:]
                current_file = raw
                if current_file not in result:
                    result[current_file] = []
                i += 1
                continue
            if line.startswith("--- "):
                i += 1
                continue
            if line.startswith("@@") and current_file is not None:
                m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if m:
                    old_start = int(m.group(1))
                    new_start = int(m.group(3))
                    new_count = int(m.group(4) or 1)
                    hunk_end = new_start + new_count - 1
                    new_changed: set[int] = set()
                    old_changed: set[int] = set()
                    new_cursor = new_start
                    old_cursor = old_start
                    i += 1
                    while i < len(lines):
                        body = lines[i]
                        if body.startswith("@@") or body.startswith("+++") or body.startswith("---"):
                            break
                        prefix = body[0] if body else " "
                        if prefix == "\\":
                            i += 1
                            continue
                        if prefix == "+":
                            new_changed.add(new_cursor)
                            new_cursor += 1
                        elif prefix == "-":
                            old_changed.add(old_cursor)
                            old_cursor += 1
                        else:
                            new_cursor += 1
                            old_cursor += 1
                        i += 1
                    result[current_file].append(((new_start, hunk_end), new_changed, old_changed))
                    continue
            i += 1

        return result

    def _trim_to_diff_context(
        self,
        source: str,
        diff_spec: DiffSpec,
        file_node: FileNode,
        parser: LanguageParser,
    ) -> str:
        """Extract only the lines/symbols from *source* relevant to the diff.

        Strategy:
        1. Parse diff hunk headers to find changed line ranges per file.
        2. For each changed range, identify overlapping symbols using
           *only* the core changed lines (``+`` additions), not the hunk
           context lines, to avoid pulling in neighboring symbols.
        3. Include the full text of those symbols and the file preamble
           (imports, docstrings, comments).
        4. Add clamped context lines between preamble and first relevant
           symbol, and between relevant symbols — never extending into
           the body of a non-relevant symbol.
        5. If >50% of file lines are covered by hunks, include the whole
           file.
        6. If no diff hunks exist for this file, emit signatures only
           (for dependent files) or the full source (for new/changed
           files with no diff representation).

        Args:
            source: Full source text of the file.
            diff_spec: The diff specification containing diff_text and
                       changed_files.
            file_node: Parsed FileNode with symbol boundaries.
            parser: LanguageParser for signature extraction fallback.

        Returns:
            Trimmed source text with only diff-relevant sections.
        """
        source_lines = source.splitlines()
        total_lines = len(source_lines)
        if not source_lines:
            return source

        # Parse hunks: each entry is ((hunk_start, hunk_end), new_additions, old_removals)
        raw_hunks = self._parse_diff_hunks(diff_spec.diff_text)
        file_hunk_data: list[tuple[tuple[int, int], set[int], set[int]]] = raw_hunks.get(file_node.path, [])

        # Fallback: try suffix match for path differences
        if not file_hunk_data:
            for hunk_path, hunk_data in raw_hunks.items():
                if file_node.path.endswith(hunk_path) or hunk_path.endswith(file_node.path):
                    file_hunk_data = hunk_data
                    break

        # No hunks for this file — determine appropriate fallback
        if not file_hunk_data:
            if file_node.path in diff_spec.changed_files:
                return source  # Changed but undiffable (new/binary) — full source
            return parser.extract_signatures(source)  # Dependent — signatures only

        # Extract full hunk ranges (for >50% check) and changed lines
        file_hunks: list[tuple[int, int]] = []
        new_changed_lines: set[int] = set()
        old_changed_lines: set[int] = set()
        for (h_start, h_end), new_lines, old_lines in file_hunk_data:
            file_hunks.append((h_start, h_end))
            new_changed_lines.update(new_lines)
            old_changed_lines.update(old_lines)

        # If no new changed lines (e.g. no ``+`` additions), fall back
        # to full hunk ranges
        if not new_changed_lines:
            for h_start, h_end in file_hunks:
                for i in range(max(1, h_start), min(total_lines, h_end) + 1):
                    new_changed_lines.add(i)

        # Also keep the full hunk range lines for the >50% check and
        # as a fallback for changed-line inclusion
        all_hunk_lines: set[int] = set()
        for h_start, h_end in file_hunks:
            for i in range(max(1, h_start), min(total_lines, h_end) + 1):
                all_hunk_lines.add(i)

        # If >50% of file lines are covered by hunks, include the whole file
        pct_changed = len(all_hunk_lines) / max(total_lines, 1)
        if pct_changed > 0.5:
            return source

        ctx = self._config.trimming_context_lines

        # ---- Find symbols that overlap with changed lines ----
        # Matching priority:
        #   1. new_lines (``+`` additions) — precise, new-file positions
        #   2. Full hunk range — ONLY when no symbols matched via new_lines
        #
        # When a symbol lacks line_end (GenericParser sets only line_start),
        # estimate the end from the next symbol or EOF.
        relevant_qualnames: set[str] = set()
        syms = [s for s in file_node.symbols if s.kind != NodeKind.MODULE and s.line_start > 0]
        for i, sym in enumerate(syms):
            _s = sym.line_start
            _e = sym.line_end if sym.line_end > 0 else (
                syms[i + 1].line_start - 1 if i + 1 < len(syms) else total_lines
            )
            for cl in new_changed_lines:
                if _s <= cl <= _e:
                    relevant_qualnames.add(sym.qualified_name)
                    break

        # Fallback to full hunk range only if new_lines matched nothing
        if not relevant_qualnames:
            for i, sym in enumerate(syms):
                _s = sym.line_start
                _e = sym.line_end if sym.line_end > 0 else (
                    syms[i + 1].line_start - 1 if i + 1 < len(syms) else total_lines
                )
                for hl in all_hunk_lines:
                    if _s <= hl <= _e:
                        relevant_qualnames.add(sym.qualified_name)
                        break

        # ---- Build the set of lines to include ----
        lines: set[int] = set()

        # 1. Preamble: leading comments, docstrings, and imports
        _preamble_starts = (
            "#", "//", "/*", "*", '"""', "'''",
            "import ", "from ", "use ", "package ", "require(",
            "#include",
        )
        preamble_end: int = 0
        for i, line_text in enumerate(source_lines, 1):
            stripped = line_text.strip()
            if not stripped:
                lines.add(i)
                preamble_end = i
                continue
            if stripped.startswith(_preamble_starts):
                lines.add(i)
                preamble_end = i
                continue
            break  # end of preamble

        # 2. Build map of non-relevant symbol bodies (line ranges to avoid
        #    for context padding)
        non_relevant_ranges: list[tuple[int, int]] = []
        for sym in file_node.symbols:
            if sym.kind == NodeKind.MODULE:
                continue
            if sym.qualified_name not in relevant_qualnames and sym.line_start > 0 and sym.line_end > 0:
                non_relevant_ranges.append((sym.line_start, sym.line_end))

        def _clamped_context(anchor_start: int, anchor_end: int, expand: int) -> list[int]:
            """Return context lines around *anchor*, clamped to avoid
            entering non-relevant symbol bodies or exceeding file bounds."""
            result: list[int] = []
            limit = max(preamble_end + 1, anchor_start - expand)
            for candidate in range(anchor_start - 1, limit - 1, -1):
                blocked = any(rs <= candidate <= re for rs, re in non_relevant_ranges)
                if blocked:
                    break
                result.append(candidate)
            limit = min(total_lines, anchor_end + expand)
            for candidate in range(anchor_end + 1, limit + 1):
                blocked = any(rs <= candidate <= re for rs, re in non_relevant_ranges)
                if blocked:
                    break
                result.append(candidate)
            return result

        # 3. Include relevant symbols: their full body + clamped context
        for i, sym in enumerate(syms):
            if sym.qualified_name not in relevant_qualnames:
                continue
            _s = sym.line_start
            _e = sym.line_end if sym.line_end > 0 else (
                syms[i + 1].line_start - 1 if i + 1 < len(syms) else total_lines
            )
            for j in range(max(1, _s), min(total_lines, _e) + 1):
                lines.add(j)
            for j in _clamped_context(_s, _e, ctx):
                lines.add(j)

        # 4. Ensure new-file changed lines are included (in case they
        #    fall outside any parsed symbol — e.g. in an unparsed language)
        for ln in new_changed_lines:
            if 1 <= ln <= total_lines:
                lines.add(ln)

        # ---- Render sorted lines with gap markers ----
        sorted_lines = sorted(lines)
        result: list[str] = []
        prev = 0
        for ln in sorted_lines:
            if prev and ln - prev > 1:
                omitted = ln - prev - 1
                result.append(f"# ... {omitted} lines omitted ...")
            result.append(source_lines[ln - 1])
            prev = ln

        return "\n".join(result)

    def select_and_render(
        self,
        ranked_files: list[ScoredFile],
        source_map: dict[str, str],
        diff_spec: DiffSpec,
    ) -> tuple[list[ScoredFile], str, int, int]:
        """Select files within budget and render the context string.

        Args:
            ranked_files: Output of RelevanceRanker.rank (sorted by score).
            source_map: Mapping of file_path → source text.
            diff_spec: Original diff specification.

        Returns:
            Tuple of (selected_files, rendered_context, original_tokens, rendered_tokens).
        """
        budget = self._config.token_budget
        selected: list[ScoredFile] = []
        parts: list[str] = []
        used_tokens = 0
        total_original = 0

        # Diff-aware trimming state
        self._current_diff_spec = diff_spec
        self._trim_stats = {}

        # Always include changed files first
        changed_set = set(diff_spec.changed_files)
        priority: list[ScoredFile] = []
        rest: list[ScoredFile] = []
        for sf in ranked_files:
            if sf.file_node.path in changed_set:
                priority.append(sf)
            else:
                rest.append(sf)

        # Entropy-based dedup: track fingerprints of selected files
        seen_fingerprints: set[str] = set()
        dedup_enabled = self._config.dedup_enabled

        for sf in priority + rest:
            source = source_map.get(sf.file_node.path, "")
            if not source:
                continue

            # Dedup: skip near-duplicate files (unless directly changed)
            if dedup_enabled and sf.file_node.path not in changed_set:
                if self._is_duplicate(source, seen_fingerprints):
                    continue

            original_tokens = estimate_tokens(source)
            total_original += original_tokens

            rendered = self._render_file(sf, source)
            rendered_tokens = estimate_tokens(rendered)

            # Skip COLD-tier files early (below warm_threshold) to save budget
            warm_threshold = getattr(self._config, 'warm_threshold', 0.25)
            if sf.score < warm_threshold:
                continue

            if used_tokens + rendered_tokens > budget:
                # Try signatures-only to fit within budget
                parser = _PARSER_REGISTRY.get(sf.file_node.language, GenericParser())
                sig_text = parser.extract_signatures(source)
                sig_tokens = estimate_tokens(sig_text)
                if used_tokens + sig_tokens <= budget:
                    rendered = f"# {sf.file_node.path} [signatures only, score={sf.score:.2f}]\n{sig_text}"
                    rendered_tokens = sig_tokens
                else:
                    # Skip file entirely
                    continue

            selected.append(sf)
            parts.append(rendered)
            used_tokens += rendered_tokens

            if used_tokens >= budget:
                break

        cache_aware = getattr(self._config, 'cache_aware', False)
        breakpoints = 0
        if cache_aware:
            context, breakpoints = self._render_cache_aware(selected, parts, diff_spec)
        else:
            context = self._build_header(diff_spec) + "\n\n".join(parts)

        # Store breakpoints in metadata for ContextResult
        self._last_breakpoints = breakpoints
        return selected, context, total_original, used_tokens

    def _render_file(self, sf: ScoredFile, source: str) -> str:
        path = sf.file_node.path
        mode = sf.output_mode

        # Determine tier for visual labeling
        hot_threshold = getattr(self._config, 'hot_threshold', 0.8)
        warm_threshold = getattr(self._config, 'warm_threshold', 0.25)
        if sf.score >= hot_threshold:
            tier_label = "HOT"
        elif sf.score >= warm_threshold:
            tier_label = "WARM"
        else:
            tier_label = "COLD"

        origin_tag = sf.source_confidence.value.upper()
        header = (
            f"## {path} [{tier_label}] [origin: {origin_tag}]\n"
            f"<!-- score={sf.score:.3f} rank={sf.rank} depth={sf.depth} "
            f"reasons={','.join(sf.reasons[:2])} "
            f"origin={origin_tag} -->\n"
        )
        if sf.source_confidence == SourceConfidence.INFERRED:
            header += f"<!-- CAUTION: {path} symbols were INFERRED via regex/heuristic, not AST-parsed -->\n"

        # Apply diff-aware trimming before mode-specific rendering
        # This runs after tier selection but before token budget counting
        parser = _PARSER_REGISTRY.get(sf.file_node.language, GenericParser())
        if self._config.diff_aware_trimming and self._current_diff_spec is not None:
            trimmed = self._trim_to_diff_context(
                source, self._current_diff_spec, sf.file_node, parser,
            )
            if trimmed != source:
                orig_tok = estimate_tokens(source)
                trim_tok = estimate_tokens(trimmed)
                self._trim_stats[path] = {
                    "original_file_tokens": orig_tok,
                    "trimmed_file_tokens": trim_tok,
                    "saved_tokens": orig_tok - trim_tok,
                    "trim_ratio": round(1.0 - (trim_tok / max(orig_tok, 1)), 4),
                }
                source = trimmed

        if mode == OutputMode.SIGNATURES:
            body = parser.extract_signatures(source)
            return header + f"```{sf.file_node.language.value}\n{body}\n```"

        if mode == OutputMode.COMPRESSED and self._pruner is not None:
            try:
                result = self._pruner.prune(source)  # type: ignore[attr-defined]
                return header + f"```{sf.file_node.language.value}\n{result.pruned_text}\n```"
            except Exception:
                pass  # fall through to FULL

        lang = sf.file_node.language.value
        return header + f"```{lang}\n{source}\n```"

    @staticmethod
    def _build_header(diff_spec: DiffSpec) -> str:
        lines = ["# Code Review Context (generated by graphsift)\n"]
        if diff_spec.commit_message:
            lines.append(f"**Commit:** {diff_spec.commit_message}\n")
        if diff_spec.query:
            lines.append(f"**Query:** {diff_spec.query}\n")
        lines.append(f"**Changed files:** {', '.join(diff_spec.changed_files)}\n")
        lines.append("---\n")
        return "\n".join(lines)

    def _render_cache_aware(
        self,
        selected: list[ScoredFile],
        rendered_parts: list[str],
        diff_spec: DiffSpec,
    ) -> tuple[str, int]:
        """Structure output with prompt-cache breakpoints for Anthropic/OpenAI.

        Layout designed for maximum cache reuse across reviews:
          [CACHE ZONE 1] — Signatures of WARM files (cacheable across reviews)
          [CACHE ZONE 2] — Full source of HOT files (cacheable per PR session)
          [DYNAMIC ZONE] — Query, diff text (never cached)

        Returns (rendered_context, num_breakpoints).
        """
        provider = getattr(self._config, 'cache_provider', 'anthropic')

        hot_parts: list[str] = []
        warm_parts: list[str] = []
        cold_parts: list[str] = []

        hot_threshold = getattr(self._config, 'hot_threshold', 0.8)
        warm_threshold = getattr(self._config, 'warm_threshold', 0.25)

        for i, sf in enumerate(selected):
            part = rendered_parts[i] if i < len(rendered_parts) else ""
            if sf.score >= hot_threshold:
                hot_parts.append(part)
            elif sf.score >= warm_threshold:
                warm_parts.append(part)
            else:
                cold_parts.append(part)

        sections: list[str] = []
        breakpoints = 0

        # Header
        header = self._build_header(diff_spec)
        sections.append(header)

        # CACHE ZONE 1: Signatures skeleton (highly cacheable)
        if warm_parts:
            breakpoints += 1
            if provider == "anthropic":
                sections.append("<!-- cache_control: ephemeral -->")
            sections.append("## Signed Reference Signatures (WARM tier)\n")
            sections.extend(warm_parts)

        # CACHE ZONE 2: Full source of HOT files
        if hot_parts:
            breakpoints += 1
            if provider == "anthropic":
                sections.append("<!-- cache_control: ephemeral -->")
            sections.append("## Core Changed Files (HOT tier)\n")
            sections.extend(hot_parts)

        # DYNAMIC ZONE: Query-specific
        if diff_spec.query:
            sections.append(f"\n## Review Query\n{diff_spec.query}\n")
        if diff_spec.diff_text:
            sections.append(f"\n## Diff\n```diff\n{diff_spec.diff_text}\n```\n")

        # COLD footnotes (minimal)
        if cold_parts:
            sections.append("\n## Additional Context (COLD tier)\n")
            sections.extend(cold_parts)

        return "\n\n".join(sections), breakpoints


# ---------------------------------------------------------------------------
# ContextBuilder — top-level orchestrator
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Orchestrates the full graphsift pipeline.

    Pipeline:
      1. Accept pre-parsed FileNodes (caller supplies source reading)
      2. Build DependencyGraph (import + inheritance + decorator edges)
      3. BFS ranked traversal from changed files
      4. Multi-signal relevance ranking
      5. Token-budget-aware selection + rendering (with tokenpruner)
      6. Return ContextResult

    The caller owns all file I/O — the library never opens files.

    Args:
        config: ContextConfig.
        graph: Optional pre-built DependencyGraph (for incremental updates).

    Example::

        builder = ContextBuilder(ContextConfig(token_budget=50_000))
        for path, source in my_files.items():
            builder.index_file(path, source)

        diff = DiffSpec(changed_files=["src/auth.py"], query="review this")
        result = builder.build(diff, source_map=my_files)
        print(result)
        # ContextResult(selected=8/143, tokens=11,200, saved=94%)
    """

    def __init__(
        self,
        config: ContextConfig | None = None,
        graph: DependencyGraph | None = None,
        store: GraphStore | None = None,
    ) -> None:
        self._config = config or ContextConfig()
        self._graph = graph or DependencyGraph(
            max_depth=self._config.max_depth
        )
        self._ranker = RelevanceRanker()
        self._selector = ContextSelector(self._config)
        self._index_stats = IndexStats()
        self._lock = threading.RLock()
        self._store = store
        # Incremental indexing: path → sha256 of last indexed version
        self._sha_cache: dict[str, str] = {}
        # Cross-session memory cache: diff_spec_hash → context_data
        self._memory_cache: dict[str, dict] = {}
        # Whether warm_cache() has been called
        self._cache_warmed = False

    def __repr__(self) -> str:
        return f"ContextBuilder(budget={self._config.token_budget:,}, {self._graph})"

    def index_file(self, path: str, source: str) -> FileNode:
        """Parse and index a single source file.

        Args:
            path: File path (used as identifier).
            source: File source text.

        Returns:
            Parsed FileNode.

        Raises:
            ParseError: If the file cannot be parsed.
        """
        lang = detect_language(path)
        parser = _PARSER_REGISTRY.get(lang, GenericParser())
        file_node = parser.parse_file(path, source)
        self._graph.add_file(file_node)
        return file_node

    def index_files(self, source_map: dict[str, str]) -> IndexStats:
        """Index multiple files and build all edges.

        Args:
            source_map: Dict mapping file path → source text.

        Returns:
            IndexStats with counts of files, symbols, edges.
        """
        return self._index_files_impl(source_map, incremental=False)

    def index_files_incremental(self, source_map: dict[str, str]) -> IndexStats:
        """Incrementally index files, skipping unchanged files via SHA-256 check.

        Only files whose content hash differs from the last indexed version are
        re-parsed. This matches the sub-2-second update behaviour of
        code-review-graph for large repos.

        Args:
            source_map: Dict mapping file path → source text (full repo snapshot).

        Returns:
            IndexStats with counts; ``files_skipped`` includes unchanged files.
        """
        return self._index_files_impl(source_map, incremental=True)

    def index_roots(
        self,
        root_source_maps: list[dict[str, str]],
        *,
        incremental: bool = False,
    ) -> list[IndexStats]:
        """Index multiple repository roots into a single shared graph.

        Enables monorepo support: each root is a separate source map
        (e.g. different packages or services), but all share the same
        DependencyGraph so cross-package imports resolve correctly.

        Args:
            root_source_maps: List of source maps, one per monorepo root.
            incremental: If True, skip unchanged files (SHA-256 check).

        Returns:
            List of IndexStats, one per root.
        """
        results: list[IndexStats] = []
        for sm in root_source_maps:
            stats = self._index_files_impl(sm, incremental=incremental, build_edges=False)
            results.append(stats)

        # Build edges once across all roots combined
        self._graph.build_import_edges()
        self._graph.build_inheritance_edges()
        self._graph.build_decorator_edges()

        logger.info(
            "graphsift: monorepo index complete",
            extra={"roots": len(root_source_maps), "total_files": sum(s.files_indexed for s in results)},
        )
        return results

    def _index_files_impl(
        self,
        source_map: dict[str, str],
        incremental: bool,
        build_edges: bool = True,
    ) -> IndexStats:
        import time  # noqa: PLC0415

        t0 = time.monotonic()
        files_indexed = 0
        files_skipped = 0
        symbols = 0
        lang_counts: dict[str, int] = defaultdict(int)

        for path, source in source_map.items():
            if self._should_skip(path):
                files_skipped += 1
                continue

            # Incremental: skip if SHA matches cached value
            if incremental:
                new_sha = hashlib.sha256(source.encode(errors="replace")).hexdigest()
                with self._lock:
                    cached_sha = self._sha_cache.get(path)
                if cached_sha == new_sha:
                    files_skipped += 1
                    continue

            try:
                fn = self.index_file(path, source)
                files_indexed += 1
                symbols += len(fn.symbols)
                lang_counts[fn.language.value] += 1
                if incremental:
                    with self._lock:
                        self._sha_cache[path] = fn.sha256
            except (ParseError, Exception) as exc:
                logger.warning(
                    "graphsift: skipping file",
                    extra={"path": path, "error": str(exc)},
                )
                files_skipped += 1

        total_edges = 0
        if build_edges:
            import_edges = self._graph.build_import_edges()
            inherit_edges = self._graph.build_inheritance_edges()
            dec_edges = self._graph.build_decorator_edges()
            total_edges = import_edges + inherit_edges + dec_edges

        duration = (time.monotonic() - t0) * 1000
        stats = IndexStats(
            files_indexed=files_indexed,
            files_skipped=files_skipped,
            symbols_extracted=symbols,
            edges_created=total_edges,
            duration_ms=round(duration, 2),
            languages=dict(lang_counts),
        )
        with self._lock:
            self._index_stats = stats

        logger.info(
            "graphsift: index complete",
            extra={
                "files": files_indexed,
                "symbols": symbols,
                "edges": total_edges,
                "ms": round(duration, 2),
                "incremental": incremental,
            },
        )
        return stats

    def warm_cache(self, limit: int = 10) -> int:
        """Pre-load recent session contexts into an in-memory cache for instant response.

        Call this once at session start before ``build()`` to avoid SQLite lookups
        during the critical path. Only entries whose ``session_id`` matches
        the configured ``session_id`` are loaded.

        Args:
            limit: Maximum number of recent entries to warm.

        Returns:
            Number of entries loaded into the in-memory cache.
        """
        if self._store is None or not self._config.session_id:
            self._cache_warmed = True
            return 0

        session_id = self._config.session_id
        try:
            recent = self._store.list_recent_sessions(limit=limit)
        except Exception as exc:
            logger.debug("graphsift: warm_cache failed: %s", exc)
            self._cache_warmed = True
            return 0

        loaded = 0
        for entry in recent:
            if entry["session_id"] != session_id:
                continue
            try:
                data = self._store.load_session_context(
                    session_id,
                    entry["diff_spec_hash"],
                    max_age_days=self._config.cache_ttl_days,
                )
                if data is not None:
                    self._memory_cache[entry["diff_spec_hash"]] = data
                    loaded += 1
            except Exception:
                continue

        self._cache_warmed = True
        logger.info(
            "graphsift: warmed %d/%d session memory entries",
            loaded, len(recent),
        )
        return loaded

    def build(
        self,
        diff_spec: DiffSpec,
        source_map: dict[str, str],
    ) -> ContextResult:
        """Build the ranked context for a diff.

        If session memory is configured (store + session_id), checks for a
        cached result first. Cache key = hash of (sorted changed_files + query
        + commit_message). Hit returns instantly with ``from_cache`` metadata.

        Args:
            diff_spec: Which files changed and optional query.
            source_map: Dict mapping file path → source text (for rendering).

        Returns:
            ContextResult with selected files and rendered LLM context.

        Raises:
            ValidationError: If diff_spec has no changed files.
            GraphError: If graph traversal fails.
        """
        if not diff_spec.changed_files:
            raise ValidationError("DiffSpec must have at least one changed_file.")

        # ── Check session memory cache ────────────────────────────────────
        diff_hash = _build_diff_hash(diff_spec)
        cached = self._check_cache(diff_spec, diff_hash)
        if cached is not None:
            return cached

        # ── Full build ────────────────────────────────────────────────────
        try:
            graph_scores = self._graph.ranked_neighbors(
                diff_spec.changed_files,
                include_dynamic=self._config.include_dynamic,
            )
        except Exception as exc:
            raise GraphError(f"Graph traversal failed: {exc}") from exc

        all_files = self._graph.all_files()
        ranked = self._ranker.rank(diff_spec, graph_scores, all_files, self._config)

        selected, context, orig_tokens, rendered_tokens = self._selector.select_and_render(
            ranked, source_map, diff_spec
        )

        reduction = 1.0 - (rendered_tokens / max(orig_tokens, 1))

        result = ContextResult(
            diff_spec=diff_spec,
            selected_files=selected,
            rendered_context=context,
            cache_breakpoints=getattr(self._selector, '_last_breakpoints', 0),
            total_original_tokens=orig_tokens,
            total_rendered_tokens=rendered_tokens,
            reduction_ratio=round(reduction, 4),
            files_scanned=len(all_files),
            files_selected=len(selected),
            metadata={
                "from_cache": False,
                "trim_stats": getattr(self._selector, '_trim_stats', {}),
            },
        )

        # ── Persist to session memory ─────────────────────────────────────
        self._save_to_memory(diff_hash, result)
        return result

    def _check_cache(
        self,
        diff_spec: DiffSpec,
        diff_hash: str,
    ) -> ContextResult | None:
        """Check in-memory and persistent caches for a matching context.

        Returns a ContextResult with ``from_cache`` metadata if found,
        or None to trigger a full build.
        """
        # 1. In-memory cache (fastest — warmed via warm_cache())
        if diff_hash in self._memory_cache:
            data = self._memory_cache[diff_hash]
            logger.info(
                "graphsift: cache HIT (in-memory) for diff hash %s",
                diff_hash[:12],
            )
            return self._cached_result(diff_spec, data, source="memory")

        # 2. Persistent store (SQLite — cross-session)
        if self._store is not None and self._config.session_id:
            try:
                data = self._store.load_session_context(
                    self._config.session_id,
                    diff_hash,
                    max_age_days=self._config.cache_ttl_days,
                )
                if data is not None:
                    # Promote to in-memory cache for faster subsequent access
                    self._memory_cache[diff_hash] = data
                    logger.info(
                        "graphsift: cache HIT (sqlite) for diff hash %s",
                        diff_hash[:12],
                    )
                    return self._cached_result(diff_spec, data, source="sqlite")
            except Exception as exc:
                logger.debug("graphsift: cache lookup failed: %s", exc)

        return None

    def _cached_result(
        self,
        diff_spec: DiffSpec,
        data: dict,
        source: str,
    ) -> ContextResult:
        """Reconstruct a ContextResult from cached data."""
        selected_files_raw = data.get("selected_file_paths", [])
        # Build lightweight ScoredFile stubs (file_node has path + minimal fields)
        scored = []
        for fp in selected_files_raw:
            fn = self._graph.get_file(fp)
            if fn is None:
                continue
            scored.append(
                ScoredFile(
                    file_node=fn,
                    score=data.get("_cached_score", 0.0),
                    rank=0,
                    reasons=["from session memory"],
                    output_mode=OutputMode.SMART,
                )
            )

        return ContextResult(
            diff_spec=diff_spec,
            selected_files=scored,
            rendered_context=data.get("rendered_context", ""),
            cache_breakpoints=data.get("cache_breakpoints", 0),
            total_original_tokens=data.get("total_original_tokens", 0),
            total_rendered_tokens=data.get("total_rendered_tokens", 0),
            reduction_ratio=data.get("reduction_ratio", 0.0),
            files_scanned=data.get("files_scanned", 0),
            files_selected=len(scored),
            metadata={
                "from_cache": True,
                "cache_source": source,
                "cached_at": data.get("_cached_at", ""),
                "cached_session_id": self._config.session_id,
                "diff_spec_hash": data.get("diff_spec_hash", ""),
                "memory_id": data.get("_memory_id"),
            },
        )

    def _save_to_memory(
        self,
        diff_hash: str,
        result: ContextResult,
    ) -> None:
        """Save a build result to session memory (in-memory + SQLite)."""
        if self._store is None or not self._config.session_id:
            return

        context_data = {
            "diff_spec_hash": diff_hash,
            "rendered_context": result.rendered_context,
            "cache_breakpoints": result.cache_breakpoints,
            "selected_file_paths": [sf.file_node.path for sf in result.selected_files],
            "files_selected": result.files_selected,
            "files_scanned": result.files_scanned,
            "total_original_tokens": result.total_original_tokens,
            "total_rendered_tokens": result.total_rendered_tokens,
            "reduction_ratio": result.reduction_ratio,
            "_cached_score": result.selected_files[0].score if result.selected_files else 0.0,
        }

        # Update in-memory cache
        self._memory_cache[diff_hash] = context_data

        # Persist to SQLite
        try:
            self._store.save_session_context(
                self._config.session_id,
                diff_hash,
                context_data,
            )
        except Exception as exc:
            logger.debug("graphsift: failed to save session memory: %s", exc)

    def graph_stats(self) -> dict[str, int]:
        """Return current graph statistics."""
        return self._graph.stats()

    def index_stats(self) -> IndexStats:
        """Return stats from last index_files call."""
        with self._lock:
            return self._index_stats

    def _should_skip(self, path: str) -> bool:
        p_norm = path.replace("\\", "/").lower()
        parts = p_norm.split("/")
        filename = parts[-1]
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename

        for pat in self._config.exclude_patterns:
            pat_clean = pat.rstrip("/*").lower()
            # Match exact segment (directory or filename stem)
            if pat_clean in parts or pat_clean == stem:
                return True
            # Glob suffix: *.egg-info → any part ending with .egg-info
            if pat_clean.startswith("*"):
                suffix = pat_clean[1:]
                if any(seg.endswith(suffix) for seg in parts):
                    return True
        return False
