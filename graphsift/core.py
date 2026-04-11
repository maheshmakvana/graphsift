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
import logging
import math
import re
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Protocol, runtime_checkable

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

    Args:
        bm25_weight: Weight for BM25 keyword signal (0–1).
        graph_weight: Weight for graph distance signal (0–1).
    """

    def __init__(self, bm25_weight: float = 0.3, graph_weight: float = 0.7) -> None:
        self._bm25_w = bm25_weight
        self._graph_w = graph_weight

    def __repr__(self) -> str:
        return f"RelevanceRanker(bm25={self._bm25_w}, graph={self._graph_w})"

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

            combined = min(1.0, max(0.0, combined))

            # Determine output mode
            if config.output_mode == OutputMode.SMART:
                mode = OutputMode.FULL if combined >= config.smart_threshold else OutputMode.SIGNATURES
            else:
                mode = config.output_mode

            scored.append(ScoredFile(
                file_node=fnode,
                score=round(combined, 4),
                rank=0,
                reasons=reasons,
                depth=depth,
                output_mode=mode,
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

    def __init__(self, config: ContextConfig | None = None) -> None:
        self._config = config or ContextConfig()
        self._pruner = self._load_pruner()

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

        # Always include changed files first
        changed_set = set(diff_spec.changed_files)
        priority: list[ScoredFile] = []
        rest: list[ScoredFile] = []
        for sf in ranked_files:
            if sf.file_node.path in changed_set:
                priority.append(sf)
            else:
                rest.append(sf)

        for sf in priority + rest:
            source = source_map.get(sf.file_node.path, "")
            if not source:
                continue

            original_tokens = estimate_tokens(source)
            total_original += original_tokens

            rendered = self._render_file(sf, source)
            rendered_tokens = estimate_tokens(rendered)

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

        context = self._build_header(diff_spec) + "\n\n".join(parts)
        return selected, context, total_original, used_tokens

    def _render_file(self, sf: ScoredFile, source: str) -> str:
        path = sf.file_node.path
        mode = sf.output_mode

        header = (
            f"## {path}\n"
            f"<!-- score={sf.score:.3f} rank={sf.rank} depth={sf.depth} "
            f"reasons={','.join(sf.reasons[:2])} -->\n"
        )

        if mode == OutputMode.SIGNATURES:
            parser = _PARSER_REGISTRY.get(sf.file_node.language, GenericParser())
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
    ) -> None:
        self._config = config or ContextConfig()
        self._graph = graph or DependencyGraph(
            max_depth=self._config.max_depth
        )
        self._ranker = RelevanceRanker()
        self._selector = ContextSelector(self._config)
        self._index_stats = IndexStats()
        self._lock = threading.RLock()
        # Incremental indexing: path → sha256 of last indexed version
        self._sha_cache: dict[str, str] = {}

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

    def build(
        self,
        diff_spec: DiffSpec,
        source_map: dict[str, str],
    ) -> ContextResult:
        """Build the ranked context for a diff.

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

        return ContextResult(
            diff_spec=diff_spec,
            selected_files=selected,
            rendered_context=context,
            total_original_tokens=orig_tokens,
            total_rendered_tokens=rendered_tokens,
            reduction_ratio=round(reduction, 4),
            files_scanned=len(all_files),
            files_selected=len(selected),
        )

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
