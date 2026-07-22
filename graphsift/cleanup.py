"""Stale-reference scanner for graphsift — detect and fix references to deleted files.

After a file is deleted, remaining source files may contain:
- ``import`` / ``from ... import`` statements that will raise ``ImportError``
- Symbol name references (function calls, class references) that will raise ``NameError``
- String path references (e.g. ``"./deleted_module"``) that silently resolve to nothing

Usage::

    from graphsift.cleanup import StaleRefScanner

    scanner = StaleRefScanner("/path/to/project")
    report = scanner.scan_after_deletion(
        deleted_paths=["/path/to/project/src/old_module.py"],
    )
    print(report.total, "stale references found")

    # Preview auto-fixes, then apply them
    result = scanner.apply_fixes(report, dry_run=True)
    scanner.apply_fixes(report, dry_run=False)
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from graphsift.core import PythonParser, detect_language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ScanResult(BaseModel):
    """A single stale reference found in a remaining source file."""

    file_path: str = Field(description="File containing the stale reference")
    line_number: int = Field(description="Line number (1-based)")
    line_text: str = Field(description="The actual line content")
    severity: str = Field(description='"HIGH" | "MEDIUM" | "LOW"')
    kind: str = Field(
        description='"import" | "from_import" | "name_ref" | "string_ref"'
    )
    symbol: str = Field(description="The stale symbol being referenced")
    suggested_fix: str = Field(
        description="The replacement or removal suggestion"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence that this is a real stale reference (0.0-1.0)",
    )


class StaleRefReport(BaseModel):
    """Full report from a stale-reference scan."""

    findings: list[ScanResult] = Field(default_factory=list)
    total: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_kind: dict[str, int] = Field(default_factory=dict)
    auto_fixable: int = 0
    dry_run: bool = True


# ---------------------------------------------------------------------------
# Constants: patterns and builtin/stdlib sets
# ---------------------------------------------------------------------------

# Python import statement patterns — each returns (module_name,) as group(1)
# or for "from X import Y" the module is in group(1) and names in group(2)+.
_PY_IMPORT_PATTERNS: list[re.Pattern[str]] = [
    # import foo; import foo.bar; import foo as bar
    re.compile(r"^\s*import\s+(\w+(?:\.\w+)*)(?:\s+as\s+\w+)?\s*(?:$|#)"),
    # from foo import bar; from .foo import bar; from ..foo.bar import baz
    re.compile(
        r"^\s*from\s+(\.{0,2})(\w+(?:\.\w+)*)\s+import\s+(.+?)(?:\s*#.*)?$"
    ),
]

# Python "from X import" sub-pattern — used after matching _PY_IMPORT_PATTERNS[1]
# to extract imported symbol names from the comma-separated list.
_PY_FROM_IMPORT_NAMES = re.compile(r"(\w+)(?:\s+as\s+\w+)?")

# Python inline module-attribute access (MEDIUM)
_PY_ATTR_REF = re.compile(r"(\w+(?:\.\w+)*)\.(\w+)\s*\(")

# Python name reference — a bare name that might be a function call or type hint
_PY_NAME_CALL = re.compile(r"(?<![.\w])'?>?(\w+)\s*\(")
_PY_NAME_TYPE = re.compile(r":\s*(\w+)")
_PY_NAME_PAREN = re.compile(r"(\w+)\)")

# JS/TS import patterns
_JS_IMPORT_MODULE = re.compile(
    r"""import\s+(?:\{[^}]*\}|\w+(?:\s*,\s*\{[^}]*\})?)\s+from\s+["']([^"']+)["']"""
)
_JS_EXPORT_MODULE = re.compile(
    r"""export\s+\{[^}]*\}\s+from\s+["']([^"']+)["']"""
)
_JS_EXPORT_STAR = re.compile(r"""export\s+\*\s+from\s+["']([^"']+)["']""")
_JS_REQUIRE = re.compile(
    r"""(?:const|let|var|import)\s+\w+(?:\s*,\s*\{[^}]*\})?\s*=\s*require\(["']([^"']+)["']\)"""
)
_JS_IMPORT_EQ_REQUIRE = re.compile(
    r"""import\s+\w+\s*=\s*require\(["']([^"']+)["']\)"""
)
_JS_DYNAMIC_IMPORT = re.compile(r"""import\(["']([^"']+)["']\)""")
_JSX_COMPONENT = re.compile(r"""<\s*(\w+)\s*[/>]""")

# Generic string path reference (LOW severity)
_STRING_PATH_REF = re.compile(r"""["']((?:\.\.?)?[\/][^\s"']*)["']""")

# ---------------------------------------------------------------------------
# Python stdlib modules (never stale if referenced as top-level module)
# ---------------------------------------------------------------------------

_STDLIB_MODULES: frozenset[str] = frozenset({
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii", "binhex",
    "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk",
    "cmath", "cmd", "code", "codecs", "codeop", "collections", "colorsys",
    "compileall", "concurrent", "configparser", "contextlib", "contextvars",
    "copy", "copyreg", "cProfile", "crypt", "csv", "ctypes", "curses",
    "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
    "distutils", "doctest", "email", "encodings", "enum", "errno",
    "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch", "fractions",
    "ftplib", "functools", "gc", "getopt", "getpass", "gettext", "glob",
    "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect", "io",
    "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
    "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
    "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
    "platform", "plistlib", "poplib", "posix", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pyclbr",
    "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site", "smtpd",
    "smtplib", "sndhdr", "socket", "socketserver", "sqlite3", "ssl",
    "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "tomllib", "trace", "traceback", "tracemalloc", "tty", "turtle",
    "turtledemo", "types", "typing", "unicodedata", "unittest", "urllib",
    "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
})

# Names that are too generic to be reliable stale-reference indicators
_GENERIC_NAMES: frozenset[str] = frozenset({
    "config", "utils", "helpers", "common", "base", "core", "main",
    "app", "index", "types", "constants", "settings", "env", "lib",
    "src", "data", "model", "view", "controller", "service", "repo",
    "manager", "factory", "provider", "registry", "context", "mixins",
    "decorators", "signals", "middleware", "handlers", "routes", "init",
    "build", "dist", "node_modules", "vendor", "public",
})

# Minimum length for a symbol name to be considered specific (unless it is a
# known exported symbol with high confidence from AST parsing).
_MIN_SYMBOL_LENGTH = 4


# ---------------------------------------------------------------------------
# StaleRefScanner
# ---------------------------------------------------------------------------


class StaleRefScanner:
    """Scan remaining source files for references to recently deleted files.

    Args:
        project_root: Absolute path to the project root. Used to compute
            importable module names from file paths.
    """

    def __init__(self, project_root: str = "") -> None:
        self.project_root = os.path.abspath(project_root) if project_root else ""
        self._python_parser = PythonParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_after_deletion(
        self,
        deleted_paths: list[str],
        source_map: dict[str, str] | None = None,
    ) -> StaleRefReport:
        """Scan remaining source files for stale references to deleted files.

        Args:
            deleted_paths: Absolute paths of files that were deleted.
            source_map: Dict of remaining file path → source text.
                If ``None``, the scanner walks ``self.project_root`` to
                discover all remaining source files (excluding deleted ones).

        Returns:
            StaleRefReport with all findings.
        """
        deleted_paths = [os.path.abspath(p) for p in deleted_paths]

        # 1. Extract exported symbols & module names from deleted files
        deleted_exports: dict[str, set[str]] = {}
        deleted_modules: set[str] = set()
        deleted_basenames: set[str] = set()
        deleted_stems: set[str] = set()

        for dpath in deleted_paths:
            try:
                source = self._read_file_source(dpath)
            except OSError:
                logger.debug("Cannot read deleted file (already gone): %s", dpath)
                source = ""

            symbols = self._parse_exported_symbols(dpath, source)
            if symbols:
                deleted_exports[dpath] = symbols

            modname = self._module_name_from_path(dpath)
            deleted_modules.add(modname)

            basename = os.path.basename(dpath)
            deleted_basenames.add(basename)

            stem = Path(dpath).stem
            deleted_stems.add(stem)

        # 2. Load source_map if not provided
        if source_map is None:
            source_map = self._load_remaining_source_map(deleted_paths)

        # 3. Scan each remaining file
        all_exports: set[str] = set()
        for syms in deleted_exports.values():
            all_exports.update(syms)

        findings: list[ScanResult] = []

        for file_path, source in source_map.items():
            file_path = os.path.abspath(file_path)
            if file_path in deleted_paths:
                continue

            findings.extend(
                self._scan_file(
                    file_path=file_path,
                    source=source,
                    deleted_modules=deleted_modules,
                    deleted_exports=all_exports,
                    deleted_stems=deleted_stems,
                    deleted_basenames=deleted_basenames,
                )
            )

        # 4. Build report
        return self._build_report(findings, dry_run=True)

    def apply_fixes(
        self,
        report: StaleRefReport,
        findings: list[ScanResult] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Apply fixes for stale references.

        For HIGH severity findings (import/from-import statements):
        - Remove the entire import line by commenting it out.

        For MEDIUM severity (name references):
        - Report only; not auto-fixed (too risky).

        For LOW severity:
        - Report only.

        Args:
            report: The full stale-ref report.
            findings: Optional subset of findings to fix. If ``None``, all
                auto-fixable (HIGH severity) findings are processed.
            dry_run: If ``True``, return what WOULD be done without modifying
                files. If ``False``, make a ``.bak`` backup first, then modify.

        Returns:
            Dict with keys:
            - ``files_modified``: int
            - ``lines_commented``: int
            - ``files_backed_up``: list[str]
            - ``errors``: list[str]
        """
        target_findings = (
            findings
            if findings is not None
            else [f for f in report.findings if f.severity == "HIGH"]
        )

        result: dict[str, Any] = {
            "files_modified": 0,
            "lines_commented": 0,
            "files_backed_up": [],
            "errors": [],
        }

        # Group findings by file
        by_file: dict[str, list[ScanResult]] = {}
        for f in target_findings:
            by_file.setdefault(f.file_path, []).append(f)

        for file_path, file_findings in by_file.items():
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError as exc:
                result["errors"].append(f"Cannot read {file_path}: {exc}")
                continue

            modified = False
            # Sort findings by line descending so line removal doesn't shift offsets
            sorted_findings = sorted(
                file_findings, key=lambda x: x.line_number, reverse=True
            )

            for finding in sorted_findings:
                idx = finding.line_number - 1  # 0-based
                if idx < 0 or idx >= len(lines):
                    continue

                if finding.severity == "HIGH":
                    # Comment-out the import line
                    stripped = lines[idx].lstrip()
                    indent = lines[idx][: len(lines[idx]) - len(stripped)]
                    comment = f"{indent}# STALE-REF: {stripped.rstrip()}\n"
                    lines[idx] = comment
                    modified = True
                    result["lines_commented"] += 1

            if not modified:
                continue

            if dry_run:
                result["files_modified"] += 1
                continue

            # Backup and write
            try:
                bak_path = file_path + ".bak"
                shutil.copy2(file_path, bak_path)
                result["files_backed_up"].append(bak_path)
            except OSError as exc:
                result["errors"].append(
                    f"Cannot backup {file_path}: {exc}"
                )
                continue

            try:
                with open(file_path, "w", encoding="utf-8") as fh:
                    fh.writelines(lines)
                result["files_modified"] += 1
            except OSError as exc:
                result["errors"].append(
                    f"Cannot write {file_path}: {exc}"
                )
                # Restore from backup
                try:
                    shutil.copy2(bak_path, file_path)
                except OSError:
                    pass

        return result

    # ------------------------------------------------------------------
    # Internal: symbol and module-name extraction
    # ------------------------------------------------------------------

    def _parse_exported_symbols(self, path: str, source: str) -> set[str]:
        """Parse a source file and return the set of exported/defined symbol names.

        For Python, uses the AST parser to extract top-level function and class
        definitions. For JS/TS and other languages, uses regex-based extraction.

        Args:
            path: Absolute file path.
            source: Source text of the file (may be empty if file is already
                deleted from disk).

        Returns:
            Set of symbol name strings. Empty if file cannot be parsed or is
            a non-source file.
        """
        if not source.strip():
            return set()

        lang = detect_language(path)
        symbols: set[str] = set()

        if lang.value == "python":
            try:
                tree = ast.parse(source, filename=path)
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        symbols.add(node.name)
            except SyntaxError:
                logger.debug("Syntax error parsing %s for exported symbols", path)
        elif lang.value in ("javascript", "typescript"):
            # Use regex patterns similar to GenericParser
            func_pat = re.compile(
                r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("
            )
            cls_pat = re.compile(r"(?:export\s+)?class\s+(\w+)")
            const_pat = re.compile(
                r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*="
            )
            for pat in (func_pat, cls_pat, const_pat):
                for m in pat.finditer(source):
                    name = m.group(1)
                    if len(name) >= 2:
                        symbols.add(name)
        else:
            # Generic — try simple regex for defs and classes
            func_pat = re.compile(
                r"(?:^|\n)\s*(?:pub\s+)?(?:async\s+)?(?:fn|def|function|sub)\s+(\w+)"
            )
            cls_pat = re.compile(
                r"(?:^|\n)\s*(?:pub\s+)?(?:class|struct|trait|interface|type)\s+(\w+)"
            )
            for pat in (func_pat, cls_pat):
                for m in pat.finditer(source):
                    name = m.group(1)
                    if len(name) >= 2:
                        symbols.add(name)

        return symbols

    def _module_name_from_path(self, path: str) -> str:
        """Convert an absolute file path to an importable Python module name.

        Examples::

            /repo/src/app/main.py       →  src.app.main
            /repo/src/app/__init__.py   →  src.app

        Args:
            path: Absolute file path.

        Returns:
            Dotted module name.
        """
        p = Path(os.path.abspath(path))

        # Strip project_root prefix if present
        if self.project_root:
            try:
                rel = p.relative_to(self.project_root)
            except ValueError:
                rel = p
        else:
            rel = p

        # Normalize separator to forward slash, strip extension
        parts = rel.as_posix().rsplit(".", 1)[0]

        # Convert / to .
        modname = parts.replace("/", ".")

        # Handle __init__ → package name (strip trailing .__init__)
        if modname.endswith(".__init__"):
            modname = modname[: -len(".__init__")]

        # Strip leading dot(s) from relative paths
        modname = modname.lstrip(".")

        return modname

    # ------------------------------------------------------------------
    # Internal: file scanning
    # ------------------------------------------------------------------

    def _scan_file(
        self,
        file_path: str,
        source: str,
        deleted_modules: set[str],
        deleted_exports: set[str],
        deleted_stems: set[str],
        deleted_basenames: set[str],
    ) -> list[ScanResult]:
        """Scan a single remaining file for stale references."""
        findings: list[ScanResult] = []
        lines = source.splitlines(keepends=False)
        lang = detect_language(file_path)

        if lang.value == "python":
            findings.extend(
                self._scan_python_imports(
                    file_path, lines, deleted_modules, deleted_stems
                )
            )
            findings.extend(
                self._scan_python_name_refs(
                    file_path, lines, deleted_exports, deleted_stems
                )
            )

        if lang.value in ("javascript", "typescript"):
            findings.extend(
                self._scan_js_imports(
                    file_path, lines, deleted_basenames, deleted_stems
                )
            )
            findings.extend(
                self._scan_js_name_refs(
                    file_path, lines, deleted_exports, deleted_stems
                )
            )

        # Low severity: string path references (any language)
        findings.extend(
            self._scan_string_refs(file_path, lines, deleted_basenames, deleted_stems)
        )

        return findings

    # ------------------------------------------------------------------
    # Python import scanner
    # ------------------------------------------------------------------

    def _scan_python_imports(
        self,
        file_path: str,
        lines: list[str],
        deleted_modules: set[str],
        deleted_stems: set[str],
    ) -> list[ScanResult]:
        """Find Python import statements referencing deleted modules."""
        findings: list[ScanResult] = []

        for lineno, line in enumerate(lines, start=1):
            # Skip comments and empty lines
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Pattern 1: ``import deleted_module`` or ``import deleted_module as X``
            m = re.match(r"^\s*import\s+(\S+)", line)
            if m:
                imported = m.group(1)
                # Handle "as" alias
                mod_name = imported.split(" as ")[0].strip()
                # Handle multi-import: import foo, bar
                for part in mod_name.split(","):
                    part = part.strip()
                    if self._is_stale_python_module(part, deleted_modules, deleted_stems):
                        # Check the module's first component
                        if self._module_is_stdlib(part):
                            continue
                        findings.append(
                            ScanResult(
                                file_path=file_path,
                                line_number=lineno,
                                line_text=line.rstrip(),
                                severity="HIGH",
                                kind="import",
                                symbol=part,
                                suggested_fix=f"# Remove stale import: {part}",
                                confidence=0.95,
                            )
                        )
                continue

            # Pattern 2: ``from deleted_module import X`` or ``from .deleted_module import X``
            m2 = re.match(r"^\s*from\s+(\S+)\s+import\s+(.+)$", line)
            if m2:
                from_mod = m2.group(1).lstrip(".")  # strip leading dots for matching
                import_part = m2.group(2).strip()

                if self._is_stale_python_module(from_mod, deleted_modules, deleted_stems):
                    if self._module_is_stdlib(from_mod):
                        continue
                    # Extract imported names (strip commas and "as" aliases)
                    imported_names: list[str] = []
                    for name_part in re.split(r",\s*", import_part):
                        name_part = name_part.strip()
                        # Handle "from X import Y as Z"
                        name_part = name_part.split(" as ")[0].strip()
                        # Handle parenthesized imports
                        name_part = name_part.strip("()")
                        if name_part:
                            imported_names.append(name_part)

                    kind = "from_import"
                    symbol_str = f"{from_mod}.{', '.join(imported_names)}"
                    findings.append(
                        ScanResult(
                            file_path=file_path,
                            line_number=lineno,
                            line_text=line.rstrip(),
                            severity="HIGH",
                            kind=kind,
                            symbol=symbol_str,
                            suggested_fix=f"# Remove stale from-import: {from_mod}",
                            confidence=1.0,
                        )
                    )

        return findings

    def _is_stale_python_module(
        self,
        mod_name: str,
        deleted_modules: set[str],
        deleted_stems: set[str],
    ) -> bool:
        """Check if a Python module name matches a deleted module or its stem."""
        # Direct match: import deleted.module
        if mod_name in deleted_modules:
            return True

        # Check if the first component of the module matches a deleted stem
        # e.g., import cleanup → where cleanup.py was deleted
        first_part = mod_name.split(".")[0]
        if first_part in deleted_stems:
            # Only flag if the full module isn't a known package
            if mod_name not in deleted_modules:
                # Match if the first part is the deleted file's stem AND
                # the full module path looks like it could belong to the project
                # (i.e., not a well-known third-party package)
                if self._module_is_stdlib(first_part):
                    return False
                return True

        return False

    @staticmethod
    def _module_is_stdlib(mod_name: str) -> bool:
        """Return True if *mod_name* (first component) is a Python stdlib module."""
        first = mod_name.split(".")[0]
        return first in _STDLIB_MODULES

    # ------------------------------------------------------------------
    # Python name reference scanner
    # ------------------------------------------------------------------

    def _scan_python_name_refs(
        self,
        file_path: str,
        lines: list[str],
        deleted_exports: set[str],
        deleted_stems: set[str],
    ) -> list[ScanResult]:
        """Find inline references to deleted symbols in Python files."""
        findings: list[ScanResult] = []

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Skip import lines (already handled)
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue

            # Check for ``deleted_module.some_call(`` — attribute access pattern
            for m in _PY_ATTR_REF.finditer(line):
                module_part = m.group(1)
                if module_part in deleted_stems:
                    findings.append(
                        ScanResult(
                            file_path=file_path,
                            line_number=lineno,
                            line_text=line.rstrip(),
                            severity="MEDIUM",
                            kind="name_ref",
                            symbol=f"{module_part}.{m.group(2)}",
                            suggested_fix=f"# Check reference: {module_part}.{m.group(2)}",
                            confidence=0.7,
                        )
                    )

            # Check for bare function calls that match deleted exports
            for m in _PY_NAME_CALL.finditer(line):
                name = m.group(1)
                if (
                    name in deleted_exports
                    and self._is_specific_name(name)
                    and not self._module_is_stdlib(name)
                ):
                    findings.append(
                        ScanResult(
                            file_path=file_path,
                            line_number=lineno,
                            line_text=line.rstrip(),
                            severity="MEDIUM",
                            kind="name_ref",
                            symbol=name,
                            suggested_fix=f"# Check reference: {name}()",
                            confidence=0.6,
                        )
                    )

        return findings

    # ------------------------------------------------------------------
    # JS/TS import scanner
    # ------------------------------------------------------------------

    def _scan_js_imports(
        self,
        file_path: str,
        lines: list[str],
        deleted_basenames: set[str],
        deleted_stems: set[str],
    ) -> list[ScanResult]:
        """Find JS/TS import/require statements referencing deleted files."""
        findings: list[ScanResult] = []
        text = "\n".join(lines)

        # Check each pattern
        for pattern, kind, severity, conf in [
            (_JS_IMPORT_MODULE, "import", "HIGH", 1.0),
            (_JS_EXPORT_MODULE, "import", "HIGH", 1.0),
            (_JS_EXPORT_STAR, "import", "HIGH", 1.0),
            (_JS_REQUIRE, "import", "HIGH", 0.95),
            (_JS_IMPORT_EQ_REQUIRE, "import", "HIGH", 0.95),
            (_JS_DYNAMIC_IMPORT, "import", "MEDIUM", 0.6),
        ]:
            for m in pattern.finditer(text):
                import_path = m.group(1)
                # Extract the filename from the import path
                imported_file = self._resolve_import_target(import_path)
                if imported_file and (
                    imported_file in deleted_stems
                    or imported_file in {Path(b).stem for b in deleted_basenames}
                ):
                    # Find the line number
                    line_num = text[: m.start()].count("\n") + 1
                    findings.append(
                        ScanResult(
                            file_path=file_path,
                            line_number=line_num,
                            line_text=lines[line_num - 1].rstrip(),
                            severity=severity,
                            kind=kind,
                            symbol=import_path,
                            suggested_fix=(
                                f"// Remove stale import: {import_path}"
                                if severity == "HIGH"
                                else f"// Check import: {import_path}"
                            ),
                            confidence=conf,
                        )
                    )

        return findings

    def _scan_js_name_refs(
        self,
        file_path: str,
        lines: list[str],
        deleted_exports: set[str],
        deleted_stems: set[str],
    ) -> list[ScanResult]:
        """Find JSX component references and symbol uses of deleted exports."""
        findings: list[ScanResult] = []

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("//", "/*", "*")):
                continue
            # Skip import/require lines
            if re.match(r"^\s*(import|const|let|var)\s", line):
                continue

            # Check for JSX component usage like <DeletedComponent />
            for m in _JSX_COMPONENT.finditer(line):
                name = m.group(1)
                if (
                    name in deleted_exports
                    and self._is_specific_name(name)
                ):
                    findings.append(
                        ScanResult(
                            file_path=file_path,
                            line_number=lineno,
                            line_text=line.rstrip(),
                            severity="MEDIUM",
                            kind="name_ref",
                            symbol=name,
                            suggested_fix=f"{{/* Check deleted component: {name} */}}",
                            confidence=0.65,
                        )
                    )
                elif name in deleted_stems and self._is_specific_name(name):
                    # The component name matches a deleted file stem
                    findings.append(
                        ScanResult(
                            file_path=file_path,
                            line_number=lineno,
                            line_text=line.rstrip(),
                            severity="MEDIUM",
                            kind="name_ref",
                            symbol=name,
                            suggested_fix=f"{{/* Check deleted component: {name} */}}",
                            confidence=0.5,
                        )
                    )

        return findings

    # ------------------------------------------------------------------
    # String path reference scanner (LOW severity, any language)
    # ------------------------------------------------------------------

    def _scan_string_refs(
        self,
        file_path: str,
        lines: list[str],
        deleted_basenames: set[str],
        deleted_stems: set[str],
    ) -> list[ScanResult]:
        """Find string literals containing paths that reference deleted files."""
        findings: list[ScanResult] = []

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            for m in _STRING_PATH_REF.finditer(line):
                path_ref = m.group(1)
                # Extract the filename from the path reference
                ref_parts = path_ref.replace("\\", "/").split("/")
                ref_filename = ref_parts[-1] if ref_parts else ""
                ref_stem = Path(ref_filename).stem

                if ref_stem in deleted_stems or ref_stem in {
                    Path(b).stem for b in deleted_basenames
                }:
                    # Deduplicate: skip if this line already has a HIGH/MEDIUM finding
                    findings.append(
                        ScanResult(
                            file_path=file_path,
                            line_number=lineno,
                            line_text=line.rstrip(),
                            severity="LOW",
                            kind="string_ref",
                            symbol=path_ref,
                            suggested_fix=f"# Update stale path reference: {path_ref}",
                            confidence=0.4,
                        )
                    )

        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_import_target(import_path: str) -> str:
        """Extract the target filename stem from an import path.

        Handles::

            ./utils/helper    → helper
            ../components/Bar → Bar
            @org/pkg/tools    → tools

        Args:
            import_path: The string from inside the import statement.

        Returns:
            The stem (filename without extension), or empty string.
        """
        # Normalize backslashes
        norm = import_path.replace("\\", "/")
        parts = norm.split("/")
        last = parts[-1] if parts else ""
        # Strip any extension (.js, .ts, .tsx, etc.)
        stem = Path(last).stem
        return stem

    @staticmethod
    def _is_specific_name(name: str) -> bool:
        """Check whether *name* is specific enough to avoid false positives.

        Filters out:
        - Names < 4 characters
        - Names in the GENERIC_NAMES set
        - all-lowercase single-word names that look like variables

        Args:
            name: The symbol name to check.

        Returns:
            ``True`` if the name is likely specific enough.
        """
        if len(name) < _MIN_SYMBOL_LENGTH:
            return False
        if name.lower() in _GENERIC_NAMES:
            return False
        return True

    def _load_remaining_source_map(
        self, deleted_paths: list[str]
    ) -> dict[str, str]:
        """Walk the project root and load source files excluding deleted ones."""
        from graphsift.adapters.filesystem import load_source_map

        deleted_set = set(os.path.abspath(p) for p in deleted_paths)
        source_map: dict[str, str] = {}

        if not self.project_root or not os.path.isdir(self.project_root):
            return source_map

        for path_str, source in load_source_map(self.project_root).items():
            if os.path.abspath(path_str) not in deleted_set:
                source_map[path_str] = source

        return source_map

    @staticmethod
    def _read_file_source(path: str) -> str:
        """Read file source, returning empty string on failure."""
        from graphsift.read_cache import SafeFileIO

        if os.path.isfile(path):
            return SafeFileIO.read(path)
        return ""

    @staticmethod
    def _build_report(
        findings: list[ScanResult], dry_run: bool
    ) -> StaleRefReport:
        """Aggregate findings into a StaleRefReport."""
        by_severity: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        by_kind: dict[str, int] = {}
        auto_fixable = 0

        for f in findings:
            sev = f.severity
            if sev in by_severity:
                by_severity[sev] += 1
            by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
            if sev == "HIGH":
                auto_fixable += 1

        return StaleRefReport(
            findings=findings,
            total=len(findings),
            by_severity={k: v for k, v in by_severity.items() if v > 0},
            by_kind=by_kind,
            auto_fixable=auto_fixable,
            dry_run=dry_run,
        )
