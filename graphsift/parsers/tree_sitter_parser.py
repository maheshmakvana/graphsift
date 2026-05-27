"""Tree-sitter AST parser for 11 languages with graceful GenericParser fallback.

Design:
  - Lazy-loads tree-sitter on first instantiation.
  - Each supported language grammar is imported on demand.
  - Tree-walking approach using tree-sitter CST nodes.
  - Falls through to ``GenericParser`` when tree-sitter or a grammar is
    unavailable so callers always get a ``FileNode`` back.
  - Dynamic imports are detected via regex (same patterns across all parsers).

Usage::

    parser = TreeSitterParser()
    fn = parser.parse_file("src/auth.py", source_text)
    # If tree-sitter is not installed, falls through to GenericParser
"""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import Any

from graphsift.core import GenericParser, detect_language, estimate_tokens, register_parser
from graphsift.exceptions import ParseError
from graphsift.models import FileNode, GraphNode, Language, NodeKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language grammar module map
# ---------------------------------------------------------------------------

_LANGUAGE_GRAMMAR_MAP: dict[Language, str] = {
    Language.PYTHON: "tree_sitter_python",
    Language.JAVASCRIPT: "tree_sitter_javascript",
    Language.TYPESCRIPT: "tree_sitter_typescript",
    Language.GO: "tree_sitter_go",
    Language.RUST: "tree_sitter_rust",
    Language.JAVA: "tree_sitter_java",
    Language.C: "tree_sitter_c",
    Language.CPP: "tree_sitter_cpp",
    Language.RUBY: "tree_sitter_ruby",
    Language.PHP: "tree_sitter_php",
    Language.BASH: "tree_sitter_bash",
}

# ---------------------------------------------------------------------------
# Dynamic import regex patterns (shared across all languages)
# ---------------------------------------------------------------------------

_DYN_IMPORT_PATTERNS: list[re.Pattern] = [
    re.compile(r'import\s*\(\s*["\']([^"\']+)["\']\s*\)'),
    re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)'),
    re.compile(r'importlib\.import_module\s*\(\s*["\']([^"\']+)["\']\s*\)'),
    re.compile(r'__import__\s*\(\s*["\']([^"\']+)["\']\s*\)'),
    re.compile(r'importlib\.util\.spec_from_file_location\([^,]+,\s*["\']([^"\']+)["\']\s*\)'),
    re.compile(r'plugin\.Open\s*\(\s*["\']([^"\']+)["\']\s*\)'),
    re.compile(r'libloading::Library::new\s*\(\s*["\']([^"\']+)["\']\s*\)'),
    re.compile(r'(?:source|\.)\s+([\w./\-]+)'),
]


# ===================================================================
# TreeSitterParser
# ===================================================================


class TreeSitterParser:
    """Tree-sitter based language parser with fallback to ``GenericParser``.

    Implements the ``LanguageParser`` protocol (``parse_file``,
    ``extract_signatures``).

    Tree-sitter and each language grammar are optionally installed:
    ``pip install tree-sitter tree-sitter-python tree-sitter-javascript ...``

    If tree-sitter is missing, or if a specific language grammar is missing,
    the parser silently falls through to the regex-based ``GenericParser``.

    Args:
        None
    """

    def __init__(self) -> None:
        self._ts: Any = None
        self._ts_available = False
        self._grammars: dict[Language, Any] = {}
        self._generic = GenericParser()
        self._init_tree_sitter()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_tree_sitter(self) -> None:
        """Lazy-import tree-sitter. Raises a clear error if unavailable."""
        try:
            # tree-sitter >= 0.23
            import tree_sitter as _ts  # noqa: PLC0415
            self._ts = _ts
            self._ts_available = True
        except ImportError:
            logger.debug("tree-sitter not installed — falling back to GenericParser")

    def _get_grammar(self, lang: Language) -> Any | None:
        """Load the tree-sitter grammar for *lang*, or return ``None``.

        Resolution order:
          1. Individual grammar package (e.g. ``tree_sitter_python``)
          2. ``tree_sitter_language_pack`` (bundled grammars)
          3. ``None`` → ``GenericParser`` fallback
        """
        if lang in self._grammars:
            return self._grammars[lang]

        if not self._ts_available:
            return None

        pkg_name = _LANGUAGE_GRAMMAR_MAP.get(lang)
        if pkg_name is None:
            return None

        # --- Try 1: individual grammar package ---
        try:
            mod = importlib.import_module(pkg_name)
            # Each grammar module exposes a ``language()`` callable
            # that returns a ``tree_sitter.Language`` instance.
            lang_obj = mod.language()
            self._grammars[lang] = lang_obj
            logger.debug("Loaded tree-sitter grammar for %s", lang.value)
            return lang_obj
        except ImportError:
            pass

        # --- Try 2: bundled tree-sitter-language-pack ---
        try:
            lang_pack = importlib.import_module("tree_sitter_language_pack")
            get_language = getattr(lang_pack, "get_language", None)
            if get_language is not None:
                lang_obj = get_language(lang.value)
                self._grammars[lang] = lang_obj
                logger.debug(
                    "Loaded tree-sitter grammar for %s via language pack",
                    lang.value,
                )
                return lang_obj
        except (ImportError, Exception):
            pass

        logger.debug(
            "tree-sitter grammar not available for %s (%s) — using GenericParser",
            lang.value,
            pkg_name,
        )
        return None

    # ------------------------------------------------------------------
    # LanguageParser protocol
    # ------------------------------------------------------------------

    def parse_file(self, path: str, source: str) -> FileNode:
        """Parse *source* for the language detected from *path*.

        If tree-sitter (or the language grammar) is not available, delegates
        to ``GenericParser``.

        Args:
            path: File path (used for language detection and node IDs).
            source: Full source code text.

        Returns:
            ``FileNode`` with extracted symbols, imports, dynamic_imports.
        """
        lang = detect_language(path)
        ts_lang = self._get_grammar(lang)
        if ts_lang is None:
            return self._generic.parse_file(path, source)

        try:
            parser = self._ts.Parser(ts_lang)
            tree = parser.parse(source.encode("utf-8"))
            root = tree.root_node
        except Exception as exc:
            logger.warning(
                "tree-sitter parse failed for %s: %s — falling back to GenericParser",
                path,
                exc,
            )
            return self._generic.parse_file(path, source)

        return self._extract(root, ts_lang, lang, path, source)

    def extract_signatures(self, source: str) -> str:
        """Return signatures-only view (no bodies).

        Uses tree-sitter to identify function/class/method boundaries and
        returns the signature line (up to the opening ``{``, ``:``, or body).

        Tries multiple language grammars and returns the first non-empty
        result since we don't have a file path for language detection.

        Args:
            source: Full source code text.

        Returns:
            Condensed string with only function/class signatures.
        """
        # Try languages in order — return first non-empty result
        attempts = [
            (Language.PYTHON, self._extract_py_signatures),
            (Language.JAVASCRIPT, self._extract_js_signatures),
            (Language.GO, None),
            (Language.RUST, None),
        ]
        for lang, extractor in attempts:
            ts_lang = self._get_grammar(lang)
            if ts_lang is None:
                continue
            try:
                parser = self._ts.Parser(ts_lang)
                tree = parser.parse(source.encode("utf-8"))
                root = tree.root_node
                if extractor is not None:
                    result = extractor(root, source, ts_lang)
                else:
                    result = self._capture_signatures(root, source, ts_lang)
                if result.strip():
                    return result
            except Exception:
                continue

        # Fallback to generic
        return self._generic.extract_signatures(source)

    # ------------------------------------------------------------------
    # Capture helper
    # ------------------------------------------------------------------

    def _capture(
        self, root_node: Any, ts_lang: Any, query_src: str
    ) -> list[Any]:
        """Run a tree-sitter query and return captured nodes.

        Supports both the new ``QueryCursor`` API (tree-sitter >= 0.25)
        and the deprecated ``Language.query()`` / ``Query.captures()`` API
        for backwards compatibility.
        """
        # Track whether the new API was attempted (even if empty results)
        new_api_attempted = False
        try:
            from tree_sitter import Query, QueryCursor  # noqa: PLC0415
            query = Query(ts_lang, query_src)
            qc = QueryCursor(query)
            matches = qc.matches(root_node)
            nodes: list[Any] = []
            for _pattern_idx, captures_dict in matches:
                for _cap_name, cap_nodes in captures_dict.items():
                    nodes.extend(cap_nodes)
            new_api_attempted = True
            return nodes
        except Exception:
            pass

        # Fallback: old API (Language.query + Query.captures)
        # Only use if the new API was not available (not just empty results)
        if not new_api_attempted:
            try:
                query = ts_lang.query(query_src)
                return [node for node, _ in query.captures(root_node)]
            except Exception:
                pass
        return []

    def _capture_signatures(
        self, root_node: Any, source: str, ts_lang: Any
    ) -> str:
        """Generic signature extraction using tree-sitter.

        Works for any language by capturing function_definition,
        function_declaration, function_item, method_declaration, and
        class/struct nodes, then taking the text up to the body.
        """
        lines: list[str] = []

        # Try common function-like node types
        for func_type in (
            "function_definition",
            "function_declaration",
            "function_item",
            "method_declaration",
            "arrow_function",
        ):
            for node in self._capture(root_node, ts_lang, f"({func_type}) @n"):
                body = node.child_by_field_name("body")
                if body is not None:
                    sig = source[node.start_byte : body.start_byte].strip()
                    if sig and sig not in lines:
                        lines.append(sig)

        # Try class/struct-like node types
        for cls_type in (
            "class_definition",
            "class_declaration",
            "struct_item",
            "struct_specifier",
            "class_specifier",
        ):
            for node in self._capture(root_node, ts_lang, f"({cls_type}) @n"):
                body = node.child_by_field_name("body")
                if body is not None:
                    sig = source[node.start_byte : body.start_byte].strip()
                    if sig and sig not in lines:
                        lines.append(sig)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Node text helper
    # ------------------------------------------------------------------

    @staticmethod
    def _node_text(source: str, node: Any) -> str:
        """Return source text spanned by *node*."""
        return source[node.start_byte : node.end_byte]

    @staticmethod
    def _line_num(node: Any) -> int:
        """Return 1-indexed start line."""
        return node.start_point[0] + 1

    # ==============================================================
    # Dispatch
    # ==============================================================

    def _extract(
        self,
        root_node: Any,
        ts_lang: Any,
        lang: Language,
        path: str,
        source: str,
    ) -> FileNode:
        """Dispatch to language-specific extractor."""
        extractors = {
            Language.PYTHON: self._extract_python,
            Language.JAVASCRIPT: self._extract_js_ts,
            Language.TYPESCRIPT: self._extract_js_ts,
            Language.GO: self._extract_go,
            Language.RUST: self._extract_rust,
            Language.JAVA: self._extract_java,
            Language.C: self._extract_c_cpp,
            Language.CPP: self._extract_c_cpp,
            Language.RUBY: self._extract_ruby,
            Language.PHP: self._extract_php,
            Language.BASH: self._extract_bash,
        }

        extractor = extractors.get(lang)
        if extractor is None:
            return self._generic.parse_file(path, source)

        symbols, imports, dynamic_imports = extractor(root_node, ts_lang, path, source)

        sha = hashlib_256(source)
        return FileNode(
            path=path,
            language=lang,
            size_bytes=len(source.encode("utf-8", errors="replace")),
            line_count=len(source.splitlines()),
            sha256=sha,
            symbols=symbols,
            imports=imports,
            dynamic_imports=dynamic_imports,
            token_estimate=estimate_tokens(source),
        )

    # ==============================================================
    # Python extractor
    # ==============================================================

    def _extract_python(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []
        seen_imports: set[str] = set()

        # --- Decorator map: (inner_start, inner_end) → [decorator_names] ---
        decorator_map: dict[tuple[int, int], list[str]] = {}
        for dnode in self._capture(root_node, ts_lang, "(decorated_definition) @n"):
            decs: list[str] = []
            inner = None
            for child in dnode.children:
                if child.type == "decorator":
                    dtext = self._node_text(source, child).lstrip("@")
                    decs.append(dtext.strip())
                elif child.type in ("function_definition", "class_definition"):
                    inner = child
            if inner is not None and decs:
                decorator_map[(inner.start_byte, inner.end_byte)] = decs

        # --- Import statements ---
        for inode in self._capture(root_node, ts_lang, "(import_statement) @n"):
            for child in inode.named_children:
                if child.type == "dotted_name":
                    mod = self._node_text(source, child)
                    if mod not in seen_imports:
                        seen_imports.add(mod)
                        imports.append(mod)

        # --- Import-from statements ---
        for ifnode in self._capture(
            root_node, ts_lang, "(import_from_statement) @n"
        ):
            module_node = ifnode.child_by_field_name("module_name")
            module_name = (
                self._node_text(source, module_node) if module_node else ""
            )
            if module_name and module_name not in seen_imports:
                seen_imports.add(module_name)
                imports.append(module_name)
            for child in ifnode.named_children:
                if child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node is not None and module_name:
                        full = f"{module_name}.{self._node_text(source, name_node)}"
                        if full not in seen_imports:
                            seen_imports.add(full)
                            imports.append(full)

        # --- Functions ---
        for fnode in self._capture(root_node, ts_lang, "(function_definition) @n"):
            self._process_py_function(fnode, source, path, symbols, decorator_map)

        # --- Classes ---
        for cnode in self._capture(root_node, ts_lang, "(class_definition) @n"):
            self._process_py_class(cnode, source, path, symbols, decorator_map)

        # --- Module-level assignments ---
        var_query = """
            (expression_statement
                (assignment
                    left: (identifier) @name
                    right: (_) @value)) @variable
        """
        for vnode in self._capture(root_node, ts_lang, var_query):
            # Find the identifier child
            for child in vnode.named_children:
                if child.type == "assignment":
                    left = child.child_by_field_name("left")
                    if left is not None and left.type == "identifier":
                        varname = self._node_text(source, left)
                        line = self._line_num(vnode)
                        symbols.append(GraphNode(
                            node_id=f"{path}::{varname}",
                            file_path=path,
                            kind=NodeKind.VARIABLE,
                            name=varname,
                            qualified_name=varname,
                            line_start=line,
                            language=Language.PYTHON,
                        ))

        # --- Dynamic imports via regex ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    def _process_py_function(
        self,
        fnode: Any,
        source: str,
        path: str,
        symbols: list[GraphNode],
        decorator_map: dict[tuple[int, int], list[str]],
    ) -> None:
        name_node = fnode.child_by_field_name("name")
        params_node = fnode.child_by_field_name("parameters")
        return_type_node = fnode.child_by_field_name("return_type")
        if name_node is None:
            return

        name = self._node_text(source, name_node)
        qual = name
        kind = NodeKind.FUNCTION

        # Check if inside a class (method)
        parent = fnode.parent
        while parent is not None and parent.type not in (
            "class_definition", "decorated_definition", "block",
        ):
            parent = parent.parent

        if parent is not None and parent.type == "class_definition":
            cn = parent.child_by_field_name("name")
            if cn is not None:
                qual = f"{self._node_text(source, cn)}.{name}"
                kind = NodeKind.METHOD
        elif parent is not None and parent.type == "decorated_definition":
            # Decorated function — check if decorated_definition is inside a class
            dp = parent.parent
            while dp is not None and dp.type not in (
                "class_definition", "block",
            ):
                dp = dp.parent
            if dp is not None and dp.type == "class_definition":
                cn = dp.child_by_field_name("name")
                if cn is not None:
                    qual = f"{self._node_text(source, cn)}.{name}"
                    kind = NodeKind.METHOD
            elif dp is not None and dp.type == "block":
                pp = dp.parent
                if pp is not None and pp.type == "class_definition":
                    cn = pp.child_by_field_name("name")
                    if cn is not None:
                        qual = f"{self._node_text(source, cn)}.{name}"
                        kind = NodeKind.METHOD
        elif parent is not None and parent.type == "block":
            # Check if block's parent is a class_definition
            pp = parent.parent
            if pp is not None and pp.type == "class_definition":
                cn = pp.child_by_field_name("name")
                if cn is not None:
                    qual = f"{self._node_text(source, cn)}.{name}"
                    kind = NodeKind.METHOD

        sig = self._build_py_signature(name, params_node, return_type_node, source)
        is_async = self._check_py_async(fnode, source)

        line_start = self._line_num(fnode)
        line_end = fnode.end_point[0] + 1

        decs = decorator_map.get((fnode.start_byte, fnode.end_byte), [])

        symbols.append(GraphNode(
            node_id=f"{path}::{qual}",
            file_path=path,
            kind=kind,
            name=name,
            qualified_name=qual,
            line_start=line_start,
            line_end=line_end,
            language=Language.PYTHON,
            signature=sig,
            decorators=decs,
            is_async=is_async,
        ))

    def _process_py_class(
        self,
        cnode: Any,
        source: str,
        path: str,
        symbols: list[GraphNode],
        decorator_map: dict[tuple[int, int], list[str]],
    ) -> None:
        name_node = cnode.child_by_field_name("name")
        if name_node is None:
            return

        name = self._node_text(source, name_node)

        # Extract bases from argument_list
        bases: list[str] = []
        for child in cnode.children:
            if child.type == "argument_list":
                for arg in child.named_children:
                    bases.append(self._node_text(source, arg))

        line_start = self._line_num(cnode)
        line_end = cnode.end_point[0] + 1
        decs = decorator_map.get((cnode.start_byte, cnode.end_byte), [])

        symbols.append(GraphNode(
            node_id=f"{path}::{name}",
            file_path=path,
            kind=NodeKind.CLASS,
            name=name,
            qualified_name=name,
            line_start=line_start,
            line_end=line_end,
            language=Language.PYTHON,
            decorators=decs,
            metadata={"bases": bases},
        ))

    @staticmethod
    def _check_py_async(fnode: Any, source: str) -> bool:
        """Return ``True`` if the function definition is async."""
        text = source[fnode.start_byte : fnode.end_byte]
        return text.startswith("async")

    @staticmethod
    def _build_py_signature(
        name: str,
        params_node: Any,
        return_type_node: Any,
        source: str,
    ) -> str:
        """Build a compact function signature string."""
        params = ""
        if params_node is not None:
            params = source[params_node.start_byte : params_node.end_byte]
        ret = ""
        if return_type_node is not None:
            ret = f" -> {source[return_type_node.start_byte : return_type_node.end_byte]}"
        return f"def {name}{params}{ret}"

    def _extract_py_signatures(
        self, root_node: Any, source: str, ts_lang: Any
    ) -> str:
        """Extract signatures from a Python tree."""
        lines: list[str] = []

        # Class signatures
        for node in self._capture(root_node, ts_lang, "(class_definition) @n"):
            body = node.child_by_field_name("body")
            if body is not None:
                sig = source[node.start_byte : body.start_byte].strip().rstrip(":")
                lines.append(f"{sig}:")
            # Also collect method signatures within
            for child in node.named_children:
                if child.type == "block":
                    for sub in self._capture(
                        child, ts_lang, "(function_definition) @n"
                    ):
                        body2 = sub.child_by_field_name("body")
                        if body2 is not None:
                            s = source[sub.start_byte : body2.start_byte].strip().rstrip(":")
                            lines.append(f"    {s}:")

        # Top-level function signatures (that aren't methods inside a class)
        decorated_seen: set[int] = set()
        for dnode in self._capture(
            root_node, ts_lang, "(decorated_definition) @n"
        ):
            outer_source = self._node_text(source, dnode)
            for child in dnode.children:
                if child.type == "function_definition":
                    body = child.child_by_field_name("body")
                    if body is not None:
                        sig = source[child.start_byte : body.start_byte].strip().rstrip(":")
                        lines.append(f"{sig}:")
                    decorated_seen.add(child.start_byte)

        for node in self._capture(root_node, ts_lang, "(function_definition) @n"):
            if node.start_byte in decorated_seen:
                continue
            body = node.child_by_field_name("body")
            if body is not None:
                sig = source[node.start_byte : body.start_byte].strip().rstrip(":")
                lines.append(f"{sig}:")

        return "\n".join(lines)

    # ==============================================================
    # JavaScript / TypeScript extractor
    # ==============================================================

    def _extract_js_ts(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []
        seen_imports: set[str] = set()

        lang = Language.JAVASCRIPT if path.endswith((".js", ".mjs", ".cjs")) else Language.TYPESCRIPT

        # --- Function declarations ---
        for fnode in self._capture(root_node, ts_lang, "(function_declaration) @n"):
            name_node = fnode.child_by_field_name("name")
            params_node = fnode.child_by_field_name("parameters")
            if name_node is None:
                continue
            name = self._node_text(source, name_node)
            sig = self._node_text(source, fnode).split("{")[0].strip()
            line_start = self._line_num(fnode)
            line_end = fnode.end_point[0] + 1
            is_async = source[fnode.start_byte : fnode.start_byte + 5] == "async"
            symbols.append(GraphNode(
                node_id=f"{path}::{name}",
                file_path=path,
                kind=NodeKind.FUNCTION,
                name=name,
                qualified_name=name,
                line_start=line_start,
                line_end=line_end,
                language=lang,
                signature=sig,
                is_async=is_async,
            ))

        # --- Arrow functions assigned to variables ---
        arrow_query = """
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @name
                    value: (arrow_function) @fn)) @arrow
        """
        for anode in self._capture(root_node, ts_lang, arrow_query):
            # Find the name child (variable_declarator.name)
            for child in anode.named_children:
                if child.type == "variable_declarator":
                    nname = child.child_by_field_name("name")
                    val = child.child_by_field_name("value")
                    if nname is not None and val is not None:
                        arrow_name = self._node_text(source, nname)
                        sig_text = (
                            f"const {arrow_name} = "
                            f"{self._node_text(source, val).split('=>')[0].strip()} =>"
                        )
                        line_start = self._line_num(anode)
                        line_end = anode.end_point[0] + 1
                        is_async = source[
                            val.start_byte : val.start_byte + 5
                        ] == "async"
                        symbols.append(GraphNode(
                            node_id=f"{path}::{arrow_name}",
                            file_path=path,
                            kind=NodeKind.FUNCTION,
                            name=arrow_name,
                            qualified_name=arrow_name,
                            line_start=line_start,
                            line_end=line_end,
                            language=lang,
                            signature=sig_text,
                            is_async=is_async,
                        ))

        # --- Classes ---
        for cnode in self._capture(root_node, ts_lang, "(class_declaration) @n"):
            name_node = cnode.child_by_field_name("name")
            if name_node is None:
                continue
            name = self._node_text(source, name_node)
            # Extract base class from heritage clause
            bases: list[str] = []
            for child in cnode.children:
                if child.type == "class_heritage":
                    for sub in child.named_children:
                        bases.append(self._node_text(source, sub))
            line_start = self._line_num(cnode)
            line_end = cnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{name}",
                file_path=path,
                kind=NodeKind.CLASS,
                name=name,
                qualified_name=name,
                line_start=line_start,
                line_end=line_end,
                language=lang,
                metadata={"bases": bases},
            ))

        # --- Methods inside classes ---
        for mnode in self._capture(root_node, ts_lang, "(method_definition) @n"):
            name_node = mnode.child_by_field_name("name")
            params_node = mnode.child_by_field_name("parameters")
            if name_node is None:
                continue
            mname = self._node_text(source, name_node)
            # Find parent class
            parent = mnode.parent
            while parent is not None and parent.type not in ("class_body", "class_declaration"):
                parent = parent.parent
            class_name = ""
            if parent is not None:
                cd = parent if parent.type == "class_declaration" else parent.parent
                if cd is not None and cd.type == "class_declaration":
                    cn = cd.child_by_field_name("name")
                    if cn is not None:
                        class_name = self._node_text(source, cn)
            qual = f"{class_name}.{mname}" if class_name else mname
            sig = self._node_text(source, mnode).split("{")[0].strip()
            line_start = self._line_num(mnode)
            line_end = mnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{qual}",
                file_path=path,
                kind=NodeKind.METHOD,
                name=mname,
                qualified_name=qual,
                line_start=line_start,
                line_end=line_end,
                language=lang,
                signature=sig,
            ))

        # --- Import statements ---
        for inode in self._capture(root_node, ts_lang, "(import_statement) @n"):
            for child in inode.named_children:
                if child.type == "string":
                    src = self._node_text(source, child).strip("\"'")
                    if src not in seen_imports:
                        seen_imports.add(src)
                        imports.append(src)

        # --- Dynamic imports via regex ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    def _extract_js_signatures(
        self, root_node: Any, source: str, ts_lang: Any
    ) -> str:
        lines: list[str] = []
        for node in self._capture(root_node, ts_lang, "(function_declaration) @n"):
            body = node.child_by_field_name("body")
            if body is not None:
                sig = source[node.start_byte : body.start_byte].strip()
                lines.append(sig)
        for node in self._capture(root_node, ts_lang, "(class_declaration) @n"):
            body = node.child_by_field_name("body")
            if body is not None:
                sig = source[node.start_byte : body.start_byte].strip()
                lines.append(sig)
        return "\n".join(lines)

    # ==============================================================
    # Go extractor
    # ==============================================================

    def _extract_go(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []
        seen_imports: set[str] = set()

        # --- Import declarations ---
        for inode in self._capture(root_node, ts_lang, "(import_declaration) @n"):
            # Collect import_spec nodes which may be directly in import_spec_list
            spec_nodes: list[Any] = []
            for child in inode.named_children:
                if child.type == "import_spec":
                    spec_nodes.append(child)
                elif child.type == "import_spec_list":
                    for spec_child in child.named_children:
                        if spec_child.type == "import_spec":
                            spec_nodes.append(spec_child)
            for spec_node in spec_nodes:
                path_node = spec_node.child_by_field_name("path")
                if path_node is not None:
                    imp_path = self._node_text(source, path_node).strip("\"")
                    if imp_path not in seen_imports:
                        seen_imports.add(imp_path)
                        imports.append(imp_path)

        # --- Function declarations ---
        for fnode in self._capture(
            root_node, ts_lang, "(function_declaration) @n"
        ):
            name_node = fnode.child_by_field_name("name")
            if name_node is None:
                continue
            name = self._node_text(source, name_node)
            sig = self._node_text(source, fnode).split("{")[0].strip()
            line_start = self._line_num(fnode)
            line_end = fnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{name}",
                file_path=path,
                kind=NodeKind.FUNCTION,
                name=name,
                qualified_name=name,
                line_start=line_start,
                line_end=line_end,
                language=Language.GO,
                signature=sig,
            ))

        # --- Method declarations ---
        for mnode in self._capture(
            root_node, ts_lang, "(method_declaration) @n"
        ):
            name_node = mnode.child_by_field_name("name")
            receiver_node = mnode.child_by_field_name("receiver")
            if name_node is None or receiver_node is None:
                continue
            mname = self._node_text(source, name_node)
            # Extract receiver type name
            recv_text = self._node_text(source, receiver_node)
            # Parse receiver text like "(r *MyStruct)" or "(s MyStruct)"
            recv_match = re.search(r"\*?(\w+)$", recv_text.strip("()"))
            type_name = recv_match.group(1) if recv_match else ""
            qual = f"{type_name}.{mname}" if type_name else mname
            sig = self._node_text(source, mnode).split("{")[0].strip()
            line_start = self._line_num(mnode)
            line_end = mnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{qual}",
                file_path=path,
                kind=NodeKind.METHOD,
                name=mname,
                qualified_name=qual,
                line_start=line_start,
                line_end=line_end,
                language=Language.GO,
                signature=sig,
                metadata={"receiver_type": type_name} if type_name else {},
            ))

        # --- Struct type specs ---
        for snode in self._capture(root_node, ts_lang, """
            (type_declaration
                (type_spec
                    name: (type_identifier) @name
                    type: (struct_type))) @struct
        """):
            name_node = self._find_named_child(snode, "type_spec", "type_identifier")
            if name_node is not None:
                sname = self._node_text(source, name_node)
                line_start = self._line_num(snode)
                symbols.append(GraphNode(
                    node_id=f"{path}::{sname}",
                    file_path=path,
                    kind=NodeKind.CLASS,
                    name=sname,
                    qualified_name=sname,
                    line_start=line_start,
                    language=Language.GO,
                    metadata={"go_kind": "struct"},
                ))

        # --- Interface type specs ---
        for inode in self._capture(root_node, ts_lang, """
            (type_declaration
                (type_spec
                    name: (type_identifier) @name
                    type: (interface_type))) @interface
        """):
            name_node = self._find_named_child(inode, "type_spec", "type_identifier")
            if name_node is not None:
                iname = self._node_text(source, name_node)
                line_start = self._line_num(inode)
                symbols.append(GraphNode(
                    node_id=f"{path}::{iname}",
                    file_path=path,
                    kind=NodeKind.CLASS,
                    name=iname,
                    qualified_name=iname,
                    line_start=line_start,
                    language=Language.GO,
                    metadata={"go_kind": "interface", "is_interface": True},
                ))

        # --- Dynamic imports ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    # ==============================================================
    # Rust extractor
    # ==============================================================

    def _extract_rust(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []
        seen_imports: set[str] = set()

        # --- Use declarations ---
        for unode in self._capture(root_node, ts_lang, "(use_declaration) @n"):
            imp = self._extract_rust_use_path(unode, source)
            if imp and imp not in seen_imports:
                seen_imports.add(imp)
                imports.append(imp)

        # --- Function items ---
        for fnode in self._capture(root_node, ts_lang, "(function_item) @n"):
            name_node = fnode.child_by_field_name("name")
            if name_node is None:
                continue
            name = self._node_text(source, name_node)
            sig = self._node_text(source, fnode).split("{")[0].strip()
            line_start = self._line_num(fnode)
            line_end = fnode.end_point[0] + 1
            is_pub = source[fnode.start_byte : fnode.start_byte + 3] == "pub"
            is_async = "async" in source[fnode.start_byte : fnode.start_byte + 60].split("{")[0]
            qual = name
            # Check if inside impl block
            parent = fnode.parent
            while parent is not None:
                if parent.type == "impl_item":
                    type_node = parent.child_by_field_name("type")
                    if type_node is not None:
                        qual = f"{self._node_text(source, type_node)}.{name}"
                    break
                parent = parent.parent
            kind = NodeKind.METHOD if "." in qual else NodeKind.FUNCTION
            symbols.append(GraphNode(
                node_id=f"{path}::{qual}",
                file_path=path,
                kind=kind,
                name=name,
                qualified_name=qual,
                line_start=line_start,
                line_end=line_end,
                language=Language.RUST,
                signature=sig,
                is_async=is_async,
                metadata={"is_pub": is_pub} if is_pub else {},
            ))

        # --- Struct items ---
        for snode in self._capture(root_node, ts_lang, "(struct_item) @n"):
            name_node = snode.child_by_field_name("name")
            if name_node is not None:
                sname = self._node_text(source, name_node)
                line_start = self._line_num(snode)
                symbols.append(GraphNode(
                    node_id=f"{path}::{sname}",
                    file_path=path,
                    kind=NodeKind.CLASS,
                    name=sname,
                    qualified_name=sname,
                    line_start=line_start,
                    language=Language.RUST,
                    metadata={"rust_kind": "struct"},
                ))

        # --- Enum items ---
        for enode in self._capture(root_node, ts_lang, "(enum_item) @n"):
            name_node = enode.child_by_field_name("name")
            if name_node is not None:
                ename = self._node_text(source, name_node)
                line_start = self._line_num(enode)
                symbols.append(GraphNode(
                    node_id=f"{path}::{ename}",
                    file_path=path,
                    kind=NodeKind.CLASS,
                    name=ename,
                    qualified_name=ename,
                    line_start=line_start,
                    language=Language.RUST,
                    metadata={"rust_kind": "enum"},
                ))

        # --- Trait items ---
        for tnode in self._capture(root_node, ts_lang, "(trait_item) @n"):
            name_node = tnode.child_by_field_name("name")
            if name_node is not None:
                tname = self._node_text(source, name_node)
                line_start = self._line_num(tnode)
                symbols.append(GraphNode(
                    node_id=f"{path}::{tname}",
                    file_path=path,
                    kind=NodeKind.CLASS,
                    name=tname,
                    qualified_name=tname,
                    line_start=line_start,
                    language=Language.RUST,
                    metadata={"rust_kind": "trait", "is_interface": True},
                ))

        # --- Dynamic imports ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    @staticmethod
    def _extract_rust_use_path(unode: Any, source: str) -> str:
        """Extract the import path from a Rust ``use_declaration`` node."""
        for child in unode.named_children:
            if child.type in ("scoped_identifier", "scoped_use_list"):
                # use std::collections::HashMap  or  use serde::{...}
                text = source[child.start_byte : child.end_byte]
                return text
            if child.type == "identifier":
                # use crate or use foo
                text = source[child.start_byte : child.end_byte]
                return text
        # Fallback: full text minus "use " and ";"
        full = source[unode.start_byte : unode.end_byte]
        return full.replace("use ", "", 1).rstrip(";").strip()

    # ==============================================================
    # Java extractor
    # ==============================================================

    def _extract_java(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []
        seen_imports: set[str] = set()

        # --- Import declarations ---
        for inode in self._capture(root_node, ts_lang, "(import_declaration) @n"):
            for child in inode.named_children:
                if child.type == "scoped_identifier":
                    imp = self._node_text(source, child)
                    if imp not in seen_imports:
                        seen_imports.add(imp)
                        imports.append(imp)

        # --- Class declarations ---
        for cnode in self._capture(root_node, ts_lang, "(class_declaration) @n"):
            name_node = cnode.child_by_field_name("name")
            if name_node is None:
                continue
            name = self._node_text(source, name_node)
            # Extract superclass from superclass clause
            bases: list[str] = []
            for child in cnode.children:
                if child.type == "superclass":
                    for sub in child.named_children:
                        bname = self._node_text(source, sub)
                        if bname:
                            bases.append(bname)
            line_start = self._line_num(cnode)
            line_end = cnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{name}",
                file_path=path,
                kind=NodeKind.CLASS,
                name=name,
                qualified_name=name,
                line_start=line_start,
                line_end=line_end,
                language=Language.JAVA,
                metadata={"bases": bases} if bases else {},
            ))

        # --- Interface declarations ---
        for inode in self._capture(root_node, ts_lang, "(interface_declaration) @n"):
            name_node = inode.child_by_field_name("name")
            if name_node is not None:
                iname = self._node_text(source, name_node)
                line_start = self._line_num(inode)
                symbols.append(GraphNode(
                    node_id=f"{path}::{iname}",
                    file_path=path,
                    kind=NodeKind.CLASS,
                    name=iname,
                    qualified_name=iname,
                    line_start=line_start,
                    language=Language.JAVA,
                    metadata={"is_interface": True},
                ))

        # --- Method declarations ---
        for mnode in self._capture(root_node, ts_lang, "(method_declaration) @n"):
            name_node = mnode.child_by_field_name("name")
            if name_node is None:
                continue
            mname = self._node_text(source, name_node)
            qual = mname
            # Check parent class
            parent = mnode.parent
            while parent is not None and parent.type != "class_body":
                parent = parent.parent
            if parent is not None and parent.parent is not None:
                cn = parent.parent.child_by_field_name("name")
                if cn is not None:
                    qual = f"{self._node_text(source, cn)}.{mname}"
            sig = self._node_text(source, mnode).split("{")[0].strip()
            line_start = self._line_num(mnode)
            line_end = mnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{qual}",
                file_path=path,
                kind=NodeKind.METHOD,
                name=mname,
                qualified_name=qual,
                line_start=line_start,
                line_end=line_end,
                language=Language.JAVA,
                signature=sig,
            ))

        # --- Dynamic imports ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    # ==============================================================
    # C / C++ extractor (shared)
    # ==============================================================

    def _extract_c_cpp(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []
        seen_imports: set[str] = set()
        lang = Language.C if path.endswith((".c", ".h")) else Language.CPP

        # --- Preprocessor includes ---
        for inode in self._capture(root_node, ts_lang, "(preproc_include) @n"):
            path_node = inode.child_by_field_name("path")
            if path_node is not None:
                inc = self._node_text(source, path_node).strip("\"<>")
                if inc not in seen_imports:
                    seen_imports.add(inc)
                    imports.append(inc)

        # --- Function definitions ---
        for fnode in self._capture(
            root_node, ts_lang, "(function_definition) @n"
        ):
            declarator = fnode.child_by_field_name("declarator")
            if declarator is None:
                continue
            # The function name is inside the declarator
            name = self._extract_c_func_name(declarator, source)
            if not name:
                continue
            sig = self._node_text(source, fnode).split("{")[0].strip()
            line_start = self._line_num(fnode)
            line_end = fnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{name}",
                file_path=path,
                kind=NodeKind.FUNCTION,
                name=name,
                qualified_name=name,
                line_start=line_start,
                line_end=line_end,
                language=lang,
                signature=sig,
            ))

        # --- Class specifiers (C++) ---
        if lang == Language.CPP:
            for cnode in self._capture(
                root_node, ts_lang, "(class_specifier) @n"
            ):
                name_node = cnode.child_by_field_name("name")
                if name_node is not None:
                    cname = self._node_text(source, name_node)
                    line_start = self._line_num(cnode)
                    symbols.append(GraphNode(
                        node_id=f"{path}::{cname}",
                        file_path=path,
                        kind=NodeKind.CLASS,
                        name=cname,
                        qualified_name=cname,
                        line_start=line_start,
                        language=Language.CPP,
                    ))

            # --- Namespace definitions ---
            for nnode in self._capture(
                root_node, ts_lang, "(namespace_definition) @n"
            ):
                name_node = nnode.child_by_field_name("name")
                if name_node is not None:
                    ns = self._node_text(source, name_node)
                    line_start = self._line_num(nnode)
                    symbols.append(GraphNode(
                        node_id=f"{path}::{ns}",
                        file_path=path,
                        kind=NodeKind.MODULE,
                        name=ns,
                        qualified_name=ns,
                        line_start=line_start,
                        language=Language.CPP,
                    ))

        # --- Dynamic imports ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    @staticmethod
    def _extract_c_func_name(declarator: Any, source: str) -> str:
        """Extract function name from a C/C++ function declarator."""
        # Function declarator: `function_declarator` → `identifier`
        if declarator.type == "identifier":
            return source[declarator.start_byte : declarator.end_byte]
        if declarator.type == "function_declarator":
            # Look for the direct child that is an identifier or nested declarator
            for child in declarator.children:
                if child.type == "identifier":
                    return source[child.start_byte : child.end_byte]
                if child.type in ("pointer_declarator", "function_declarator"):
                    return TreeSitterParser._extract_c_func_name(child, source)
            # Also check named children
            for ch in declarator.named_children:
                if ch.type == "identifier":
                    return source[ch.start_byte : ch.end_byte]
                if ch.type in ("pointer_declarator", "function_declarator"):
                    return TreeSitterParser._extract_c_func_name(ch, source)
        if declarator.type == "pointer_declarator":
            for child in declarator.children:
                if child.type == "identifier":
                    return source[child.start_byte : child.end_byte]
                if child.type in ("function_declarator", "pointer_declarator"):
                    return TreeSitterParser._extract_c_func_name(child, source)
        return ""

    # ==============================================================
    # Ruby extractor
    # ==============================================================

    def _extract_ruby(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []
        seen_imports: set[str] = set()

        # --- Method definitions ---
        for mnode in self._capture(root_node, ts_lang, "(method) @n"):
            name_node = mnode.child_by_field_name("name")
            if name_node is None:
                continue
            mname = self._node_text(source, name_node)
            params_node = mnode.child_by_field_name("parameters")
            params = (
                self._node_text(source, params_node)
                if params_node is not None
                else "()"
            )
            sig = f"def {mname}{params}"
            line_start = self._line_num(mnode)
            line_end = mnode.end_point[0] + 1
            # Check if defined inside a class
            parent = mnode.parent
            while parent is not None and parent.type not in ("class", "module"):
                parent = parent.parent
            qual = mname
            kind = NodeKind.FUNCTION
            if parent is not None and parent.type == "class":
                cn = parent.child_by_field_name("name")
                if cn is not None:
                    qual = f"{self._node_text(source, cn)}.{mname}"
                    kind = NodeKind.METHOD
            symbols.append(GraphNode(
                node_id=f"{path}::{qual}",
                file_path=path,
                kind=kind,
                name=mname,
                qualified_name=qual,
                line_start=line_start,
                line_end=line_end,
                language=Language.RUBY,
                signature=sig,
            ))

        # --- Class definitions ---
        for cnode in self._capture(root_node, ts_lang, "(class) @n"):
            name_node = cnode.child_by_field_name("name")
            if name_node is not None:
                cname = self._node_text(source, name_node)
                # Check for superclass
                super_node = cnode.child_by_field_name("superclass")
                bases = (
                    [self._node_text(source, super_node)]
                    if super_node is not None
                    else []
                )
                line_start = self._line_num(cnode)
                symbols.append(GraphNode(
                    node_id=f"{path}::{cname}",
                    file_path=path,
                    kind=NodeKind.CLASS,
                    name=cname,
                    qualified_name=cname,
                    line_start=line_start,
                    language=Language.RUBY,
                    metadata={"bases": bases} if bases else {},
                ))

        # --- Module definitions ---
        for mnode in self._capture(root_node, ts_lang, "(module) @n"):
            name_node = mnode.child_by_field_name("name")
            if name_node is not None:
                mname = self._node_text(source, name_node)
                line_start = self._line_num(mnode)
                symbols.append(GraphNode(
                    node_id=f"{path}::{mname}",
                    file_path=path,
                    kind=NodeKind.MODULE,
                    name=mname,
                    qualified_name=mname,
                    line_start=line_start,
                    language=Language.RUBY,
                ))

        # --- require calls ---
        for rnode in self._capture(root_node, ts_lang, """
            (call method: (identifier) @method
                  arguments: (argument_list (string (string_content) @path))) @require
        """):
            method_name = None
            for child in rnode.children:
                if child.type == "identifier":
                    method_name = self._node_text(source, child)
                    break
            if method_name in ("require", "require_relative", "autoload", "load"):
                for child in rnode.children:
                    if child.type == "argument_list":
                        for arg in child.named_children:
                            if arg.type in ("string", "string_content"):
                                req = (
                                    self._node_text(source, arg)
                                    .strip("\"'")
                                )
                                if req and req not in seen_imports:
                                    seen_imports.add(req)
                                    imports.append(req)

        # --- Dynamic imports ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    # ==============================================================
    # PHP extractor
    # ==============================================================

    def _extract_php(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []
        seen_imports: set[str] = set()

        # --- Function definitions ---
        for fnode in self._capture(
            root_node, ts_lang, "(function_definition) @n"
        ):
            name_node = fnode.child_by_field_name("name")
            if name_node is None:
                continue
            name = self._node_text(source, name_node)
            params_node = fnode.child_by_field_name("parameters")
            params = (
                self._node_text(source, params_node)
                if params_node is not None
                else "()"
            )
            sig = f"function {name}{params}"
            line_start = self._line_num(fnode)
            line_end = fnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{name}",
                file_path=path,
                kind=NodeKind.FUNCTION,
                name=name,
                qualified_name=name,
                line_start=line_start,
                line_end=line_end,
                language=Language.PHP,
                signature=sig,
            ))

        # --- Class declarations ---
        for cnode in self._capture(
            root_node, ts_lang, "(class_declaration) @n"
        ):
            name_node = cnode.child_by_field_name("name")
            if name_node is None:
                continue
            cname = self._node_text(source, name_node)
            bases: list[str] = []
            for child in cnode.children:
                if child.type == "base_clause":
                    for bc in child.named_children:
                        bname = self._node_text(source, bc)
                        if bname:
                            bases.append(bname)
            line_start = self._line_num(cnode)
            line_end = cnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{cname}",
                file_path=path,
                kind=NodeKind.CLASS,
                name=cname,
                qualified_name=cname,
                line_start=line_start,
                line_end=line_end,
                language=Language.PHP,
                metadata={"bases": bases} if bases else {},
            ))

        # --- Method declarations ---
        for mnode in self._capture(
            root_node, ts_lang, "(method_declaration) @n"
        ):
            name_node = mnode.child_by_field_name("name")
            if name_node is None:
                continue
            mname = self._node_text(source, name_node)
            # Determine parent class
            parent = mnode.parent
            while parent is not None and parent.type not in ("class_body", "class_declaration"):
                parent = parent.parent
            qual = mname
            if parent is not None:
                cd = parent if parent.type == "class_declaration" else None
                if cd is None and parent.parent is not None:
                    cd = parent.parent if parent.parent.type == "class_declaration" else None
                if cd is not None:
                    cn = cd.child_by_field_name("name")
                    if cn is not None:
                        qual = f"{self._node_text(source, cn)}.{mname}"
            params_node = mnode.child_by_field_name("parameters")
            params = (
                self._node_text(source, params_node)
                if params_node is not None
                else "()"
            )
            sig = f"function {mname}{params}"
            line_start = self._line_num(mnode)
            line_end = mnode.end_point[0] + 1
            symbols.append(GraphNode(
                node_id=f"{path}::{qual}",
                file_path=path,
                kind=NodeKind.METHOD,
                name=mname,
                qualified_name=qual,
                line_start=line_start,
                line_end=line_end,
                language=Language.PHP,
                signature=sig,
            ))

        # --- Include/require expressions ---
        for inode in self._capture(root_node, ts_lang, """
            (expression_statement
                (include_expression
                    (string) @path)) @import
        """):
            for child in inode.named_children:
                if child.type == "include_expression":
                    for sub in child.named_children:
                        if sub.type == "string":
                            inc = self._node_text(source, sub).strip("\"'")
                            if inc and inc not in seen_imports:
                                seen_imports.add(inc)
                                imports.append(inc)

        # --- Dynamic imports ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    # ==============================================================
    # Bash extractor
    # ==============================================================

    def _extract_bash(
        self,
        root_node: Any,
        ts_lang: Any,
        path: str,
        source: str,
    ) -> tuple[list[GraphNode], list[str], list[str]]:
        symbols: list[GraphNode] = []
        imports: list[str] = []
        dynamic_imports: list[str] = []

        # --- Function definitions ---
        for fnode in self._capture(
            root_node, ts_lang, "(function_definition) @n"
        ):
            name_node = fnode.child_by_field_name("name")
            if name_node is None:
                continue
            name = self._node_text(source, name_node).strip("()")
            sig = f"function {name}()"
            line_start = self._line_num(fnode)
            symbols.append(GraphNode(
                node_id=f"{path}::{name}",
                file_path=path,
                kind=NodeKind.FUNCTION,
                name=name,
                qualified_name=name,
                line_start=line_start,
                language=Language.BASH,
                signature=sig,
            ))

        # --- Variable assignments (declaration_command) ---
        for vnode in self._capture(root_node, ts_lang, """
            (declaration_command
                (variable_name) @name) @variable
        """):
            vname = self._node_text(source, vnode).strip("=")
            line_start = self._line_num(vnode)
            symbols.append(GraphNode(
                node_id=f"{path}::{vname}",
                file_path=path,
                kind=NodeKind.VARIABLE,
                name=vname,
                qualified_name=vname,
                line_start=line_start,
                language=Language.BASH,
            ))

        # --- Source imports (detected via simple_command) ---
        for snode in self._capture(root_node, ts_lang, """
            (simple_command
                command: (command_name (word) @cmd)
                argument: (word) @path) @source
        """):
            cmd = None
            arg = None
            for child in snode.children:
                if child.type == "command_name":
                    for sub in child.named_children:
                        cmd = self._node_text(source, sub).strip()
                        break
                elif child.type == "word":
                    # This might be the argument
                    if cmd is not None:
                        arg = self._node_text(source, child).strip()
            if cmd in ("source", ".") and arg:
                if arg not in imports:
                    imports.append(arg)

        # --- Dynamic imports via regex ---
        self._extract_dynamic_imports(source, dynamic_imports)

        return symbols, imports, dynamic_imports

    # ==============================================================
    # Shared helpers
    # ==============================================================

    @staticmethod
    def _find_named_child(
        node: Any, child_type: str, grandchild_type: str
    ) -> Any | None:
        """Find a named child of *child_type* and its named grandchild of *grandchild_type*."""
        for child in node.named_children:
            if child.type == child_type:
                for gc in child.named_children:
                    if gc.type == grandchild_type:
                        return gc
        return None

    def _extract_dynamic_imports(
        self, source: str, dynamic_imports: list[str]
    ) -> None:
        """Extract dynamic import strings via regex."""
        for pat in _DYN_IMPORT_PATTERNS:
            for m in pat.finditer(source):
                mod = m.group(1)
                if mod not in dynamic_imports:
                    dynamic_imports.append(mod)


# ===================================================================
# Module-level helpers
# ===================================================================


def hashlib_256(source: str) -> str:
    """SHA-256 hex digest."""
    import hashlib
    return hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()


# ===================================================================
# Convenience: register tree-sitter for all available grammars
# ===================================================================


def register_tree_sitter_parsers() -> dict[Language, bool]:
    """Try to register ``TreeSitterParser`` for every available language grammar.

    This function iterates over all supported languages, attempts to load
    the corresponding tree-sitter grammar module, and registers the
    ``TreeSitterParser`` in the global parser registry for each successfully
    loaded grammar.

    Call this once at application startup if you want tree-sitter accuracy
    for all languages where the grammar happens to be installed::

        from graphsift.parsers import register_tree_sitter_parsers

        results = register_tree_sitter_parsers()
        # results == {Language.PYTHON: True, Language.JAVASCRIPT: False, ...}

    Returns:
        Dict mapping each ``Language`` to whether registration succeeded.
    """
    results: dict[Language, bool] = {}
    parser = TreeSitterParser()

    for lang in _LANGUAGE_GRAMMAR_MAP:
        ts_lang = parser._get_grammar(lang)
        if ts_lang is not None:
            register_parser(lang, parser)
            results[lang] = True
            logger.info("Registered TreeSitterParser for %s", lang.value)
        else:
            results[lang] = False
            logger.debug("Could not register TreeSitterParser for %s", lang.value)

    return results
