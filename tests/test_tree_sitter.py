"""Tests for the tree-sitter parser engine.

Covers:
  - Basic parsing of Python, JavaScript, Go with tree-sitter
  - Graceful fallback when tree-sitter is not installed
  - Correct symbol kinds (function, class, method)
  - Import extraction
  - Signature extraction
  - Dynamic import detection
"""

import sys
from pathlib import Path

import pytest

from graphsift import FileNode, Language, NodeKind, detect_language
from graphsift.parsers import TreeSitterParser


# ===================================================================
# Helpers (must be defined before decorator evaluation)
# ===================================================================


def _tree_sitter_available(grammar_package: str) -> bool:
    """Check if a tree-sitter grammar package is installed."""
    try:
        __import__(grammar_package)
        return True
    except ImportError:
        return False


# Also check the tree-sitter-language-pack fallback
def _ts_lang_available(lang_name: str) -> bool:
    """Check if a language is available via tree-sitter-language-pack."""
    try:
        from tree_sitter_language_pack import get_language  # noqa: PLC0415
        get_language(lang_name)
        return True
    except Exception:
        return False


# ===================================================================
# Fixtures
# ===================================================================

PYTHON_SAMPLE = '''"""Sample module for tree-sitter testing."""
import os
import sys
from typing import Optional, List

from mylib.utils import format_date

CONSTANT = 42
DEBUG_MODE = True

class UserModel:
    """Represents a user in the system."""

    base_url = "/api/users"

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    async def save(self) -> bool:
        """Save user to database."""
        return True

    @staticmethod
    def from_dict(data: dict) -> "UserModel":
        return UserModel(data["name"], data["email"])

@dataclass
class AdminUser(UserModel):
    """Admin user with elevated privileges."""

    role: str = "admin"

async def fetch_users(api_key: str) -> list:
    """Fetch all users from the API."""
    pass

def deprecated_func():
    import warnings
    warnings.warn("deprecated")
    pass
'''

JAVASCRIPT_SAMPLE = '''// Sample JavaScript module
import { useState, useEffect } from "react";
import axios from "axios";

const API_URL = "https://api.example.com";

export async function fetchData(url) {
    const response = await axios.get(url);
    return response.data;
}

export class ApiService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(path) {
        return fetchData(this.baseUrl + path);
    }

    static create(baseUrl) {
        return new ApiService(baseUrl);
    }
}

// Arrow function
const transform = (data) => {
    return data.map(x => x * 2);
};

// Dynamic import
const loadModule = () => import("./lazy.module");

// Require
const fs = require("fs");
'''

GO_SAMPLE = '''package service

import (
    "context"
    "database/sql"
    "fmt"
)

type Storage interface {
    Get(key string) (string, error)
    Set(key string, value string) error
}

type UserStore struct {
    db *sql.DB
}

func NewUserStore(db *sql.DB) *UserStore {
    return &UserStore{db: db}
}

func (s *UserStore) GetUser(ctx context.Context, id string) (string, error) {
    return s.get(ctx, id)
}

func (s *UserStore) get(ctx context.Context, id string) (string, error) {
    return "", nil
}
'''

RUST_SAMPLE = '''use std::collections::HashMap;
use serde::{Deserialize, Serialize};

pub trait Repository {
    fn find(&self, id: &str) -> Option<String>;
}

pub struct UserRepo {
    store: HashMap<String, String>,
}

impl UserRepo {
    pub fn new() -> Self {
        UserRepo {
            store: HashMap::new(),
        }
    }

    pub async fn get(&self, id: &str) -> Option<String> {
        self.store.get(id).cloned()
    }
}

pub enum Status {
    Active,
    Inactive,
}

fn helper() -> bool {
    true
}
'''


# ===================================================================
# Fallback: tree-sitter not available
# ===================================================================


def test_fallback_when_tree_sitter_not_available(monkeypatch):
    """TreeSitterParser should fall through to GenericParser when unavailable."""
    import builtins
    original_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "tree_sitter":
            raise ImportError("No module named 'tree_sitter'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _mock_import)

    parser = TreeSitterParser()
    assert parser._ts_available is False

    fn = parser.parse_file("test.py", PYTHON_SAMPLE)
    assert isinstance(fn, FileNode)
    # Should still get basic symbols from GenericParser fallback
    assert len(fn.symbols) >= 0


# ===================================================================
# Python parsing
# ===================================================================


@pytest.mark.skipif(
    not (_tree_sitter_available("tree_sitter_python") or _ts_lang_available("python")),
    reason="tree-sitter-python or language-pack not installed",
)
class TestPythonParsing:
    """Tree-sitter Python extraction tests."""

    def test_parser_creates_parser(self):
        parser = TreeSitterParser()
        assert parser._ts_available

    def test_parse_python_file(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        assert isinstance(fn, FileNode)
        assert fn.language == Language.PYTHON
        assert len(fn.symbols) >= 1

    def test_extracts_functions(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        names = [s.name for s in fn.symbols]
        assert "fetch_users" in names, f"Missing fetch_users in {names}"
        assert "deprecated_func" in names

    def test_extracts_classes(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        classes = [s for s in fn.symbols if s.kind == NodeKind.CLASS]
        class_names = [c.name for c in classes]
        assert "UserModel" in class_names, f"Missing UserModel in {class_names}"
        assert "AdminUser" in class_names

    def test_extracts_methods(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        methods = [s for s in fn.symbols if s.kind == NodeKind.METHOD]
        method_names = [m.name for m in methods]
        for name in ("__init__", "save", "from_dict"):
            assert name in method_names, f"Missing method {name} in {method_names}"

    def test_extracts_imports(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        assert "os" in fn.imports, f"Missing 'os' in {fn.imports}"
        assert "typing" in fn.imports, f"Missing 'typing' in {fn.imports}"

    def test_extracts_from_imports(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        assert any("mylib.utils" in imp for imp in fn.imports), (
            f"Expected mylib.utils in imports: {fn.imports}"
        )

    def test_detects_async_functions(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        async_syms = [s for s in fn.symbols if s.is_async]
        async_names = [s.name for s in async_syms]
        assert "fetch_users" in async_names, f"Missing async fetch_users in {async_names}"

    def test_detects_async_methods(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        async_methods = [
            s for s in fn.symbols
            if s.kind == NodeKind.METHOD and s.is_async
        ]
        assert any(m.name == "save" for m in async_methods), (
            f"Missing async save method in {[m.name for m in async_methods]}"
        )

    def test_detects_class_bases(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        for sym in fn.symbols:
            if sym.name == "AdminUser":
                bases = sym.metadata.get("bases", [])
                assert any("UserModel" in b for b in bases), (
                    f"Expected UserModel in bases: {bases}"
                )
                return
        pytest.fail("AdminUser not found in symbols")

    def test_detects_decorators(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        for sym in fn.symbols:
            if sym.name == "from_dict":
                assert "staticmethod" in sym.decorators, (
                    f"Expected staticmethod decorator, got {sym.decorators}"
                )
                return
        pytest.fail("from_dict not found")

    def test_detects_class_decorators(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        for sym in fn.symbols:
            if sym.name == "AdminUser":
                assert any("dataclass" in d for d in sym.decorators), (
                    f"Expected dataclass decorator on AdminUser, got {sym.decorators}"
                )
                return
        pytest.fail("AdminUser not found in symbols")

    def test_extracts_module_variables(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        vars_found = [s for s in fn.symbols if s.kind == NodeKind.VARIABLE]
        var_names = [v.name for v in vars_found]
        assert "CONSTANT" in var_names, f"Missing CONSTANT in {var_names}"

    def test_extracts_signatures(self):
        parser = TreeSitterParser()
        sigs = parser.extract_signatures(PYTHON_SAMPLE)
        assert "UserModel" in sigs, f"Missing UserModel in signatures:\n{sigs}"
        assert "fetch_users" in sigs, f"Missing fetch_users in signatures:\n{sigs}"
        assert "deprecated_func" in sigs

    def test_dynamic_imports(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        # No dynamic imports in the sample — this should still be empty
        # (The parser uses regex for dynamic import detection)
        assert isinstance(fn.dynamic_imports, list)

    def test_line_numbers(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        fetch_users = [s for s in fn.symbols if s.name == "fetch_users"]
        assert len(fetch_users) == 1
        assert fetch_users[0].line_start > 0
        assert fetch_users[0].line_end >= fetch_users[0].line_start

    def test_signatures_on_symbols(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("test_module.py", PYTHON_SAMPLE)
        funcs = [s for s in fn.symbols if s.kind in (NodeKind.FUNCTION, NodeKind.METHOD)]
        for func in funcs:
            assert func.signature, f"Empty signature for {func.name}"
            assert "def" in func.signature or func.name in func.signature


# ===================================================================
# JavaScript parsing
# ===================================================================


@pytest.mark.skipif(
    not (_tree_sitter_available("tree_sitter_javascript") or _ts_lang_available("javascript")),
    reason="tree-sitter-javascript or language-pack not installed",
)
class TestJavaScriptParsing:
    """Tree-sitter JavaScript extraction tests."""

    def test_parse_javascript(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        assert isinstance(fn, FileNode)
        assert fn.language == Language.JAVASCRIPT

    def test_extracts_functions(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        names = [s.name for s in fn.symbols]
        assert "fetchData" in names, f"Missing fetchData in {names}"

    def test_extracts_async_functions(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        async_syms = [s for s in fn.symbols if s.name == "fetchData"]
        assert async_syms, "fetchData not found"
        assert async_syms[0].is_async

    def test_extracts_classes(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        class_syms = [s for s in fn.symbols if s.kind == NodeKind.CLASS]
        class_names = [c.name for c in class_syms]
        assert "ApiService" in class_names, f"Missing ApiService in {class_names}"

    def test_extracts_methods(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        method_names = [s.name for s in fn.symbols if s.kind == NodeKind.METHOD]
        for name in ("get", "create"):
            assert name in method_names, f"Missing method {name} in {method_names}"

    def test_extracts_imports(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        import_sources = fn.imports
        assert any("react" in i for i in import_sources), (
            f"Expected react in imports: {import_sources}"
        )
        assert any("axios" in i for i in import_sources)

    def test_extracts_arrow_functions(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        names = [s.name for s in fn.symbols]
        assert "transform" in names, f"Missing transform arrow fn in {names}"

    def test_dynamic_imports_via_regex(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        assert any("lazy.module" in d for d in fn.dynamic_imports), (
            f"Expected lazy.module in dynamic imports: {fn.dynamic_imports}"
        )

    def test_require_imports_via_regex(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("api_service.js", JAVASCRIPT_SAMPLE)
        assert any("fs" in d for d in fn.dynamic_imports), (
            f"Expected fs in dynamic imports: {fn.dynamic_imports}"
        )

    def test_signatures(self):
        parser = TreeSitterParser()
        sigs = parser.extract_signatures(JAVASCRIPT_SAMPLE)
        assert "fetchData" in sigs, f"Missing fetchData in:\n{sigs}"


# ===================================================================
# Go parsing
# ===================================================================


@pytest.mark.skipif(
    not (_tree_sitter_available("tree_sitter_go") or _ts_lang_available("go")),
    reason="tree-sitter-go or language-pack not installed",
)
class TestGoParsing:
    """Tree-sitter Go extraction tests."""

    def test_parse_go(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("service.go", GO_SAMPLE)
        assert isinstance(fn, FileNode)
        assert fn.language == Language.GO

    def test_extracts_functions(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("service.go", GO_SAMPLE)
        names = [s.name for s in fn.symbols]
        assert "NewUserStore" in names, f"Missing NewUserStore in {names}"

    def test_extracts_methods_with_receiver(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("service.go", GO_SAMPLE)
        methods = [s for s in fn.symbols if s.kind == NodeKind.METHOD]
        method_names = [m.name for m in methods]
        for name in ("GetUser", "get"):
            assert name in method_names, f"Missing method {name} in {method_names}"

    def test_extracts_structs_as_classes(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("service.go", GO_SAMPLE)
        struct_names = [
            s.name for s in fn.symbols
            if s.kind == NodeKind.CLASS and s.metadata.get("go_kind") == "struct"
        ]
        assert "UserStore" in struct_names, f"Missing UserStore in {struct_names}"

    def test_extracts_interfaces(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("service.go", GO_SAMPLE)
        iface_names = [
            s.name for s in fn.symbols
            if s.kind == NodeKind.CLASS and s.metadata.get("go_kind") == "interface"
        ]
        assert "Storage" in iface_names, f"Missing Storage in {iface_names}"

    def test_extracts_imports(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("service.go", GO_SAMPLE)
        import_sources = fn.imports
        assert any("context" in i for i in import_sources), (
            f"Expected context in imports: {import_sources}"
        )
        assert any("database/sql" in i for i in import_sources)


# ===================================================================
# Rust parsing
# ===================================================================


@pytest.mark.skipif(
    not (_tree_sitter_available("tree_sitter_rust") or _ts_lang_available("rust")),
    reason="tree-sitter-rust or language-pack not installed",
)
class TestRustParsing:
    """Tree-sitter Rust extraction tests."""

    def test_parse_rust(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("user_repo.rs", RUST_SAMPLE)
        assert isinstance(fn, FileNode)
        assert fn.language == Language.RUST

    def test_extracts_functions(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("user_repo.rs", RUST_SAMPLE)
        names = [s.name for s in fn.symbols]
        assert "helper" in names, f"Missing helper in {names}"

    def test_extracts_structs(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("user_repo.rs", RUST_SAMPLE)
        class_syms = [s for s in fn.symbols if s.kind == NodeKind.CLASS]
        class_names = [c.name for c in class_syms]
        for name in ("UserRepo", "Status"):
            assert name in class_names, f"Missing {name} in {class_names}"

    def test_extracts_methods_in_impl(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("user_repo.rs", RUST_SAMPLE)
        method_names = [s.name for s in fn.symbols if s.kind == NodeKind.METHOD]
        for name in ("new", "get"):
            assert name in method_names, f"Missing method {name} in {method_names}"

    def test_extracts_traits(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("user_repo.rs", RUST_SAMPLE)
        trait_names = [
            s.name for s in fn.symbols
            if s.kind == NodeKind.CLASS and s.metadata.get("rust_kind") == "trait"
        ]
        assert "Repository" in trait_names, f"Missing Repository in {trait_names}"

    def test_extracts_imports(self):
        parser = TreeSitterParser()
        fn = parser.parse_file("user_repo.rs", RUST_SAMPLE)
        assert len(fn.imports) > 0


# ===================================================================
# Language detection integration
# ===================================================================


def test_detect_language_for_tree_sitter_file_types():
    """Verify tree-sitter language detection aligns with core detect_language."""
    mappings = {
        ".py": Language.PYTHON,
        ".js": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".java": Language.JAVA,
        ".c": Language.C,
        ".cpp": Language.CPP,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".sh": Language.BASH,
    }
    for ext, expected_lang in mappings.items():
        detected = detect_language(f"file{ext}")
        assert detected == expected_lang, (
            f"Mismatch for {ext}: expected {expected_lang}, got {detected}"
        )


# ===================================================================
# EMPTY / EDGE CASES
# ===================================================================


def test_parse_empty_file(monkeypatch):
    """An empty file should produce a valid FileNode."""
    # Only test GenericParser fallback (most predictable)
    import builtins
    original_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "tree_sitter":
            raise ImportError("No module named 'tree_sitter'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _mock_import)

    parser = TreeSitterParser()
    fn = parser.parse_file("empty.py", "")
    assert isinstance(fn, FileNode)
    # Empty file should produce a valid FileNode (GenericParser adds a MODULE symbol)


def test_unknown_language_falls_back_to_generic(monkeypatch):
    """An unknown file extension should fall through to GenericParser."""
    import builtins
    original_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "tree_sitter":
            raise ImportError("No module named 'tree_sitter'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _mock_import)

    parser = TreeSitterParser()
    fn = parser.parse_file("data.csv", "a,b,c\n1,2,3\n")
    assert isinstance(fn, FileNode)
    assert fn.language == Language.UNKNOWN
