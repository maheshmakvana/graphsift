"""Fixtures for graphsift tests — comprehensive source factories and helpers."""

import os
import tempfile
import gc
import pytest

from graphsift import ContextBuilder, ContextConfig, DiffSpec, OutputMode


# ---------------------------------------------------------------------------
# Static source files for unit tests (original fixtures)
# ---------------------------------------------------------------------------

PYTHON_SOURCE_AUTH = '''"""Auth module."""
import hashlib
import os
from typing import Optional

class AuthManager:
    """Manages user authentication."""

    def __init__(self, secret: str):
        self.secret = secret

    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify(self, password: str, hashed: str) -> bool:
        return self.hash_password(password) == hashed

def create_token(user_id: str, secret: str) -> str:
    """Create an auth token."""
    return hashlib.sha256(f"{user_id}{secret}".encode()).hexdigest()
'''

PYTHON_SOURCE_USER = '''"""User module — imports from auth."""
from auth import AuthManager, create_token

class UserService:
    """Manages users."""

    def __init__(self):
        self.auth = AuthManager(secret="supersecret")
        self._users: dict = {}

    def register(self, username: str, password: str) -> str:
        hashed = self.auth.hash_password(password)
        self._users[username] = hashed
        return create_token(username, "supersecret")

    def login(self, username: str, password: str) -> bool:
        stored = self._users.get(username)
        if not stored:
            return False
        return self.auth.verify(password, stored)
'''

PYTHON_SOURCE_TEST = '''"""Tests for auth module."""
import pytest
from auth import AuthManager, create_token

def test_hash_password():
    mgr = AuthManager(secret="s")
    hashed = mgr.hash_password("hello")
    assert len(hashed) == 64

def test_verify():
    mgr = AuthManager(secret="s")
    h = mgr.hash_password("pw")
    assert mgr.verify("pw", h)

def test_create_token():
    token = create_token("user1", "secret")
    assert len(token) == 64
'''

PYTHON_SOURCE_UTILS = '''"""Utility functions — not related to auth."""
def format_date(ts: int) -> str:
    """Format a timestamp."""
    from datetime import datetime
    return datetime.fromtimestamp(ts).isoformat()

def slugify(text: str) -> str:
    """Convert text to URL slug."""
    return text.lower().replace(" ", "-")
'''


# ---------------------------------------------------------------------------
# Factory: auto-generated source files
# ---------------------------------------------------------------------------

def generate_source_files(
    count: int = 10,
    include_classes: bool = True,
    include_functions: bool = True,
    include_imports: bool = True,
    include_tests: bool = False,
    base_path: str = "src",
) -> dict[str, str]:
    """Generate *count* Python source files with classes, functions, imports.

    Creates a realistic codebase with cross-file dependencies for testing
    graph building, ranking, and context selection.

    Args:
        count: Number of source files to generate.
        include_classes: Whether to include class definitions.
        include_functions: Whether to include top-level functions.
        include_imports: Whether to include cross-file imports.
        include_tests: Whether to include test files (count // 4).
        base_path: Base directory for generated files.

    Returns:
        Dict mapping file paths to source code strings.
    """
    source_map = {}

    for i in range(count):
        mod_name = f"module_{i:04d}"
        file_path = f"{base_path}/{mod_name}.py"

        parts = []

        # Docstring
        parts.append(f'"""{mod_name} — auto-generated."""')
        parts.append("")

        # Imports (cross-reference neighboring modules)
        if include_imports and count > 1:
            imports = []
            for j in range(max(0, i - 3), i):
                imports.append(f"from module_{j:04d} import HelperClass{j}, top_func_{j}")
            for j in range(i + 1, min(count, i + 4)):
                imports.append(f"from module_{j:04d} import HelperClass{j}")
            if imports:
                parts.extend(imports[:5])  # Limit to 5 imports
                parts.append("")

        # Imports from stdlib
        parts.append("import os")
        parts.append("import sys")
        parts.append("from typing import Optional, List, Dict")
        parts.append("")

        # Class
        if include_classes:
            parts.append(f"class HelperClass{i}:")
            parts.append(f'    """Helper class {i}."""')
            parts.append(f"    def __init__(self, value: int = {i}):")
            parts.append("        self.value = value")
            parts.append("")
            parts.append(f'    def process(self, data: str) -> str:')
            parts.append(f'        """Process the data."""')
            parts.append(f'        return f"processed-{{data}}-{i}"')
            parts.append("")
            parts.append(f"    @staticmethod")
            parts.append(f"    def static_helper() -> int:")
            parts.append(f"        return {i * 2}")
            parts.append("")

        # Another class with dependencies
        if include_classes and i > 1:
            parts.append(f"class Consumer{i}:")
            parts.append(f'    """Consumer that uses helper classes."""')
            parts.append(f"    def __init__(self):")
            parts.append(f"        self.helper = HelperClass{i}()")
            parts.append("")
            parts.append(f"    def consume(self, item: str) -> str:")
            parts.append(f"        return self.helper.process(item)")
            parts.append("")

        # Top-level function
        if include_functions:
            parts.append(f"def top_func_{i}(arg: str = 'default') -> str:")
            parts.append(f'    """Top-level function {i}."""')
            parts.append(f"    helper = HelperClass{i}()")
            parts.append(f"    return helper.process(arg)")
            parts.append("")

            parts.append(f"def get_value_{i}() -> int:")
            parts.append(f'    """Get configured value."""')
            parts.append(f"    return {i * 10}")
            parts.append("")

        source_map[file_path] = "\n".join(parts)

    # Add test files
    if include_tests and count >= 4:
        test_count = max(1, count // 4)
        for i in range(test_count):
            mod_idx = i * 4
            test_path = f"tests/test_module_{mod_idx:04d}.py"
            test_source = f'''"""Tests for module_{mod_idx:04d}."""
import pytest
from src.module_{mod_idx:04d} import HelperClass{mod_idx}, top_func_{mod_idx}

class TestHelperClass:
    """Test the helper class."""

    def test_process(self):
        obj = HelperClass{mod_idx}(value=42)
        result = obj.process("test")
        assert "processed" in result

    def test_static_helper(self):
        result = HelperClass{mod_idx}.static_helper()
        assert result >= 0

def test_top_func():
    """Test the top-level function."""
    result = top_func_{mod_idx}("input")
    assert result is not None
'''
            source_map[test_path] = test_source

    return source_map


def generate_large_source_map(file_count: int = 100) -> dict[str, str]:
    """Generate a large source map for performance testing.

    Creates files with cross-references for realistic graph building.
    """
    return generate_source_files(
        count=file_count,
        include_classes=True,
        include_functions=True,
        include_imports=True,
        include_tests=(file_count >= 20),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_map():
    """Original static source map with 4 Python files."""
    return {
        "src/auth.py": PYTHON_SOURCE_AUTH,
        "src/user.py": PYTHON_SOURCE_USER,
        "tests/test_auth.py": PYTHON_SOURCE_TEST,
        "src/utils.py": PYTHON_SOURCE_UTILS,
    }


@pytest.fixture
def builder(source_map):
    """Pre-configured ContextBuilder with indexed source map."""
    b = ContextBuilder(ContextConfig(token_budget=50_000, output_mode=OutputMode.FULL))
    b.index_files(source_map)
    return b


@pytest.fixture
def builder_with_large_map(request):
    """ContextBuilder with a larger, auto-generated source map.

    Use with parametrize to control size::

        @pytest.mark.parametrize("builder_with_large_map", [50], indirect=True)
        def test_something(builder_with_large_map):
            builder, source_map = builder_with_large_map
    """
    size = getattr(request, "param", 50)
    source_map = generate_source_files(size)
    config = ContextConfig(token_budget=100_000)
    builder = ContextBuilder(config)
    builder.index_files(source_map)
    return builder, source_map


@pytest.fixture
def diff_spec():
    """Standard DiffSpec for testing."""
    return DiffSpec(
        changed_files=["src/auth.py"],
        query="Review the authentication changes",
        commit_message="refactor: improve auth token generation",
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_db():
    """Temporary SQLite database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def pre_built_graph(builder):
    """Pre-built DependencyGraph with known properties.

    The graph indexes 4 files:
    - src/auth.py (AuthManager class, hash_password, create_token)
    - src/user.py (UserService class, imports auth)
    - tests/test_auth.py (tests importing auth)
    - src/utils.py (standalone utility functions)
    """
    return builder._graph


@pytest.fixture
def known_graph_properties(pre_built_graph):
    """Known properties of the pre-built graph for assertions."""
    stats = pre_built_graph.stats()
    return {
        "graph": pre_built_graph,
        "file_count": stats.get("files", 0),
        "node_count": stats.get("nodes", 0),
        "edge_count": stats.get("edges", 0),
        "has_auth_file": any("auth.py" in n.file_path for n in pre_built_graph._nodes.values()),
        "has_user_file": any("user.py" in n.file_path for n in pre_built_graph._nodes.values()),
    }


@pytest.fixture
def performance_timer():
    """Context manager for measuring execution time.

    Usage::

        with performance_timer as timer:
            do_something()
        print(f"Took {timer.elapsed:.3f}s")
    """

    class Timer:
        def __enter__(self):
            import time
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            import time
            self.elapsed = time.perf_counter() - self.start

    return Timer()


# ---------------------------------------------------------------------------
# Register custom markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line("markers", "fuzz: Fuzz test with random inputs.")
    config.addinivalue_line("markers", "stress: Stress/performance test.")
    config.addinivalue_line("markers", "slow: Slow test that may take >10s.")
    config.addinivalue_line("markers", "integration: End-to-end integration test.")
    config.addinivalue_line("markers", "property: Property-based test with hypothesis.")
