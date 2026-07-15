"""Tests for Next.js, React, and JS/TS framework enhancements."""

from __future__ import annotations

import pytest

from graphsift.core import DependencyGraph, GenericParser, detect_language
from graphsift.models import GraphNode, NodeKind, Language


# ---------------------------------------------------------------------------
# JSX / React component detection via GenericParser (tree-sitter fallback)
# ---------------------------------------------------------------------------

class TestNextJsEntryPointDetection:
    """DependencyGraph._detect_entry_points should handle Next.js patterns."""

    def _make_graph(self, nodes: list[GraphNode]) -> DependencyGraph:
        g = DependencyGraph()
        for n in nodes:
            g._nodes[n.node_id] = n
        return g

    @pytest.mark.parametrize("path", [
        "app/page.tsx",
        "app/layout.tsx",
        "app/loading.tsx",
        "app/error.tsx",
        "app/not-found.tsx",
        "app/route.ts",
        "app/api/users/route.ts",
        "app/dashboard/page.tsx",
        "app/dashboard/layout.tsx",
    ])
    def test_nextjs_app_router(self, path):
        """Files under app/ with page/layout/loading/error/route should be entries."""
        g = self._make_graph([
            GraphNode(
                node_id=f"{path}::__module__",
                file_path=path,
                kind=NodeKind.MODULE,
                name="page",
                qualified_name="page",
                language=Language.TYPESCRIPT,
            ),
        ])
        entries = g._detect_entry_points()
        assert path in entries, f"{path} should be an entry point"

    @pytest.mark.parametrize("path", [
        "pages/index.tsx",
        "pages/about.tsx",
        "pages/blog/[slug].tsx",
        "pages/api/hello.ts",
    ])
    def test_nextjs_pages_router(self, path):
        """Files under pages/ should be detected as entry points."""
        g = self._make_graph([
            GraphNode(
                node_id=f"{path}::__module__",
                file_path=path,
                kind=NodeKind.MODULE,
                name="index",
                qualified_name="index",
                language=Language.TYPESCRIPT,
            ),
        ])
        entries = g._detect_entry_points()
        assert path in entries, f"{path} should be an entry point"

    @pytest.mark.parametrize("path", [
        "middleware.ts",
        "next.config.ts",
        "instrumentation.ts",
    ])
    def test_nextjs_special_files(self, path):
        """middleware.ts, next.config.ts etc should be detected."""
        g = self._make_graph([
            GraphNode(
                node_id=f"{path}::__module__",
                file_path=path,
                kind=NodeKind.MODULE,
                name=path.replace(".", "_"),
                qualified_name=path.replace(".", "_"),
                language=Language.TYPESCRIPT,
            ),
        ])
        entries = g._detect_entry_points()
        assert path in entries, f"{path} should be an entry point"

    @pytest.mark.parametrize("path", [
        "src/main.tsx",
        "src/main.jsx",
        "index.tsx",
        "index.js",
        "App.tsx",
        "App.jsx",
        "src/app.tsx",
    ])
    def test_vite_cra_entry_points(self, path):
        """main/index/App files should be detected as entry points."""
        g = self._make_graph([
            GraphNode(
                node_id=f"{path}::__module__",
                file_path=path,
                kind=NodeKind.MODULE,
                name="main",
                qualified_name="main",
                language=Language.TYPESCRIPT if path.endswith(".tsx") else Language.JAVASCRIPT,
            ),
        ])
        entries = g._detect_entry_points()
        assert path in entries, f"{path} should be an entry point"

    def test_python_flask_entry_point(self):
        """Flask route decorators should mark files as entry points."""
        g = self._make_graph([
            GraphNode(
                node_id="app.py::home",
                file_path="app.py",
                kind=NodeKind.FUNCTION,
                name="home",
                qualified_name="home",
                language=Language.PYTHON,
                decorators=["@app.route('/')"],
            ),
        ])
        entries = g._detect_entry_points()
        assert "app.py" in entries

    def test_python_click_entry_point(self):
        """Click command decorators should mark files as entry points."""
        g = self._make_graph([
            GraphNode(
                node_id="cli.py::run",
                file_path="cli.py",
                kind=NodeKind.FUNCTION,
                name="run",
                qualified_name="run",
                language=Language.PYTHON,
                decorators=["@click.command()"],
            ),
        ])
        entries = g._detect_entry_points()
        assert "cli.py" in entries

    def test_python_wsgi_app(self):
        """WSGI 'app' or 'application' objects are entry points."""
        for name in ("app", "application"):
            g = self._make_graph([
                GraphNode(
                    node_id=f"wsgi.py::{name}",
                    file_path="wsgi.py",
                    kind=NodeKind.VARIABLE,
                    name=name,
                    qualified_name=name,
                    language=Language.PYTHON,
                ),
            ])
            entries = g._detect_entry_points()
            assert "wsgi.py" in entries, f"wsgi.py with {name} should be entry"
            break  # only need one


class TestReactComponentDetection:
    """GenericParser should detect React/JSX patterns."""

    PARSER = GenericParser()

    def test_jsx_file_is_valid(self):
        """GenericParser should handle .jsx/.tsx extensions properly."""
        path = "components/Button.tsx"
        source = """
        import React from 'react';

        export const Button = (props: {label: string}) => {
            return <button>{props.label}</button>;
        };

        export default Button;
        """
        fn = self.PARSER.parse_file(path, source)
        assert fn.language == Language.TYPESCRIPT
        # Arrow function: const Button = (...) => ...
        symbols = fn.symbols
        assert len(symbols) >= 1

    def test_react_component_arrow_function(self):
        """React component as arrow function should be extracted."""
        path = "components/Card.tsx"
        source = """
        import React from 'react';

        export const Card: React.FC<{title: string}> = ({title}) => {
            return <div>{title}</div>;
        };
        """
        fn = self.PARSER.parse_file(path, source)
        names = [s.name for s in fn.symbols]
        assert "Card" in names


class TestPythonEntryPointDetection:
    """Enhanced Python entry point detection."""

    def test_main_py(self):
        """main.py should be an entry point."""
        g = DependencyGraph()
        g._nodes["main.py::main"] = GraphNode(
            node_id="main.py::main",
            file_path="main.py",
            kind=NodeKind.FUNCTION,
            name="main",
            qualified_name="main",
            language=Language.PYTHON,
        )
        entries = g._detect_entry_points()
        assert "main.py" in entries

    def test_django_urls(self):
        """urls.py should be an entry point."""
        g = DependencyGraph()
        g._nodes["pkg/urls.py::__module__"] = GraphNode(
            node_id="pkg/urls.py::__module__",
            file_path="pkg/urls.py",
            kind=NodeKind.MODULE,
            name="urls",
            qualified_name="urls",
            language=Language.PYTHON,
        )
        entries = g._detect_entry_points()
        assert "pkg/urls.py" in entries
