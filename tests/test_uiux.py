"""Tests for graphsift.uiux — the thin wrapper over the MIT-licensed
ui-ux-pro-max-skill engine (no upstream code is vendored into graphsift).

These tests never require the engine to be installed: they exercise discovery,
the install hint, and subprocess delegation against a minimal fake engine.
"""

import json
import os
from pathlib import Path

import pytest

from graphsift import uiux


def _fake_engine(tmp_path: Path, body: str = "") -> Path:
    """Write a minimal stand-in search.py that emits JSON."""
    script = tmp_path / "search.py"
    script.write_text(
        "import sys, json\n"
        + (body + "\n" if body else "")
        + "print(json.dumps({'design_system': {'style': {'Name': 'Fake Style'}}, 'count': 1}))\n",
        encoding="utf-8",
    )
    return script


def test_install_hint_names_source_and_command(monkeypatch):
    hint = uiux.install_hint()
    assert "ui-ux-pro-max-skill" in hint
    assert "MIT" in hint
    assert "uipro init --ai claude" in hint
    assert "GRAPHSIFT_UIUX_SKILL" in hint


def test_domain_and_stack_facts():
    assert "style" in uiux.DOMAINS
    assert "ux" in uiux.DOMAINS
    assert "shadcn" in uiux.STACKS
    assert "react" in uiux.STACKS
    assert len(uiux.DOMAINS) >= 12
    assert len(uiux.STACKS) >= 22


def test_find_search_script_honors_env_override(tmp_path, monkeypatch):
    fake = _fake_engine(tmp_path)
    monkeypatch.setenv("GRAPHSIFT_UIUX_SKILL", str(fake))
    assert uiux.find_search_script() == fake


def test_find_search_script_env_pointing_at_directory(tmp_path, monkeypatch):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    fake = scripts_dir / "search.py"
    fake.write_text("print('x')\n", encoding="utf-8")
    monkeypatch.setenv("GRAPHSIFT_UIUX_SKILL", str(scripts_dir))
    assert uiux.find_search_script() == fake


def test_run_json_with_fake_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHSIFT_UIUX_SKILL", str(_fake_engine(tmp_path)))
    result = uiux.run_json(["--design-system", "--json"])
    assert result["design_system"]["style"]["Name"] == "Fake Style"


def test_run_json_missing_engine_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHSIFT_UIUX_SKILL", str(tmp_path / "does-not-exist.py"))
    # Make discovery deterministic: no candidate roots.
    monkeypatch.setattr(uiux, "_candidate_roots", lambda: [])
    result = uiux.run_json(["--design-system", "--json"])
    assert "error" in result
    assert "not installed" in result["error"]


def test_run_json_nonzero_exit_reports_stderr(tmp_path, monkeypatch):
    bad = tmp_path / "search.py"
    bad.write_text("import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)\n", encoding="utf-8")
    monkeypatch.setenv("GRAPHSIFT_UIUX_SKILL", str(bad))
    result = uiux.run_json(["foo"])
    assert result["error"] == "boom"


def _stack_fake_engine(tmp_path: Path) -> Path:
    """Fake engine that only matches queries 'react' or 'components'."""
    script = tmp_path / "search.py"
    script.write_text(
        "import sys, json\n"
        "query = sys.argv[1]\n"
        "hits = [{'Guideline': query, 'Do': 'x'}] if query in ('react', 'components') else []\n"
        "print(json.dumps({'stack': 'react', 'count': len(hits), 'results': hits}))\n",
        encoding="utf-8",
    )
    return script


def test_mcp_stack_guide_falls_back_to_stack_name(tmp_path, monkeypatch):
    """uiux_stack_guide with no query must not return an empty set when the
    default query matches nothing — it should fall back to the stack name."""
    from graphsift.mcp_server import _tool_uiux_stack_guide

    monkeypatch.setenv("GRAPHSIFT_UIUX_SKILL", str(_stack_fake_engine(tmp_path)))
    result = _tool_uiux_stack_guide({"stack": "react"})
    assert result["count"] == 1
    assert result["query"] == "react"


def test_mcp_stack_guide_honors_custom_query(tmp_path, monkeypatch):
    """An explicit query is used as-is (no fallback rewriting)."""
    from graphsift.mcp_server import _tool_uiux_stack_guide

    monkeypatch.setenv("GRAPHSIFT_UIUX_SKILL", str(_stack_fake_engine(tmp_path)))
    result = _tool_uiux_stack_guide({"stack": "react", "query": "custom"})
    assert result["query"] == "custom"


def test_mcp_stack_guide_missing_stack_errors():
    from graphsift.mcp_server import _tool_uiux_stack_guide

    result = _tool_uiux_stack_guide({})
    assert "error" in result
    assert "stack parameter is required" in result["error"]


# --- install_engine (auto-install path used by `graphsift install`) ---------


def test_install_engine_npm_missing(monkeypatch):
    monkeypatch.setattr(uiux.shutil, "which", lambda cmd: None)
    code, msg = uiux.install_engine()
    assert code == 1
    assert "npm not found" in msg


def test_install_engine_success(monkeypatch):
    which_results = {"npm": "/usr/bin/npm", "uipro": "/usr/bin/uipro",
                     "ui-ux-pro-max-cli": None}
    monkeypatch.setattr(uiux.shutil, "which", lambda cmd: which_results.get(cmd))
    monkeypatch.setattr(uiux.subprocess, "call", lambda cmd: 0)
    code, msg = uiux.install_engine()
    assert code == 0
    assert "installed" in msg.lower()


def test_install_engine_npm_install_fails(monkeypatch):
    monkeypatch.setattr(uiux.shutil, "which", lambda cmd: "/usr/bin/npm" if cmd == "npm" else None)
    monkeypatch.setattr(uiux.subprocess, "call", lambda cmd: 1)
    code, msg = uiux.install_engine()
    assert code == 1
    assert "npm install ui-ux-pro-max-cli failed" in msg


def test_install_engine_uipro_init_fails(monkeypatch):
    which_results = {"npm": "/usr/bin/npm", "uipro": "/usr/bin/uipro",
                     "ui-ux-pro-max-cli": None}
    monkeypatch.setattr(uiux.shutil, "which", lambda cmd: which_results.get(cmd))
    monkeypatch.setattr(uiux.subprocess, "call",
                        lambda cmd: 0 if "install" in cmd else 1)
    code, msg = uiux.install_engine()
    assert code == 1
    assert "uipro init --ai claude failed" in msg
