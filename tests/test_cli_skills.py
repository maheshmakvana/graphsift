"""Tests for graphsift's Claude Code skill management.

Covers `_cleanup_legacy_global_skills` — the self-healing that removes stale
user-global graphsift skills/commands that duplicate the project-scoped ones
in the Claude Code slash menu.
"""

from graphsift import cli


def _make_legacy_globals(tmp_path):
    """Create a fake home dir with legacy user-global graphsift leftovers
    (directory-form skill, single-file skill, command) plus an unrelated skill."""
    skills = tmp_path / ".claude" / "skills"
    commands = tmp_path / ".claude" / "commands"
    (skills / "graphsift-build").mkdir(parents=True)
    (skills / "graphsift-build" / "SKILL.md").write_text("x", encoding="utf-8")
    (skills / "graphsift-compress.md").write_text("x", encoding="utf-8")
    (commands / "graphsift-review").mkdir(parents=True)
    (commands / "graphsift-review" / "README.md").write_text("x", encoding="utf-8")
    (skills / "unrelated-skill").mkdir(parents=True)
    (skills / "unrelated-skill" / "SKILL.md").write_text("x", encoding="utf-8")
    return tmp_path


def test_cleanup_removes_legacy_global_skills(monkeypatch, tmp_path):
    root = _make_legacy_globals(tmp_path)
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: root))

    removed = cli._cleanup_legacy_global_skills()

    assert removed == 3  # dir skill + single-file skill + command dir
    assert not (root / ".claude" / "skills" / "graphsift-build").exists()
    assert not (root / ".claude" / "skills" / "graphsift-compress.md").exists()
    assert not (root / ".claude" / "commands" / "graphsift-review").exists()
    # Unrelated skills must be untouched.
    assert (root / ".claude" / "skills" / "unrelated-skill").exists()


def test_cleanup_is_idempotent(monkeypatch, tmp_path):
    root = _make_legacy_globals(tmp_path)
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: root))

    assert cli._cleanup_legacy_global_skills() == 3
    assert cli._cleanup_legacy_global_skills() == 0  # nothing left to remove


def test_cleanup_noop_when_no_globals(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))
    assert cli._cleanup_legacy_global_skills() == 0


def test_cleanup_handles_md_named_directory(monkeypatch, tmp_path):
    """A directory literally named *.md (legacy install quirk) must be removed."""
    skills = tmp_path / ".claude" / "skills"
    md_dir = skills / "graphsift-compress.md"
    md_dir.mkdir(parents=True)
    (md_dir / "SKILL.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: tmp_path))

    assert cli._cleanup_legacy_global_skills() == 1
    assert not md_dir.exists()


def _install_args(tmp_path, **overrides):
    from types import SimpleNamespace
    base = dict(
        project_root=str(tmp_path),
        no_hooks=True, no_skills=True, bash_wrapper=False, no_uiux_engine=False,
        all=False, claude_code=False, claude_desktop=False, cursor=False,
        windsurf=False, continue_=False, codex=False, copilot=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_install_auto_installs_engine_when_missing(monkeypatch, tmp_path, capsys):
    """`graphsift install` must auto-install the ui-ux-pro-max engine when the
    engine is missing — no manual `graphsift uiux --install` needed."""
    from graphsift import uiux
    monkeypatch.setattr(uiux, "find_search_script", lambda: None)
    monkeypatch.setattr(uiux, "install_engine", lambda: (0, "installed fake"))
    assert cli.cmd_install(_install_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "installing the MIT-licensed" in out
    assert "installed fake" in out


def test_install_engine_missing_but_install_fails_gracefully(monkeypatch, tmp_path, capsys):
    """A failed auto-install (e.g. no npm) must not abort `graphsift install`."""
    from graphsift import uiux
    monkeypatch.setattr(uiux, "find_search_script", lambda: None)
    monkeypatch.setattr(uiux, "install_engine", lambda: (1, "npm not found on PATH."))
    assert cli.cmd_install(_install_args(tmp_path)) == 0
    captured = capsys.readouterr()
    assert "auto-install failed" in captured.out + captured.err


def test_install_no_uiux_engine_flag_skips(monkeypatch, tmp_path, capsys):
    """--no-uiux-engine must skip both the check and the install."""
    from graphsift import uiux
    monkeypatch.setattr(uiux, "find_search_script", lambda: None)
    monkeypatch.setattr(uiux, "install_engine", lambda: (0, "should not run"))
    cli.cmd_install(_install_args(tmp_path, no_uiux_engine=True))
    out = capsys.readouterr().out
    assert "should not run" not in out


def test_write_skill_emits_lowercase_boolean_frontmatter(tmp_path):
    """Booleans in frontmatter must serialize as lowercase YAML (false/true)."""
    target = tmp_path / "skills" / "graphsift-uiux" / "SKILL.md"
    cli._write_skill(
        target,
        title="T",
        description="D",
        steps=["s1"],
        example="e",
        frontmatter={"name": "graphsift-uiux", "user-invocable": False},
    )
    content = target.read_text(encoding="utf-8")
    assert "name: graphsift-uiux" in content
    assert "user-invocable: false" in content
    assert "user-invocable: False" not in content
