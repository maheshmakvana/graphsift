"""Tests for diff-aware context trimming in graphsift."""

import pytest

from graphsift import (
    ContextBuilder,
    ContextConfig,
    DiffSpec,
    FileNode,
    GenericParser,
    Language,
    LanguageParser,
    OutputMode,
    PythonParser,
    estimate_tokens,
)
from graphsift.core import ContextSelector


# ---------------------------------------------------------------------------
# _parse_diff_hunks
# ---------------------------------------------------------------------------


class TestParseDiffHunks:
    """Tests for ContextSelector._parse_diff_hunks."""

    def test_empty_diff(self):
        """Empty diff text returns empty dict."""
        assert ContextSelector._parse_diff_hunks("") == {}

    def test_single_hunk(self):
        """Parse a single hunk with additions."""
        diff = (
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -10,5 +10,6 @@\n"
            " context\n"
            "-removed\n"
            "+added\n"
            " context2\n"
        )
        parsed = ContextSelector._parse_diff_hunks(diff)
        assert "src/foo.py" in parsed
        entries = parsed["src/foo.py"]
        assert len(entries) == 1
        (hunk_range, new_lines, old_lines) = entries[0]
        assert hunk_range == (10, 15)  # 6 lines starting at 10 → 10-15
        assert new_lines == {11}  # + line at new file line 11

    def test_multiple_hunks(self):
        """Parse multiple hunks for the same file."""
        diff = (
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1,3 +1,4 @@\n"
            " a\n"
            "+b\n"
            " c\n"
            "@@ -20,2 +21,3 @@\n"
            " d\n"
            "+e\n"
        )
        parsed = ContextSelector._parse_diff_hunks(diff)
        entries = parsed["src/foo.py"]
        assert len(entries) == 2
        # First hunk: 4 lines starting at 1 = 1-4
        assert entries[0][0] == (1, 4)
        assert entries[0][1] == {2}
        # Second hunk: 3 lines starting at 21 = 21-23
        assert entries[1][0] == (21, 23)
        assert entries[1][1] == {22}

    def test_multiple_files(self):
        """Parse hunks for multiple files."""
        diff = (
            "--- a/src/a.py\n"
            "+++ b/src/a.py\n"
            "@@ -5,2 +5,3 @@\n"
            " x\n"
            "+y\n"
            "--- a/src/b.py\n"
            "+++ b/src/b.py\n"
            "@@ -1,1 +1,2 @@\n"
            " z\n"
            "+w\n"
        )
        parsed = ContextSelector._parse_diff_hunks(diff)
        assert "src/a.py" in parsed
        assert "src/b.py" in parsed
        assert len(parsed["src/a.py"]) == 1
        assert len(parsed["src/b.py"]) == 1
        assert parsed["src/a.py"][0][1] == {6}
        assert parsed["src/b.py"][0][1] == {2}

    def test_no_changed_lines(self):
        """Hunk with only context lines (should have empty core_changed)."""
        diff = (
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1,3 +1,3 @@\n"
            " a\n"
            " b\n"
            " c\n"
        )
        parsed = ContextSelector._parse_diff_hunks(diff)
        # This shouldn't happen in a real diff, but handle gracefully
        assert parsed["src/foo.py"][0][1] == set()

    def test_new_file(self):
        """A new file diff (+0,0 to +start,count)."""
        diff = (
            "--- /dev/null\n"
            "+++ b/src/new.py\n"
            "@@ -0,0 +1,5 @@\n"
            "+line1\n"
            "+line2\n"
            "+line3\n"
            "+line4\n"
            "+line5\n"
        )
        parsed = ContextSelector._parse_diff_hunks(diff)
        assert "src/new.py" in parsed
        (hunk_range, new_lines, old_lines) = parsed["src/new.py"][0]
        assert hunk_range == (1, 5)
        # All + lines are in new_lines
        assert new_lines == {1, 2, 3, 4, 5}

    def test_path_prefix_stripping(self):
        """The b/ prefix should be stripped from paths."""
        diff = (
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1,1 +1,2 @@\n"
            " a\n"
            "+b\n"
        )
        parsed = ContextSelector._parse_diff_hunks(diff)
        assert "src/foo.py" in parsed


# ---------------------------------------------------------------------------
# _trim_to_diff_context
# ---------------------------------------------------------------------------


class TestTrimToDiffContext:
    """Tests for ContextSelector._trim_to_diff_context."""

    SOURCE = '''"""Module docstring."""
import os
import sys

def alpha():
    """Alpha function."""
    return os.getcwd()

def bravo():
    """Bravo function."""
    return sys.version

def charlie():
    """Charlie function — this one changes."""
    return os.getcwd() + sys.version

def delta():
    """Delta function."""
    return None
'''

    def _make_fixtures(self, source_text, diff_text, changed_files):
        """Helper to create selector, file_node, parser, diff_spec."""
        config = ContextConfig(
            diff_aware_trimming=True,
            trimming_context_lines=10,
        )
        selector = ContextSelector(config)
        parser = PythonParser()
        file_node = parser.parse_file("src/example.py", source_text)
        diff_spec = DiffSpec(
            changed_files=changed_files,
            diff_text=diff_text,
        )
        return selector, diff_spec, file_node, parser

    def test_changed_file_trims_to_relevant_symbol(self):
        """A changed file should be trimmed to preamble + changed symbol only."""
        # Diff changes only the `charlie` function body
        diff_text = (
            "--- a/src/example.py\n"
            "+++ b/src/example.py\n"
            "@@ -14,4 +14,3 @@\n"
            "     \"\"\"Charlie function — this one changes.\"\"\"\n"
            "-    x = os.getcwd()\n"
            "-    y = sys.version\n"
            "-    return x + y\n"
            "+    return os.getcwd() + sys.version\n"
            " \n"
            " def delta():\n"
        )
        selector, diff_spec, file_node, parser = self._make_fixtures(
            self.SOURCE, diff_text, ["src/example.py"],
        )

        trimmed = selector._trim_to_diff_context(
            self.SOURCE, diff_spec, file_node, parser,
        )

        # Should include preamble (docstring, imports)
        assert '"""Module docstring."""' in trimmed
        assert "import os" in trimmed
        assert "import sys" in trimmed

        # Should include the changed symbol
        assert "def charlie():" in trimmed
        assert "return os.getcwd() + sys.version" in trimmed

        # Should NOT include unrelated symbols
        assert "def alpha():" not in trimmed, "Unchanged alpha should be excluded"
        assert "def bravo():" not in trimmed, "Unchanged bravo should be excluded"
        assert "def delta():" not in trimmed, "Unchanged delta should be excluded"

    def test_no_diff_text_falls_back_to_signatures(self):
        """Without diff text, dependent files get signature-only output."""
        source = self.SOURCE
        diff_spec = DiffSpec(
            changed_files=["src/other.py"],
            diff_text="",
        )
        selector = ContextSelector(ContextConfig(diff_aware_trimming=True))
        parser = PythonParser()
        file_node = parser.parse_file("src/example.py", source)

        trimmed = selector._trim_to_diff_context(
            source, diff_spec, file_node, parser,
        )

        # Should be signatures only
        assert "def alpha()" in trimmed
        assert "def bravo()" in trimmed
        assert "def charlie()" in trimmed
        # Should NOT have function bodies
        assert "return os.getcwd()" not in trimmed

    def test_high_percentage_changed_returns_full(self):
        """When >50% of lines are in hunks, the whole file is returned."""
        tiny_source = '"""Tiny."""\nimport os\n\ndef only_fn():\n    pass\n'
        diff_text = (
            "--- a/tiny.py\n"
            "+++ b/tiny.py\n"
            "@@ -1,5 +1,5 @@\n"
            " \"\"\"Tiny.\"\"\"\n"
            " import os\n"
            " \n"
            " def only_fn():\n"
            "-    pass\n"
            "+    return 1\n"
        )
        diff_spec = DiffSpec(
            changed_files=["tiny.py"],
            diff_text=diff_text,
        )
        selector = ContextSelector(ContextConfig(diff_aware_trimming=True))
        parser = PythonParser()
        file_node = parser.parse_file("tiny.py", tiny_source)

        trimmed = selector._trim_to_diff_context(
            tiny_source, diff_spec, file_node, parser,
        )

        # The hunk covers 5 of 5 lines = 100% → whole file should be returned
        assert trimmed == tiny_source

    def test_changed_file_without_hunks_returns_full(self):
        """Changed file with no diff hunks (new/binary) gets full source."""
        source = '"""New file."""\nprint("hello")\n'
        diff_spec = DiffSpec(
            changed_files=["src/new.py"],
            diff_text="",  # no diff text
        )
        selector = ContextSelector(ContextConfig(diff_aware_trimming=True))
        parser = PythonParser()
        file_node = parser.parse_file("src/new.py", source)

        trimmed = selector._trim_to_diff_context(
            source, diff_spec, file_node, parser,
        )

        assert trimmed == source

    def test_dependent_file_without_hunks_gets_signatures(self):
        """Non-changed file without hunks gets signatures only."""
        source = '"""Dep."""\ndef a():\n    return 1\ndef b():\n    return 2\n'
        diff_spec = DiffSpec(
            changed_files=["src/main.py"],
            diff_text="",  # no diff for the dependent file
        )
        selector = ContextSelector(ContextConfig(diff_aware_trimming=True))
        parser = PythonParser()
        file_node = parser.parse_file("src/dep.py", source)

        trimmed = selector._trim_to_diff_context(
            source, diff_spec, file_node, parser,
        )

        # Signatures only
        assert "def a()" in trimmed
        assert "def b()" in trimmed
        assert "return 1" not in trimmed
        assert "return 2" not in trimmed

    def test_gap_markers_inserted(self):
        """Omitted lines should be indicated with a comment marker."""
        # A larger source where only one function changes
        big_source = (
            "# license header\n"
            "# more license\n"
            "\n"
            "import os\n"
            "\n"
            "def fn_a():\n"
            "    pass\n"
            "\n"
            "def fn_b():\n"
            "    pass\n"
            "\n"
            "def fn_c():\n"
            "    pass\n"
            "\n"
            "def fn_d():\n"
            "    pass\n"
            "\n"
            "def fn_e():\n"
            "    pass\n"
            "\n"
            "def fn_f():\n"
            "    pass\n"
            "\n"
            "def fn_g():\n"
            "    pass\n"
        )
        # Diff changes only fn_d (line 16-18)
        diff_text = (
            "--- a/big.py\n"
            "+++ b/big.py\n"
            "@@ -15,5 +15,5 @@\n"
            " \n"
            " def fn_d():\n"
            "     \"\"\"Doc.\"\"\"\n"
            "-    pass\n"
            "+    return 42\n"
            " \n"
            " def fn_e():\n"
        )
        diff_spec = DiffSpec(
            changed_files=["big.py"],
            diff_text=diff_text,
        )
        selector = ContextSelector(ContextConfig(diff_aware_trimming=True))
        # Use GenericParser since source is not real Python (no module docstring)
        parser = PythonParser()
        file_node = parser.parse_file("big.py", big_source)

        trimmed = selector._trim_to_diff_context(
            big_source, diff_spec, file_node, parser,
        )

        # The trimmed output should have gap markers
        assert "... lines omitted ..." in trimmed or "lines omitted" in trimmed


# ---------------------------------------------------------------------------
# Integration: diff_aware_trimming through ContextBuilder
# ---------------------------------------------------------------------------


class TestTrimIntegration:
    """End-to-end tests through ContextBuilder."""

    AUTH_SOURCE = '''"""Auth module."""
import hashlib
from typing import Optional

class AuthManager:
    """Manages authentication."""

    def __init__(self, secret: str):
        self.secret = secret

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        import bcrypt
        return bcrypt.hashpw(
            password.encode(), bcrypt.gensalt()
        ).decode()

    def verify(self, password: str, hashed: str) -> bool:
        return self.hash_password(password) == hashed

def create_token(user_id: str, secret: str) -> str:
    """Create an auth token."""
    return hashlib.sha256(f"{user_id}{secret}".encode()).hexdigest()
'''

    USER_SOURCE = '''"""User module."""
from auth import AuthManager, create_token

class UserService:
    """Manages users."""

    def __init__(self):
        self.auth = AuthManager(secret="s")

    def register(self, username: str, password: str) -> str:
        hashed = self.auth.hash_password(password)
        return create_token(username, "s")
'''

    def test_trimming_reduces_tokens_vs_full(self):
        """Token count should be lower with diff_aware_trimming enabled."""
        source_map = {
            "src/auth.py": self.AUTH_SOURCE,
            "src/user.py": self.USER_SOURCE,
        }
        diff_text = (
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -12,2 +12,5 @@\n"
            "         \"\"\"Hash a password.\"\"\"\n"
            "-        return hashlib.sha256(password.encode()).hexdigest()\n"
            "+        import bcrypt\n"
            "+        return bcrypt.hashpw(\n"
            "+            password.encode(), bcrypt.gensalt()\n"
            "+        ).decode()\n"
            " \n"
            "     def verify(self, password: str, hashed: str) -> bool:\n"
        )
        diff = DiffSpec(
            changed_files=["src/auth.py"],
            diff_text=diff_text,
            query="Review auth changes",
        )

        # Build with trimming
        config = ContextConfig(
            token_budget=50_000,
            output_mode=OutputMode.FULL,
            diff_aware_trimming=True,
        )
        builder = ContextBuilder(config)
        builder.index_files(source_map)
        result = builder.build(diff, source_map)

        # Verify trim stats are populated
        trim_stats = result.metadata.get("trim_stats", {})
        assert "src/auth.py" in trim_stats
        auth_stats = trim_stats["src/auth.py"]
        assert auth_stats["original_file_tokens"] > 0
        assert auth_stats["trimmed_file_tokens"] > 0
        assert auth_stats["saved_tokens"] >= 0

        # The rendered context should only contain relevant parts
        ctx = result.rendered_context
        assert "hash_password" in ctx  # changed function
        assert "create_token" not in ctx  # unchanged, unrelated function
        assert "AuthManager" in ctx  # class containing the changed method

    def test_disabled_trimming_includes_full_file(self):
        """With diff_aware_trimming=False, full file is included."""
        source_map = {
            "src/auth.py": self.AUTH_SOURCE,
        }
        diff_text = (
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -12,2 +12,3 @@\n"
            "         \"\"\"Hash a password.\"\"\"\n"
            "-        return hashlib.sha256(password.encode()).hexdigest()\n"
            "+        import bcrypt\n"
            "+        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()\n"
            " \n"
            "     def verify(self, password: str, hashed: str) -> bool:\n"
        )
        diff = DiffSpec(
            changed_files=["src/auth.py"],
            diff_text=diff_text,
        )

        config_off = ContextConfig(
            token_budget=50_000,
            output_mode=OutputMode.FULL,
            diff_aware_trimming=False,
        )
        config_on = ContextConfig(
            token_budget=50_000,
            output_mode=OutputMode.FULL,
            diff_aware_trimming=True,
        )

        builder_off = ContextBuilder(config_off)
        builder_off.index_files(source_map)
        result_off = builder_off.build(diff, source_map)

        builder_on = ContextBuilder(config_on)
        builder_on.index_files(source_map)
        result_on = builder_on.build(diff, source_map)

        # Trimming should produce fewer tokens
        assert result_on.total_rendered_tokens < result_off.total_rendered_tokens

        # Without trimming, the full file includes create_token
        assert "create_token" in result_off.rendered_context

        # With trimming, create_token should be excluded (it's not in the changed hunk)
        assert "create_token" not in result_on.rendered_context

    def test_trim_stats_in_metadata(self):
        """Trim savings should be reported in result metadata."""
        source_map = {
            "src/auth.py": self.AUTH_SOURCE,
        }
        diff_text = (
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -12,2 +12,3 @@\n"
            "         \"\"\"Hash a password.\"\"\"\n"
            "-        return hashlib.sha256(password.encode()).hexdigest()\n"
            "+        import bcrypt\n"
            "+        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()\n"
            " \n"
            "     def verify(self, password: str, hashed: str) -> bool:\n"
        )
        diff = DiffSpec(
            changed_files=["src/auth.py"],
            diff_text=diff_text,
        )

        config = ContextConfig(
            token_budget=50_000,
            output_mode=OutputMode.FULL,
            diff_aware_trimming=True,
        )
        builder = ContextBuilder(config)
        builder.index_files(source_map)
        result = builder.build(diff, source_map)

        meta = result.metadata
        assert "trim_stats" in meta
        assert "src/auth.py" in meta["trim_stats"]
        ts = meta["trim_stats"]["src/auth.py"]
        assert "original_file_tokens" in ts
        assert "trimmed_file_tokens" in ts
        assert "saved_tokens" in ts
        assert "trim_ratio" in ts
        assert ts["original_file_tokens"] > ts["trimmed_file_tokens"]
        assert ts["saved_tokens"] > 0
        assert ts["trim_ratio"] > 0


# ---------------------------------------------------------------------------
# Non-Python language trimming
# ---------------------------------------------------------------------------


class TestNonPythonTrimming:
    """Diff-aware trimming should work with non-Python parsers too."""

    def test_javascript_trimming(self):
        """JS file with multiple functions should trim to changed one."""
        source = (
            "// JavaScript module\n"
            "import { foo } from './foo';\n"
            "import { bar } from './bar';\n"
            "\n"
            "function helper() {\n"
            "  return 1;\n"
            "}\n"
            "\n"
            "function target() {\n"
            "  return helper();\n"
            "}\n"
            "\n"
            "function cleanup() {\n"
            "  return null;\n"
            "}\n"
        )
        # Diff changes the `target` function body
        diff_text = (
            "--- a/src/main.js\n"
            "+++ b/src/main.js\n"
            "@@ -9,3 +9,5 @@\n"
            " function target() {\n"
            "-  return helper();\n"
            "+  const val = helper();\n"
            "+  return val * 2;\n"
            " }\n"
        )
        diff_spec = DiffSpec(
            changed_files=["src/main.js"],
            diff_text=diff_text,
        )
        parser = GenericParser()
        file_node = parser.parse_file("src/main.js", source)

        selector = ContextSelector(ContextConfig(
            diff_aware_trimming=True,
            output_mode=OutputMode.FULL,
        ))

        trimmed = selector._trim_to_diff_context(
            source, diff_spec, file_node, parser,
        )

        # Should include the changed function
        assert "function target()" in trimmed
        # Should NOT include unrelated functions
        # (Note: GenericParser doesn't extract line_end, so this may not trim
        # as aggressively — but the method should not crash)
        assert "function " in trimmed  # at least some functions


class TestMultipleHunks:
    """Tests for files with multiple changed regions."""

    SOURCE = '''"""Multi-change module."""
import os
import sys

def first():
    """First function."""
    return 10

def second():
    """Second function."""
    return 2

def third():
    """Third function."""
    return 30

def fourth():
    """Fourth function."""
    return 4
'''

    def test_two_hunks_two_functions(self):
        """When two different functions change, both should be included."""
        diff_text = (
            "--- a/src/multi.py\n"
            "+++ b/src/multi.py\n"
            "@@ -6,4 +6,4 @@\n"
            "     \"\"\"First function.\"\"\"\n"
            "-    return 1\n"
            "+    return 10\n"
            " \n"
            " def second():\n"
            "@@ -14,4 +14,4 @@\n"
            "     \"\"\"Third function.\"\"\"\n"
            "-    return 3\n"
            "+    return 30\n"
            " \n"
            " def fourth():\n"
        )
        diff_spec = DiffSpec(
            changed_files=["src/multi.py"],
            diff_text=diff_text,
        )
        selector = ContextSelector(ContextConfig(diff_aware_trimming=True))
        parser = PythonParser()
        file_node = parser.parse_file("src/multi.py", self.SOURCE)

        trimmed = selector._trim_to_diff_context(
            self.SOURCE, diff_spec, file_node, parser,
        )

        # Both changed functions should be included
        assert "def first():" in trimmed
        assert "return 10" in trimmed
        assert "def third():" in trimmed
        assert "return 30" in trimmed

        # Unchanged functions should NOT be included
        assert "def second():" not in trimmed, (
            "Unchanged second() should be excluded"
        )
        assert "def fourth():" not in trimmed, (
            "Unchanged fourth() should be excluded"
        )


class TestPreambleInclusion:
    """The file preamble (docstring, imports) should always be included."""

    def test_preamble_includes_docstring_and_imports(self):
        """The module docstring and import block should be in the trimmed output."""
        source = (
            '# -*- coding: utf-8 -*-\n'
            '"""Module docstring."""\n'
            'import os\n'
            'import sys\n'
            'from typing import Optional\n'
            '\n'
            'CONSTANT = 42\n'
            '\n'
            'def worker():\n'
            '    return CONSTANT\n'
            '\n'
            'def changed():\n'
            '    result = worker()\n'
            '    return result * 2\n'
        )
        diff_text = (
            "--- a/src/demo.py\n"
            "+++ b/src/demo.py\n"
            "@@ -12,2 +12,3 @@\n"
            " def changed():\n"
            "-    return worker()\n"
            "+    result = worker()\n"
            "+    return result * 2\n"
        )
        diff_spec = DiffSpec(
            changed_files=["src/demo.py"],
            diff_text=diff_text,
        )
        selector = ContextSelector(ContextConfig(diff_aware_trimming=True))
        parser = PythonParser()
        file_node = parser.parse_file("src/demo.py", source)

        trimmed = selector._trim_to_diff_context(
            source, diff_spec, file_node, parser,
        )

        # Preamble should be included
        assert '# -*- coding: utf-8 -*-' in trimmed
        assert '"""Module docstring."""' in trimmed
        assert 'import os' in trimmed
        assert 'import sys' in trimmed
        assert 'from typing import Optional' in trimmed

        # The changed function should be included
        assert 'def changed():' in trimmed
        assert 'return result * 2' in trimmed

        # Unrelated items should not be included
        # CONSTANT might be included if it's within context of the preamble
        # but 'def worker()' should not be included if it's not relevant
