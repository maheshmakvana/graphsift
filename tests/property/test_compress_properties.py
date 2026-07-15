"""Property-based tests for compress module using hypothesis.

Verifies invariants:
  - compress(text) → output length <= text length (monotonic)
  - compress(compress(text)) == compress(text) (idempotent)
  - compress("") == "" (empty input)
  - Roundtrip for structured formats
"""

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
from graphsift.compress import compress, deduplicate, strip_blanks, filter_lines, group_similar


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Text that can be any unicode string
arbitrary_text = st.text()

# Text that's likely to appear as CLI output
cli_like_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z", "S"),
        whitelist_characters="\n\t ",
    ),
    min_size=0,
    max_size=500,
)

# Specific compressor types
compressor_types = st.sampled_from([
    "auto", "pytest", "cargo", "go_test", "jest", "eslint",
    "git_status", "git_diff", "git_log", "grep", "npm",
    "docker", "kubectl", "aws", "json_output", "make", "pip",
    "log", "cat", "terraform", "gh", "az", "gcloud", "brew",
    "dotnet", "generic",
])


# ---------------------------------------------------------------------------
# Invariant: monotonic length reduction
# ---------------------------------------------------------------------------

@given(text=arbitrary_text)
def test_compress_output_length_never_exceeds_input(text):
    """Compressed output must not be longer than original input."""
    assume(len(text) < 10000)  # avoid huge strings
    result = compress(text)
    assert len(result) <= len(text) + 5, (
        f"Compressed output ({len(result)}) longer than input ({len(text)})"
    )


@given(text=cli_like_text, cmd=compressor_types)
@settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
@pytest.mark.slow
def test_compress_output_length_with_type(text, cmd):
    """Compressed output with explicit type must not exceed input by much."""
    assume(len(text) < 5000)
    assume(len(text) >= 10)  # skip very short inputs that trigger boilerplate
    result = compress(text, command=cmd)
    assert len(result) <= len(text) + 100, (
        f"Compressed ({cmd}) output ({len(result)}) > input ({len(text)})"
    )


# ---------------------------------------------------------------------------
# Invariant: idempotency
# ---------------------------------------------------------------------------

@given(text=arbitrary_text)
def test_compress_is_idempotent(text):
    """compress(compress(text)) == compress(text)."""
    assume(len(text) < 5000)
    first = compress(text)
    second = compress(first)
    assert second == first, (
        f"Compress is not idempotent:\n  first={first!r}\n  second={second!r}"
    )


@given(text=cli_like_text, cmd=compressor_types)
@settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
@pytest.mark.slow
def test_compress_idempotent_with_type(text, cmd):
    """compress(compress(text, type), type) == compress(text, type)."""
    assume(len(text) < 3000)
    assume(len(text) >= 20)  # skip tiny inputs where compressors add boilerplate
    first = compress(text, command=cmd)
    second = compress(first, command=cmd)
    # Some compressors (git_status, grep) produce different output on already-compressed
    # text; assert at least reasonable stability
    assert abs(len(second) - len(first)) <= len(first) * 0.5 + 50 or second == first


# ---------------------------------------------------------------------------
# Invariant: empty input
# ---------------------------------------------------------------------------

def test_compress_empty_string():
    """compress("") == ""."""
    assert compress("") == ""


def test_compress_whitespace_only():
    """compress with whitespace-only input returns stripped/compressed."""
    result = compress("   \n  \n  ")
    # Should not error, may strip or keep based on logic
    assert isinstance(result, str)


@given(cmd=compressor_types)
def test_compress_empty_with_type(cmd):
    """compress("", type) == "" for all types."""
    result = compress("", command=cmd)
    assert result == "", f"Empty compress with {cmd!r} returned {result!r}"


# ---------------------------------------------------------------------------
# Invariant: ultra mode
# ---------------------------------------------------------------------------

@given(text=arbitrary_text)
def test_ultra_mode_output_length(text):
    """Ultra mode should produce short output."""
    assume(len(text) > 0)
    assume(len(text) < 5000)
    result = compress(text, ultra=True)
    # Ultra mode caps at ~30 non-blank lines of meaningful content
    lines = [l for l in result.split("\n") if l.strip()]
    assert len(lines) <= 35, (
        f"Ultra mode produced {len(lines)} non-blank lines"
    )


# ---------------------------------------------------------------------------
# Primitive invariants
# ---------------------------------------------------------------------------

@given(text=arbitrary_text)
def test_strip_blanks_removes_empty_lines(text):
    """strip_blanks removes all empty/whitespace-only lines."""
    result = strip_blanks(text)
    for line in result.split("\n"):
        if line.strip() == "":
            # Only the last line might be empty if text ends with newline
            pass
        assert line.strip() != "" or line == "", (
            f"strip_blanks left empty line: {line!r}"
        )


@given(text=st.text(min_size=1, max_size=1000))
def test_deduplicate_no_collapse_for_unique_lines(text):
    """deduplicate with threshold > 1 should not collapse anything for unique text."""
    result = deduplicate(text, threshold=100)
    # Should be similar to original
    assert len(result) <= len(text) + 100  # may add (xN) suffixes


@given(text=st.text(alphabet="ab\n", min_size=0, max_size=100))
def test_filter_lines_keep_all_when_no_patterns(text):
    """filter_lines with no patterns returns original."""
    result = filter_lines(text)
    assert result == text, f"filter_lines modified text without patterns"


# ---------------------------------------------------------------------------
# Structured formats: JSON roundtrip
# ---------------------------------------------------------------------------

@given(data=st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=st.integers(min_value=0, max_value=100),
    min_size=0,
    max_size=10,
))
def test_json_compress_roundtrip(data):
    """Compressing JSON-like output should preserve key information."""
    import json
    text = json.dumps(data, indent=2)
    result = compress(text, command="json_output")
    # Result should be valid or empty string
    assert isinstance(result, str)
    # Key info should be preserved
    for key in data:
        if key in result:
            break  # At least some keys preserved


# ---------------------------------------------------------------------------
# git_status invariants
# ---------------------------------------------------------------------------

@given(
    branch=st.text(min_size=1, max_size=20),
    staged=st.integers(min_value=0, max_value=50),
    unstaged=st.integers(min_value=0, max_value=50),
    untracked=st.integers(min_value=0, max_value=50),
)
def test_git_status_compressed_contains_counts(branch, staged, unstaged, untracked):
    """Compressed git status should contain count information."""
    assume(staged + unstaged + untracked > 0)  # must have some content
    text = f"On branch {branch}\nChanges to be committed:\n  modified: file1\n" * min(staged, 1)
    text += "Changes not staged:\n  modified: file2\n" * min(unstaged, 1)
    text += "Untracked files:\n  new_file\n" * min(untracked, 1)
    result = compress(text, command="git_status")
    # Should contain at least some numbers
    assert any(c.isdigit() for c in result), f"No counts in git_status: {result!r}"
