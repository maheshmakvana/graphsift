import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graphsift.tool_budgets import ToolBudget
from graphsift.read_cache import ReadCache
from graphsift.verify_hooks import Verifier
from graphsift.evidence_check import EvidenceChecker
from graphsift.prompt_templates import FixBugTemplate, AddFeatureTemplate, RefactorTemplate, get_template
from graphsift.tiered_memory import TieredMemory
from graphsift.compact_context import ConversationCompactor


# ── ToolBudget ──────────────────────────────────────────────────────────

def test_tool_budget_bash_cap():
    budget = ToolBudget()
    text = "\n".join(f"line {i}" for i in range(200))
    capped = budget.apply("bash", text)
    assert len(capped.split("\n")) <= 82  # 80 lines + 2 for omit message


def test_tool_budget_read_cap():
    budget = ToolBudget()
    text = "\n".join(f"line {i}" for i in range(500))
    capped = budget.apply("read", text)
    assert len(capped.split("\n")) <= 302  # 300 lines + omit msg


def test_tool_budget_grep_cap():
    budget = ToolBudget()
    text = "\n".join(f"line {i}" for i in range(200))
    capped = budget.apply("grep", text)
    assert len(capped.split("\n")) <= 122  # 120 lines + omit msg


def test_tool_budget_ansi_strip():
    budget = ToolBudget()
    text = "\x1b[31mred\x1b[0m line"
    capped = budget.apply("bash", text)
    assert "\x1b" not in capped


def test_tool_budget_blank_collapse():
    budget = ToolBudget()
    text = "a\n\n\n\n\nb"
    capped = budget.apply("bash", text)
    # should collapse 5 blanks into 2 (a\n\n\nb = a, blank, blank, b)
    assert "a\n\n\nb" in capped  # 2 blank lines between a and b
    assert capped.count("\n") <= 4  # a + blank + blank + b = 3 newlines max


def test_tool_budget_custom():
    budget = ToolBudget()
    budget.set_budget("bash", 10)
    text = "\n".join(f"line {i}" for i in range(50))
    capped = budget.apply("bash", text)
    assert len(capped.split("\n")) <= 12  # 10 lines + omit msg


# ── ReadCache ───────────────────────────────────────────────────────────

def test_read_cache_first_read():
    cache = ReadCache()
    content = cache.read("test.py", lambda: "print('hello')")
    assert content == "print('hello')"


def test_read_cache_second_read_stub():
    cache = ReadCache()
    cache.read("test.py", lambda: "print('hello')")
    stub = cache.read("test.py", lambda: "print('hello')")
    assert "same content" in stub or "fingerprint match" in stub


def test_read_cache_changed_content():
    cache = ReadCache()
    cache.read("test.py", lambda: "print('hello')")
    new = cache.read("test.py", lambda: "print('world')")
    assert "world" in new


def test_read_cache_invalidate():
    cache = ReadCache()
    cache.read("test.py", lambda: "print('hello')")
    cache.invalidate("test.py")
    content = cache.read("test.py", lambda: "print('hello')")
    assert content == "print('hello')"


def test_read_cache_clear():
    cache = ReadCache()
    cache.read("a.py", lambda: "x")
    cache.read("b.py", lambda: "y")
    assert cache.stubs_served == 0
    cache.clear()
    assert cache.stubs_served == 0
    assert cache.read("a.py", lambda: "x") == "x"


def test_read_cache_stubs_counter():
    cache = ReadCache()
    cache.read("x.py", lambda: "data")
    cache.read("x.py", lambda: "data")
    assert cache.stubs_served == 1
    cache.read("x.py", lambda: "data")
    assert cache.stubs_served == 2


# ── Verifier ────────────────────────────────────────────────────────────

def test_verify_python_syntax_ok():
    verifier = Verifier()
    result = verifier.check("graphsift/_version.py")
    assert result.syntax_ok is True


def test_verify_file_not_found():
    verifier = Verifier()
    result = verifier.check("nonexistent_file_12345.py")
    assert result.syntax_ok is False


def test_verify_python_syntax_error():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("def broken(  ")
        tmp = f.name
    try:
        verifier = Verifier()
        result = verifier.check(tmp)
        assert result.syntax_ok is False
    finally:
        os.unlink(tmp)


# ── EvidenceChecker ─────────────────────────────────────────────────────

def test_evidence_no_citations():
    checker = EvidenceChecker()
    citations = checker.check_response("No references here.")
    assert len(citations) == 0


def test_evidence_finds_citations():
    checker = EvidenceChecker()
    text = "Fix in `src/main.py:42`"
    citations = checker.check_response(text)
    assert len(citations) >= 1
    assert citations[0].file_path == "src/main.py"


def test_evidence_finds_multiple():
    checker = EvidenceChecker()
    text = "Fix a.py:10 and b.py:20"
    citations = checker.check_response(text)
    assert len(citations) == 2


def test_evidence_dedup():
    checker = EvidenceChecker()
    text = "Bug in src/a.py:10 and also src/a.py:10"
    citations = checker.check_response(text)
    assert len(citations) == 1


# ── PromptTemplates ────────────────────────────────────────────────────

def test_fix_template_contains_bug():
    tpl = FixBugTemplate()
    prompt = tpl.render(bug="null pointer", file="main.py")
    assert "null pointer" in prompt


def test_fix_template_json_output():
    tpl = FixBugTemplate()
    prompt = tpl.render(bug="x", file="y")
    assert "JSON" in prompt


def test_add_template_contains_feature():
    tpl = AddFeatureTemplate()
    prompt = tpl.render(feature="dark mode", files=["theme.py"])
    assert "dark mode" in prompt
    assert "theme.py" in prompt


def test_refactor_template_constraint():
    tpl = RefactorTemplate()
    prompt = tpl.render(target="auth.py")
    assert "Behavior must not change" in prompt


def test_get_template_returns_correct():
    assert isinstance(get_template("fix"), FixBugTemplate)
    assert isinstance(get_template("add"), AddFeatureTemplate)
    assert isinstance(get_template("refactor"), RefactorTemplate)


def test_get_template_unknown():
    import pytest
    with pytest.raises(ValueError, match="fix"):
        get_template("unknown")


# ── TieredMemory ────────────────────────────────────────────────────────

def test_tiered_memory_add_axiom():
    mem = TieredMemory(repo_root=".")
    mem.add_axiom("Use type hints")
    assert "Use type hints" in mem.get_tier("axioms")


def test_tiered_memory_axiom_cap():
    mem = TieredMemory(repo_root=".")
    for i in range(20):
        mem.add_axiom(f"Axiom {i}")
    assert len(mem.get_tier("axioms")) <= 12


def test_tiered_memory_topic_empty():
    mem = TieredMemory(repo_root=".")
    assert mem.get_tier("topic") == []


def test_tiered_memory_archive_empty():
    mem = TieredMemory(repo_root=".")
    assert mem.get_tier("archive") == []


def test_tiered_memory_token_estimate():
    mem = TieredMemory(repo_root=".")
    mem.add_axiom("Hello world")
    assert mem.estimate_tokens("axioms") > 0


# ── ConversationCompactor.should_compact ───────────────────────────────

def test_should_compact_false():
    compactor = ConversationCompactor(max_context_tokens=100000)
    assert compactor.should_compact(50000, threshold_pct=80) is False


def test_should_compact_true():
    compactor = ConversationCompactor(max_context_tokens=100000)
    assert compactor.should_compact(90000, threshold_pct=80) is True


def test_should_compact_custom_threshold():
    compactor = ConversationCompactor(max_context_tokens=100000)
    assert compactor.should_compact(60000, threshold_pct=50) is True
    assert compactor.should_compact(40000, threshold_pct=50) is False


def test_should_compact_custom_max():
    compactor = ConversationCompactor(max_context_tokens=50000)
    assert compactor.should_compact(40000) is True  # 80% of 50k = 40k
