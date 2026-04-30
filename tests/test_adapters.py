"""Tests for multi-provider LLM adapters."""

import pytest

from graphsift import (
    AdapterError,
    ClaudeCodeReviewAdapter,
    ClaudeContextAdapter,
    CodexCodeReviewAdapter,
    CodexContextAdapter,
    GeminiCodeReviewAdapter,
    GeminiContextAdapter,
    OpenAICodeReviewAdapter,
    OpenAICompatibleCodeReviewAdapter,
    OpenAIContextAdapter,
    ValidationError,
)


class _FakeAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"provider": "anthropic", "kwargs": kwargs}


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeAnthropicMessages()


class _FakeOpenAICompletions:
    def __init__(self, should_fail: bool = False):
        self.calls = []
        self._should_fail = should_fail

    def create(self, **kwargs):
        if self._should_fail:
            raise RuntimeError("boom")
        self.calls.append(kwargs)
        return {"provider": "openai", "kwargs": kwargs}


class _FakeOpenAIClient:
    def __init__(self, should_fail: bool = False):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeOpenAICompletions(should_fail=should_fail)


class _FakeGeminiModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return {"provider": "gemini", "kwargs": kwargs}


class _FakeGeminiClient:
    def __init__(self):
        self.models = _FakeGeminiModels()


def test_claude_code_review_adapter(builder, source_map):
    client = _FakeAnthropicClient()
    adapter = ClaudeCodeReviewAdapter(client, builder)

    response, meta = adapter.review(
        changed_files=["src/auth.py"],
        source_map=source_map,
    )

    assert response["provider"] == "anthropic"
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-4-1"
    assert "src/auth.py" in call["messages"][0]["content"]
    assert meta["files_selected"] >= 1


def test_claude_context_adapter_prepends_context(builder, source_map):
    client = _FakeAnthropicClient()
    adapter = ClaudeContextAdapter(client, builder)

    _, meta = adapter.messages_create(
        changed_files=["src/auth.py"],
        source_map=source_map,
        messages=[{"role": "user", "content": "Review this"}],
        model="claude-sonnet-4-0",
    )

    message = client.messages.calls[0]["messages"][0]["content"]
    assert "src/auth.py" in message
    assert "Review this" in message
    assert meta["rendered_tokens"] > 0


def test_openai_code_review_adapter(builder, source_map):
    client = _FakeOpenAIClient()
    adapter = OpenAICodeReviewAdapter(client, builder)

    response, meta = adapter.review(
        changed_files=["src/auth.py"],
        source_map=source_map,
    )

    assert response["provider"] == "openai"
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-4.1"
    assert call["messages"][0]["role"] == "system"
    assert "src/auth.py" in call["messages"][1]["content"]
    assert meta["reduction_ratio"] >= -1.0


def test_openai_context_adapter_injects_first_user_message(builder, source_map):
    client = _FakeOpenAIClient()
    adapter = OpenAIContextAdapter(client, builder)

    adapter.chat_completions_create(
        changed_files=["src/auth.py"],
        source_map=source_map,
        messages=[
            {"role": "system", "content": "Keep it short"},
            {"role": "user", "content": "Find bugs"},
        ],
        model="gpt-4.1-mini",
    )

    call = client.chat.completions.calls[0]
    assert call["messages"][0]["role"] == "system"
    assert "src/auth.py" in call["messages"][1]["content"]
    assert "Find bugs" in call["messages"][1]["content"]


def test_codex_adapters(builder, source_map):
    client = _FakeOpenAIClient()
    review_adapter = CodexCodeReviewAdapter(client, builder)
    context_adapter = CodexContextAdapter(client, builder)

    response, _ = review_adapter.review(
        changed_files=["src/auth.py"],
        source_map=source_map,
    )
    context_adapter.chat_completions_create(
        changed_files=["src/auth.py"],
        source_map=source_map,
        messages=[{"role": "user", "content": "Inspect this diff"}],
        model="gpt-5-codex",
    )

    assert response["provider"] == "openai"
    assert client.chat.completions.calls[0]["model"] == "gpt-5-codex"
    assert "src/auth.py" in client.chat.completions.calls[1]["messages"][0]["content"]


def test_openai_compatible_adapter_uses_custom_provider_name(builder, source_map):
    client = _FakeOpenAIClient()
    adapter = OpenAICompatibleCodeReviewAdapter(
        client,
        builder,
        provider_name="groq",
    )

    response, _ = adapter.review(
        changed_files=["src/auth.py"],
        source_map=source_map,
        model="llama-3.3-70b-versatile",
    )

    assert response["provider"] == "openai"
    assert "groq" in repr(adapter)


def test_gemini_code_review_adapter(builder, source_map):
    client = _FakeGeminiClient()
    adapter = GeminiCodeReviewAdapter(client, builder)

    response, meta = adapter.review(
        changed_files=["src/auth.py"],
        source_map=source_map,
    )

    assert response["provider"] == "gemini"
    call = client.models.calls[0]
    assert call["model"] == "gemini-2.5-pro"
    assert "system_instruction" in call
    assert "src/auth.py" in call["contents"]
    assert meta["files_scanned"] >= 1


def test_gemini_context_adapter_injects_context(builder, source_map):
    client = _FakeGeminiClient()
    adapter = GeminiContextAdapter(client, builder)

    adapter.generate_content(
        changed_files=["src/auth.py"],
        source_map=source_map,
        contents="Summarize the change",
        model="gemini-2.5-flash",
    )

    call = client.models.calls[0]
    assert "Summarize the change" in call["contents"]
    assert "src/auth.py" in call["contents"]


def test_adapter_validation_errors(builder, source_map):
    with pytest.raises(ValidationError):
        OpenAICodeReviewAdapter(object(), builder)

    client = _FakeAnthropicClient()
    adapter = ClaudeCodeReviewAdapter(client, builder)
    with pytest.raises(ValidationError):
        adapter.review(changed_files=[], source_map=source_map)


def test_openai_adapter_wraps_upstream_errors(builder, source_map):
    client = _FakeOpenAIClient(should_fail=True)
    adapter = OpenAICodeReviewAdapter(client, builder)

    with pytest.raises(AdapterError):
        adapter.review(changed_files=["src/auth.py"], source_map=source_map)
