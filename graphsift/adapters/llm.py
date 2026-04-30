"""Provider-agnostic LLM adapters for graphsift.

These adapters inject graphsift-ranked context into common chat/message APIs
used by Claude, Codex/GPT, Gemini, and OpenAI-compatible providers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..core import ContextBuilder
from ..exceptions import AdapterError, ValidationError
from ..models import DiffSpec

logger = logging.getLogger(__name__)


def _build_context_payload(
    builder: ContextBuilder,
    *,
    changed_files: list[str],
    source_map: dict[str, str],
    query: str = "",
    commit_message: str = "",
    diff_text: str = "",
) -> tuple[str, dict[str, Any]]:
    if not changed_files:
        raise ValidationError("changed_files must not be empty.")
    if not source_map:
        raise ValidationError("source_map must not be empty.")

    diff_spec = DiffSpec(
        changed_files=changed_files,
        query=query,
        commit_message=commit_message,
        diff_text=diff_text,
    )
    ctx_result = builder.build(diff_spec, source_map)
    meta: dict[str, Any] = {
        "files_selected": ctx_result.files_selected,
        "files_scanned": ctx_result.files_scanned,
        "original_tokens": ctx_result.total_original_tokens,
        "rendered_tokens": ctx_result.total_rendered_tokens,
        "reduction_ratio": ctx_result.reduction_ratio,
        "top_files": [
            {"path": sf.file_node.path, "score": sf.score}
            for sf in ctx_result.selected_files[:5]
        ],
    }
    return ctx_result.rendered_context, meta


def _prepend_to_first_user_message(
    messages: list[dict[str, Any]],
    context_text: str,
) -> list[dict[str, Any]]:
    if not messages:
        raise ValidationError("messages must not be empty.")

    enriched = [dict(message) for message in messages]
    first_user_idx = next(
        (i for i, message in enumerate(enriched) if message.get("role") == "user"),
        None,
    )
    if first_user_idx is None:
        enriched.insert(0, {"role": "user", "content": context_text})
        return enriched

    original_content = enriched[first_user_idx].get("content", "")
    enriched[first_user_idx]["content"] = f"{context_text}\n\n{original_content}".strip()
    return enriched


class _BaseAdapter:
    _DEFAULT_SYSTEM = (
        "You are an expert code reviewer. Analyse the provided code context "
        "and the changed files. Identify bugs, security issues, performance "
        "problems, and design concerns. Be specific and actionable."
    )

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        *,
        system_prompt: str | None = None,
        provider_name: str,
    ) -> None:
        self._client = client
        self._builder = builder
        self._system = system_prompt or self._DEFAULT_SYSTEM
        self._provider_name = provider_name

    def _review_prompt(self, query: str, context_text: str) -> str:
        return f"{query}\n\n{context_text}"

    def _context_meta(
        self,
        *,
        changed_files: list[str],
        source_map: dict[str, str],
        query: str = "",
        commit_message: str = "",
        diff_text: str = "",
    ) -> tuple[str, dict[str, Any]]:
        context_text, meta = _build_context_payload(
            self._builder,
            changed_files=changed_files,
            source_map=source_map,
            query=query,
            commit_message=commit_message,
            diff_text=diff_text,
        )
        logger.info(
            "%s_review: sending context",
            self._provider_name,
            extra={
                "rendered_tokens": meta["rendered_tokens"],
                "reduction": f"{meta['reduction_ratio']:.0%}",
            },
        )
        return context_text, meta


class ClaudeCodeReviewAdapter(_BaseAdapter):
    """Code review adapter for Anthropic's messages API."""

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        system_prompt: str | None = None,
    ) -> None:
        if not hasattr(client, "messages"):
            raise ValidationError("Client must expose a .messages attribute.")
        super().__init__(
            client,
            builder,
            system_prompt=system_prompt,
            provider_name="claude",
        )

    def __repr__(self) -> str:
        return f"ClaudeCodeReviewAdapter(builder={self._builder!r})"

    def review(
        self,
        changed_files: list[str],
        source_map: dict[str, str],
        *,
        model: str = "claude-opus-4-1",
        max_tokens: int = 4096,
        query: str = "Please review these changes.",
        commit_message: str = "",
        diff_text: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        context_text, meta = self._context_meta(
            changed_files=changed_files,
            source_map=source_map,
            query=query,
            commit_message=commit_message,
            diff_text=diff_text,
        )
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system,
                messages=[{
                    "role": "user",
                    "content": self._review_prompt(query, context_text),
                }],
                **kwargs,
            )
        except Exception as exc:
            raise AdapterError(f"Anthropic API call failed: {exc}") from exc
        return response, meta


class ClaudeContextAdapter(_BaseAdapter):
    """Inject graphsift context into Anthropic messages."""

    def __init__(self, client: Any, builder: ContextBuilder) -> None:
        if not hasattr(client, "messages"):
            raise ValidationError("Client must expose a .messages attribute.")
        super().__init__(client, builder, provider_name="claude")

    def __repr__(self) -> str:
        return "ClaudeContextAdapter()"

    def messages_create(
        self,
        *,
        changed_files: list[str],
        source_map: dict[str, str],
        messages: list[dict[str, Any]],
        query: str = "",
        commit_message: str = "",
        diff_text: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        context_text, meta = self._context_meta(
            changed_files=changed_files,
            source_map=source_map,
            query=query,
            commit_message=commit_message,
            diff_text=diff_text,
        )
        enriched = _prepend_to_first_user_message(messages, context_text)
        try:
            response = self._client.messages.create(messages=enriched, **kwargs)
        except Exception as exc:
            raise AdapterError(f"Anthropic API call failed: {exc}") from exc
        return response, meta


class OpenAICodeReviewAdapter(_BaseAdapter):
    """Code review adapter for OpenAI-style chat completions APIs."""

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        system_prompt: str | None = None,
        *,
        provider_name: str = "openai",
    ) -> None:
        if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
            raise ValidationError("Client must expose .chat.completions.")
        super().__init__(
            client,
            builder,
            system_prompt=system_prompt,
            provider_name=provider_name,
        )

    def __repr__(self) -> str:
        return f"OpenAICodeReviewAdapter(provider={self._provider_name})"

    def review(
        self,
        changed_files: list[str],
        source_map: dict[str, str],
        *,
        model: str = "gpt-4.1",
        query: str = "Please review these changes.",
        commit_message: str = "",
        diff_text: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        context_text, meta = self._context_meta(
            changed_files=changed_files,
            source_map=source_map,
            query=query,
            commit_message=commit_message,
            diff_text=diff_text,
        )
        messages = [
            {"role": "system", "content": self._system},
            {
                "role": "user",
                "content": self._review_prompt(query, context_text),
            },
        ]
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            raise AdapterError(f"{self._provider_name} API call failed: {exc}") from exc
        return response, meta


class OpenAIContextAdapter(_BaseAdapter):
    """Inject graphsift context into OpenAI-style chat completions."""

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        *,
        provider_name: str = "openai",
    ) -> None:
        if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
            raise ValidationError("Client must expose .chat.completions.")
        super().__init__(client, builder, provider_name=provider_name)

    def __repr__(self) -> str:
        return f"OpenAIContextAdapter(provider={self._provider_name})"

    def chat_completions_create(
        self,
        *,
        changed_files: list[str],
        source_map: dict[str, str],
        messages: list[dict[str, Any]],
        query: str = "",
        commit_message: str = "",
        diff_text: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        context_text, meta = self._context_meta(
            changed_files=changed_files,
            source_map=source_map,
            query=query,
            commit_message=commit_message,
            diff_text=diff_text,
        )
        enriched = _prepend_to_first_user_message(messages, context_text)
        try:
            response = self._client.chat.completions.create(
                messages=enriched,
                **kwargs,
            )
        except Exception as exc:
            raise AdapterError(f"{self._provider_name} API call failed: {exc}") from exc
        return response, meta


class CodexCodeReviewAdapter(OpenAICodeReviewAdapter):
    """Code review adapter for Codex/GPT models via OpenAI chat completions."""

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            client,
            builder,
            system_prompt=system_prompt,
            provider_name="codex",
        )

    def review(
        self,
        changed_files: list[str],
        source_map: dict[str, str],
        *,
        model: str = "gpt-5-codex",
        query: str = "Please review these changes.",
        commit_message: str = "",
        diff_text: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        return super().review(
            changed_files,
            source_map,
            model=model,
            query=query,
            commit_message=commit_message,
            diff_text=diff_text,
            **kwargs,
        )


class CodexContextAdapter(OpenAIContextAdapter):
    """Context adapter for Codex/GPT models via OpenAI chat completions."""

    def __init__(self, client: Any, builder: ContextBuilder) -> None:
        super().__init__(client, builder, provider_name="codex")


class OpenAICompatibleCodeReviewAdapter(OpenAICodeReviewAdapter):
    """Code review adapter for OpenAI-compatible providers."""

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        system_prompt: str | None = None,
        provider_name: str = "openai_compatible",
    ) -> None:
        super().__init__(
            client,
            builder,
            system_prompt=system_prompt,
            provider_name=provider_name,
        )


class OpenAICompatibleContextAdapter(OpenAIContextAdapter):
    """Context adapter for OpenAI-compatible chat APIs."""

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        *,
        provider_name: str = "openai_compatible",
    ) -> None:
        super().__init__(client, builder, provider_name=provider_name)


class GeminiCodeReviewAdapter(_BaseAdapter):
    """Code review adapter for the Google GenAI Gemini SDK."""

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        system_prompt: str | None = None,
        invoke: Callable[..., Any] | None = None,
    ) -> None:
        if invoke is None:
            if not hasattr(client, "models") or not hasattr(client.models, "generate_content"):
                raise ValidationError(
                    "Client must expose .models.generate_content or a custom invoke callable."
                )
            invoke = client.models.generate_content
        super().__init__(
            client,
            builder,
            system_prompt=system_prompt,
            provider_name="gemini",
        )
        self._invoke = invoke

    def __repr__(self) -> str:
        return f"GeminiCodeReviewAdapter(builder={self._builder!r})"

    def review(
        self,
        changed_files: list[str],
        source_map: dict[str, str],
        *,
        model: str = "gemini-2.5-pro",
        query: str = "Please review these changes.",
        commit_message: str = "",
        diff_text: str = "",
        config: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        context_text, meta = self._context_meta(
            changed_files=changed_files,
            source_map=source_map,
            query=query,
            commit_message=commit_message,
            diff_text=diff_text,
        )
        contents = self._review_prompt(query, context_text)
        request_kwargs = dict(kwargs)
        if config is not None:
            request_kwargs["config"] = config
        try:
            response = self._invoke(
                model=model,
                contents=contents,
                system_instruction=self._system,
                **request_kwargs,
            )
        except TypeError:
            if config is None:
                request_kwargs["config"] = {"system_instruction": self._system}
            response = self._invoke(
                model=model,
                contents=contents,
                **request_kwargs,
            )
        except Exception as exc:
            raise AdapterError(f"Gemini API call failed: {exc}") from exc
        return response, meta


class GeminiContextAdapter(_BaseAdapter):
    """Inject graphsift context into Gemini prompt generation calls."""

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        invoke: Callable[..., Any] | None = None,
    ) -> None:
        if invoke is None:
            if not hasattr(client, "models") or not hasattr(client.models, "generate_content"):
                raise ValidationError(
                    "Client must expose .models.generate_content or a custom invoke callable."
                )
            invoke = client.models.generate_content
        super().__init__(client, builder, provider_name="gemini")
        self._invoke = invoke

    def __repr__(self) -> str:
        return "GeminiContextAdapter()"

    def generate_content(
        self,
        *,
        changed_files: list[str],
        source_map: dict[str, str],
        contents: str,
        query: str = "",
        commit_message: str = "",
        diff_text: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        context_text, meta = self._context_meta(
            changed_files=changed_files,
            source_map=source_map,
            query=query,
            commit_message=commit_message,
            diff_text=diff_text,
        )
        enriched = f"{context_text}\n\n{contents}".strip()
        try:
            response = self._invoke(contents=enriched, **kwargs)
        except Exception as exc:
            raise AdapterError(f"Gemini API call failed: {exc}") from exc
        return response, meta
