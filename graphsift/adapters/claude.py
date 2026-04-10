"""Claude / Anthropic adapter for graphsift.

Wraps the Anthropic messages API to transparently inject graphsift-selected
context before sending. Caller owns the Anthropic client.

Example::

    import anthropic
    from graphsift import ContextBuilder, ContextConfig, DiffSpec
    from graphsift.adapters.claude import ClaudeCodeReviewAdapter

    client = anthropic.Anthropic()
    builder = ContextBuilder(ContextConfig(token_budget=50_000))
    for path, src in my_files.items():
        builder.index_file(path, src)

    adapter = ClaudeCodeReviewAdapter(client, builder)
    response, meta = adapter.review(
        changed_files=["src/auth.py"],
        source_map=my_files,
        model="claude-opus-4-6",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import ContextBuilder
from ..exceptions import AdapterError, ValidationError
from ..models import ContextConfig, DiffSpec

logger = logging.getLogger(__name__)


class ClaudeCodeReviewAdapter:
    """Token-efficient code review adapter for the Anthropic API.

    Selects the minimal relevant context from a codebase, compresses it
    via tokenpruner, then sends it to Claude.

    Args:
        client: Instantiated ``anthropic.Anthropic`` or compatible client.
        builder: Pre-configured ContextBuilder (already indexed).
        system_prompt: System prompt for the review (caller-supplied).

    Raises:
        ValidationError: If client does not expose ``.messages``.
    """

    _DEFAULT_SYSTEM = (
        "You are an expert code reviewer. Analyse the provided code context "
        "and the changed files. Identify bugs, security issues, performance "
        "problems, and design concerns. Be specific and actionable."
    )

    def __init__(
        self,
        client: Any,
        builder: ContextBuilder,
        system_prompt: str | None = None,
    ) -> None:
        if not hasattr(client, "messages"):
            raise ValidationError("Client must expose a .messages attribute.")
        self._client = client
        self._builder = builder
        self._system = system_prompt or self._DEFAULT_SYSTEM

    def __repr__(self) -> str:
        return f"ClaudeCodeReviewAdapter(builder={self._builder!r})"

    def review(
        self,
        changed_files: list[str],
        source_map: dict[str, str],
        *,
        model: str = "claude-opus-4-6",
        max_tokens: int = 4096,
        query: str = "Please review these changes.",
        commit_message: str = "",
        diff_text: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Run a code review with minimal token usage.

        Args:
            changed_files: List of changed file paths.
            source_map: Dict of file_path → source text.
            model: Claude model ID.
            max_tokens: Max response tokens.
            query: Specific review question.
            commit_message: Commit message for context.
            diff_text: Raw unified diff (optional).
            **kwargs: Extra args forwarded to ``client.messages.create``.

        Returns:
            Tuple of (Claude response, metadata dict with token savings).

        Raises:
            AdapterError: If the API call fails.
            ValidationError: If inputs are invalid.
        """
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

        ctx_result = self._builder.build(diff_spec, source_map)

        user_message = (
            f"{query}\n\n"
            f"{ctx_result.rendered_context}"
        )

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

        logger.info(
            "claude_review: sending context",
            extra={
                "rendered_tokens": ctx_result.total_rendered_tokens,
                "reduction": f"{ctx_result.reduction_ratio:.0%}",
            },
        )

        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system,
                messages=[{"role": "user", "content": user_message}],
                **kwargs,
            )
        except Exception as exc:
            raise AdapterError(f"Anthropic API call failed: {exc}") from exc

        return response, meta


class ClaudeContextAdapter:
    """General-purpose adapter: inject graphsift context into any messages call.

    Unlike ClaudeCodeReviewAdapter (specialised for code review), this adapter
    lets you build arbitrary prompts with graphsift-selected context injected.

    Args:
        client: Anthropic client.
        builder: Pre-indexed ContextBuilder.
    """

    def __init__(self, client: Any, builder: ContextBuilder) -> None:
        if not hasattr(client, "messages"):
            raise ValidationError("Client must expose a .messages attribute.")
        self._client = client
        self._builder = builder

    def __repr__(self) -> str:
        return "ClaudeContextAdapter()"

    def messages_create(
        self,
        *,
        changed_files: list[str],
        source_map: dict[str, str],
        messages: list[dict[str, str]],
        query: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Inject graphsift context into a messages list and call Claude.

        Args:
            changed_files: Changed file paths.
            source_map: file path → source text.
            messages: Existing messages list (context prepended to first user msg).
            query: Optional query string for ranking.
            **kwargs: Passed to ``client.messages.create``.

        Returns:
            Tuple of (response, metadata).
        """
        if not messages:
            raise ValidationError("messages must not be empty.")

        diff_spec = DiffSpec(changed_files=changed_files, query=query)
        ctx_result = self._builder.build(diff_spec, source_map)

        # Prepend context to the first user message
        enriched = list(messages)
        first_user_idx = next(
            (i for i, m in enumerate(enriched) if m.get("role") == "user"), None
        )
        if first_user_idx is not None:
            original_content = enriched[first_user_idx]["content"]
            enriched[first_user_idx] = {
                "role": "user",
                "content": f"{ctx_result.rendered_context}\n\n{original_content}",
            }

        meta: dict[str, Any] = {
            "files_selected": ctx_result.files_selected,
            "rendered_tokens": ctx_result.total_rendered_tokens,
            "reduction_ratio": ctx_result.reduction_ratio,
        }

        try:
            response = self._client.messages.create(messages=enriched, **kwargs)
        except Exception as exc:
            raise AdapterError(f"Anthropic API call failed: {exc}") from exc

        return response, meta
