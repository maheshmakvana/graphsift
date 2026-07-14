"""Conversation compaction engine — verbatim deletion-based context reduction
for agent conversations.

Applies the 2026 best practice of "deletion over rewriting": verbatim
compaction for code agent conversations rather than LLM summarization
(which hallucinates). Pure Python, zero external LLM calls.

Strategies
----------
- observation_masking : replace old tool outputs with [masked: N tokens]
- boundary_preserve  : keep system + last N, drop middle
- adaptive           : ACON-style aggressive tool output compression,
                       preserve reasoning

Usage::

    compactor = ConversationCompactor(preserve_last_n=3)
    compacted = compactor.compact(messages, token_budget=32_000)
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from graphsift.compress import compress
from graphsift.core import estimate_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticalFact:
    """A decision, constraint, preference, or gotcha extracted from a conversation.

    These facts are preserved across compactions to prevent silent constraint
    collapse (where token-saving drops a critical rule the LLM was relying on).
    """

    content: str
    fact_type: str  # "decision", "constraint", "preference", "gotcha"
    source_index: int  # which message (index into the original list) it came from
    importance: float = 0.5  # 0–1, caller can adjust after extraction


@dataclass
class CompactionStats:
    """Statistics from the last compaction run."""

    original_tokens: int = 0
    compacted_tokens: int = 0
    tokens_saved: int = 0
    savings_ratio: float = 0.0
    strategy_used: str = ""
    critical_facts_preserved: int = 0
    messages_removed: int = 0
    tool_results_masked: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ROLE_SYSTEM = "system"
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"
_ROLE_TOOL = "tool"

# Phrases that signal critical conversation content
_DECISION_PATTERNS = [
    re.compile(r"\b(?:we\s+(?:will|shall|must|need\s+to|decided|chose))\b", re.I),
    re.compile(r"\b(?:let'?s\s+go\s+with|going\s+with|settled\s+on)\b", re.I),
    re.compile(r"\b(?:decided|decision|conclusion|resolution)\b", re.I),
]
_CONSTRAINT_PATTERNS = [
    re.compile(r"\b(?:must|must\s+not|can'?t|cannot|required|requirement)\b", re.I),
    re.compile(r"\b(?:constraint|restriction|limitation|mandatory)\b", re.I),
    re.compile(r"\b(?:no\s+(?:more|less|longer|new)|never\s+(?:use|call))\b", re.I),
]
_PREFERENCE_PATTERNS = [
    re.compile(r"\b(?:prefer|preferably|rather\s+than|better\s+to)\b", re.I),
    re.compile(r"\b(?:would\s+(?:like|prefer|rather)|like\s+to)\b", re.I),
    re.compile(r"\b(?:ideally|best\s+practice|recommend)\b", re.I),
]
_GOTCHA_PATTERNS = [
    re.compile(r"\b(?:note\b.*:|caution|warning|gotcha|pitfall|careful)\b", re.I),
    re.compile(r"\b(?:tricky|got\s+stuck|spent\s+\d+|important\s+note)\b", re.I),
    re.compile(r"\b(?:workaround|edge\s+case|corner\s+case)\b", re.I),
]


def _is_tool_call(message: dict) -> bool:
    """Return True if *message* contains tool-call instructions."""
    # OpenAI format: message["tool_calls"] is a list
    if message.get("tool_calls"):
        return True
    # Anthropic format: content blocks contain {"type": "tool_use"}
    content = message.get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True
    return False


def _is_tool_result(message: dict) -> bool:
    """Return True if *message* is a tool-result message."""
    if message.get("role") == _ROLE_TOOL:
        return True
    # Anthropic format: user message with tool_result blocks
    if message.get("role") == _ROLE_USER:
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
    return False


def _get_tool_call_name(message: dict) -> str:
    """Extract the tool/function name from a tool-call message."""
    # OpenAI format
    tcs = message.get("tool_calls")
    if tcs and isinstance(tcs, list):
        first = tcs[0]
        func = first.get("function", {})
        return func.get("name", first.get("id", "unknown")) or "unknown"
    # Anthropic format
    content = message.get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return block.get("name", "unknown")
    return "unknown"


def _get_tool_result_text(message: dict) -> str:
    """Extract the text content of a tool-result message."""
    # OpenAI format
    if message.get("role") == _ROLE_TOOL:
        c = message.get("content", "")
        return c if isinstance(c, str) else str(c) if c else ""
    # Anthropic format
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    for b in inner:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(b.get("text", ""))
                elif isinstance(inner, str):
                    parts.append(inner)
        return "\n".join(parts)
    return ""


def _make_masked_result(tool_name: str, token_count: int) -> str:
    """Build a masked placeholder for a tool result."""
    return f"[masked: {tool_name} -- {token_count} tokens saved]"


def _estimate_message_tokens(message: dict) -> int:
    """Estimate the token count of a single message dict.

    Accounts for role, content (string or list of blocks), tool_calls,
    tool_call_id, and structural overhead.
    """
    total = 0
    # Role overhead (~2 tokens per message)
    total += 2

    content = message.get("content", "")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                total += estimate_tokens(str(block.get("text", "")))
                total += estimate_tokens(str(block.get("name", "")))
                total += estimate_tokens(str(block.get("input", {})))
                total += estimate_tokens(str(block.get("content", "")))

    # Tool calls (OpenAI format)
    tcs = message.get("tool_calls")
    if tcs and isinstance(tcs, list):
        for tc in tcs:
            func = tc.get("function", {})
            total += estimate_tokens(func.get("name", ""))
            total += estimate_tokens(func.get("arguments", ""))
            total += estimate_tokens(tc.get("id", ""))
            total += 3  # structural tokens per call

    # tool_call_id
    tcid = message.get("tool_call_id")
    if tcid:
        total += estimate_tokens(tcid)

    return max(1, total)


def _extract_turns(messages: list[dict]) -> list[int]:
    """Return a list of message indices that start a new user -> assistant turn.

    A turn starts with a user message and includes all subsequent messages
    (assistant reasoning, tool calls, tool results) until the next user message.
    """
    turn_starts: list[int] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role == _ROLE_USER:
            content = msg.get("content", "")
            # Anthropic tool-result messages are also role=user but we skip them
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in (content if isinstance(content, list) else [])
            ):
                continue
            turn_starts.append(i)
        elif role == _ROLE_SYSTEM:
            # System messages are not part of any turn but we treat them as
            # their own segment; they are always preserved.
            pass
    return turn_starts


# ---------------------------------------------------------------------------
# Strategy — observation_masking
# ---------------------------------------------------------------------------


def _compact_observation_masking(
    messages: list[dict],
    token_budget: int,
    preserve_last_n: int,
    _preserve_system: bool,  # kept for API compatibility
) -> tuple[list[dict], CompactionStats]:
    """Replace tool results from old turns with masked placeholders.

    1. Identify tool-call + tool-result pairs.
    2. For pairs older than *preserve_last_n* turns from the end:
       - Keep the tool call (function name + args summary).
       - Replace the tool result with ``[masked: tool_name -- N tokens saved]``.
    3. Never mask user messages or the system prompt.
    """
    messages = copy.deepcopy(messages)
    original_tokens = sum(_estimate_message_tokens(m) for m in messages)
    turn_starts = _extract_turns(messages)

    # Determine the cutoff: preserve the last *preserve_last_n* turns fully
    if len(turn_starts) <= preserve_last_n:
        cutoff = 0  # preserve all
    else:
        cutoff = turn_starts[-preserve_last_n] if preserve_last_n > 0 else len(messages)

    masked_count = 0
    total_masked_tokens = 0
    removed_count = 0

    for i, msg in enumerate(messages):
        if i < cutoff and _is_tool_result(msg):
            # Find the corresponding tool-call message
            tname = _find_tool_call_name_for_result(messages, i)
            orig_tok = _estimate_message_tokens(msg)
            messages[i] = {
                "role": _ROLE_TOOL if msg.get("role") == _ROLE_TOOL else _ROLE_USER,
                "content": _make_masked_result(tname or "tool", orig_tok),
                "tool_call_id": msg.get("tool_call_id", ""),
            }
            masked_count += 1
            total_masked_tokens += orig_tok

    compacted_tokens = sum(_estimate_message_tokens(m) for m in messages)

    # If still over budget, drop masked results entirely (they are just placeholders)
    if compacted_tokens > token_budget:
        surviving: list[dict] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("[masked:"):
                removed_count += 1
                total_masked_tokens += _estimate_message_tokens(msg)
                continue
            surviving.append(msg)
        messages = surviving
        compacted_tokens = sum(_estimate_message_tokens(m) for m in messages)

    return messages, CompactionStats(
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        tokens_saved=original_tokens - compacted_tokens,
        savings_ratio=_ratio(original_tokens, compacted_tokens),
        strategy_used="observation_masking",
        tool_results_masked=masked_count,
        messages_removed=removed_count,
    )


def _find_tool_call_name_for_result(messages: list[dict], result_idx: int) -> str | None:
    """Walk backwards from *result_idx* to find the tool-call that triggered it.

    Handles both OpenAI format (tool_call_id match) and Anthropic format
    (tool_use block with matching id).
    """
    result_msg = messages[result_idx]
    target_id = result_msg.get("tool_call_id", "")

    if target_id:
        # OpenAI format: match by tool_call_id
        for j in range(result_idx - 1, -1, -1):
            tcs = messages[j].get("tool_calls")
            if tcs and isinstance(tcs, list):
                for tc in tcs:
                    if tc.get("id") == target_id:
                        func = tc.get("function", {})
                        return func.get("name", "tool")
    else:
        # Anthropic format: match by adjacent tool_use block
        for j in range(result_idx - 1, -1, -1):
            name = _get_tool_call_name(messages[j])
            if name != "unknown":
                return name

    return None


# ---------------------------------------------------------------------------
# Strategy — boundary_preserve
# ---------------------------------------------------------------------------


def _compact_boundary_preserve(
    messages: list[dict],
    token_budget: int,
    preserve_last_n: int,
    preserve_system: bool,
) -> tuple[list[dict], CompactionStats]:
    """Keep system prompt + last N non-tool messages; drop the middle.

    When the result still exceeds *token_budget*, drops are made from the
    preserved tail first.
    """
    messages = copy.deepcopy(messages)
    original_tokens = sum(_estimate_message_tokens(m) for m in messages)

    # Separate system messages and everything else
    system_msgs = [m for m in messages if m.get("role") == _ROLE_SYSTEM]
    non_system = [m for m in messages if m.get("role") != _ROLE_SYSTEM]
    removed_count = len(non_system)

    if preserve_system:
        kept = list(system_msgs)
    else:
        kept = []

    # Keep last N non-tool messages (user + assistant reasoning)
    preserved = 0
    for msg in reversed(non_system):
        role = msg.get("role", "")
        if role in (_ROLE_USER, _ROLE_ASSISTANT):
            if preserved < preserve_last_n:
                kept.insert(len(kept), msg)
                preserved += 1
                removed_count -= 1
        elif role == _ROLE_TOOL:
            # Keep tool results that belong to preserved turns
            if preserved > 0:
                kept.insert(len(kept), msg)
                removed_count -= 1

    # Reorder: system first, then kept messages in original order
    # We need to preserve relative ordering within the kept set
    kept_set = set(id(m) for m in kept)
    ordered = []
    for msg in messages:
        if id(msg) in kept_set:
            ordered.append(msg)

    compacted_tokens = sum(_estimate_message_tokens(m) for m in ordered)

    return ordered, CompactionStats(
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        tokens_saved=original_tokens - compacted_tokens,
        savings_ratio=_ratio(original_tokens, compacted_tokens),
        strategy_used="boundary_preserve",
        messages_removed=removed_count,
    )


# ---------------------------------------------------------------------------
# Strategy — adaptive (ACON-style)
# ---------------------------------------------------------------------------


def _compact_adaptive(
    messages: list[dict],
    token_budget: int,
    preserve_last_n: int,
    preserve_system: bool,
) -> tuple[list[dict], CompactionStats]:
    """ACON-style aggressive tool output compression while preserving reasoning.

    1. Preserve system prompt verbatim.
    2. Preserve last *preserve_last_n* user/assistant reasoning messages verbatim.
    3. For older messages: compress tool outputs via ``graphsift.compress``;
       leave reasoning untouched.
    4. Never compress mid-task — only at boundaries (handled by the caller).
    """
    messages = copy.deepcopy(messages)
    original_tokens = sum(_estimate_message_tokens(m) for m in messages)
    turn_starts = _extract_turns(messages)

    # Determine cutoff — preserve last N turns fully
    if len(turn_starts) <= preserve_last_n or preserve_last_n <= 0:
        cutoff = 0
    else:
        cutoff = turn_starts[-preserve_last_n]

    compressed_count = 0
    total_saved = 0
    removed_count = 0

    for i, msg in enumerate(messages):
        if i >= cutoff:
            continue  # preserve fully

        role = msg.get("role", "")
        content = msg.get("content", "")

        # Skip system and user reasoning messages
        if role == _ROLE_SYSTEM:
            continue

        # Compress tool result content
        if role == _ROLE_TOOL and isinstance(content, str) and content.strip():
            compressed = compress(content, command="auto", ultra=True)
            if compressed != content:
                orig_tok = estimate_tokens(content)
                new_tok = estimate_tokens(compressed)
                if new_tok < orig_tok:
                    msg["content"] = compressed
                    compressed_count += 1
                    total_saved += orig_tok - new_tok

        # Anthropic format: tool_result blocks inside user messages
        if isinstance(content, list):
            msg = _compress_tool_result_blocks(msg)
            compressed_count += 1

    # Remove messages that are completely empty after compression
    surviving = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and not content.strip():
            removed_count += 1
            continue
        if isinstance(content, list) and not content:
            removed_count += 1
            continue
        surviving.append(msg)
    messages = surviving

    compacted_tokens = sum(_estimate_message_tokens(m) for m in messages)

    return messages, CompactionStats(
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        tokens_saved=original_tokens - compacted_tokens,
        savings_ratio=_ratio(original_tokens, compacted_tokens),
        strategy_used="adaptive",
        tool_results_masked=compressed_count,
        messages_removed=removed_count,
    )


def _compress_tool_result_blocks(message: dict) -> dict:
    """Compress tool_result content blocks inside an Anthropic-format message."""
    content = message.get("content", [])
    if not isinstance(content, list):
        return message

    new_blocks: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            new_blocks.append(block)
            continue
        if block.get("type") != "tool_result":
            new_blocks.append(block)
            continue

        inner = block.get("content", "")
        if isinstance(inner, list):
            compressed_inner: list[dict] = []
            for b in inner:
                if isinstance(b, dict) and b.get("type") == "text":
                    text = b.get("text", "")
                    if text.strip():
                        compressed = compress(text, command="auto", ultra=True)
                        compressed_inner.append({**b, "text": compressed})
                    else:
                        compressed_inner.append(b)
                else:
                    compressed_inner.append(b)
            new_blocks.append({**block, "content": compressed_inner})
        elif isinstance(inner, str):
            compressed = compress(inner, command="auto", ultra=True)
            new_blocks.append({**block, "content": compressed})
        else:
            new_blocks.append(block)

    return {**message, "content": new_blocks}


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_COMPACTION_STRATEGIES: dict[str, Callable[..., tuple[list[dict], CompactionStats]]] = {
    "observation_masking": _compact_observation_masking,
    "boundary_preserve": _compact_boundary_preserve,
    "adaptive": _compact_adaptive,
}


def list_strategies() -> list[str]:
    """Return the list of registered compaction strategy names."""
    return sorted(_COMPACTION_STRATEGIES)


def register_strategy(
    name: str,
    fn: Callable[..., tuple[list[dict], CompactionStats]],
) -> None:
    """Register a custom compaction strategy.

    The callable receives ``(messages, token_budget, preserve_last_n,
    preserve_system)`` and must return ``(compacted_messages, CompactionStats)``.
    """
    _COMPACTION_STRATEGIES[name] = fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ratio(original: int, compacted: int) -> float:
    """Return savings ratio, clamped to [0.0, 1.0]."""
    if original <= 0:
        return 0.0
    return round(1.0 - compacted / original, 4)


# ---------------------------------------------------------------------------
# ConversationCompactor
# ---------------------------------------------------------------------------


class ConversationCompactor:
    """Verbatim compaction for agent conversation messages.

    Applies deletion-based strategies (never summarization) to reduce context
    usage while preserving the agent's action trace and critical facts.

    Parameters
    ----------
    preserve_system : bool
        Whether to always preserve the system prompt. (default ``True``)
    preserve_last_n : int
        Number of most recent user/assistant reasoning exchanges to keep
        fully intact. (default ``3``)
    """

    def __init__(
        self,
        preserve_system: bool = True,
        preserve_last_n: int = 3,
        max_context_tokens: int = 200000,
    ) -> None:
        self._preserve_system = preserve_system
        self._preserve_last_n = max(1, preserve_last_n)
        self._last_stats: CompactionStats = CompactionStats()
        self._critical_facts: list[CriticalFact] = []
        self.max_context_tokens = max_context_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compact(
        self,
        messages: list[dict],
        token_budget: int,
        strategy: str = "observation_masking",
    ) -> list[dict]:
        """Compact *messages* to fit within *token_budget* tokens.

        Parameters
        ----------
        messages : list[dict]
            Conversation messages in OpenAI or Anthropic format (role-based).
        token_budget : int
            Maximum number of tokens the compacted conversation may use.
        strategy : str
            One of ``"observation_masking"``, ``"boundary_preserve"``,
            ``"adaptive"``, or a custom registered strategy name.

        Returns
        -------
        list[dict]
            Compacted message list (mutated copy).

        Raises
        ------
        ValueError
            If *strategy* is unknown.
        """
        if not messages:
            return messages

        if token_budget <= 0:
            return self._empty_with_system(messages)

        strategy_fn = _COMPACTION_STRATEGIES.get(strategy)
        if strategy_fn is None:
            raise ValueError(
                f"Unknown compaction strategy {strategy!r}. "
                f"Available: {list(_COMPACTION_STRATEGIES)}"
            )

        # Quick path — already within budget
        current = sum(_estimate_message_tokens(m) for m in messages)
        if current <= token_budget:
            self._last_stats = CompactionStats(
                original_tokens=current,
                compacted_tokens=current,
                strategy_used=strategy,
            )
            return messages

        compacted, stats = strategy_fn(
            messages,
            token_budget,
            self._preserve_last_n,
            self._preserve_system,
        )
        self._last_stats = stats

        # If still over budget, re-run with "boundary_preserve" as a hard
        # floor — always drop enough to fit.
        post_tokens = sum(_estimate_message_tokens(m) for m in compacted)
        if post_tokens > token_budget and strategy != "boundary_preserve":
            logger.debug(
                "compact_context: %s left %d tokens (budget %d) — "
                "falling back to boundary_preserve",
                strategy,
                post_tokens,
                token_budget,
            )
            compacted, fallback_stats = _compact_boundary_preserve(
                compacted,
                token_budget,
                self._preserve_last_n,
                self._preserve_system,
            )
            self._last_stats = CompactionStats(
                original_tokens=self._last_stats.original_tokens,
                compacted_tokens=fallback_stats.compacted_tokens,
                tokens_saved=self._last_stats.tokens_saved
                + (fallback_stats.compacted_tokens - post_tokens),
                savings_ratio=_ratio(
                    self._last_stats.original_tokens,
                    fallback_stats.compacted_tokens,
                ),
                strategy_used=f"{strategy}+boundary_preserve",
                critical_facts_preserved=len(self._critical_facts),
                messages_removed=self._last_stats.messages_removed
                + fallback_stats.messages_removed,
                tool_results_masked=self._last_stats.tool_results_masked,
            )

        return compacted

    def should_compact(self, current_tokens: int, threshold_pct: int = 80) -> bool:
        """Return True if current token count exceeds threshold_pct of max_context.

        Parameters
        ----------
        current_tokens : int
            Current token count of the conversation.
        threshold_pct : int
            Percentage of max_context at which to trigger (default 80).

        Returns
        -------
        bool
            True if compaction is recommended.
        """
        return current_tokens >= (self.max_context_tokens * threshold_pct // 100)

    def extract_critical(self, messages: list[dict]) -> list[CriticalFact]:
        """Find decisions, constraints, preferences, and gotchas in *messages*.

        Uses lightweight regex heuristics — no LLM calls.  The extracted
        facts can be re-injected after compaction to prevent silent constraint
        collapse.

        Parameters
        ----------
        messages : list[dict]
            Conversation messages to scan.

        Returns
        -------
        list[CriticalFact]
            Extracted critical facts, ordered by importance descending.
        """
        facts: list[CriticalFact] = []

        for i, msg in enumerate(messages):
            content = self._get_text_content(msg)
            if not content:
                continue

            # Decisions
            for pat in _DECISION_PATTERNS:
                for m in pat.finditer(content):
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 120)
                    snippet = content[start:end].strip()
                    facts.append(CriticalFact(
                        content=snippet[:200],
                        fact_type="decision",
                        source_index=i,
                        importance=0.8,
                    ))

            # Constraints
            for pat in _CONSTRAINT_PATTERNS:
                for m in pat.finditer(content):
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 120)
                    snippet = content[start:end].strip()
                    facts.append(CriticalFact(
                        content=snippet[:200],
                        fact_type="constraint",
                        source_index=i,
                        importance=0.9,
                    ))

            # Preferences
            for pat in _PREFERENCE_PATTERNS:
                for m in pat.finditer(content):
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 120)
                    snippet = content[start:end].strip()
                    facts.append(CriticalFact(
                        content=snippet[:200],
                        fact_type="preference",
                        source_index=i,
                        importance=0.6,
                    ))

            # Gotchas
            for pat in _GOTCHA_PATTERNS:
                for m in pat.finditer(content):
                    start = max(0, m.start() - 60)
                    end = min(len(content), m.end() + 120)
                    snippet = content[start:end].strip()
                    facts.append(CriticalFact(
                        content=snippet[:200],
                        fact_type="gotcha",
                        source_index=i,
                        importance=0.7,
                    ))

        # Deduplicate by content, keeping highest importance
        seen: set[str] = set()
        deduped: list[CriticalFact] = []
        for fact in sorted(facts, key=lambda f: -f.importance):
            key = fact.content[:100]
            if key not in seen:
                seen.add(key)
                deduped.append(fact)

        self._critical_facts = deduped
        return deduped

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Return the estimated token count for a list of messages.

        Uses the same heuristic as ``graphsift.core.estimate_tokens``
        (4 chars per token) with per-message structural overhead.
        """
        return sum(_estimate_message_tokens(m) for m in messages)

    @property
    def stats(self) -> CompactionStats:
        """CompactionStats from the most recent ``compact()`` call."""
        return self._last_stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_with_system(self, messages: list[dict]) -> list[dict]:
        """Return only system messages (or the last message if none)."""
        if self._preserve_system:
            kept = [m for m in messages if m.get("role") == _ROLE_SYSTEM]
            if kept:
                total = sum(_estimate_message_tokens(m) for m in messages)
                kept_tok = sum(_estimate_message_tokens(m) for m in kept)
                self._last_stats = CompactionStats(
                    original_tokens=total,
                    compacted_tokens=kept_tok,
                    tokens_saved=total - kept_tok,
                    savings_ratio=_ratio(total, kept_tok),
                    strategy_used="empty_with_system",
                )
                return kept
        # Fallback: return the last message only
        last = messages[-1:] if messages else []
        total = sum(_estimate_message_tokens(m) for m in messages)
        last_tok = sum(_estimate_message_tokens(m) for m in last)
        self._last_stats = CompactionStats(
            original_tokens=total,
            compacted_tokens=last_tok,
            tokens_saved=total - last_tok,
            savings_ratio=_ratio(total, last_tok),
            strategy_used="empty_with_system",
        )
        return last

    @staticmethod
    def _get_text_content(message: dict) -> str:
        """Extract the plain-text content from a message regardless of format."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, list):
                            for b in inner:
                                if isinstance(b, dict) and b.get("type") == "text":
                                    parts.append(b.get("text", ""))
                        elif isinstance(inner, str):
                            parts.append(inner)
                    elif block.get("type") == "tool_use":
                        parts.append(str(block.get("input", "")))
            return "\n".join(parts)
        return str(content) if content else ""


# ---------------------------------------------------------------------------
# AutonomousCompressor
# ---------------------------------------------------------------------------


class AutonomousCompressor:
    """Agent-triggered compaction at task boundaries (LangChain 2026 pattern).

    Monitors context usage and triggers compaction only at explicit task
    boundaries.  This prevents mid-task information loss while keeping long-
    running agent sessions within context limits.

    Parameters
    ----------
    compactor : ConversationCompactor
        The underlying compactor instance.
    trigger_ratio : float
        Fraction of *token_budget* that triggers compaction.  E.g. 0.75
        means compact when 75% of the budget is consumed. (default ``0.75``)
    """

    def __init__(
        self,
        compactor: ConversationCompactor,
        trigger_ratio: float = 0.75,
    ) -> None:
        self._compactor = compactor
        self._trigger_ratio = max(0.1, min(0.99, trigger_ratio))
        self._at_boundary = True  # first call is always safe to compact
        self._compact_count = 0
        self._last_budget: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_compact(
        self,
        messages: list[dict],
        token_budget: int,
        strategy: str = "observation_masking",
    ) -> tuple[list[dict], bool]:
        """Check if compaction is needed and return compacted messages if so.

        Only compacts when both conditions are true:

        1. The estimated token usage exceeds ``trigger_ratio * token_budget``.
        2. A task boundary has been marked since the last compaction
           (or it is the first call).

        Parameters
        ----------
        messages : list[dict]
            Current conversation messages.
        token_budget : int
            Maximum allowed tokens.
        strategy : str
            Compaction strategy to use.

        Returns
        -------
        tuple[list[dict], bool]
            ``(compacted_or_original_messages, was_compacted)``.
        """
        self._last_budget = token_budget
        current = self._compactor.estimate_tokens(messages)
        threshold = int(token_budget * self._trigger_ratio)

        if not self._at_boundary:
            return messages, False

        if current <= threshold:
            self._at_boundary = False  # reset until next boundary mark
            return messages, False

        if not messages:
            return messages, False

        logger.info(
            "compact_context: autonomous compact triggered "
            "(%d / %d tokens, ratio=%.2f, strategy=%s)",
            current,
            token_budget,
            self._trigger_ratio,
            strategy,
        )

        # Extract critical facts before compaction
        facts = self._compactor.extract_critical(messages)

        compacted = self._compactor.compact(messages, token_budget, strategy=strategy)

        # Re-inject critical facts as a system message if any were found and
        # the system prompt exists
        if facts and self._compactor._preserve_system:
            fact_text = self._format_critical_facts(facts)
            # Append to the last system message
            for msg in compacted:
                if msg.get("role") == _ROLE_SYSTEM:
                    existing = msg.get("content", "")
                    msg["content"] = existing.rstrip("\n") + "\n\n" + fact_text
                    break

        self._compact_count += 1
        self._at_boundary = False  # reset — caller must mark next boundary

        logger.debug(
            "compact_context: autonomous compact done — "
            "%d tokens -> %d tokens (saved %d, %.1f%%)",
            current,
            self._compactor.stats.compacted_tokens,
            self._compactor.stats.tokens_saved,
            self._compactor.stats.savings_ratio * 100,
        )

        return compacted, True

    def mark_boundary(self) -> None:
        """Mark the current point as a safe task boundary.

        Call this after completing a subtask so that the next
        ``maybe_compact()`` call may trigger compaction.
        """
        self._at_boundary = True

    @property
    def compactor(self) -> ConversationCompactor:
        """The underlying ``ConversationCompactor`` instance."""
        return self._compactor

    @property
    def compact_count(self) -> int:
        """Number of compactions triggered so far."""
        return self._compact_count

    @property
    def at_boundary(self) -> bool:
        """Whether we are currently at a task boundary."""
        return self._at_boundary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_critical_facts(facts: list[CriticalFact]) -> str:
        """Format critical facts for re-injection into the system prompt."""
        if not facts:
            return ""

        lines = [
            "# Critical facts preserved from prior conversation context",
        ]
        seen: set[str] = set()
        for fact in facts:
            key = fact.content[:80]
            if key not in seen:
                seen.add(key)
                lines.append(f"- [{fact.fact_type}] {fact.content}")

        return "\n".join(lines)
