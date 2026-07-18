"""Tiered memory — axioms, rules, topic files, archives.

Manages a hierarchical memory structure for token-efficient context loading.
Each tier has a different loading strategy and token budget, allowing LLM
agents to access the right level of detail without blowing the context window.

Tiers (in order of loading priority):
    1. **axioms** — Always loaded, capped at 12 items. Core truths about the project.
    2. **rules** — Auto-loaded from ``.claude/rules/`` directory. Project conventions.
    3. **topic** — Keys only are pre-loaded; full content loaded on demand.
    4. **archive** — Grep-only. Not loaded automatically; searched when needed.

Usage::
    mem = TieredMemory(repo_root)
    axioms = mem.get_tier("axioms")      # Always loaded, <=12 items
    rules = mem.get_tier("rules")        # Auto-loaded from .claude/rules/
    topic = mem.get_tier("topic")        # Keys for on-demand loading
    archive = mem.get_tier("archive")    # Grep only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from graphsift.read_cache import SafeFileIO


@dataclass
class TieredMemory:
    """Manages a tiered memory structure for token-efficient context loading."""

    repo_root: str
    axioms: list[str] = field(default_factory=list)
    rules_dir: str = ".claude/rules"
    topic_files: dict[str, str] = field(default_factory=dict)

    _axiom_max: int = 12

    # ── Public API ──────────────────────────────────────────────────────

    def get_tier(self, tier: str) -> list[str]:
        """Return items for a given tier ('axioms', 'rules', 'topic', 'archive')."""
        root = Path(self.repo_root)
        if tier == "axioms":
            return self.axioms[: self._axiom_max]
        elif tier == "rules":
            return self._load_rules(root)
        elif tier == "topic":
            return list(self.topic_files.keys())
        elif tier == "archive":
            return []  # archive is grep-only, returns nothing by default
        return []

    def add_axiom(self, axiom: str) -> None:
        """Add an axiom (will persist in memory for the session)."""
        if axiom not in self.axioms:
            self.axioms.append(axiom)

    def load_topic(self, topic: str) -> Optional[str]:
        """Load a topic file's content from disk."""
        path = self.topic_files.get(topic)
        if path and Path(path).exists():
            return SafeFileIO.read(path)
        return None

    def estimate_tokens(self, tier: str) -> int:
        """Rough token estimate for a tier's content."""
        items = self.get_tier(tier)
        text = "\n".join(items) if isinstance(items, list) else str(items)
        return len(text) // 4  # rough: ~4 chars per token

    # ── Internal ────────────────────────────────────────────────────────

    def _load_rules(self, root: Path) -> list[str]:
        rules_path = root / self.rules_dir
        if not rules_path.exists():
            return []
        rules: list[str] = []
        for f in sorted(rules_path.glob("*.md")):
            rules.append(SafeFileIO.read(f).strip())
        return rules
