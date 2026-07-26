"""Configuration schema for loop-engineering patterns.

Defines LoopConfig — the settings object that controls how loop patterns
behave: cadence, maturity, budgets, and safety limits.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphsift.read_cache import SafeFileIO

from graphsift.loop_engineering import MaturityLevel, PatternType, PATTERN_REGISTRY


@dataclass
class LoopPatternSettings:
	"""Per-pattern configuration settings."""

	enabled: bool = True
	cadence_seconds: int | None = None  # None = use registry default
	maturity: str = "L1"
	token_budget: int | None = None
	max_runs_per_day: int = 24
	max_consecutive_failures: int = 5
	allowed_actions: list[str] = field(default_factory=lambda: [
		"fix_unused_import", "fix_type_annotation", "fix_dead_code",
		"update_minor_dep", "update_patch_dep", "remove_temp_file",
		"cleanup_branch", "fix_syntax_error", "add_docstring",
	])


@dataclass
class LoopConfig:
	"""Top-level configuration for the loop-engineering subsystem.

	Stored as JSON at ``.graphsift/loop-config.json`` in the project root.

	Usage::

		config = LoopConfig.load(project_root="/path/to/repo")
		config.patterns["ci_sweeper"].maturity = "L2"
		config.save()
	"""

	patterns: dict[str, LoopPatternSettings] = field(default_factory=dict)
	worktree_base: str = ""  # empty = default (~/.graphsift/worktrees/<hash>)
	state_dir: str = ""  # empty = default (~/.graphsift/loops/<hash>)
	max_daily_tokens: int = 500_000
	auto_start_scheduler: bool = False
	verbose_logging: bool = False
	# Danger-level denylist
	denylist: list[str] = field(default_factory=lambda: [
		"delete_branch_main", "force_push", "modify_ci_config",
		"modify_deploy_config", "modify_security_config",
		"delete_production", "modify_dependencies_major", "modify_secrets",
	])

	def __post_init__(self) -> None:
		"""Ensure all 7 patterns have default settings."""
		for ptype in PatternType:
			name = ptype.value
			if name not in self.patterns:
				info = PATTERN_REGISTRY.get(ptype, {})
				self.patterns[name] = LoopPatternSettings(
					enabled=True,
					cadence_seconds=info.get("default_cadence_seconds", 3600),
					maturity=info.get("week1_maturity", MaturityLevel.L1_REPORT).value,
					token_budget=info.get("token_budget", 50_000),
				)

	@classmethod
	def load(cls, project_root: str | None = None) -> LoopConfig:
		"""Load config from ``.graphsift/loop-config.json`` in *project_root*.

		Returns defaults if the file doesn't exist.
		"""
		config_path = cls._config_path(project_root)
		if config_path.exists():
			try:
				data = SafeFileIO.read_json(config_path)
				return cls(**data)
			except (json.JSONDecodeError, TypeError, KeyError):
				pass
		return cls()

	def save(self, project_root: str | None = None) -> None:
		"""Save config to ``.graphsift/loop-config.json``."""
		config_path = self._config_path(project_root)
		config_path.parent.mkdir(parents=True, exist_ok=True)
		SafeFileIO.write_json(config_path, self.to_dict())

	def to_dict(self) -> dict[str, Any]:
		"""Serialize to dict for JSON persistence."""
		return {
			"patterns": {
				name: {
					"enabled": ps.enabled,
					"cadence_seconds": ps.cadence_seconds,
					"maturity": ps.maturity,
					"token_budget": ps.token_budget,
					"max_runs_per_day": ps.max_runs_per_day,
					"max_consecutive_failures": ps.max_consecutive_failures,
					"allowed_actions": ps.allowed_actions,
				}
				for name, ps in self.patterns.items()
			},
			"worktree_base": self.worktree_base,
			"state_dir": self.state_dir,
			"max_daily_tokens": self.max_daily_tokens,
			"auto_start_scheduler": self.auto_start_scheduler,
			"verbose_logging": self.verbose_logging,
			"denylist": self.denylist,
		}

	@staticmethod
	def _config_path(project_root: str | None = None) -> Path:
		root = Path(project_root or os.getcwd()).resolve()
		return root / ".graphsift" / "loop-config.json"
