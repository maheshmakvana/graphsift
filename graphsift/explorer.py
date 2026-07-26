"""Deep context enricher — discovers hidden dependencies for changed files.

Finds config files, env vars, Docker/CI references, test files, and git
co-change patterns related to changed source files. Extends what the
dependency graph finds by looking outside pure import/call graphs into
the project's configuration, infrastructure, and history.

Matches Goose's aggressive context gathering — finding every relevant
config, test, and dependency before making changes.

Usage::

    enricher = ContextEnricher(root="/repo", store=store)
    result = enricher.enrich(changed_files=["src/auth.py"])
    print(result.summary)
    for d in result.discoveries:
        print(f"  [{d.type}] {d.path} (relevance={d.relevance:.2f})")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphsift.executor import ProcessRunner
from graphsift.read_cache import SafeFileIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery types (plain strings for lightweight use)
# ---------------------------------------------------------------------------

class DiscoveryType:
    """Type constants for discoveries."""
    CONFIG = "config"
    ENV_VAR = "env_var"
    TEST_FILE = "test_file"
    CO_CHANGE = "co_change"
    DOCKER_REF = "docker_ref"
    CI_REF = "ci_ref"


@dataclass
class Discovery:
    """A single item discovered during context enrichment."""
    type: str
    path: str
    relevance: float = 0.5
    evidence: str = ""
    content_snippet: str = ""
    source: str = ""


@dataclass
class EnrichmentResult:
    """Aggregated results from a full enrichment run."""
    changed_files: list[str]
    discoveries: list[Discovery] = field(default_factory=list)
    config_files: list[Discovery] = field(default_factory=list)
    test_files: list[Discovery] = field(default_factory=list)
    co_changed_files: list[Discovery] = field(default_factory=list)
    env_refs: list[Discovery] = field(default_factory=list)
    docker_refs: list[Discovery] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.discoveries)

    @property
    def summary(self) -> str:
        return (
            f"Enriched {len(self.changed_files)} file(s): "
            f"{len(self.config_files)} config, "
            f"{len(self.test_files)} tests, "
            f"{len(self.co_changed_files)} co-changes, "
            f"{len(self.env_refs)} env refs, "
            f"{len(self.docker_refs)} docker refs"
        )


# ---------------------------------------------------------------------------
# Discoverers
# ---------------------------------------------------------------------------

class ConfigDiscoverer:
    """Find config files that reference the changed source files."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def discover(self, changed_files: list[str]) -> list[Discovery]:
        """Search well-known config files for references."""
        discoveries: list[Discovery] = []
        changed_set = set(changed_files)
        changed_names = {Path(cf).name for cf in changed_files if cf}

        config_targets: list[tuple[str, re.Pattern, str]] = [
            ("pyproject.toml", re.compile(r"[\"']([^\"']*\.py)[\"']"), "python"),
            ("package.json", re.compile(r"[\"']([^\"']+(?:\.js|\.ts|\.jsx|\.tsx))[\"']"), "node"),
            ("Dockerfile", re.compile(r"(?:COPY|ADD)\s+([^\s]+)"), "docker"),
        ]

        for config_name, pattern, _source in config_targets:
            config_path = self.root / config_name
            if not config_path.exists():
                continue
            try:
                content = SafeFileIO.read(config_path)
            except Exception:
                continue

            for match in pattern.finditer(content):
                ref = match.group(1)
                for cf in changed_files:
                    if (cf in ref or Path(cf).name in ref
                            or Path(cf).stem in ref):
                        discoveries.append(Discovery(
                            type=DiscoveryType.CONFIG,
                            path=str(config_path.relative_to(self.root)),
                            relevance=0.8,
                            evidence=f"References changed file: {cf}",
                            content_snippet=content[:200],
                            source=config_name,
                        ))
                        break
        return discoveries


class TestDiscoverer:
    """Find test files related to changed source files."""
    __test__ = False  # pytest: not a test class

    def __init__(self, root: Path) -> None:
        self.root = root

    def discover(self, changed_files: list[str]) -> list[Discovery]:
        """Map changed files to likely test file counterparts."""
        discoveries: list[Discovery] = []

        for cf in changed_files:
            cf_path = Path(cf)
            name = cf_path.name
            stem = cf_path.stem
            parent = str(cf_path.parent) if str(cf_path.parent) != "." else ""

            # Common test file naming patterns to check
            candidates = [
                self.root / "tests" / f"test_{name}",
                self.root / "tests" / parent / f"test_{name}",
                self.root / parent / f"test_{name}",
                self.root / cf.replace(f".{stem}", "_test."),
            ]

            for tc in candidates:
                if tc.exists() and tc.is_file():
                    try:
                        rel = str(tc.relative_to(self.root))
                    except ValueError:
                        rel = str(tc)
                    discoveries.append(Discovery(
                        type=DiscoveryType.TEST_FILE,
                        path=rel,
                        relevance=0.9,
                        evidence=f"Corresponds to changed file: {cf}",
                        source=cf,
                    ))

        return discoveries


class CoChangeDiscoverer:
    """Use git history to find files that co-change."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._runner = ProcessRunner(cwd=str(self.root), timeout=30)

    def discover(
        self, changed_files: list[str], max_results: int = 10
    ) -> list[Discovery]:
        """Find files historically changed together with the given files."""
        discoveries: list[Discovery] = []

        if not (self.root / ".git").exists():
            return discoveries

        for cf in changed_files[:3]:
            try:
                result = self._runner.run_simple(
                    [
                        "git", "log", "--all", "--name-only",
                        "--pretty=format:", "-n", "50", "--", cf,
                    ],
                    timeout=30,
                )
                co_counts: dict[str, int] = {}
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if line and line != cf and Path(line).suffix in {
                        ".py", ".js", ".ts", ".tsx", ".jsx",
                        ".go", ".rs", ".java", ".rb",
                    }:
                        co_counts[line] = co_counts.get(line, 0) + 1

                total = max(sum(co_counts.values()), 1)
                for co_path, count in sorted(
                    co_counts.items(), key=lambda x: -x[1]
                )[:max_results]:
                    ratio = count / total
                    if ratio > 0.05:
                        discoveries.append(Discovery(
                            type=DiscoveryType.CO_CHANGE,
                            path=co_path,
                            relevance=min(ratio, 1.0),
                            evidence=(
                                f"Co-changed with {cf} in "
                                f"{count}/{int(total)} commits"
                            ),
                            source=cf,
                        ))
            except Exception:
                pass

        return discoveries


class EnvDiscoverer:
    """Find environment variable references in changed files."""

    _ENV_ACCESS_RE = re.compile(
        r'(?:os\.environ|os\.getenv|os\.env)\s*\['
        r'\s*["\']([^"\']+)["\']'
    )
    _ENV_GET_RE = re.compile(
        r'os\.getenv\(["\']([^"\']+)["\']'
    )

    def __init__(
        self,
        root: Path,
        source_map: dict[str, str] | None = None,
    ) -> None:
        self.root = root
        self.source_map = source_map or {}

    def discover(self, changed_files: list[str]) -> list[Discovery]:
        """Scan changed files for environment variable references."""
        discoveries: list[Discovery] = []
        seen: set[str] = set()

        for cf in changed_files:
            source = self.source_map.get(cf)
            if source is None:
                try:
                    fpath = self.root / cf
                    if fpath.exists():
                        source = SafeFileIO.read(fpath)
                except Exception:
                    continue

            if not source:
                continue

            for pattern in (self._ENV_ACCESS_RE, self._ENV_GET_RE):
                for match in pattern.finditer(source):
                    var = match.group(1)
                    if var not in seen:
                        seen.add(var)
                        discoveries.append(Discovery(
                            type=DiscoveryType.ENV_VAR,
                            path=var,
                            relevance=0.7,
                            evidence=f"Referenced via env access in {cf}",
                            content_snippet=f"{var} (from {cf})",
                            source=cf,
                        ))

        return discoveries


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ContextEnricher:
    """Orchestrates all discovery types for a set of changed files.

    Args:
        root: Repository root path.
        store: Optional ``GraphStore`` for persisted data.
        source_map: Optional dict of file path → source text.
    """

    def __init__(
        self,
        root: str = "",
        store: Any = None,
        source_map: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root or ".").resolve()
        self.store = store
        self.source_map = source_map or {}
        self._config_discoverer = ConfigDiscoverer(self.root)
        self._test_discoverer = TestDiscoverer(self.root)
        self._co_change_discoverer = CoChangeDiscoverer(self.root)
        self._env_discoverer = EnvDiscoverer(self.root, self.source_map)

    def enrich(
        self,
        changed_files: list[str],
        find_config: bool = True,
        find_tests: bool = True,
        find_co_changes: bool = True,
        find_env: bool = True,
    ) -> EnrichmentResult:
        """Run all enabled discoverers.

        Args:
            changed_files: The changed file paths.
            find_config: Look for config file references.
            find_tests: Look for related test files.
            find_co_changes: Look for git co-change patterns.
            find_env: Look for environment variable references.

        Returns:
            ``EnrichmentResult`` with all discoveries.
        """
        result = EnrichmentResult(changed_files=changed_files)

        if find_config:
            configs = self._config_discoverer.discover(changed_files)
            result.config_files = configs
            result.discoveries.extend(configs)

        if find_tests:
            tests = self._test_discoverer.discover(changed_files)
            result.test_files = tests
            result.discoveries.extend(tests)

        if find_co_changes:
            cochg = self._co_change_discoverer.discover(changed_files)
            result.co_changed_files = cochg
            result.discoveries.extend(cochg)

        if find_env:
            envs = self._env_discoverer.discover(changed_files)
            result.env_refs = envs
            result.discoveries.extend(envs)

        return result


__all__ = [
    "Discovery",
    "DiscoveryType",
    "EnrichmentResult",
    "ConfigDiscoverer",
    "TestDiscoverer",
    "CoChangeDiscoverer",
    "EnvDiscoverer",
    "ContextEnricher",
]
