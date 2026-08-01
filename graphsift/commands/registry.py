"""Plugin registry — discovers, validates, and executes external command plugins."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PluginManifest:
    """Manifest for an external command plugin."""

    id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    binary: str = ""
    default_timeout_ms: int = 30000
    mutating: bool = False
    network: bool = False
    schema: dict = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: Path) -> PluginManifest:
        """Parse a command.toml manifest file."""
        text = path.read_text(encoding="utf-8")

        # Minimal TOML parser — handles the subset we need
        def _get(key: str, default: Any = "") -> Any:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{key} ="):
                    val = stripped.split("=", 1)[1].strip()
                    if val.startswith('"') and val.endswith('"'):
                        return val[1:-1]
                    if val.startswith("true") or val.startswith("false"):
                        return val.lower().startswith("true")
                    try:
                        return int(val)
                    except ValueError:
                        return val
            return default

        def _get_dotted(dotted: str, default: Any = "") -> Any:
            """Get a value with dotted key like execution.binary."""
            for line in text.splitlines():
                stripped = line.strip()
                # Match dotted prefix
                if stripped.startswith(f"{dotted} ="):
                    val = stripped.split("=", 1)[1].strip()
                    if val.startswith('"') and val.endswith('"'):
                        return val[1:-1]
                    if val.startswith("true") or val.startswith("false"):
                        return val.lower().startswith("true")
                    try:
                        return int(val)
                    except ValueError:
                        return val
            return default

        return cls(
            id=_get("id", ""),
            name=_get("name", ""),
            version=_get("version", "1.0.0"),
            description=_get("description", ""),
            binary=_get_dotted("execution.binary", ""),
            default_timeout_ms=int(_get_dotted("execution.default_timeout_ms", 30000)),
            mutating=bool(_get_dotted("mutating", False)),
            network=bool(_get_dotted("network", False)),
        )

    def to_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "binary": self.binary,
            "default_timeout_ms": self.default_timeout_ms,
            "mutating": self.mutating,
            "network": self.network,
            "schema": self.schema,
        }


class PluginRegistry:
    """Registry of external command plugins.

    Discovers plugins from the commands directory, validates their manifests,
    and provides execution via JSON stdin/stdout subprocess protocol.
    """

    def __init__(self, commands_dir: str | None = None) -> None:
        self._plugins: dict[str, PluginManifest] = {}
        self._prompts: dict[str, str] = {}
        self._commands_dir = Path(commands_dir) if commands_dir else Path(__file__).parent
        self._load_plugins()

    def _load_plugins(self) -> None:
        """Discover plugins from the commands directory."""
        if not self._commands_dir.is_dir():
            logger.debug("Commands directory not found: %s", self._commands_dir)
            return

        for entry in sorted(self._commands_dir.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / "command.toml"
            if not manifest_path.exists():
                continue

            try:
                manifest = PluginManifest.from_toml(manifest_path)
                self._plugins[manifest.id] = manifest

                # Load prompt.md if it exists
                prompt_path = entry / "prompt.md"
                if prompt_path.exists():
                    self._prompts[manifest.id] = prompt_path.read_text(encoding="utf-8")

                logger.debug("Loaded plugin: %s (%s)", manifest.id, manifest.binary)
            except Exception as exc:
                logger.warning("Failed to load plugin from %s: %s", entry, exc)

    def list_plugins(self) -> list[dict]:
        """List all registered plugins."""
        return [p.to_dict() for p in self._plugins.values()]

    def get_plugin(self, plugin_id: str) -> PluginManifest | None:
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)

    def get_plugin_prompt(self, plugin_id: str) -> str:
        """Get the prompt.md content for a plugin."""
        return self._prompts.get(plugin_id, "")

    def _binary_available(self, binary: str) -> bool:
        """Check if a binary is available on the PATH."""
        return shutil.which(binary) is not None

    def execute(
        self,
        plugin_id: str,
        arguments: dict,
        session_dir: str = "",
        timeout_ms: int | None = None,
    ) -> dict:
        """Execute a plugin via subprocess with JSON stdin/stdout protocol.

        Args:
            plugin_id: ID of the plugin to execute.
            arguments: Arguments to pass to the plugin.
            session_dir: Working directory for the plugin.
            timeout_ms: Timeout in milliseconds (overrides manifest default).

        Returns:
            Dict with keys: success, output, error, duration_ms
        """
        manifest = self._plugins.get(plugin_id)
        if manifest is None:
            return {
                "success": False,
                "output": "",
                "error": f"Unknown plugin: {plugin_id}",
                "duration_ms": 0,
            }

        # Check binary availability
        binary_path = shutil.which(manifest.binary)
        if not binary_path:
            return {
                "success": False,
                "output": "",
                "error": (
                    f"Binary '{manifest.binary}' not found for plugin '{plugin_id}'. "
                    f"Install it to use this plugin."
                ),
                "duration_ms": 0,
            }

        # Build the request payload
        call_id = str(uuid.uuid4())[:12]
        request = {
            "kind": "execute",
            "payload": {
                "arguments": arguments,
                "session_dir": session_dir,
                "call_id": call_id,
            },
        }

        # Determine timeout
        effective_timeout = (
            timeout_ms if timeout_ms is not None else manifest.default_timeout_ms
        )

        # Execute
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [binary_path],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                encoding="utf-8", errors="replace",
                timeout=effective_timeout / 1000,
            )

            duration_ms = int((time.monotonic() - start) * 1000)

            # Parse response
            if proc.returncode != 0:
                return {
                    "success": False,
                    "output": proc.stderr,
                    "error": f"Plugin exited with code {proc.returncode}",
                    "duration_ms": duration_ms,
                }

            try:
                response = json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                return {
                    "success": False,
                    "output": proc.stdout,
                    "error": f"Invalid JSON response: {exc}",
                    "duration_ms": duration_ms,
                }

            payload = response.get("payload", {})
            return {
                "success": payload.get("success", False),
                "output": payload.get("output", ""),
                "error": payload.get("error"),
                "call_id": payload.get("call_id", call_id),
                "duration_ms": duration_ms,
            }

        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            return {
                "success": False,
                "output": "",
                "error": f"Plugin timed out after {effective_timeout}ms",
                "duration_ms": duration_ms,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": f"Binary '{manifest.binary}' not found at path",
                "duration_ms": 0,
            }
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.exception("Plugin execution failed: %s", plugin_id)
            return {
                "success": False,
                "output": "",
                "error": str(exc),
                "duration_ms": duration_ms,
            }
