"""Schema versioning and migration system for graphsift.

Provides a central :class:`SchemaRegistry` that tracks all schema versions
for a given model family and supports forward/backward data migration
between versions.

Usage::

    from graphsift.migrations import SchemaRegistry
    from graphsift.schemas.graph_schema import GraphNodeV1, GraphNodeV2

    SchemaRegistry.register("GraphNode", [GraphNodeV1, GraphNodeV2])

    # Migrate data from v1 -> v2
    v2_data = SchemaRegistry.migrate("GraphNode", v1_dict, 1, 2)
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pydantic import BaseModel

from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """Central registry for all schema versions with migration paths.

    Each named schema family (e.g. "GraphNode", "ContextConfig") can have
    multiple versioned models registered. Migration functions are auto-detected
    from the schema modules using a naming convention:

    - ``migrate_<family_lower>_v1_to_v2``
    - ``migrate_<family_lower>_v<from>_to_v<to>``

    If no explicit migration function is found, a best-effort merge is
    performed: fields present in the source data are preserved, missing fields
    take their defaults from the target schema.
    """

    _schemas: dict[str, list[type[BaseModel]]] = {}
    _migration_fns: dict[str, dict[tuple[int, int], callable]] = {}

    @classmethod
    def register(
        cls, name: str, versions: list[type[BaseModel]]
    ) -> None:
        """Register schema versions in order. Index = version number (1-based).

        The first element in the list is version 1, the second version 2, etc.

        Args:
            name: Family name (e.g. ``"GraphNode"``, ``"ContextConfig"``).
            versions: Ordered list of Pydantic model classes.

        Raises:
            ValueError: If versions list is empty.
        """
        if not versions:
            raise ValueError(
                f"Cannot register empty versions list for '{name}'"
            )

        cls._schemas[name] = list(versions)
        logger.debug(
            "SchemaRegistry: registered '%s' with %d version(s)",
            name,
            len(versions),
        )

        # Auto-discover migration functions from the module that defines the
        # target version class, or any module in the schemas package.
        cls._discover_migrations(name)

    @classmethod
    def get_schema(
        cls, name: str, version: int
    ) -> type[BaseModel] | None:
        """Get the registered schema class for a given version.

        Args:
            name: Schema family name.
            version: Version number (1-based).

        Returns:
            The Pydantic model class, or None if not found.
        """
        versions = cls._schemas.get(name)
        if not versions or version < 1 or version > len(versions):
            return None
        return versions[version - 1]

    @classmethod
    def current_version(cls, name: str) -> int:
        """Return the latest registered version number for a schema family.

        Args:
            name: Schema family name.

        Returns:
            Latest version number, or 0 if not registered.
        """
        versions = cls._schemas.get(name)
        return len(versions) if versions else 0

    @classmethod
    def migrate(
        cls,
        name: str,
        data: dict,
        from_version: int,
        to_version: int,
    ) -> dict:
        """Migrate data from one schema version to another.

        Supports forward and backward migration. Each version step is applied
        sequentially (e.g. 1->3 = 1->2 then 2->3).

        Args:
            name: Schema family name.
            data: The data dict to migrate.
            from_version: Current version of the data.
            to_version: Target version to migrate to.

        Returns:
            Migrated data dict.

        Raises:
            ValidationError: If the schema family is not registered, versions
                are out of range, or migration fails.
        """
        if name not in cls._schemas:
            raise ValidationError(
                f"Schema family '{name}' is not registered. "
                f"Call SchemaRegistry.register('{name}', versions) first."
            )

        versions = cls._schemas[name]
        max_v = len(versions)

        if from_version < 1 or from_version > max_v:
            raise ValidationError(
                f"from_version={from_version} is out of range "
                f"for '{name}' (1-{max_v})"
            )
        if to_version < 1 or to_version > max_v:
            raise ValidationError(
                f"to_version={to_version} is out of range "
                f"for '{name}' (1-{max_v})"
            )

        if from_version == to_version:
            return dict(data)

        result = dict(data)
        step = 1 if to_version > from_version else -1
        versions_range = range(from_version, to_version, step)

        for v in versions_range:
            next_v = v + step
            migrated = cls._apply_migration(name, result, v, next_v)
            # Ensure schema_version is updated
            migrated["schema_version"] = next_v
            result = migrated

        return result

    @classmethod
    def validate_data(
        cls, name: str, data: dict, version: int | None = None
    ) -> dict:
        """Validate data against a registered schema, with coercion.

        If *version* is None, the latest version is used.
        Unknown extra fields are stripped.

        Args:
            name: Schema family name.
            data: Data dict to validate.
            version: Target schema version (default: latest).

        Returns:
            Validated (and coerced) data dict.

        Raises:
            ValidationError: If data is invalid.
        """
        versions = cls._schemas.get(name)
        if not versions:
            raise ValidationError(f"Schema family '{name}' is not registered.")

        target_v = version if version is not None else len(versions)
        if target_v < 1 or target_v > len(versions):
            raise ValidationError(
                f"Version {target_v} out of range for '{name}' "
                f"(1-{len(versions)})"
            )

        schema_cls = versions[target_v - 1]

        # Check current schema_version in data and auto-migrate if needed
        data_v = data.get("schema_version", target_v)
        if data_v != target_v:
            data = cls.migrate(name, data, data_v, target_v)

        # Validate by constructing the model
        try:
            # Only pass fields known to the target schema
            known_fields = set(schema_cls.model_fields.keys())
            filtered = {k: v for k, v in data.items() if k in known_fields}
            instance = schema_cls(**filtered)
            return instance.model_dump()
        except Exception as exc:
            raise ValidationError(
                f"Data validation failed for '{name}' v{target_v}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _discover_migrations(cls, name: str) -> None:
        """Scan the graph_schema/context_schema/memory_schema modules for
        migration functions matching the naming convention.

        Expected format: ``migrate_<lowername>_v<from>_to_v<to>``
        """
        # Look in the schemas subpackage for migration functions
        try:
            from . import schemas as _schemas_pkg  # noqa: F811
        except ImportError:
            return

        name_lower = name.lower()
        prefix = f"migrate_{name_lower}_v"

        for attr_name in dir(_schemas_pkg):
            if not attr_name.startswith(prefix):
                continue
            fn = getattr(_schemas_pkg, attr_name, None)
            if not callable(fn):
                continue

            # Parse version tuple from suffix: "v1_to_v2" -> (1, 2)
            suffix = attr_name[len(prefix):]
            parts = suffix.split("_to_v")
            if len(parts) != 2:
                continue
            try:
                v_from = int(parts[0])
                v_to = int(parts[1])
            except ValueError:
                continue

            cls._migration_fns.setdefault(name, {})[(v_from, v_to)] = fn

        logger.debug(
            "SchemaRegistry: discovered %d migration(s) for '%s'",
            len(cls._migration_fns.get(name, {})),
            name,
        )

    @classmethod
    def _apply_migration(
        cls,
        name: str,
        data: dict,
        from_version: int,
        to_version: int,
    ) -> dict:
        """Apply a single-step migration (v_from -> v_to).

        If a dedicated migration function is registered, it is used.
        Otherwise a best-effort merge is performed: known fields are
        preserved, missing fields take defaults.
        """
        # Check for explicit migration function
        migration_key = (from_version, to_version)
        fn = cls._migration_fns.get(name, {}).get(migration_key)

        if fn is not None:
            try:
                return fn(data)
            except Exception as exc:
                raise ValidationError(
                    f"Migration '{name}' v{from_version}->v{to_version} "
                    f"failed: {exc}"
                ) from exc

        # Fallback: best-effort merge
        return cls._merge_migration(name, data, to_version)

    @classmethod
    def _merge_migration(
        cls, name: str, data: dict, target_version: int
    ) -> dict:
        """Merge data into target schema, preserving known fields and applying
        defaults for missing fields.

        This is a soft migration that doesn't require explicit migration
        functions — it simply fills in defaults for fields added in later
        versions.
        """
        versions = cls._schemas.get(name)
        if not versions:
            return dict(data)

        target_idx = target_version - 1
        if target_idx < 0 or target_idx >= len(versions):
            return dict(data)

        target_cls = versions[target_idx]
        known_fields = set(target_cls.model_fields.keys())

        result = {}
        for field_name, field_info in target_cls.model_fields.items():
            if field_name in data:
                result[field_name] = data[field_name]
            elif field_info.default is not None and not callable(field_info.default):
                result[field_name] = field_info.default
            elif field_info.default_factory is not None:
                try:
                    result[field_name] = field_info.default_factory()
                except Exception:
                    result[field_name] = None
            else:
                # Could be a required field — try to get a reasonable default
                result[field_name] = None

        # Carry forward any extra fields from source that target doesn't know about
        for k, v in data.items():
            if k not in known_fields:
                result[k] = v

        result["schema_version"] = target_version
        return result


class SchemaEvolution:
    """High-level helper for evolving stored data between schema versions.

    Ties together :class:`SchemaRegistry` with the storage adapter so that
    data loaded from SQLite is automatically migrated to the current schema
    version.

    Usage::

        from graphsift.migrations import SchemaEvolution

        evo = SchemaEvolution()
        evo.register_defaults()
        result = evo.migrate_data("ContextConfig", loaded_dict)
    """

    def __init__(self) -> None:
        self._registered = False

    def register_defaults(self) -> None:
        """Register all known schema families with their versioned models."""
        if self._registered:
            return
        from .schemas.graph_schema import GraphNodeV1, GraphNodeV2, GraphEdgeV1, GraphEdgeV2
        from .schemas.context_schema import ContextConfigV1, ContextConfigV2, ContextResultV1, ContextResultV2
        from .schemas.memory_schema import MemoryFactV1, MemoryFactV2, SessionInfoV1, SessionInfoV2

        SchemaRegistry.register("GraphNode", [GraphNodeV1, GraphNodeV2])
        SchemaRegistry.register("GraphEdge", [GraphEdgeV1, GraphEdgeV2])
        SchemaRegistry.register("ContextConfig", [ContextConfigV1, ContextConfigV2])
        SchemaRegistry.register("ContextResult", [ContextResultV1, ContextResultV2])
        SchemaRegistry.register("MemoryFact", [MemoryFactV1, MemoryFactV2])
        SchemaRegistry.register("SessionInfo", [SessionInfoV1, SessionInfoV2])

        self._registered = True
        logger.info(
            "SchemaEvolution: registered %d schema families",
            len(SchemaRegistry._schemas),
        )

    def migrate_data(
        self,
        name: str,
        data: dict,
        target_version: int | None = None,
    ) -> dict:
        """Migrate a single dict to the target schema version.

        Args:
            name: Schema family name.
            data: Data dict to migrate.
            target_version: Target version (default: latest).

        Returns:
            Migrated data dict.
        """
        self.register_defaults()
        current_v = data.get("schema_version", 1)
        target_v = target_version or SchemaRegistry.current_version(name)

        if current_v == target_v:
            return dict(data)

        return SchemaRegistry.migrate(name, data, current_v, target_v)

    def get_current_version(self, name: str) -> int:
        """Get the latest version for a schema family."""
        self.register_defaults()
        return SchemaRegistry.current_version(name)

    def validate(self, name: str, data: dict) -> dict:
        """Validate and coerce data to the latest schema version."""
        self.register_defaults()
        return SchemaRegistry.validate_data(name, data)
