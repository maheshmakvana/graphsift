"""Clean exception hierarchy for the v2 graphsift API."""

from __future__ import annotations


class SiftError(Exception):
    """Base exception for all graphsift v2 API errors. Never raise directly."""


class IndexError(SiftError):
    """Indexing operation failed."""


class SearchError(SiftError):
    """Search operation failed."""


class BuildError(SiftError):
    """Context building operation failed."""


class ConfigError(SiftError):
    """Invalid or missing configuration."""


class CompressError(SiftError):
    """Compression operation failed."""


class AnalyzeError(SiftError):
    """Analysis operation failed."""


__all__ = [
    "SiftError",
    "IndexError",
    "SearchError",
    "BuildError",
    "ConfigError",
    "CompressError",
    "AnalyzeError",
]
