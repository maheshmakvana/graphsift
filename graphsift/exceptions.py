"""Typed exception hierarchy for graphsift."""


class graphsiftError(Exception):
    """Base exception — never raise directly."""


class ValidationError(graphsiftError):
    """Input or configuration validation failure."""


class ConfigurationError(graphsiftError):
    """Invalid configuration."""


class ParseError(graphsiftError):
    """Source file could not be parsed."""


class IndexError(graphsiftError):  # noqa: A001
    """Repository indexing failure."""


class GraphError(graphsiftError):
    """Dependency graph construction or traversal failure."""


class AdapterError(graphsiftError):
    """External system boundary failure."""

    class TimeoutError(graphsiftError):  # noqa: A001,N818
        """Operation timed out."""

    class RateLimitError(graphsiftError):  # noqa: N818
        """Upstream rate limit hit."""


class BudgetExceededError(graphsiftError):
    """Selected context exceeds token budget."""


class LanguageNotSupportedError(graphsiftError):
    """Language has no registered adapter."""
