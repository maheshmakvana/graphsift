"""graphsift v1 API compatibility shim.

All exports from the old top-level ``graphsift`` package are re-exported here
with deprecation warnings. New code should use ``graphsift.api.v2`` instead.
"""

from __future__ import annotations

import warnings as _warnings

from graphsift._version import __version__ as _version

# ---------------------------------------------------------------------------
# Helper: deprecate and re-export
# ---------------------------------------------------------------------------

_DEPRECATION_MSG = (
    "Import from ``graphsift.api.v1`` is deprecated. "
    "Please use ``graphsift.api.v2`` instead."
)


def __getattr__(name: str):
    """Lazily import from the original location with a deprecation warning."""
    _warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

    # Try importing from the graphsift top-level
    import graphsift as _mod

    if hasattr(_mod, name):
        return getattr(_mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Explicit re-exports for commonly used symbols (also trigger warnings)
# ---------------------------------------------------------------------------

# Core
ContextBuilder: "graphsift.core.ContextBuilder" = __getattr__("ContextBuilder")
ContextConfig: "graphsift.models.ContextConfig" = __getattr__("ContextConfig")
ContextSelector: "graphsift.core.ContextSelector" = __getattr__("ContextSelector")
DependencyGraph: "graphsift.core.DependencyGraph" = __getattr__("DependencyGraph")
RelevanceRanker: "graphsift.core.RelevanceRanker" = __getattr__("RelevanceRanker")
detect_language = __getattr__("detect_language")
estimate_tokens = __getattr__("estimate_tokens")

# Models
DiffSpec = __getattr__("DiffSpec")
FileNode = __getattr__("FileNode")
GraphNode = __getattr__("GraphNode")
GraphEdge = __getattr__("GraphEdge")
ScoredFile = __getattr__("ScoredFile")
IndexStats = __getattr__("IndexStats")
ContextResult = __getattr__("ContextResult")
Language = __getattr__("Language")
OutputMode = __getattr__("OutputMode")

# Advanced
AnalysisPipeline = __getattr__("AnalysisPipeline")
DiffValidator = __getattr__("DiffValidator")
GraphCache = __getattr__("GraphCache")
SchemaEvolution = __getattr__("SchemaEvolution")
batch_index = __getattr__("batch_index")
batch_build = __getattr__("batch_build")
stream_context = __getattr__("stream_context")

# Exceptions
graphsiftError = __getattr__("graphsiftError")
ValidationError = __getattr__("ValidationError")
ConfigurationError = __getattr__("ConfigurationError")
ParseError = __getattr__("ParseError")
IndexError = __getattr__("IndexError")
GraphError = __getattr__("GraphError")
AdapterError = __getattr__("AdapterError")
BudgetExceededError = __getattr__("BudgetExceededError")

# Utilities
compress = __getattr__("compress")
HybridSearcher = __getattr__("HybridSearcher")
FixSuggester = __getattr__("FixSuggester")

# MCP
run_server = __getattr__("run_server")

__all__ = [
    "ContextBuilder",
    "ContextConfig",
    "ContextSelector",
    "DependencyGraph",
    "RelevanceRanker",
    "detect_language",
    "estimate_tokens",
    "DiffSpec",
    "FileNode",
    "GraphNode",
    "GraphEdge",
    "ScoredFile",
    "IndexStats",
    "ContextResult",
    "Language",
    "OutputMode",
    "AnalysisPipeline",
    "DiffValidator",
    "GraphCache",
    "SchemaEvolution",
    "batch_index",
    "batch_build",
    "stream_context",
    "graphsiftError",
    "ValidationError",
    "ConfigurationError",
    "ParseError",
    "IndexError",
    "GraphError",
    "AdapterError",
    "BudgetExceededError",
    "compress",
    "HybridSearcher",
    "FixSuggester",
    "run_server",
]
