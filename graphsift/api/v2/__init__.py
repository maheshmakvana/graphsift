"""graphsift v2 API — modern unified interface.

This package provides a clean, versioned API for graphsift with:

- **Sift** — unified class combining indexing, search, context, compression, analysis
- **SiftConfig** — clean configuration model
- **IndexResult**, **ContextResult**, **AnalysisResult** — typed Pydantic results
- **ScoredFile** — simplified file scoring model
- **SiftError**, **IndexError**, **SearchError**, etc. — clean exception hierarchy

Quick start::

    from graphsift.api.v2 import Sift, SiftConfig

    sift = Sift(SiftConfig(token_budget=50_000))
    result = sift.index({"src/main.py": "def hello(): ..."})
    analysis = sift.analyze("src/")
    ctx = sift.build_context(["src/main.py"], query="explain this")
    compressed = sift.compress(some_output, "pytest")
"""

from graphsift.api.v2.exceptions import (
    AnalyzeError,
    BuildError,
    CompressError,
    ConfigError,
    IndexError,
    SearchError,
    SiftError,
)
from graphsift.api.v2.models import (
    AnalysisResult,
    CompressResult,
    ContextResult,
    IndexResult,
    ScoredFile,
    SiftConfig,
)
from graphsift.api.v2.sift import Sift

__all__ = [
    # Main class
    "Sift",
    # Config
    "SiftConfig",
    # Result models
    "IndexResult",
    "ScoredFile",
    "ContextResult",
    "AnalysisResult",
    "CompressResult",
    # Exceptions
    "SiftError",
    "IndexError",
    "SearchError",
    "BuildError",
    "ConfigError",
    "CompressError",
    "AnalyzeError",
]
