"""graphsift API — versioned public interface.

This package provides versioned API layers for graphsift.
- v1: Legacy API with deprecation warnings (re-exports from old locations)
- v2: Modern unified API (Sift class, clean models, consistent naming)
"""

from graphsift._version import __version__

API_VERSION = "2.0.0"
"""Current API version string."""

__all__ = [
    "API_VERSION",
]
