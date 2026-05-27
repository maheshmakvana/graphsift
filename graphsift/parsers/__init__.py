"""Tree-sitter parser package for graphsift.

Provides ``TreeSitterParser`` — an AST-accurate parser for 11 languages
with graceful fallback to the regex-based ``GenericParser`` when tree-sitter
(or the specific language grammar) is not installed.

Usage::

    from graphsift.parsers import TreeSitterParser

    parser = TreeSitterParser()
    fn = parser.parse_file("src/auth.py", source_text)

    # Or register globally:
    from graphsift.parsers import register_tree_sitter_parsers
    register_tree_sitter_parsers()
"""

from .tree_sitter_parser import TreeSitterParser, register_tree_sitter_parsers

__all__ = [
    "TreeSitterParser",
    "register_tree_sitter_parsers",
]
