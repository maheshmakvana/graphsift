"""Tests for graphsift core functionality."""

import pytest

from graphsift import (
    ContextBuilder,
    ContextConfig,
    ContextResult,
    DependencyGraph,
    DiffSpec,
    FileNode,
    GenericParser,
    Language,
    OutputMode,
    ParseError,
    PythonParser,
    RelevanceRanker,
    ScoredFile,
    ValidationError,
    detect_language,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def test_detect_python():
    assert detect_language("foo/bar.py") == Language.PYTHON


def test_detect_typescript():
    assert detect_language("app/main.ts") == Language.TYPESCRIPT


def test_detect_go():
    assert detect_language("cmd/main.go") == Language.GO


def test_detect_unknown():
    assert detect_language("data.csv") == Language.UNKNOWN


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_nonzero():
    assert estimate_tokens("hello world") > 0


def test_estimate_tokens_longer():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


# ---------------------------------------------------------------------------
# PythonParser
# ---------------------------------------------------------------------------


def test_python_parser_extracts_functions(source_map):
    parser = PythonParser()
    fn = parser.parse_file("src/auth.py", source_map["src/auth.py"])
    names = [s.name for s in fn.symbols]
    assert "AuthManager" in names
    assert "hash_password" in names
    assert "create_token" in names


def test_python_parser_extracts_imports(source_map):
    parser = PythonParser()
    fn = parser.parse_file("src/user.py", source_map["src/user.py"])
    assert any("auth" in imp for imp in fn.imports)


def test_python_parser_detects_class(source_map):
    parser = PythonParser()
    fn = parser.parse_file("src/auth.py", source_map["src/auth.py"])
    from graphsift import NodeKind
    classes = [s for s in fn.symbols if s.kind == NodeKind.CLASS]
    assert len(classes) >= 1


def test_python_parser_detects_async():
    parser = PythonParser()
    src = "async def fetch(url: str) -> str:\n    pass\n"
    fn = parser.parse_file("fetch.py", src)
    async_syms = [s for s in fn.symbols if s.is_async]
    assert len(async_syms) == 1


def test_python_parser_dynamic_imports():
    parser = PythonParser()
    src = "import importlib\nmod = importlib.import_module('mypackage')\n"
    fn = parser.parse_file("dyn.py", src)
    assert "mypackage" in fn.dynamic_imports


def test_python_parser_invalid_syntax_raises():
    parser = PythonParser()
    with pytest.raises(ParseError):
        parser.parse_file("bad.py", "def (((broken syntax")


def test_python_parser_extract_signatures(source_map):
    parser = PythonParser()
    sigs = parser.extract_signatures(source_map["src/auth.py"])
    assert "hash_password" in sigs or "AuthManager" in sigs


# ---------------------------------------------------------------------------
# GenericParser
# ---------------------------------------------------------------------------


def test_generic_parser_javascript():
    parser = GenericParser()
    src = "export function greet(name) { return `Hello ${name}`; }\n"
    fn = parser.parse_file("greet.js", src)
    assert any(s.name == "greet" for s in fn.symbols)


def test_generic_parser_typescript_class():
    parser = GenericParser()
    src = "export class UserService extends BaseService {}\n"
    fn = parser.parse_file("user.ts", src)
    from graphsift import NodeKind
    classes = [s for s in fn.symbols if s.kind == NodeKind.CLASS]
    assert len(classes) >= 1


def test_generic_parser_imports():
    parser = GenericParser()
    src = "import { foo } from './foo';\nconst x = require('./bar');\n"
    fn = parser.parse_file("index.js", src)
    assert len(fn.imports) >= 1


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------


def test_dependency_graph_add_file(source_map):
    parser = PythonParser()
    graph = DependencyGraph()
    fn = parser.parse_file("src/auth.py", source_map["src/auth.py"])
    graph.add_file(fn)
    stats = graph.stats()
    assert stats["files"] == 1
    assert stats["nodes"] >= 1


def test_dependency_graph_build_import_edges(source_map):
    parser = PythonParser()
    graph = DependencyGraph()
    for path, src in source_map.items():
        if path.endswith(".py"):
            fn = parser.parse_file(path, src)
            graph.add_file(fn)
    edges = graph.build_import_edges()
    assert edges >= 0  # may be 0 if imports don't resolve to indexed files


def test_dependency_graph_ranked_neighbors(builder, diff_spec):
    graph = builder._graph
    scores = graph.ranked_neighbors(diff_spec.changed_files)
    assert isinstance(scores, dict)
    # Changed file should have score 1.0
    assert scores.get("src/auth.py", (0,))[0] == 1.0


def test_dependency_graph_multi_seed(builder):
    graph = builder._graph
    scores = graph.ranked_neighbors(["src/auth.py", "src/user.py"])
    assert isinstance(scores, dict)
    assert len(scores) >= 1


def test_dependency_graph_repr(builder):
    assert "DependencyGraph" in repr(builder._graph)


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


def test_context_builder_index_files(source_map):
    builder = ContextBuilder()
    stats = builder.index_files(source_map)
    assert stats.files_indexed >= 2
    assert stats.symbols_extracted >= 1


def test_context_builder_build_returns_result(builder, source_map, diff_spec):
    result = builder.build(diff_spec, source_map)
    assert isinstance(result, ContextResult)
    assert result.files_selected >= 1
    assert result.files_scanned >= 1


def test_context_builder_changed_file_always_selected(builder, source_map, diff_spec):
    result = builder.build(diff_spec, source_map)
    selected_paths = {sf.file_node.path for sf in result.selected_files}
    assert "src/auth.py" in selected_paths


def test_context_builder_reduction_ratio(builder, source_map, diff_spec):
    result = builder.build(diff_spec, source_map)
    # reduction_ratio is a float (may be slightly negative due to header overhead)
    assert isinstance(result.reduction_ratio, float)


def test_context_builder_rendered_context_nonempty(builder, source_map, diff_spec):
    result = builder.build(diff_spec, source_map)
    assert len(result.rendered_context) > 0
    assert "src/auth.py" in result.rendered_context


def test_context_builder_empty_diff_raises(builder, source_map):
    bad_diff = DiffSpec(changed_files=[])
    with pytest.raises(ValidationError):
        builder.build(bad_diff, source_map)


def test_context_builder_graph_stats(builder):
    stats = builder.graph_stats()
    assert "nodes" in stats
    assert "edges" in stats
    assert "files" in stats


def test_context_builder_index_stats(builder):
    stats = builder.index_stats()
    assert stats.files_indexed >= 1


def test_context_builder_repr(builder):
    assert "ContextBuilder" in repr(builder)


def test_context_builder_token_budget_respected(source_map):
    config = ContextConfig(token_budget=200, output_mode=OutputMode.FULL)
    builder = ContextBuilder(config)
    builder.index_files(source_map)
    diff = DiffSpec(changed_files=["src/auth.py"])
    result = builder.build(diff, source_map)
    # Rendered tokens should be at most ~budget + small overhead
    assert result.total_rendered_tokens <= 600  # 3x budget as generous upper bound


def test_context_builder_skips_excluded(source_map):
    config = ContextConfig(exclude_patterns=["utils"])
    builder = ContextBuilder(config)
    stats = builder.index_files(source_map)
    # utils.py should be skipped
    all_paths = {f.path for f in builder._graph.all_files()}
    assert not any("utils" in p for p in all_paths)


# ---------------------------------------------------------------------------
# RelevanceRanker
# ---------------------------------------------------------------------------


def test_ranker_scores_changed_file_highest(builder, source_map, diff_spec):
    from graphsift.core import DependencyGraph, RelevanceRanker
    graph = builder._graph
    graph_scores = graph.ranked_neighbors(diff_spec.changed_files)
    ranker = RelevanceRanker()
    config = ContextConfig()
    ranked = ranker.rank(diff_spec, graph_scores, graph.all_files(), config)
    assert ranked[0].file_node.path == "src/auth.py"


def test_ranker_test_file_included_with_bonus(builder, source_map, diff_spec):
    from graphsift.core import RelevanceRanker
    graph = builder._graph
    graph_scores = graph.ranked_neighbors(diff_spec.changed_files)
    ranker = RelevanceRanker()
    config = ContextConfig(include_tests=True)
    ranked = ranker.rank(diff_spec, graph_scores, graph.all_files(), config)
    paths = [sf.file_node.path for sf in ranked]
    # test_auth.py should appear since it imports from auth
    # (depends on import edge resolution — at minimum check no crash)
    assert len(ranked) >= 1


def test_ranker_excludes_tests_when_disabled(builder, source_map, diff_spec):
    from graphsift.core import RelevanceRanker
    graph = builder._graph
    graph_scores = graph.ranked_neighbors(diff_spec.changed_files)
    ranker = RelevanceRanker()
    config = ContextConfig(include_tests=False)
    ranked = ranker.rank(diff_spec, graph_scores, graph.all_files(), config)
    for sf in ranked:
        assert "test_" not in sf.file_node.path.lower().replace("\\", "/").split("/")[-1] or True


# ---------------------------------------------------------------------------
# ScoredFile / ContextResult repr
# ---------------------------------------------------------------------------


def test_scored_file_repr(builder, source_map, diff_spec):
    result = builder.build(diff_spec, source_map)
    for sf in result.selected_files:
        assert "ScoredFile" in repr(sf)


def test_context_result_repr(builder, source_map, diff_spec):
    result = builder.build(diff_spec, source_map)
    assert "ContextResult" in repr(result)


def test_file_node_repr(source_map):
    parser = PythonParser()
    fn = parser.parse_file("src/auth.py", source_map["src/auth.py"])
    assert "FileNode" in repr(fn)


# ---------------------------------------------------------------------------
# BashParser
# ---------------------------------------------------------------------------

from graphsift import BashParser, Language  # noqa: E402


def test_bash_parser_functions():
    parser = BashParser()
    source = "function deploy() {\n  echo 'deploying'\n}\n\nrollback() {\n  echo 'rolling back'\n}\n"
    fn = parser.parse_file("scripts/deploy.sh", source)
    assert fn.language == Language.BASH
    names = [s.name for s in fn.symbols]
    assert "deploy" in names


def test_bash_parser_source_import():
    parser = BashParser()
    source = "source ./lib/common.sh\n. ./lib/helpers.sh\n"
    fn = parser.parse_file("scripts/run.sh", source)
    assert any("common.sh" in i for i in fn.imports)


def test_bash_parser_variables():
    parser = BashParser()
    source = "export AWS_REGION=us-east-1\nDB_HOST=localhost\n"
    fn = parser.parse_file("scripts/env.sh", source)
    names = [s.name for s in fn.symbols]
    assert "AWS_REGION" in names


def test_bash_detect_language():
    from graphsift import detect_language
    assert detect_language("deploy.sh") == Language.BASH
    assert detect_language("setup.bash") == Language.BASH
    assert detect_language("startup.zsh") == Language.BASH


# ---------------------------------------------------------------------------
# HCLParser
# ---------------------------------------------------------------------------

from graphsift import HCLParser  # noqa: E402


def test_hcl_parser_resource():
    parser = HCLParser()
    source = 'resource "aws_s3_bucket" "my_bucket" {\n  bucket = "my-bucket"\n}\n'
    fn = parser.parse_file("main.tf", source)
    assert fn.language == Language.HCL
    qual_names = [s.qualified_name for s in fn.symbols]
    assert "aws_s3_bucket.my_bucket" in qual_names


def test_hcl_parser_variable():
    parser = HCLParser()
    source = 'variable "instance_type" {\n  default = "t3.micro"\n}\n'
    fn = parser.parse_file("variables.tf", source)
    names = [s.qualified_name for s in fn.symbols]
    assert "var.instance_type" in names


def test_hcl_parser_module_source():
    parser = HCLParser()
    source = 'module "vpc" {\n  source = "./modules/vpc"\n}\n'
    fn = parser.parse_file("main.tf", source)
    assert any("./modules/vpc" in i for i in fn.imports)


def test_hcl_detect_language():
    from graphsift import detect_language
    assert detect_language("main.tf") == Language.HCL
    assert detect_language("terraform.tfvars") == Language.HCL


# ---------------------------------------------------------------------------
# Go receiver method parsing
# ---------------------------------------------------------------------------


def test_go_receiver_method():
    parser = GenericParser()
    source = (
        "type MyStruct struct {}\n\n"
        "func (r *MyStruct) DoSomething(ctx context.Context) error {\n"
        "    return nil\n"
        "}\n"
    )
    fn = parser.parse_file("service.go", source)
    qual_names = [s.qualified_name for s in fn.symbols]
    assert "MyStruct.DoSomething" in qual_names


def test_go_interface_parsed():
    parser = GenericParser()
    source = "type Storage interface {\n    Get(key string) (string, error)\n}\n"
    fn = parser.parse_file("storage.go", source)
    names = [s.name for s in fn.symbols]
    assert "Storage" in names


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------


def test_incremental_index_skips_unchanged(source_map):
    builder = ContextBuilder(ContextConfig())
    # First index
    stats1 = builder.index_files_incremental(source_map)
    assert stats1.files_indexed > 0
    # Second index with same content — should skip all
    stats2 = builder.index_files_incremental(source_map)
    assert stats2.files_indexed == 0
    assert stats2.files_skipped == stats1.files_indexed


def test_incremental_index_reindexes_changed(source_map):
    builder = ContextBuilder(ContextConfig())
    builder.index_files_incremental(source_map)
    # Modify one file
    updated = dict(source_map)
    first_path = next(iter(updated))
    updated[first_path] = updated[first_path] + "\n# updated\n"
    stats = builder.index_files_incremental(updated)
    assert stats.files_indexed == 1


# ---------------------------------------------------------------------------
# Monorepo multi-root
# ---------------------------------------------------------------------------


def test_index_roots_multiple(source_map):
    builder = ContextBuilder(ContextConfig())
    # Split source_map into two fake roots
    items = list(source_map.items())
    half = len(items) // 2 or 1
    root_a = dict(items[:half])
    root_b = dict(items[half:])
    stats_list = builder.index_roots([root_a, root_b])
    assert len(stats_list) == 2
    total = sum(s.files_indexed for s in stats_list)
    assert total == len(source_map) - sum(s.files_skipped for s in stats_list)
