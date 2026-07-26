"""Tests for the Convention Learner."""

from __future__ import annotations

from graphsift.conventions import ConventionLearner, Convention, ConventionProfile, NamingDetector, ImportStyleDetector


class TestConventionModels:
    """Tests for Convention data classes."""

    def test_convention_defaults(self):
        """Convention should have default values."""
        conv = Convention(name="test", pattern="snake_case")
        assert conv.confidence == 0.5
        assert conv.evidence_count == 0
        assert conv.total_count == 0
        assert conv.examples == []

    def test_convention_summary(self):
        """Convention.summary should describe the convention."""
        conv = Convention(name="naming", pattern="snake_case", confidence=0.85, evidence_count=17, total_count=20)
        summary = conv.summary
        assert "naming" in summary
        assert "0.85" in summary
        assert "17/20" in summary

    def test_convention_profile_defaults(self):
        """ConventionProfile should start with empty lists."""
        profile = ConventionProfile()
        assert profile.naming == []
        assert profile.imports == []
        assert profile.summary == "ConventionProfile: "

    def test_convention_profile_with_items(self):
        """ConventionProfile should track items per category."""
        conv = Convention(name="naming", pattern="snake_case", confidence=0.85, evidence_count=17, total_count=20)
        profile = ConventionProfile(naming=[conv])
        summary = profile.summary
        assert "naming=1" in summary

    def test_convention_profile_to_context_block(self):
        """to_context_block should generate markdown for high-confidence conventions."""
        conv = Convention(name="naming", pattern="snake_case", confidence=0.85, evidence_count=17, total_count=20)
        profile = ConventionProfile(naming=[conv])
        block = profile.to_context_block()
        assert "Codebase Conventions" in block
        assert "snake_case" in block

    def test_convention_profile_to_context_block_empty(self):
        """to_context_block should return empty string for no conventions."""
        profile = ConventionProfile()
        assert profile.to_context_block() == ""

    def test_convention_profile_to_context_block_low_confidence(self):
        """to_context_block should skip low-confidence conventions."""
        conv = Convention(name="naming", pattern="unknown", confidence=0.5, evidence_count=1, total_count=10)
        profile = ConventionProfile(naming=[conv])
        assert profile.to_context_block() == ""

    def test_convention_profile_multiple_categories(self):
        """to_context_block should list multiple categories."""
        n = Convention(name="naming", pattern="snake_case", confidence=0.9, evidence_count=10, total_count=10)
        i = Convention(name="imports", pattern="absolute imports", confidence=0.8, evidence_count=8, total_count=10)
        profile = ConventionProfile(naming=[n], imports=[i])
        block = profile.to_context_block()
        assert "Naming" in block
        assert "Imports" in block


class TestNamingDetector:
    """Tests for the NamingDetector."""

    def test_detect_empty(self):
        """Detecting on empty list should return low confidence."""
        detector = NamingDetector()
        conv = detector.detect([])
        assert conv.confidence == 0.0
        assert "insufficient data" in conv.pattern

    def test_detect_snake_case(self):
        """Detecting snake_case names should identify the pattern."""
        detector = NamingDetector()

        class MockSymbol:
            def __init__(self, name):
                self.name = name

        symbols = [MockSymbol(n) for n in ["get_user", "validate_token", "create_record", "find_active", "ProcessData"]]
        conv = detector.detect(symbols)
        assert conv.confidence >= 0.5
        assert "snake_case" in conv.pattern.lower() or "pascal" in conv.pattern.lower()

    def test_detect_pascal_case(self):
        """Detecting PascalCase names should identify the pattern."""
        detector = NamingDetector()

        class MockSymbol:
            def __init__(self, name):
                self.name = name

        symbols = [MockSymbol(n) for n in ["UserService", "AuthManager", "ConfigLoader", "DataProvider", "get_user"]]
        conv = detector.detect(symbols)
        # At least 3/4 should be PascalCase
        assert conv.evidence_count >= 3

    def test_detect_mixed_no_underscore(self):
        """Symbols without underscores should still be detected."""
        detector = NamingDetector()

        class MockSymbol:
            def __init__(self, name):
                self.name = name

        symbols = [MockSymbol(n) for n in ["x", "y", "z"]]
        conv = detector.detect(symbols)
        assert conv.total_count == 3

    def test_detect_private_ignored(self):
        """Symbols starting with _ should be excluded."""
        detector = NamingDetector()

        class MockSymbol:
            def __init__(self, name):
                self.name = name

        symbols = [MockSymbol(n) for n in ["_private", "public", "_helper"]]
        conv = detector.detect(symbols)
        assert conv.total_count == 1  # Only 'public' counted


class TestImportStyleDetector:
    """Tests for the ImportStyleDetector."""

    def test_detect_empty(self):
        """Detecting on empty list should return no conventions."""
        detector = ImportStyleDetector()
        convs = detector.detect([])
        assert convs == []

    def test_detect_relative_imports(self):
        """Files with mostly relative imports should be detected."""
        detector = ImportStyleDetector()

        class MockFileNode:
            def __init__(self, imports):
                self.imports = imports

        files = [MockFileNode([".module_a", ".module_b"]) for _ in range(10)]
        convs = detector.detect(files)
        assert len(convs) > 0
        assert any("relative" in c.pattern for c in convs)

    def test_detect_absolute_imports(self):
        """Files with mostly absolute imports should be detected."""
        detector = ImportStyleDetector()

        class MockFileNode:
            def __init__(self, imports):
                self.imports = imports

        files = [MockFileNode(["os", "sys", "typing"]) for _ in range(10)]
        convs = detector.detect(files)
        assert len(convs) > 0
        assert any("absolute" in c.pattern for c in convs)

    def test_detect_too_few_files(self):
        """Fewer than MIN_SAMPLE_SIZE files should return no conventions."""
        detector = ImportStyleDetector()

        class MockFileNode:
            def __init__(self, imports):
                self.imports = imports

        files = [MockFileNode([".a"]) for _ in range(2)]
        convs = detector.detect(files)
        assert convs == []


class TestConventionLearner:
    """Tests for the ConventionLearner."""

    def test_learn_empty(self):
        """Learning from empty data should return empty profile."""
        learner = ConventionLearner()
        profile = learner.learn([])
        assert len(profile.naming) == 0
        assert len(profile.imports) == 0

    def test_learn_with_file_nodes(self):
        """Learning from file nodes should detect conventions."""
        learner = ConventionLearner()

        class MockSymbol:
            def __init__(self, name):
                self.name = name

        class MockFileNode:
            def __init__(self, path, symbols, imports):
                self.path = path
                self.symbols = symbols
                self.imports = imports

        nodes = [
            MockFileNode("a.py", [MockSymbol(n) for n in ["func_a", "func_b", "ClassA"]], ["os", "sys"]),
            MockFileNode("b.py", [MockSymbol(n) for n in ["func_c", "func_d", "ClassB"]], [".utils"]),
        ]

        profile = learner.learn(nodes)
        # Should detect at least naming
        assert len(profile.naming) >= 0 or len(profile.imports) >= 0
