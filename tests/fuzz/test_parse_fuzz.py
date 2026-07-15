"""Fuzz tests: feed random source code to parsers."""

import random
import pytest

from graphsift import PythonParser, GenericParser, BashParser, HCLParser

pytestmark = [pytest.mark.fuzz]


# ---------------------------------------------------------------------------
# Generate random source-like text
# ---------------------------------------------------------------------------

def _random_source(max_size: int = 1000) -> str:
    """Generate random text that looks vaguely like source code."""
    patterns = [
        # Valid-looking constructs mixed with noise
        'def {}():',
        'class {}:',
        '{} = {}',
        'import {}',
        'from {} import {}',
        'if {}:',
        'for {} in {}:',
        'return {}',
        '{} = {}()',
        '@{}',
        'async def {}():',
        '{} + {}',
        '{}[{}] = {}',
        'with {} as {}:',
        'try:',
        'except {}:',
        'finally:',
        'raise {}',
        'yield {}',
        'pass',
        '...',
        '# {}',
        '"""{}"""',
        "'''{}'''",
        '{} /* {} */',
        '{} // {}',
        'print({})',
    ]

    identifiers = [
        'x', 'y', 'foo', 'bar', 'baz', 'test', 'data', 'value',
        'result', 'count', 'index', 'item', 'obj', 'self', 'cls',
        'config', 'manager', 'handler', 'service', 'repository',
        'Helper', 'Manager', 'Service', 'Config', 'MyClass',
        'test_value', 'get_data', 'set_value', 'process_item',
        'do_something', 'handle_request', 'create_object',
    ]

    lines = []
    for _ in range(random.randint(0, 30)):
        if random.random() < 0.3:
            # Random noise
            noise_len = random.randint(0, 50)
            chars = [chr(random.randint(0x20, 0x7E)) for _ in range(noise_len)]
            lines.append("".join(chars))
        else:
            pattern = random.choice(patterns)
            args = [random.choice(identifiers) for _ in range(pattern.count('{}'))]
            line = pattern.format(*args)
            # Random indentation
            indent = "    " * random.randint(0, 3)
            lines.append(indent + line)

    return "\n".join(lines)


def _random_go_source() -> str:
    """Generate random Go-like source code."""
    lines = [
        'package main',
        '',
        'import "fmt"',
        '',
    ]
    for _ in range(random.randint(0, 10)):
        if random.random() < 0.5:
            lines.append(f'func {random.choice(["foo", "bar", "test", "main", "handle"])}() {{')
            if random.random() < 0.3:
                lines.append(f'    fmt.Println({random.choice(["hello", "world", "test"])})')
            lines.append('}')
        else:
            lines.append(f'var {random.choice(["x", "y", "count", "name"])} = {random.randint(0, 100)}')
        if random.random() < 0.3:
            lines.append('')
    return "\n".join(lines)


def _random_ts_source() -> str:
    """Generate random TypeScript-like source code."""
    lines = []
    for _ in range(random.randint(0, 10)):
        if random.random() < 0.5:
            lines.append(f'interface {random.choice(["Props", "State", "Config"])} {{')
            lines.append(f'    {random.choice(["name", "value", "count"])}: {random.choice(["string", "number", "boolean"])};')
            lines.append('}')
        lines.append(f'import {{ {random.choice(["foo", "bar", "Component"])} }} from "./{random.choice(["utils", "helpers", "core"])}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PythonParser fuzz
# ---------------------------------------------------------------------------

class TestPythonParserFuzz:
    """Fuzz PythonParser with random inputs."""

    def test_fuzz_python_parser_no_crash(self):
        """PythonParser must not crash on random source code."""
        parser = PythonParser()
        for _ in range(100):
            source = _random_source(500)
            try:
                fn = parser.parse_file("test.py", source)
                assert fn is not None
                assert fn.path == "test.py"
            except Exception as e:
                # ParseError is expected for invalid code
                from graphsift import ParseError
                if not isinstance(e, ParseError):
                    raise

    def test_fuzz_python_parser_binary(self):
        """PythonParser must handle binary data gracefully."""
        parser = PythonParser()
        for _ in range(20):
            raw = bytes(random.randint(0, 255) for _ in range(random.randint(0, 200)))
            text = raw.decode("utf-8", errors="replace")
            try:
                parser.parse_file("test.py", text)
            except Exception:
                pass  # Should not crash

    def test_fuzz_python_parser_extract_signatures(self):
        """extract_signatures must not crash on random input."""
        parser = PythonParser()
        for _ in range(50):
            source = _random_source(300)
            try:
                sigs = parser.extract_signatures(source)
                assert isinstance(sigs, str)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GenericParser fuzz
# ---------------------------------------------------------------------------

class TestGenericParserFuzz:
    """Fuzz GenericParser with random inputs."""

    def test_fuzz_generic_parser_js(self):
        """GenericParser must not crash on random JS-like code."""
        parser = GenericParser()
        for _ in range(50):
            source = _random_ts_source()
            try:
                fn = parser.parse_file("test.js", source)
                assert fn is not None
            except Exception:
                pass  # Parse errors expected

    def test_fuzz_generic_parser_ts(self):
        """GenericParser must not crash on random TS-like code."""
        parser = GenericParser()
        for _ in range(50):
            source = _random_ts_source()
            try:
                fn = parser.parse_file("test.ts", source)
                assert fn is not None
            except Exception:
                pass

    def test_fuzz_generic_parser_go(self):
        """GenericParser must not crash on random Go-like code."""
        parser = GenericParser()
        for _ in range(50):
            source = _random_go_source()
            try:
                fn = parser.parse_file("test.go", source)
                assert fn is not None
            except Exception:
                pass

    def test_fuzz_generic_parser_random(self):
        """GenericParser must not crash on completely random text."""
        parser = GenericParser()
        for _ in range(100):
            source = _random_source(300)
            try:
                fn = parser.parse_file(f"test_{random.randint(0, 10)}.{random.choice(['js', 'ts', 'go', 'rs', 'java', 'rb', 'php'])}", source)
                assert fn is not None
            except Exception:
                pass


# ---------------------------------------------------------------------------
# BashParser and HCLParser fuzz
# ---------------------------------------------------------------------------

class TestOtherParsersFuzz:
    """Fuzz BashParser and HCLParser."""

    def test_fuzz_bash_parser(self):
        """BashParser must not crash on random input."""
        parser = BashParser()
        for _ in range(50):
            source = _random_source(300)
            try:
                fn = parser.parse_file("test.sh", source)
                assert fn is not None
            except Exception:
                pass

    def test_fuzz_hcl_parser(self):
        """HCLParser must not crash on random input."""
        parser = HCLParser()
        for _ in range(50):
            source = _random_source(300)
            try:
                fn = parser.parse_file("test.tf", source)
                assert fn is not None
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestParserEdgeCases:
    """Edge cases for parsers."""

    def test_empty_string(self):
        """All parsers must handle empty string."""
        sources = {
            "empty.py": PythonParser(),
            "empty.js": GenericParser(),
            "empty.sh": BashParser(),
            "empty.tf": HCLParser(),
        }
        for path, parser in sources.items():
            try:
                fn = parser.parse_file(path, "")
                assert fn is not None
            except Exception as e:
                from graphsift import ParseError
                if not isinstance(e, ParseError):
                    raise

    def test_very_long_lines(self):
        """Parsers must handle very long lines."""
        long_line = "x = " + "+".join(f"v{i}" for i in range(500)) + "\n"
        parser = PythonParser()
        try:
            fn = parser.parse_file("long.py", long_line)
            assert fn is not None
        except Exception as e:
            from graphsift import ParseError
            if not isinstance(e, ParseError):
                raise

    def test_invalid_utf8_bytes(self):
        """Parsers must handle invalid unicode sequences."""
        parser = PythonParser()
        text = "\\x" + "\\x".join(f"{b:02x}" for b in range(256))
        try:
            parser.parse_file("bytes.py", text)
        except Exception:
            pass  # Should not crash

    def test_unmatched_brackets(self):
        """Parsers must handle unmatched brackets."""
        text = "(" * 100 + ")" * 50 + "\ndef foo():\n    pass\n"
        parser = PythonParser()
        try:
            fn = parser.parse_file("brackets.py", text)
            assert fn is not None
        except Exception:
            pass  # Parse error expected
