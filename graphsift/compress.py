"""Output compression — rtk-style 60-90% token reduction for CLI output.

Strategies: smart filtering, grouping, truncation, deduplication.
Pure Python, zero external dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

_NOISE_RE = re.compile(
    r"^\s*(?:"
    r"[\d:]+\s*[-/]\s*[\d:]+"               # timestamps
    r"|\[[=#\-]*(?:\s*\d+%?\s*)?[=#\-]*\]"  # progress bars [====] [== 45% ==]
    r"|[\d.]+\s*%"                           # percentage progress
    r"|[=\-]+\s*\d+%\s*[=\-]+"              # ==== 45% ==== style
    r")\s*$"
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub("", text)


def _is_noise(line: str) -> bool:
    """Return True if line is a progress bar, spinner, timestamp, or blank."""
    return bool(_NOISE_RE.match(line))


# ---------------------------------------------------------------------------
# Core compression primitives
# ---------------------------------------------------------------------------


def deduplicate(text: str, threshold: int = 1) -> str:
    """Collapse consecutive identical lines, appending (xN) suffix.

    Only lines appearing more than *threshold* consecutive times are collapsed.
    """
    lines = text.split("\n")
    if not lines:
        return text
    out: list[str] = []
    prev = lines[0]
    count = 1
    for line in lines[1:]:
        if line == prev:
            count += 1
        else:
            if count > threshold:
                out.append(f"{prev}  (x{count})")
            else:
                out.extend([prev] * count)
            prev = line
            count = 1
    if count > threshold:
        out.append(f"{prev}  (x{count})")
    else:
        out.extend([prev] * count)
    return "\n".join(out)


def truncate_middle(text: str, head: int = 20, tail: int = 10) -> str:
    """Keep first *head* lines and last *tail* lines of text."""
    lines = text.split("\n")
    if len(lines) <= head + tail:
        return text
    omitted = len(lines) - head - tail
    return "\n".join(lines[:head] + [f"... ({omitted} lines omitted)"] + (lines[-tail:] if tail else []))


def filter_lines(
    text: str,
    keep_patterns: list[str] | None = None,
    drop_patterns: list[str] | None = None,
) -> str:
    """Keep lines matching *keep_patterns* or not matching *drop_patterns*.

    When *keep_patterns* is set, only matching lines are kept.
    When *drop_patterns* is set, matching lines are removed.
    When both are set, keep is applied first.
    """
    lines = text.split("\n")
    if keep_patterns:
        keep_re = re.compile("|".join(keep_patterns))
        lines = [l for l in lines if keep_re.search(l)]
    if drop_patterns:
        drop_re = re.compile("|".join(drop_patterns))
        lines = [l for l in lines if not drop_re.search(l)]
    return "\n".join(lines)


def group_similar(text: str, pattern: str, label: str) -> str:
    """Group lines matching *pattern* into a summary line: '<label>: <count>'.

    Returns the original text with matched lines replaced by a single summary line.
    """
    lines = text.split("\n")
    re_pat = re.compile(pattern)
    matched: list[str] = []
    kept: list[str] = []
    for line in lines:
        if re_pat.search(line):
            matched.append(line)
        else:
            kept.append(line)
    if matched:
        kept.insert(0, f"{label}: {len(matched)}")
    return "\n".join(kept)


def strip_blanks(text: str) -> str:
    """Remove empty and whitespace-only lines."""
    return "\n".join(l for l in text.split("\n") if l.strip())


# ---------------------------------------------------------------------------
# Detect output type from content signature
# ---------------------------------------------------------------------------

_SIGNATURES: list[tuple[str, re.Pattern[str]]] = [
    ("pytest", re.compile(r"(?m)^={2,}.*(?:FAILURES|ERRORS|PASSED|test session|short test summary|collected \d+ items?)")),
    ("cargo", re.compile(r"(?m)^(?:error\[|warning\[|Compiling |Finished |cargo:)")),
    ("go_test", re.compile(r"(?m)^(?:---\s*(?:FAIL|PASS|\w+)\s|\s*panic:|ok\s+|FAIL\s+)")),
    ("jest", re.compile(r"(?m)^(?:FAIL|PASS)\s|\s●\s|Tests:")),
    ("eslint", re.compile(r"(?m)\d+:\d+\s+(?:error|warning)\s")),
    ("git_status", re.compile(r"(?m)^(?:On branch|Changes (?:not staged|to be committed)|Untracked files)")),
    ("git_diff", re.compile(r"^diff --git ")),
    ("git_log", re.compile(r"^commit [a-f0-9]{7,40}")),
    ("grep", re.compile(r"(?m)^[^:\n\r]+\.[a-zA-Z]{1,6}:\d+:")),
    ("npm", re.compile(r"(?m)^(?:npm ERR|npm WARN|npm notice|added|removed|audited|deprecated)")),
    ("docker", re.compile(r"(?m)^(?:CONTAINER ID|Error:|unknown flag)")),
    ("kubectl", re.compile(r"(?m)^(?:NAME\s+|No resources|Error from server|kubectl)")),
    ("aws", re.compile(r"(?m)^aws.*(?:error|Error|usage:|An error occurred)")),
    ("json_output", re.compile(r"^\s*[\{\[]")),
    ("make", re.compile(r"(?m)^(?:make:|make\[\d+\]|\*\*\*|error:)")),
    ("pip", re.compile(r"(?m)^(?:Successfully|Requirement|Collecting|Installing|ERROR|WARNING)")),
    ("log", re.compile(r"(?m)^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")),
    ("cat", re.compile(r"(?m)^[a-zA-Z]:\\")),
]


def detect_type(text: str) -> str:
    """Inspect first 500 characters and return the best matching type or 'generic'."""
    head = text[:500]
    for name, pat in _SIGNATURES:
        if pat.search(head):
            return name
    return "generic"


# ---------------------------------------------------------------------------
# Per-tool compressors
# ---------------------------------------------------------------------------


def compress_pytest(text: str) -> str:
    """Compress pytest output: assertion lines + one-line summary. Strip tracebacks and PASSED."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Keep assertion failure lines
        if re.match(r"^(?:>|E )", stripped):
            # Shorten: keep only the meaningful part after >
            cleaned = re.sub(r"^>+\s*", "", stripped)
            if cleaned not in result:
                result.append(f"  {cleaned}")
            continue
        # Keep FAIL/ERROR lines from short test summary
        if re.match(r"FAIL[ED]{0,2}\s", stripped) or re.match(r"ERROR\s", stripped):
            # Shorten: keep test name + assertion
            parts = stripped.split(" - ", 1)
            if len(parts) == 2:
                result.append(f"FAIL {parts[0]} - {parts[1]}")
            else:
                result.append(stripped)
            continue
        # Keep the final summary line
        if re.match(r"^=+ \d+ .+ in [\d.]+[a-z]+ =+$", stripped):
            # Extract just the counts without ===
            cleaned = re.sub(r"^=+\s*|\s*=+$", "", stripped)
            result.append(cleaned)
            continue
    if not result:
        passed_match = re.search(r"(\d+) passed", text)
        failed_match = re.search(r"(\d+) failed", text)
        if passed_match and not failed_match:
            return f"All {passed_match.group(1)} tests passed."
        return compress_generic(text)
    return "\n".join(result)


def compress_cargo(text: str) -> str:
    """Compress cargo output: errors, warnings, Compiling/Finished lines."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"error(?:\[E\d+\])?|warning(?:\[\w+\])?", stripped):
            result.append(line)
        elif re.match(r"(?:Compiling|Finished|error:)", stripped):
            result.append(line)
        elif "aborting due to" in stripped or "could not compile" in stripped:
            result.append(line)
    return "\n".join(result) if result else compress_generic(text)


def compress_go_test(text: str) -> str:
    """Compress Go test output: only FAIL lines, panics, and final summary."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"---\s*FAIL", stripped):
            result.append(line)
        elif stripped.startswith("panic:") or stripped.startswith("goroutine "):
            result.append(line)
        elif re.match(r"^FAIL(?:\s|:)", stripped):
            result.append(line)
        elif re.search(r"^FAIL$", stripped):
            result.append(line)
    return "\n".join(result) if result else compress_generic(text)


def compress_jest(text: str) -> str:
    """Compress Jest output: FAIL/PASS, Tests: summary, failure markers."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:FAIL|PASS|PENDING)\s", stripped):
            result.append(line)
        elif re.match(r"Tests?:\s+\d+", stripped):
            result.append(line)
        elif stripped.startswith("●") or stripped.startswith("●"):
            result.append(line)
        elif re.match(r"Snapshot Summary", stripped):
            result.append(line)
        elif re.match(r"\w+/\w+.*\d+:\d+:\d+", stripped):
            result.append(line)
    return "\n".join(result) if result else compress_generic(text)


def compress_eslint(text: str) -> str:
    """Compress ESLint output: file path + counts only. No individual issues."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    file_err: dict[str, int] = {}
    file_warn: dict[str, int] = {}
    file_order: list[str] = []
    total_errors = 0
    total_warnings = 0
    count_line = ""

    for line in lines:
        stripped = line.strip()
        if re.search(r"[/\\][\w.-]+\.(?:[jt]sx?|vue|m?js|cjs|mts|cts)$", stripped):
            fpath = stripped
            if fpath not in file_err:
                file_err[fpath] = 0
                file_warn[fpath] = 0
                file_order.append(fpath)
        elif re.match(r"\d+:\d+\s+(error)\s", stripped):
            total_errors += 1
            if file_order:
                file_err[file_order[-1]] = file_err.get(file_order[-1], 0) + 1
        elif re.match(r"\d+:\d+\s+(warning)\s", stripped):
            total_warnings += 1
            if file_order:
                file_warn[file_order[-1]] = file_warn.get(file_order[-1], 0) + 1
        elif re.search(r"(?:\d+\s+|✖\s*\d+\s+)(?:problems?|errors?|warnings?)", stripped, re.I):
            count_line = stripped

    result: list[str] = []
    for fpath in file_order:
        e = file_err.get(fpath, 0)
        w = file_warn.get(fpath, 0)
        if e == 0 and w == 0:
            continue
        parts = [f"{fpath} ({e + w}: "]
        if e > 0:
            parts.append(f"{e}E")
        if w > 0:
            parts.append(f"{w}W")
        parts.append(")")
        result.append(" ".join(parts))

    if count_line:
        result.append("")
        result.append(count_line)

    return "\n".join(result) if result else compress_generic(text)


def compress_git_status(text: str) -> str:
    """Compress git status: branch + counts only."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    branch = ""
    staged = 0
    unstaged = 0
    untracked = 0
    section = None
    for line in lines:
        if line.startswith("On branch "):
            branch = line.strip()
        elif line.startswith("Your branch "):
            branch += " | " + line.strip()
        elif line.startswith("Changes to be committed"):
            section = "staged"
        elif line.startswith("Changes not staged"):
            section = "unstaged"
        elif line.startswith("Untracked files"):
            section = "untracked"
        elif line.strip() and line[0] in (" ", "\t") and not line.strip().startswith("("):
            if section == "staged":
                staged += 1
            elif section == "unstaged":
                unstaged += 1
            elif section == "untracked":
                untracked += 1

    out = branch + "\n" if branch else ""
    out += f"{staged} staged, {unstaged} unstaged, {untracked} untracked"
    return out


def compress_git_diff(text: str) -> str:
    """Compress git diff: per-file path + first 3 changed lines per hunk, 1 hunk per file."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    current_file = ""
    in_hunk = False
    changed_in_hunk = 0
    hunks_for_file = 0
    for line in lines:
        if line.startswith("diff --git "):
            # Extract the b/ path (second file path in "diff --git a/X b/X")
            parts = line.split()
            current_file = parts[-1] if len(parts) >= 4 else parts[2] if len(parts) >= 3 else ""
            in_hunk = False
            hunks_for_file = 0
        elif line.startswith("@@ "):
            hunks_for_file += 1
            if hunks_for_file <= 1:
                in_hunk = True
                changed_in_hunk = 0
                # Add file header once per file
                result.append(current_file)
        elif in_hunk and hunks_for_file <= 1 and (line.startswith("+") or line.startswith("-")):
            changed_in_hunk += 1
            if changed_in_hunk <= 3:
                result.append(line)
            elif changed_in_hunk == 4:
                result.append("  ...")
    return "\n".join(result) if result else compress_generic(text)


def compress_git_log(text: str) -> str:
    """Compress git log: 5 most recent commits, hash + subject only."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    commits: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("commit "):
            full_hash = lines[i].split()[1] if len(lines[i].split()) > 1 else ""
            short_hash = full_hash[:7]
            # Skip Author and Date lines, find the first message line
            first_msg = ""
            j = i + 1
            while j < len(lines):
                line = lines[j]
                if line.startswith("    ") and line.strip():
                    first_msg = line.strip()
                    break
                j += 1
            if first_msg:
                commits.append(f"{short_hash} {first_msg[:60]}")
            elif full_hash:
                commits.append(short_hash)
            i = j + 1 if first_msg else i + 1
        else:
            i += 1
    kept = commits[:5]
    if len(commits) > 5:
        kept.append(f"... ({len(commits) - 5} more commits)")
    return "\n".join(kept) if kept else compress_generic(text)


def compress_grep(text: str) -> str:
    """Compress grep output: count by match content, show multi-match + summary."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    content_counts: dict[str, int] = {}
    for line in lines:
        parts = line.split(":", 2)
        content = parts[2].strip() if len(parts) > 2 else line.strip()
        content_counts[content] = content_counts.get(content, 0) + 1

    multi = [(c, n) for c, n in content_counts.items() if n > 1]
    singles = sum(1 for _, n in content_counts.items() if n == 1)

    result: list[str] = []
    for content, count in sorted(multi, key=lambda x: -x[1])[:10]:
        result.append(f"({count}x) {content[:100]}")

    if singles > 0:
        result.append(f"({singles} more unique matches)")

    total_files = len(set(l.split(":")[0] for l in lines if ":" in l))
    if total_files > 0:
        result.insert(0, f"{total_files} files, {len(lines)} matches")

    return "\n".join(result) if result else compress_generic(text)


def compress_npm(text: str) -> str:
    """Compress npm output: error header + conflict summary + final counts."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    conflict_pkg = ""
    for line in lines:
        stripped = line.strip()
        if re.match(r"npm ERR! code ", stripped):
            result.append(stripped)
        elif re.match(r"npm ERR! While resolving:", stripped):
            # Extract package name from "While resolving: pkg@ver"
            m = re.search(r"While resolving:\s*(\S+)", stripped)
            if m:
                conflict_pkg = m.group(1)
        elif re.match(r"npm ERR! Found:", stripped):
            result.append(stripped)
        elif re.match(r"npm ERR! Could not resolve", stripped):
            result.append(stripped)
        elif re.match(r"npm ERR! Fix with", stripped):
            result.append(stripped)
        elif re.match(r"(?:added|removed|changed|audited)\s", stripped, re.I):
            result.append(line)
        elif re.match(r"found \d+ vulnerabilit", stripped, re.I):
            result.append(line)
    return "\n".join(result) if result else compress_generic(text)


def compress_docker(text: str) -> str:
    """Compress docker output: ID + status + name per container, errors, capped at 40."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^Error:|unknown flag|docker:", stripped):
            result.append(stripped)
        elif re.match(r"^\w+\s{2,}", stripped):
            # Container/image data row: keep first and last field
            count += 1
            if count > 40:
                continue
            parts = stripped.split()
            first = parts[0]
            last = parts[-1]
            result.append(f"{first}  {last}")
    if count > 0:
        result.insert(0, f"Items: {count}")
    if count > 40:
        result.append(f"... ({count - 40} more containers)")
    return "\n".join(result) if result else compress_generic(text)


def compress_kubectl(text: str) -> str:
    """Compress kubectl output: header + first 5 rows, compress whitespace."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    in_table = False
    data_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        first_token = stripped.split()[0] if stripped.split() else ""
        if first_token == "NAME":
            result.append(re.sub(r"\s+", " ", stripped))
            in_table = True
        elif in_table:
            data_count += 1
            if data_count <= 5:
                result.append(re.sub(r"\s+", " ", stripped))
        elif re.match(r"Error from server|error:|No resources found", stripped):
            result.append(stripped)
    if data_count > 5:
        result.append(f"... ({data_count - 5} more rows)")
    return "\n".join(result) if result else compress_generic(text)


def compress_aws(text: str) -> str:
    """Compress AWS CLI output: compact JSON or trim to 60 lines."""
    text = _strip_ansi(text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        lines = text.split("\n")
        kept = [l for l in lines if re.search(r"(?:error|Error|usage|Invalid|Unknown)", l)]
        if not kept:
            kept = lines[:60]
        if len(kept) > 60:
            kept = kept[:60] + [f"... ({len(lines) - 60} more lines)"]
        return "\n".join(kept)

    def _compact(obj, depth: int = 0) -> object:
        if depth > 4:
            return "..."
        if isinstance(obj, dict):
            if len(obj) > 8:
                return {k: _compact(obj[k], depth + 1) for k in list(obj)[:8]} | {"...": f"+{len(obj) - 8} keys"}
            return {k: _compact(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            if len(obj) > 5:
                return [_compact(obj[0], depth + 1), "...", f"{len(obj)} items"]
            return [_compact(v, depth + 1) for v in obj]
        if isinstance(obj, str) and len(obj) > 100:
            return obj[:100] + "..."
        return obj

    compacted = _compact(data)
    return json.dumps(compacted, indent=2, default=str)


def compress_json_output(text: str) -> str:
    """Compress JSON output: compact small JSON; strip large JSON keeping keys + primitives."""
    text = _strip_ansi(text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        if text.strip().startswith(("{", "[")):
            lines = text.split("\n")
            if len(lines) > 60:
                return "\n".join(lines[:60]) + f"\n... ({len(lines) - 60} more lines)"
        return compress_generic(text)

    # Small JSON — compact to single line
    if len(text) < 500:
        compact = json.dumps(data, separators=(",", ":"), default=str)
        if compact != text.strip():
            return compact
        # If compact is identical, try with sorted keys
        if isinstance(data, dict):
            return json.dumps(data, separators=(",", ":"), sort_keys=True, default=str)
        return compact

    def _strip(obj, depth: int = 0) -> object:
        if depth > 5:
            return "..."
        if isinstance(obj, dict):
            result: dict[str, object] = {}
            for k, v in obj.items():
                result[str(k)] = _strip(v, depth + 1)
            return result
        if isinstance(obj, list):
            if len(obj) > 4:
                return [_strip(obj[0], depth + 1), "...", f"{len(obj)} items"]
            return [_strip(v, depth + 1) for v in obj]
        if isinstance(obj, str) and len(obj) > 60:
            return obj[:60] + "..."
        # Keep primitives (numbers, booleans, None)
        return obj

    stripped = _strip(data)
    result = json.dumps(stripped, indent=2, default=str)
    if len(result) > len(text):
        return text
    return result


def compress_make(text: str) -> str:
    """Compress make output: error:, ***, make: lines."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"(?:make(?:\[\d+\])?:\s|make:)", stripped):
            result.append(line)
        elif "***" in stripped:
            result.append(line)
        elif re.match(r"error:", stripped, re.I):
            result.append(line)
    return "\n".join(result) if result else compress_generic(text)


def compress_pip(text: str) -> str:
    """Compress pip output: final summary + errors only."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"Successfully installed", stripped):
            result.append(line)
        elif re.match(r"ERROR:|Could not|error:", stripped):
            result.append(line)
        elif re.match(r"WARNING:", stripped):
            result.append(line)
    return "\n".join(result) if result else compress_generic(text)


def compress_log(text: str) -> str:
    """Compress log output: strip timestamps, keep ERROR/FATAL/CRITICAL, dedup WARNING."""
    text = _strip_ansi(text)
    lines = text.split("\n")
    critical: list[str] = []
    warnings: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Strip timestamp prefix (e.g., "2026-05-27 09:17:22,345 ")
        msg = re.sub(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[,\.]\d{3,6}\s*", "", stripped)
        # Strip log level prefix (e.g., "[ERROR] ")
        short = re.sub(r"^\[(?:INFO|DEBUG|WARNING|WARN|ERROR|CRITICAL|FATAL)\]\s+", "", msg)
        if re.search(r"\b(?:ERROR|FATAL|CRITICAL|panic|Traceback)\b", msg):
            critical.append(short)
        elif re.search(r"\b(?:WARN|WARNING)\b", msg):
            warnings.append(short)
    result = critical[:10]
    if len(critical) > 10:
        result.append(f"... ({len(critical) - 10} more errors)")
    if warnings:
        deduped = list(dict.fromkeys(warnings))[:5]
        result.append(f"\n{len(warnings)} warnings:")
        for w in deduped:
            result.append(f"  {w[:100]}")
        if len(warnings) > 5:
            result.append(f"  ... ({len(warnings) - 5} more)")
    return "\n".join(result) if result else compress_generic(text)


def compress_cat(text: str) -> str:
    """Compress cat/file output: truncate to 40 head + 20 tail."""
    text = _strip_ansi(text)
    return truncate_middle(text, head=40, tail=20)


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------


def compress_generic(text: str) -> str:
    """Generic compression: strip blanks, deduplicate, truncate at 200 lines."""
    text = _strip_ansi(text)
    cleaned = strip_blanks(text)
    if not cleaned:
        return cleaned
    cleaned = deduplicate(cleaned)
    lines = cleaned.split("\n")
    if len(lines) > 200:
        return "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
    return cleaned


# ---------------------------------------------------------------------------
# Compressor registry
# ---------------------------------------------------------------------------

COMPRESSORS: dict[str, Callable[[str], str]] = {
    "pytest": compress_pytest,
    "cargo": compress_cargo,
    "go_test": compress_go_test,
    "jest": compress_jest,
    "eslint": compress_eslint,
    "git_status": compress_git_status,
    "git_diff": compress_git_diff,
    "git_log": compress_git_log,
    "grep": compress_grep,
    "npm": compress_npm,
    "docker": compress_docker,
    "kubectl": compress_kubectl,
    "aws": compress_aws,
    "json_output": compress_json_output,
    "make": compress_make,
    "pip": compress_pip,
    "log": compress_log,
    "cat": compress_cat,
    "generic": compress_generic,
}

# ---------------------------------------------------------------------------
# Tee recovery — save originals to disk
# ---------------------------------------------------------------------------

_TEE_DIR: str | None = None


def set_tee_dir(path: str) -> None:
    """Set the directory where original outputs are saved for tee recovery."""
    global _TEE_DIR
    _TEE_DIR = str(path)


def tee_save(text: str, label: str = "output") -> Path | None:
    """Save full output to tee directory, return Path or None."""
    if _TEE_DIR is None:
        return None
    tee_path = Path(_TEE_DIR) / f"{label}.txt"
    tee_path.parent.mkdir(parents=True, exist_ok=True)
    tee_path.write_text(text, encoding="utf-8")
    return tee_path


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def compress(text: str, command: str = "auto", ultra: bool = False) -> str:
    """Route text to the appropriate compressor.

    Parameters
    ----------
    text : str
        Raw output text to compress.
    command : str
        One of the COMPRESSORS keys, or "auto" for automatic detection.
    ultra : bool
        When True, truncate result to 30 non-blank lines maximum.

    Returns
    -------
    str
        Compressed text.
    """
    if not text.strip():
        return text

    resolved = command
    if resolved == "auto":
        resolved = detect_type(text)

    compressor = COMPRESSORS.get(resolved, compress_generic)
    result = compressor(text)

    if ultra:
        lines = [l for l in result.split("\n") if l.strip()]
        if len(lines) > 30:
            result = "\n".join(lines[:30]) + f"\n... ({len(lines) - 30} more lines)"

    return result


def compress_tee(
    text: str,
    command: str = "auto",
    ultra: bool = False,
    label: str = "output",
) -> tuple[str, Path | None]:
    """Compress with tee recovery: returns (compressed_text, tee_path_or_None)."""
    path = tee_save(text, label=label)
    result = compress(text, command=command, ultra=ultra)
    return result, path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """CLI entry point: reads stdin, writes compressed output to stdout."""
    parser = argparse.ArgumentParser(
        description="graphsift compress — rtk-style output compression (60-90% token reduction)",
    )
    parser.add_argument(
        "--type", "-t",
        default="auto",
        choices=list(COMPRESSORS) + ["auto"],
        help="Output type (default: auto-detect)",
    )
    parser.add_argument(
        "--ultra", "-u",
        action="store_true",
        help="Ultra-compact: cap output at 30 non-blank lines",
    )
    parser.add_argument(
        "--tee", "-e",
        default=None,
        help="Directory to save original (uncompressed) output for tee recovery",
    )
    parser.add_argument(
        "--tee-label",
        default="output",
        help="Filename label for tee save (default: output)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available compressor types and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Passthrough — show all output, no compression",
    )
    args = parser.parse_args()

    if args.list:
        print("Available compressors:")
        for name, func in COMPRESSORS.items():
            print(f"  {name:20s}  {func.__doc__ and func.__doc__.strip().split(chr(10))[0] or ''}")
        return

    text = sys.stdin.read()

    if args.all:
        sys.stdout.write(text)
        return

    if args.tee:
        set_tee_dir(args.tee)

    result = compress(text, command=args.type, ultra=args.ultra)
    sys.stdout.write(result)


if __name__ == "__main__":
    _cli()
