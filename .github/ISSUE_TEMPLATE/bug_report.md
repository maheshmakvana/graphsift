---
name: Bug report
about: Create a report to help us improve graphsift
title: "[BUG] "
labels: bug
assignees: ""
---

## Describe the Bug

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:

```python
# Minimal code example that reproduces the issue
from graphsift import ContextBuilder, ContextConfig

config = ContextConfig(token_budget=1000)
builder = ContextBuilder(config)
# ... minimal reproduction ...
```

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

What actually happened. Include full error output if applicable.

```
Paste error traceback or unexpected output here
```

## Environment

- **graphsift version:** <!-- e.g., 2.2.0 -->
- **Python version:** <!-- e.g., 3.11.4 -->
- **OS:** <!-- e.g., macOS 14.5, Ubuntu 22.04, Windows 11 -->
- **Install method:** <!-- `pip install graphsift`, `pip install -e ".[dev]"`, etc. -->
- **Git available?** <!-- Yes/No (relevant for TemporalGraph features) -->

## Additional Context

- [ ] Can reproduce with the latest version
- [ ] Can reproduce with minimal dependencies (`pip install graphsift` only)
- [ ] This is a regression from a previous version

Add any other context, screenshots, or logs here.

## Verbose Output

If applicable, run with increased verbosity:

```bash
# If using CLI
graphsift --verbose build

# If using Python API
import logging
logging.basicConfig(level=logging.DEBUG)
```
