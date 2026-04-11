# Hacker News — Show HN Post

**Title:**
```
Show HN: graphsift – ranked code context for LLMs, 80-150x token reduction vs blast-radius tools
```

---

**Body:**

I built graphsift after running into the same wall everyone hits when wiring up Claude/GPT-4 for code review: your repo has 200 files, the LLM has a 200k token window, and the naive approach (send everything that transitively imports the changed file) blows the budget and drowns the signal.

Existing tools like code-review-graph use binary blast-radius selection — a file is either included or excluded based on whether it's reachable in the import graph. That produces F1 ≈ 0.54 in practice. Nearly half the files you send are irrelevant, and genuinely important ones (connected via decorators or dynamic imports) get missed entirely.

graphsift does three things differently:

**1. Ranked scoring instead of binary selection**
Every file gets a 0–1 relevance score: 70% graph-distance decay from the changed files (BFS with exponential falloff per hop) + 30% BM25 keyword overlap against the query and commit message. Files are selected greedily until the token budget is exhausted — not included/excluded by a threshold.

**2. Edges that actually matter**
Most tools only track IMPORTS and CALLS. graphsift adds DECORATES edges (so `@require_auth` in auth.py pulling in middleware.py is caught) and DYNAMIC_IMPORT edges (so `importlib.import_module("plugins.storage")` doesn't silently drop context). It also does full multi-file diff — union blast radius across all changed files simultaneously, not just one at a time.

**3. Budget-aware output modes**
High-score files (≥0.5) get full source. Low-score files get signatures only (function/class stubs, ~10% tokens). With tokenpruner installed, low-score files get compressed further. On a 143-file FastAPI app, changing one auth file goes from 180k tokens raw → 800-1200 tokens selected. That's the difference between "can't fit in context" and "fits easily with room for the LLM response."

**Numbers on a realistic repo (143-file FastAPI app, single auth file changed):**
- Raw source: 180,000 tokens
- code-review-graph: 6,000–8,000 tokens (still noisy, F1=0.54)
- graphsift: 800–1,200 tokens (F1≈0.85)

It ships with a full MCP server (compatible with Claude Code and Claude desktop), a CLI (`graphsift install` wires everything up automatically), SQLite persistence with incremental indexing, and 10 advanced features (cache, pipeline, validator, async batch, rate limiter, streaming, diff engine, circuit breaker, retry, schema evolution).

14 languages supported: Python (native ast), JS, TS, Go (receiver methods), Rust, Java, C++, Bash, Terraform/HCL, Helm.

Pure library — no framework, no global state, no main(). One dependency (pydantic).

```bash
pip install graphsift
graphsift install   # wires MCP into Claude Code automatically
```

GitHub: https://github.com/maheshmakvana/graphsift
PyPI: https://pypi.org/project/graphsift/

Happy to answer questions about the ranking algorithm or the MCP integration.
