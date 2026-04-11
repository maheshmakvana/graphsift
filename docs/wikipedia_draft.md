# Wikipedia Article Draft — graphsift

> **IMPORTANT BEFORE SUBMITTING:**
> Wikipedia's notability policy requires significant coverage in independent, reliable, third-party sources.
> Before submitting this draft, you need at least 2–3 of the following:
> - A published article on a tech blog, dev.to, or Medium that is NOT written by the project author
> - Mentions on Hacker News, Reddit r/Python, or r/MachineLearning with substantial discussion
> - A Stack Overflow answer or question that cites graphsift
> - A GitHub repository with >100 stars from independent users
>
> Submit the draft at: https://en.wikipedia.org/wiki/Wikipedia:Articles_for_creation
> Use the Articles for Creation (AfC) process — do NOT create the article directly.

---

## Wikitext (paste this into Wikipedia's editor)

```
{{Short description|Python library for ranked code context selection for large language models}}
{{Infobox software
| name                   = graphsift
| logo                   =
| screenshot             =
| caption                =
| developer              =
| released               = {{Start date and age|2025}}
| latest release version = 1.4.1
| latest release date    = {{Start date and age|2026|4|11}}
| programming language   = [[Python (programming language)|Python]]
| operating system       = [[Cross-platform]]
| genre                  = [[Software library]], [[Artificial intelligence]]
| license                = [[MIT License]]
| website                = {{URL|https://github.com/maheshmakvana/graphsift}}
}}

'''graphsift''' is an [[open-source software|open-source]] [[Python (programming language)|Python]] [[library (computing)|library]] for intelligent [[source code]] context selection for [[large language model]]s (LLMs). It constructs an [[abstract syntax tree]] (AST)-based dependency graph of a codebase, ranks files by relevance to a code change using a multi-signal scoring algorithm, and selects files within a configurable [[lexical token|token]] budget for injection into LLM prompts.

graphsift is designed as a [[library (computing)|pure library]] with no runtime dependencies beyond [[Pydantic]], following [[hexagonal architecture]] (ports and adapters). It supports 14 programming languages and exposes a [[Model Context Protocol]] (MCP) server compatible with [[Claude (language model)|Claude Code]] and Claude desktop.

== Background ==

[[Large language model]]s used in software development tasks — such as code review, bug detection, and automated documentation — require repository context injected into their input [[prompt engineering|prompt]]. A common approach, known as "blast-radius" selection, traverses the [[dependency graph]] of a changed file and includes every reachable file in the context. This approach has been observed to produce [[F-score|F1 scores]] of approximately 0.54 in empirical evaluations,<ref>{{Cite web |title=graphsift: ranked code context for LLMs |url=https://github.com/maheshmakvana/graphsift |access-date=2026-04-11 |website=GitHub}}</ref> meaning approximately 46% of selected files are irrelevant to the change under review.

Two structural weaknesses contribute to this low precision:

# '''False positives''': Files at multiple import hops from the changed file are included regardless of their actual relevance.
# '''False negatives''': Dependencies expressed through [[decorator (computer programming)|decorators]] or [[dynamic loading|runtime imports]] (via {{code|importlib.import_module}}) are not captured by static import graph traversal.

graphsift was developed in 2025 to address these limitations through ranked relevance scoring and richer dependency edge types.

== Technical description ==

=== Dependency graph construction ===

graphsift parses source files using language-specific parsers to extract symbols (functions, classes, methods, decorators) and seven categories of directed dependency edges:

{| class="wikitable"
|-
! Edge type !! Description
|-
| CALLS || Function or method A calls function or method B
|-
| IMPORTS || Module A statically imports module B
|-
| INHERITS || Class A inherits from class B
|-
| DECORATES || A [[decorator (computer programming)|decorator]] is applied to a function or class
|-
| REFERENCES || Symbol A references symbol B
|-
| TEST_COVERS || A test module covers an implementation module
|-
| DYNAMIC_IMPORT || A runtime import via {{code|importlib.import_module}} or {{code|__import__}}
|}

The DECORATES and DYNAMIC_IMPORT edge types distinguish graphsift from binary blast-radius tools, which typically track only IMPORTS and CALLS edges.

=== Relevance ranking ===

For each changed file in a diff, graphsift performs [[breadth-first search]] (BFS) from the corresponding graph node. Each reachable file receives a graph rank score computed by exponential distance decay:

: <math>\text{graph\_score} = (1 - \text{decay\_factor})^{\text{distance}}</math>

where the default decay factor is 0.7. This score is fused with a [[Okapi BM25|BM25]] keyword overlap score computed against the query and commit message:

: <math>\text{final\_score} = 0.3 \times \text{bm25\_score} + 0.7 \times \text{graph\_score}</math>

For multi-file diffs, scores from each changed file are combined by taking the maximum across all seed nodes, producing a union blast radius.

=== Token-budget-aware selection ===

Files are sorted by relevance score in descending order and selected greedily until the configured token budget is exhausted. Each file is rendered in one of four output modes based on its score:

{| class="wikitable"
|-
! Mode !! Applied when !! Approximate token cost
|-
| FULL || Score ≥ 0.5 || Full source text
|-
| SIGNATURES || Score < 0.5 || 10–20% of full source
|-
| COMPRESSED || tokenpruner installed || 20–40% of full source
|-
| SMART || Default mode || Automatic: FULL or SIGNATURES by threshold
|}

=== Performance ===

On a 143-file [[FastAPI]] application with one authentication file changed (50 lines), graphsift selects 3–5 files totalling 800–1,200 tokens, compared to 6,000–8,000 tokens for binary blast-radius selection and approximately 180,000 tokens for unfiltered source. The reported F1 score is approximately 0.85 versus 0.54 for binary selection.

== Features ==

=== Language support ===

graphsift supports 14 programming languages:

* [[Python (programming language)|Python]] — native {{code|ast}} module parser; detects async functions, decorators, and dynamic imports
* [[JavaScript]] and [[TypeScript]] — regex-based parser for ES6 functions and classes
* [[Go (programming language)|Go]] — detects receiver methods (e.g., {{code|func (r *Router) Handle()}}) and interfaces
* [[Rust (programming language)|Rust]] — functions and {{code|impl}} blocks
* [[Java (programming language)|Java]], [[C++]], [[C (programming language)|C]], [[Ruby (programming language)|Ruby]], [[PHP]] — generic regex parser
* [[Bash (Unix shell)|Bash]]/Shell — shell functions and {{code|source}} imports
* [[Terraform]] / [[HashiCorp Configuration Language|HCL]] — resource, variable, and module blocks
* [[Helm (package manager)|Helm]] Charts — [[Go (programming language)|Go]] templates embedded in [[YAML]]

=== Advanced features ===

graphsift ships an {{code|advanced.py}} module implementing ten production-grade utilities:

# '''GraphCache''' — [[Cache replacement policies#LRU|LRU]] + [[time to live|TTL]] cache with a {{code|.memoize()}} decorator and {{code|.stats()}} method
# '''AnalysisPipeline''' — composable step chain with {{code|.arun()}} async variant, per-step audit log, and configurable retry with [[exponential backoff]]
# '''DiffValidator''' — declarative rule DSL for validating {{code|DiffSpec}} inputs; sync and async variants
# '''async_batch_index''' — concurrent batch indexing with bounded [[semaphore (programming)|semaphore]] and per-item error isolation
# '''RateLimiter''' — [[token bucket]] algorithm; sync and async context manager; per-key singletons
# '''stream_context''' / '''async_stream_context''' — generator and async generator yielding files highest-score first; cancellation-safe
# '''ContextDiff''' — structural diff of two {{code|ContextResult}} objects with {{code|.summary()}} and {{code|.to_json()}} output
# '''CircuitBreaker''' — closed/open/half-open states; configurable failure threshold and auto-reset timeout
# '''RetryStrategy''' — exponential backoff with jitter; per-exception routing; max attempts and deadline
# '''SchemaEvolution''' — versioned migration registry for {{code|ContextResult}} and {{code|DiffSpec}} payloads; migration audit log

=== Model Context Protocol server ===

graphsift ships a full [[Model Context Protocol]] (MCP) server compatible with Claude Code and Claude desktop. Installation is automated via the {{code|graphsift install}} CLI command, which writes {{code|.mcp.json}} and configures hooks automatically.

=== Persistence ===

An optional SQLite-backed {{code|GraphStore}} adapter provides graph persistence across sessions. The schema has a six-version migration history managed internally.

== Architecture ==

graphsift follows [[hexagonal architecture]] (ports and adapters):

* {{code|core.py}} — pure domain logic with no I/O or side effects
* {{code|adapters/}} — SQLite persistence, Claude API adapter, filesystem helpers
* {{code|models.py}} — [[Pydantic]] v2 {{code|BaseModel}} definitions with {{code|frozen=True}} for value objects
* {{code|exceptions.py}} — typed exception hierarchy rooted at {{code|graphsiftError}}

All external dependencies are expressed as {{code|typing.Protocol}} abstractions. All shared mutable state is protected by {{code|threading.RLock}}. Every blocking public method has an {{code|async def}} twin using {{code|asyncio.to_thread}} for any synchronous I/O.

== See also ==

* [[Abstract syntax tree]]
* [[Dependency graph]]
* [[Large language model]]
* [[Model Context Protocol]]
* [[Okapi BM25]]
* [[Token bucket]]

== References ==

{{Reflist}}

== External links ==

* [https://github.com/maheshmakvana/graphsift graphsift on GitHub]
* [https://pypi.org/project/graphsift/ graphsift on PyPI]
```

---

## Submission instructions

1. Go to https://en.wikipedia.org/wiki/Wikipedia:Articles_for_creation
2. Click **"Submit your draft"**
3. Create a draft page at `Draft:graphsift`
4. Paste the wikitext above (everything inside the triple-backtick block)
5. Add a note in the submission: *"This is a new open-source Python library for LLM code context selection, published on PyPI. Independent coverage exists at [link to your dev.to article or HN thread]."*
6. Wait for a Wikipedia reviewer — typically 1–8 weeks

**Do not submit before you have at least one independent third-party source to cite.** The article as written cites only the GitHub repo (primary source). Wikipedia reviewers will decline it without independent coverage.
