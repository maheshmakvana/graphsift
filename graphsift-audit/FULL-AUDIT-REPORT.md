# GraphSift SEO Audit — Full Report

**Audit Date:** July 26, 2026  
**Target:** GitHub Repository + PyPI Package  
**URLs:** https://github.com/maheshmakvana/graphsift · https://pypi.org/project/graphsift/  
**Project:** graphsift v4.5.0 — LLM Token Optimization Engine

---

## Executive Summary

| Metric | Score |
|--------|:-----:|
| **Overall SEO Health** | **52/100** ⚠️ |
| GitHub Repo SEO | 55/100 |
| PyPI SEO | 60/100 |
| Content Quality | 45/100 |
| AI Search Readiness | 30/100 |
| Technical SEO | 65/100 |
| Off-Page / Authority | 20/100 |

### Business Type Detected: Developer Tool / Open-Source Library

**Top 5 Critical Issues:**
1. **Dead link chain** — `docs/ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md` referenced in 3 files but does not exist
2. **Version inconsistency** — README download stats section claims "Latest version: 1.5.3" when actual version is 4.5.0
3. **Fragmented, overlapping documentation** — 7+ docs files with heavy content overlap, diluting signal
4. **No dedicated website** — Homepage URL points to PyPI (self-referential), no GitHub Pages or documentation site
5. **Outdated docs retained** — 3 of 7 docs files reference v2.2–v3.0 in headers while current is v4.5.0

**Top 5 Quick Wins:**
1. Fix broken link to `ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md` (create file or remove references)
2. Fix version "1.5.3" → "4.5.0" in download stats section of README
3. Add `llms.txt` for AI crawler discoverability
4. Add GitHub social preview image in repo settings
5. Consolidate/remove outdated docs files

---

## 1. Technical SEO (Score: 65/100)

### Crawlability & Indexability

| Factor | Status | Notes |
|--------|:------:|-------|
| robots.txt | ❌ N/A | GitHub repos don't serve custom robots.txt |
| Sitemap | ❌ Not applicable | No website to serve a sitemap for |
| GitHub Pages | ❌ Not enabled | Could host docs site at `maheshmakvana.github.io/graphsift` |
| Repo description set | ✅ Yes | "About" section has description in GitHub |
| Topics/tags | ✅ Good | 19 relevant topics set |
| Has wiki enabled | ✅ Yes | But unused — no wiki pages created |
| Discussions enabled | ❌ No | Community engagement channel missing |

### Security

| Factor | Status | Notes |
|--------|:------:|-------|
| HTTPS | ✅ Auto | GitHub serves all content over HTTPS |
| Security policy | ✅ Yes | `SECURITY.md` present with reporting process |
| License file | ✅ MIT | Clear licensing improves trust signals |

### URL Structure

| Factor | Status | Notes |
|--------|:------:|-------|
| Repo URL | ✅ Clean | `github.com/maheshmakvana/graphsift` — clean, memorable |
| PyPI URL | ✅ Clean | `pypi.org/project/graphsift/` |
| Documentation URL | ❌ Missing | No dedicated docs subdomain or path |
| Homepage URL | ⚠️ Self-referential | Points to PyPI instead of a dedicated landing page |

---

## 2. On-Page SEO (Score: 55/100)

### GitHub Repo Description

```
Current: "#1 Token Saver for Claude, GPT-5.6 & Gemini. 80-150x code context reduction, F1 0.85.
AST dependency graph, ranked context selection, 19 CLI compressors, MCP server, agent memory.
Save LLM tokens — zero telemetry."
```

**Issues:**
- References "GPT-5.6" which doesn't exist (GPT-5 exists, but the ".6" version is unusual)
- Keyword-dense but reads naturally
- No mention of "open source" or "Python" in the description (these are in topics but not description)
- Could benefit from a stronger call-to-action ("Install: pip install graphsift")

### PyPI Package Description

**Issues:**
- **Extremely long** — thousands of words for a PyPI page, functions as a full landing page rather than a concise package description
- **Heavy marketing tone** — "#1", "Save up to 99%", "Trusted by developers worldwide" reduces credibility
- **Broken internal links** — References to `docs/V3_UPGRADE_GUIDE.md`, `docs/DECONSTRUCTING_GRAPHSIFT_ARCHITECTURE.md` etc. produce 404s when clicked from PyPI
- **Version inconsistency** — Download stats section says "Latest version: 1.5.3" while the page header shows 4.5.0
- **Tag/keyword stuffing** — ~80+ tags in pyproject.toml, many near-duplicates (e.g., "claude token saver", "claude token optimizer", "claude token compression")

### README Content Quality

| Aspect | Rating | Notes |
|--------|:------:|-------|
| Value proposition | ⚠️ Good but buried | The real value (80-150x token reduction) is visible but surrounded by marketing |
| Readability | ⚠️ Dense | Heavy use of tables, emoji headers, ASCII art — information-dense but hard to scan |
| Code examples | ✅ Good | Quick Start has real Python code examples |
| Installation clarity | ✅ Good | Multiple install variants explained clearly |
| Benchmark data | ✅ Excellent | Real numbers, peer comparison tables |
| Broken links | ❌ 1 dead link | `ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md` missing |
| Version accuracy | ❌ Inconsistent | Download section shows 1.5.3, header shows 4.5.0 |

---

## 3. Content Quality & E-E-A-T (Score: 45/100)

### E-E-A-T Assessment

| Factor | Rating | Evidence |
|--------|:------:|----------|
| **Experience** | ⚠️ Medium | Author bio with GitHub/X/LinkedIn links but no developer blog or technical writing portfolio |
| **Expertise** | ✅ High | Technical content shows deep Python + LLM knowledge. 826 tests, complex architecture |
| **Authoritativeness** | ❌ Low | 4 stars, 0 forks, 0 issues — minimal community validation. No third-party citations or testimonials |
| **Trustworthiness** | ⚠️ Medium | Zero-telemetry claims are specific and verifiable. But marketing exaggeration ("#1", "99%") undermines credibility |

### Thin Content Detection

| File | Lines | Assessment |
|------|:-----:|------------|
| README.md | ~830 | Comprehensive but overly verbose |
| docs/API_REFERENCE.md | 273 | Good reference content |
| docs/DECONSTRUCTING_GRAPHSIFT_ARCHITECTURE.md | 886 | Extremely long, outdated (v2.2) |
| docs/FEATURE_MATRIX.md | 277 | Overlaps significantly with README |
| docs/PROMPT_BENCHMARK_2026.md | 216 | Unique benchmark content — keep |
| docs/V3_UPGRADE_GUIDE.md | 218 | Outdated (v3.0 → v4.5 now) |
| docs/WHY_GRAPHSIFT_DETAILED.md | 342 | Overlaps with README |
| docs/wiki_home.md | 134 | FAQ-like, partially unique |

### Duplicate Content Risk

**HIGH** — Multiple files cover the same topics:
- README and `WHY_GRAPHSIFT_DETAILED.md` both cover "WITH vs WITHOUT graphsift" scenarios extensively
- README and `FEATURE_MATRIX.md` both list all modules, classes, and CLI commands
- README, `FEATURE_MATRIX.md`, and `WHY_GRAPHSIFT_DETAILED.md` all reference the same benchmarks
- All docs files have the same boilerplate header text ("Save Claude tokens, reduce GPT-4 costs...")

---

## 4. Schema & Structured Data (Score: 40/100)

| Factor | Status | Notes |
|--------|:------:|-------|
| GitHub schema | ✅ Auto | GitHub auto-generates schema.org markup for repos |
| PyPI schema | ✅ Auto | PyPI generates its own package schema |
| README images alt text | ✅ Good | Hero banner has descriptive alt text |
| JSON-LD in docs | ❌ None | No structured data in any project documentation |
| SoftwareApplication schema | ❌ None | Could use Schema.org SoftwareApplication for docs site |

---

## 5. AI Search Readiness (Score: 30/100)

### GEO (Generative Engine Optimization)

| Factor | Status | Notes |
|--------|:------:|-------|
| llms.txt | ❌ Missing | No instructions for AI crawlers about what to use the content for |
| llms-full.txt | ❌ Missing | No full text version for AI consumption |
| AI citation readiness | ⚠️ Partial | Benchmark data has specific numbers but no citation markers |
| Passage-level citability | ❌ Weak | Long paragraphs without clear section boundaries for AI excerpting |
| Brand mention signals | ❌ None | No structured brand data or organization schema |
| Content freshness | ✅ Good | v4.5.0 released July 23, 2026 |

### Crawlability for AI Agents

| Factor | Status | Notes |
|--------|:------:|-------|
| robots.txt for AI | ⚠️ GitHub default | Standard GitHub robots.txt applies |
| Clear content hierarchy | ⚠️ Partial | README has TOC links but the structure is buried under marketing |
| Code example discoverability | ✅ Good | Python code blocks are well-formed with language tags |
| FAQs available | ⚠️ Partial | `wiki_home.md` has some FAQ/Q&A but not formatted for AI extraction |

---

## 6. Off-Page / Authority (Score: 20/100)

| Factor | Status | Notes |
|--------|:------:|-------|
| GitHub stars | ⚠️ 4 stars | Very low social proof for the depth of the project |
| GitHub forks | ❌ 0 forks | No community forks |
| GitHub watchers | ❌ 0 | No active watchers |
| PyPI downloads | ⚠️ 10,365 total | Modest; 1,533 in last 30 days is decent growth |
| Third-party mentions | ❌ Unknown | No evidence of being listed on awesome lists, newsletters, or directories |
| Competitor differentiation | ✅ Strong | Clear comparison tables vs blast-radius, Caveman, tokenpruner |
| Testimonials | ❌ 1 anonymous quote | "Real developer testimony" is unverifiable — no name, company, or source |

---

## 7. Performance & Images (Score: 60/100)

| Factor | Status | Notes |
|--------|:------:|-------|
| Hero banner image | ✅ Present | Custom branded PNG from docs/images/hero_banner.png |
| Alt text on hero | ✅ Good | Descriptive: "graphsift — #1 LLM Token Saver — Reduce Claude GPT-4 Gemini API Costs by 99%" |
| Image optimization | ⚠️ Unknown | Check if hero_banner.png is compressed for web delivery |
| Architecture diagram | ✅ Present | docs/images/architecture_flow.png |
| Comparison chart | ✅ Present | docs/images/comparison_chart.png |
| Token savings chart | ✅ Present | docs/images/token_savings_chart.png |
| Image format | ⚠️ PNG only | Could use WebP for smaller filesizes if on a website |

---

## 8. Prioritized Action Plan

### Phase 1: Critical Fixes (Week 1)

| # | Issue | Current | Fix | SEO Impact |
|---|-------|---------|-----|:----------:|
| 1 | **Dead link** | `docs/ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md` referenced but missing | Create the file or remove all references | High |
| 2 | **Version inconsistency** | Download section says "Latest version: 1.5.3" | Fix to "4.5.0" | High |
| 3 | **Outdated docs** | 3 files reference v2.2/v3.0 | Remove or update to v4.5.0 | High |
| 4 | **Broken PyPI links** | Internal `docs/*.md` links in README 404 on PyPI | Convert to full GitHub URLs or remove | High |

### Phase 2: High-Impact Improvements (Weeks 2-3)

| # | Issue | Current | Fix | SEO Impact |
|---|-------|---------|-----|:----------:|
| 5 | **Doc consolidation** | 7 overlapping docs files | Merge to 3-4 focused docs (see below) | High |
| 6 | **llms.txt** | Missing | Add at repo root for AI crawlers | Medium |
| 7 | **GitHub social preview** | Uses auto-generated OG image | Set custom social preview image in repo settings | Medium |
| 8 | **README length** | ~830 lines, marketing-heavy | Trim to essential sections, move extended content to docs/ | Medium |

### Phase 3: Content & Authority (Month 2)

| # | Issue | Current | Fix | SEO Impact |
|---|-------|---------|-----|:----------:|
| 9 | **GitHub Pages** | Not enabled | Host docs/ site with GitHub Pages | High |
| 10 | **Community building** | 4 stars, 0 forks, no discussions | Enable Discussions, engage with users | Medium |
| 11 | **Marketing tone** | "#1", "99%", "trusted by developers" | Replace with specific, verifiable claims | Medium |
| 12 | **Missing homepage** | PyPI is the "homepage" | Create simple landing page or docs site | High |

### Phase 4: Monitoring & Iteration (Ongoing)

| # | Recommendation | Effort | Frequency |
|---|---------------|:------:|:---------:|
| 13 | Track PyPI download trends | Low | Monthly |
| 14 | Monitor GitHub stars and engagement | Low | Weekly |
| 15 | Update docs with each release | Low | Per release |
| 16 | Consider submitting to awesome lists | Medium | One-time |

---

## 9. Recommended Doc Restructure

### Current (7 files, ~2,346 lines, heavy overlap)
```
docs/
├── API_REFERENCE.md (273 lines) — keep
├── DECONSTRUCTING_GRAPHSIFT_ARCHITECTURE.md (886 lines) — OUTDATED v2.2
├── FEATURE_MATRIX.md (277 lines) — OVERLAPS README
├── PROMPT_BENCHMARK_2026.md (216 lines) — keep
├── V3_UPGRADE_GUIDE.md (218 lines) — OUTDATED v3.0
├── WHY_GRAPHSIFT_DETAILED.md (342 lines) — OVERLAPS README
└── wiki_home.md (134 lines) — partially unique
```

### Proposed (4 files, ~1,000 lines, focused)
```
docs/
├── API_REFERENCE.md — Standard reference (keep, but verify version)
├── PROMPT_BENCHMARK_2026.md — Unique research (keep)
├── GUIDE.md — Consolidated: architecture overview + FAQ + use cases
└── images/ — Image assets (keep)
```

---

## 10. Audit Data

```json
{
  "summary": {
    "health_score": 52,
    "business_type": "Developer Tool / Open-Source Library",
    "top_findings": [
      "Dead link chain to ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md across 3 files",
      "Version inconsistency: 1.5.3 vs 4.5.0 in README download section",
      "7 fragmented docs files with 60%+ content overlap",
      "No dedicated website or GitHub Pages documentation site",
      "3 docs files reference outdated versions (v2.2, v3.0)"
    ],
    "quick_wins": [
      "Fix dead ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md link",
      "Fix version 1.5.3 → 4.5.0 in download stats section",
      "Add llms.txt for AI crawler discoverability",
      "Set GitHub social preview image",
      "Consolidate/remove 3+ outdated docs files"
    ]
  },
  "categories": [
    {
      "name": "Technical SEO",
      "score": 65,
      "what_works": [
        "Clean GitHub repo URL structure",
        "MIT license present",
        "SECURITY.md with reporting process",
        "19 relevant repo topics set"
      ],
      "findings": [
        {
          "title": "No GitHub Pages documentation site",
          "severity": "High",
          "description": "No docs site is published. GitHub Pages could host API docs and a landing page at no cost.",
          "recommendation": "Enable GitHub Pages from docs/ directory or use a static site generator"
        },
        {
          "title": "Homepage URL is self-referential",
          "severity": "Medium",
          "description": "pyproject.toml homepage and documentation URL both point to PyPI or GitHub README instead of a dedicated site",
          "recommendation": "Create a GitHub Pages site and use that URL as the official homepage"
        },
        {
          "title": "Discussions not enabled",
          "severity": "Low",
          "description": "GitHub Discussions could provide community engagement and user-generated content for SEO",
          "recommendation": "Enable Discussions for Q&A, ideas, and community support"
        }
      ]
    },
    {
      "name": "Content Quality",
      "score": 45,
      "what_works": [
        "Extensive benchmark data with specific numbers",
        "Clear code examples in Quick Start",
        "Good installation instructions with multiple variants",
        "Security and privacy promises are prominently stated"
      ],
      "findings": [
        {
          "title": "Heavy marketing tone reduces credibility",
          "severity": "High",
          "description": "Phrases like '#1 Token Saver', 'Save up to 99%', 'Trusted by developers worldwide' read as marketing copy rather than factual documentation",
          "recommendation": "Replace superlatives with specific, verifiable claims. Let the benchmark data speak for itself."
        },
        {
          "title": "Fragmented documentation with heavy overlap",
          "severity": "High",
          "description": "7 docs files totaling ~2,346 lines, with 60%+ content duplicated across files. Same benchmarks appear in 4+ places.",
          "recommendation": "Consolidate to 3 core docs: API reference, prompt benchmark, and a combined guide"
        },
        {
          "title": "Outdated docs retained",
          "severity": "High",
          "description": "Architecture doc says v2.2.0, Upgrade Guide says v3.0, current is v4.5.0. Confuses readers and dilutes authority.",
          "recommendation": "Remove or comprehensively update. Outdated docs hurt more than no docs."
        },
        {
          "title": "Broken internal links on PyPI",
          "severity": "High",
          "description": "Relative links like docs/V3_UPGRADE_GUIDE.md don't resolve on PyPI — users get 404s",
          "recommendation": "Convert relative markdown links to full GitHub URLs in README"
        },
        {
          "title": "Version inconsistency in download section",
          "severity": "Critical",
          "description": "Download stats section explicitly states 'Latest version: 1.5.3' while page header shows 4.5.0",
          "recommendation": "Update the version reference to 4.5.0 to match the release"
        }
      ]
    },
    {
      "name": "On-Page SEO",
      "score": 55,
      "what_works": [
        "Descriptive repo description in GitHub About section",
        "19 relevant topics/tags",
        "Good hero banner image with descriptive alt text",
        "Comprehensive feature coverage"
      ],
      "findings": [
        {
          "title": "GPT-5.6 reference is imprecise",
          "severity": "Medium",
          "description": "Repo description references 'GPT-5.6' — there is no GPT-5.6, only GPT-5. This could confuse searchers.",
          "recommendation": "Change 'GPT-5.6' to 'GPT-5' or 'GPT-4o' in the repo description"
        },
        {
          "title": "No GitHub Pages social preview image",
          "severity": "Medium",
          "description": "Repo has a hero banner image in README but no GitHub social preview set in repo settings",
          "recommendation": "Upload a social preview image (1280×640px) in repo Settings → Social Preview"
        },
        {
          "title": "Keyword stuffing in pyproject.toml",
          "severity": "Low",
          "description": "~80+ tags with many near-duplicates (e.g., 'claude token saver', 'claude token optimizer', 'claude token compression')",
          "recommendation": "Trim to 30-40 distinct, non-overlapping keywords"
        }
      ]
    },
    {
      "name": "AI Search Readiness",
      "score": 30,
      "findings": [
        {
          "title": "No llms.txt for AI crawler guidance",
          "severity": "High",
          "description": "llms.txt file is not present. AI crawlers (ChatGPT, Perplexity, Gemini) have no guidance on how to use the content.",
          "recommendation": "Add llms.txt at repo root describing the project, key pages, and how AI should cite the content"
        },
        {
          "title": "No structured FAQ format",
          "severity": "Medium",
          "description": "FAQ content in wiki_home.md isn't formatted for AI extraction or featured snippet capture",
          "recommendation": "Format FAQ with clear Q: / A: markers and proper heading hierarchy"
        },
        {
          "title": "No CITATION.cff for academic citations",
          "severity": "Low",
          "description": "No CITATION.cff file means no standardized citation format for AI tools and academic users",
          "recommendation": "Add a CITATION.cff to enable proper attribution and academic discoverability"
        }
      ]
    },
    {
      "name": "Authority & Trust",
      "score": 20,
      "findings": [
        {
          "title": "Low social proof (4 stars, 0 forks)",
          "severity": "High",
          "description": "Despite having 826+ tests and 50+ modules, the repo has almost no community engagement signals",
          "recommendation": "Share the project on relevant communities (r/Python, Hacker News, LLM dev forums). Submit to awesome lists."
        },
        {
          "title": "Unverifiable testimonial",
          "severity": "Medium",
          "description": "The 'Real Developer Testimony' quote in README has no attribution, undermining its credibility",
          "recommendation": "Either attribute with a real name/company or remove. Use GitHub issue quotes with links if available."
        }
      ]
    }
  ],
  "action_plan": {
    "phases": [
      {
        "name": "Phase 1: Critical Fixes",
        "timeframe": "Week 1",
        "items": [
          "Create ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md or remove all dead references",
          "Fix version inconsistency (1.5.3 → 4.5.0) in README download stats section",
          "Remove 3 outdated docs files (V3_UPGRADE_GUIDE.md, DECONSTRUCTING_GRAPHSIFT_ARCHITECTURE.md, FEATURE_MATRIX.md)",
          "Convert relative docs/* links in README to full GitHub URLs for PyPI compatibility"
        ]
      },
      {
        "name": "Phase 2: High-Impact Improvements",
        "timeframe": "Weeks 2-3",
        "items": [
          "Consolidate remaining docs into 3-4 focused files",
          "Add llms.txt at repo root for AI crawler guidance",
          "Set GitHub social preview image (1280×640px)",
          "Trim README to essential content, move extended discussions to docs/"
        ]
      },
      {
        "name": "Phase 3: Content & Authority",
        "timeframe": "Month 2",
        "items": [
          "Enable GitHub Pages and publish a documentation site",
          "Enable GitHub Discussions for community engagement",
          "Create a simple landing page or direct to GitHub Pages as homepage",
          "Submit to awesome lists (awesome-python, awesome-claude, awesome-llm)"
        ]
      },
      {
        "name": "Phase 4: Monitoring & Iteration",
        "timeframe": "Ongoing",
        "items": [
          "Track PyPI download trends monthly",
          "Fix outdated docs with each major release",
          "Add CITATION.cff for academic discoverability",
          "Replace marketing superlatives with specific claims"
        ]
      }
    ]
  }
}
```

---

## Summary

GraphSift has **strong technical foundations** for SEO — clean URLs, comprehensive content, specific benchmark data, and clear licensing. However, it's held back by **fragmented docs, outdated files, dead links, version inconsistencies, and an overly marketing-heavy tone** that undermines credibility.

The biggest opportunity is **documentation consolidation + a GitHub Pages site**. With GitHub Pages, the project could have a proper landing page, indexed documentation, and structured data — all at zero cost. Fixing the critical issues (dead links, version inconsistency, outdated docs) should take less than an hour and would immediately improve the project's SEO health score from 52 to ~65.

See [ACTION-PLAN.md](ACTION-PLAN.md) for the prioritized implementation checklist.
