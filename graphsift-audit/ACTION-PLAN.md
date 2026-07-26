# GraphSift SEO — Action Plan

## Phase 1: Critical Fixes (Week 1)

### ☐ Fix dead link chain
- **Files affected:** `README.md`, `docs/WHY_GRAPHSIFT_DETAILED.md`, `docs/FEATURE_MATRIX.md`
- **Action:** Either create `docs/ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md` or remove all references
- **Effort:** 5-15 min
- **Impact:** Eliminates 404 errors for users and crawlers

### ☐ Fix version inconsistency in README
- **File:** `README.md` — download stats section
- **Current:** "Latest version: **1.5.3**"
- **Fix:** Change to "4.5.0"
- **Effort:** 1 min

### ☐ Remove 3 outdated docs files
- **Files to remove:**
  - `docs/V3_UPGRADE_GUIDE.md` — references v3.0, current is v4.5.0
  - `docs/DECONSTRUCTING_GRAPHSIFT_ARCHITECTURE.md` — 886 lines, references v2.2.0
  - `docs/FEATURE_MATRIX.md` — content fully overlaps with README
- **Effort:** 5 min
- **Impact:** Eliminates outdated content that confuses users and dilutes authority

### ☐ Convert relative docs links in README to full URLs
- **File:** `README.md` — "Further Reading" section and inline links
- **Fix:** Change `docs/V3_UPGRADE_GUIDE.md` to full GitHub URL
- **Effort:** 5 min
- **Impact:** Links work from both GitHub and PyPI

## Phase 2: High-Impact Improvements (Weeks 2-3)

### ☐ Consolidate remaining docs
- **Keep:** `docs/API_REFERENCE.md`, `docs/PROMPT_BENCHMARK_2026.md`
- **Keep as consolidated:** `docs/GUIDE.md` (merge unique content from `wiki_home.md`, `WHY_GRAPHSIFT_DETAILED.md`)
- **Remove:** redundant files after consolidation

### ☐ Add `llms.txt`
- **File:** `/llms.txt` at repo root
- **Content:** Brief description of what GraphSift is, key pages, and instructions for AI crawlers
- **Effort:** 10 min
- **Impact:** AI search visibility (ChatGPT, Perplexity, Gemini)

### ☐ Set GitHub social preview image
- **Repo Settings → Social Preview**
- **Size:** 1280×640px
- **Effort:** 15 min
- **Impact:** Better link previews when shared on social media / messaging

### ☐ Trim marketing tone from README
- **Target:** "#1", "Save up to 99%", "Trusted by developers worldwide", "Night and day"
- **Replace with:** Specific, data-backed claims from benchmark tables
- **Effort:** 30 min
- **Impact:** Improved credibility and trust signals

## Phase 3: Content & Authority (Month 2)

### ☐ Enable GitHub Pages
- **Source:** `/docs` directory or `/` with static site generator
- **URL:** `maheshmakvana.github.io/graphsift`
- **Effort:** 2-4 hours

### ☐ Enable GitHub Discussions
- **Repo Settings → Features → Discussions:** Enable
- **Effort:** 5 min

### ☐ Set GitHub Pages as homepage URL
- **pyproject.toml:** Update `Homepage` URL
- **Repo About section:** Update website URL

### ☐ Submit to awesome lists
- awesome-python, awesome-claude, awesome-llm, awesome-developer-tools

## Phase 4: Monitoring (Ongoing)

### ☐ Monthly PyPI download check
- Track at pepy.tech/projects/graphsift

### ☐ Per-release docs audit
- Verify all links work, version references are current

### ☐ Add CITATION.cff
- Enable academic citation and discoverability
