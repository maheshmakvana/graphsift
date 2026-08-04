# graphsift UI/UX Design Intelligence

`graphsift uiux` turns graphsift into a UI/UX design partner for frontend work:
landing pages, dashboards, SaaS apps, components, motion/animation, color
palettes, typography, and design reviews.

It is a **thin, opt-in wrapper** — graphsift does **not** bundle the
`ui-ux-pro-max-skill` engine or its design database. Instead it locates the
officially-installed skill on your machine and delegates to its `search.py`
(BM25 search over 84 styles, 192 WCAG-tested color palettes, 74 font pairings,
25 chart types, 98 UX guidelines, 16 GSAP motion presets and 22 stacks).

## How it auto-triggers

The `graphsift-uiux` skill ships with a dense frontmatter description and is
marked **`user-invocable: false`**, so Claude Code activates it **automatically**
whenever you ask for any UI/UX, frontend, or visual-design work — and it never
clutters the `/` slash-command menu (there is no `/graphsift-uiux` command).
Claude runs `graphsift uiux ...` itself whenever the design work matches.

## Install the engine (automatic)

graphsift delegates to the MIT-licensed `ui-ux-pro-max-skill` (© 2024 Next
Level Builder). No upstream code is copied into graphsift, so there are no
licensing/copy issues beyond normal MIT attribution.

The engine installs **automatically** — the first time you run
`graphsift install` (or update graphsift and re-run it) after the engine is
missing, graphsift runs the upstream installer itself:

```bash
npm install -g ui-ux-pro-max-cli && uipro init --ai claude
```

- Skip it with `graphsift install --no-uiux-engine`.
- `graphsift uiux --install` still works as a manual one-off.
- If npm is unavailable the install fails gracefully and the skill retries on
  your first UI/UX request.

or manually:

```bash
npm install -g ui-ux-pro-max-cli
uipro init --ai claude
```

or from the Claude Code plugin marketplace:

```
/plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill
/plugin install ui-ux-pro-max@ui-ux-pro-max-skill
```

> If installed somewhere unusual, set `GRAPHSIFT_UIUX_SKILL` to the
> `search.py` path (or the directory holding it) to skip discovery.

`graphsift install` prints the engine status during setup.

## CLI usage

```bash
# Full design system (style, palette, typography, motion, anti-patterns, checklist)
graphsift uiux "beauty spa wellness service" --design-system -p "Serenity Spa"

# Tune the design with dials (1-10 each)
graphsift uiux "internal analytics dashboard" --design-system --variance 8 --motion 7 --density 8

# Targeted domain search
graphsift uiux "glassmorphism" --domain style
graphsift uiux "loading animation" --domain ux

# Stack-specific guidelines
graphsift uiux "suspense streaming" --stack nextjs

# JSON output for scripting / agents
graphsift uiux "saas analytics dashboard" --design-system --json

# Persist a design system as source of truth (master + per-page overrides)
graphsift uiux "fintech app" --design-system --persist --output-dir ./design-system -p "Fintech"

# Utilities
graphsift uiux --list-domains
graphsift uiux --list-stacks
graphsift uiux --validate-data
```

`--design-system` takes priority; otherwise `--stack` runs a stack search, and
`--domain` (or auto-detect) runs a domain search.

## MCP tools

Three tools are registered on the graphsift MCP server, so agents can call the
design engine directly (same engine, no shelling to a command):

- **`uiux_design_system`** — `query` (+ optional `project_name`, `variance`,
  `motion`, `density`, `format`) → complete design system JSON.
- **`uiux_search`** — `query` (+ optional `domain`, `max_results`) → ranked
  BM25 matches.
- **`uiux_stack_guide`** — `stack` (+ optional `query`, `max_results`) →
  framework-specific Do/Don't guidance.

All three return `{"error": ...}` (with install instructions) when the engine
is not installed.

## What a design system contains

For a query like `"beauty spa wellness"` the engine returns:

- **Pattern** — page structure (hero-centric, CTA placement, sections)
- **Style** — name, keywords, best-for, light/dark support, performance
- **Colors** — 10 WCAG AA/AAA-tested tokens (primary, accent, background, …)
- **Typography** — Google Fonts pairing + CSS import URL
- **Motion** — key effects, durations, and a GSAP snippet when `--motion` is set
- **Anti-patterns** — what to avoid
- **Pre-delivery checklist** — no emoji icons, hover states, focus-visible,
  4.5:1 contrast, prefers-reduced-motion, responsive breakpoints

## Licensing

The engine and its design database are MIT-licensed
(`ui-ux-pro-max-skill`, © 2024 Next Level Builder). graphsift only shells out
to the installed skill; it vendors no upstream code or data. See
`graphsift/uiux/__init__.py` for the discovery logic and `install_hint()` for
setup instructions.
