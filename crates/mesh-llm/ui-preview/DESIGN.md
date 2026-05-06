---
name: MeshLLM UI Preview
description: A compact operations console for governing distributed local AI inference.
colors:
  background-dark: 'oklch(0.17 0.015 250)'
  foreground-dark: 'oklch(0.96 0.005 80)'
  muted-dark: 'oklch(0.23 0.02 250)'
  muted-foreground-dark: 'oklch(0.75 0.01 250)'
  border-dark: 'oklch(0.3 0.02 250 / 0.9)'
  border-soft-dark: 'oklch(0.3 0.02 250 / 0.45)'
  panel-dark: 'oklch(0.2 0.018 250)'
  panel-strong-dark: 'oklch(0.23 0.02 250)'
  accent-dark: 'oklch(0.8 0.14 200)'
  accent-contrast-dark: 'oklch(0.78 0.14 322)'
  accent-ink-dark: 'oklch(0.2 0.04 220)'
  accent-soft-dark: 'oklch(0.3 0.06 200)'
  good-dark: 'oklch(0.78 0.14 150)'
  warn-dark: 'oklch(0.8 0.12 80)'
  bad-dark: 'oklch(0.7 0.18 25)'
  background-light: 'oklch(0.985 0.003 80)'
  foreground-light: 'oklch(0.22 0.02 250)'
  muted-light: 'oklch(0.955 0.005 80)'
  muted-foreground-light: 'oklch(0.45 0.015 250)'
  border-light: 'oklch(0.88 0.005 80)'
  border-soft-light: 'oklch(0.93 0.005 80)'
  panel-light: 'oklch(0.975 0.004 80)'
  panel-strong-light: 'oklch(0.955 0.005 80)'
  accent-light: 'oklch(0.62 0.14 220)'
  accent-contrast-light: 'oklch(0.62 0.14 322)'
  accent-ink-light: 'oklch(0.98 0.01 220)'
  accent-soft-light: 'oklch(0.92 0.05 198)'
  good-light: 'oklch(0.58 0.13 150)'
  warn-light: 'oklch(0.62 0.14 55)'
  bad-light: 'oklch(0.62 0.16 28)'
typography:
  display:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-type-display)'
    fontWeight: 700
    lineHeight: 1.16
    letterSpacing: '-0.026em'
  headline:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-type-headline)'
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: '-0.02em'
  panel-title:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-type-control-lg)'
    fontWeight: 650
    lineHeight: 1.28
    letterSpacing: '0.012em'
  body:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-type-body)'
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: '-0.004em'
  label:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-type-label)'
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: '0.07em'
    textTransform: uppercase
  mono:
    fontFamily: 'JetBrains Mono, ui-monospace, Menlo, monospace'
    fontSize: '0.92em'
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: '0'
rounded:
  control: '6px'
  panel: '10px'
  pill: '999px'
spacing:
  shell-compact: '12px'
  shell-normal: '18px'
  shell-sparse: '22px'
  panel-x: '14px'
  panel-y: '10px'
  row-y: '11px'
  control-x: '12px'
  nav-tab-y: 'var(--nav-tab-pad-y)'
components:
  ui-control:
    backgroundColor: '{colors.panel-dark}'
    textColor: 'var(--color-fg-dim)'
    borderColor: '{colors.border-dark}'
    rounded: '{rounded.control}'
    padding: '6px 8px'
  ui-control-primary:
    backgroundColor: '{colors.accent-dark}'
    textColor: '{colors.accent-ink-dark}'
    rounded: '{rounded.control}'
    padding: '7px 12px'
  status-badge:
    backgroundColor: 'color-mix(in oklab, var(--status-color) 18%, var(--color-background))'
    textColor: 'var(--status-color)'
    borderColor: 'color-mix(in oklab, var(--status-color) 30%, var(--color-background))'
    rounded: '{rounded.pill}'
    padding: '1px 8px'
  panel-shell:
    backgroundColor: '{colors.panel-dark}'
    textColor: '{colors.foreground-dark}'
    borderColor: '{colors.border-dark}'
    rounded: '{rounded.panel}'
    padding: '10px 14px'
  native-select:
    backgroundColor: 'var(--color-surface)'
    textColor: '{colors.foreground-dark}'
    rounded: '{rounded.control}'
    padding: '0 10px'
  segmented-control:
    backgroundColor: 'var(--segmented-bg)'
    textColor: 'var(--segmented-fg)'
    rounded: '{rounded.pill}'
    padding: '2px'
---

## 1. Overview

**Creative North Star: "The Control Room Ledger"**

MeshLLM UI Preview is a private operations console for people turning heterogeneous local hardware into a legible inference mesh. It should feel compact, exact, and repeatable under pressure: an operator can inspect node health, model routing, VRAM allocation, API status, configuration, and chat movement without losing command of the system.

The product is not a consumer chat app, marketing analytics dashboard, or theatrical AI surface. It earns trust by making infrastructure governable: peers, endpoints, capacities, routes, warnings, and live packets stay readable at production density. Dark and light themes are both first-class operating modes, with the same hierarchy and seriousness.

**Key characteristics:**

- Dense, precise app chrome with readable hierarchy and minimal dead space.
- Flat bordered surfaces using tonal layers before shadows.
- One active accent at a time for focus, selection, routes, and decisive action.
- Machine values in mono so IDs, endpoints, memory, latency, and model names scan differently from prose.
- Useful motion only for state, routing, feedback, or live mesh activity, always respecting reduced motion.

## 2. Colors

The source of truth is `src/styles/globals.css`. Tokens are semantic CSS variables exposed through Tailwind `@theme`; use names like `background`, `foreground`, `panel`, `panel-strong`, `border`, `border-soft`, `accent`, `accent-contrast`, `accent-ink`, `accent-soft`, `good`, `warn`, and `bad` rather than hard-coded values in components.

### Primary

- **Live Circuit Accent** (`accent-dark`, `accent-light`): Focus outlines, active tabs, selected controls, primary actions, current routes, slider progress, and live packet signals. In dark mode the default reads cyan; in light mode it shifts toward blue for contrast.
- **Accent Contrast** (`accent-contrast-*`): Secondary accent only when paired with the primary accent, such as alternate route or model-family contrast. Never use it as a second decorative brand color.
- **Soft Route Tint** (`accent-soft-*`): Low-intensity selected fills, active row backgrounds, segmented control tint, and route context.

### Secondary

- **Capacity Green** (`good-*`): Online, serving, healthy, ready, successful operations, and available capacity.
- **Thermal Amber** (`warn-*`): Degraded, pending, warm, constrained, or needs-attention states.
- **Fault Red** (`bad-*`, `destructive`): Offline, failed, blocked, destructive, or unrecoverable states.

### Neutral

- **Console Field** (`background-*`): App background and deepest mesh canvas layer.
- **Panel Deck** (`panel-*`): Primary panels, drawers, sidebars, chat shell, and app sections.
- **Inset Deck** (`panel-strong-*`, `muted-*`): Nested rows, compact controls, selected-neutral states, and low-emphasis wells.
- **Hairline Border** (`border-*`): Default one-pixel structure for panels and controls.
- **Soft Divider** (`border-soft-*`): Row separators, top-nav border, table lines, and low-emphasis boundaries.
- **Primary Ink / Dim Ink / Faint Ink** (`foreground`, `fg-dim`, `fg-faint`): Text hierarchy. Do not lower opacity manually when an ink token exists.

### Preference accents and family colors

User accent preferences (`blue`, `cyan`, `violet`, `green`, `amber`, `pink`) are implemented with `data-accent`. They may change the hue of active controls and route signals, but they do not change the semantic status colors. Model family colors (`family-0` through `family-7`) identify model groups and must stay subordinate to status and active-route meaning.

**The One Live Signal Rule.** Accent color marks current action, focus, route, or live state. Do not spend it on decoration.

**The Three Surface Rule.** Ordinary depth uses only background, panel, and inset panel. If a fourth neutral is needed, simplify the hierarchy before adding another surface.

## 3. Typography

**Display/body font:** Inter Tight, then system fallbacks.  
**Machine font:** JetBrains Mono, then ui-monospace fallbacks.

The type system is sharp, compressed, and technical. Density modes (`compact`, `dense`, `normal`, `sparse`) adjust token sizes, shell padding, zoom, and nav dimensions, while `.density-shell` pins the preview scale to production-like pixel values. Components should use `type-*`, density, and mono utilities instead of inventing local text sizes.

### Hierarchy

- **Display** (`.type-display`, 20.5 to 22.5px in the shell): Rare page or route statements; 700 weight, 1.16 line-height, tight tracking.
- **Headline** (`.type-headline`, 16.5px): Node names, major shell labels, and high-value operational titles; 700 weight and tight tracking.
- **Panel title** (`.type-panel-title`, 13px): Panel headers, drawers, settings groups, and compact feature headers; 650 weight with slight positive tracking.
- **Body** (`.type-body`, 13.5px): Descriptive copy, row summaries, chat and configuration explanations; 1.55 line-height for dense readability.
- **Caption** (`.type-caption`, 12px): Secondary descriptions, helper text, status context, and compact UI explanations.
- **Label** (`.type-label`, 11px): Uppercase operational labels, table headers, tags, and small state headers; 600 weight and 0.07em tracking.
- **Machine** (`.type-machine`, `.mono`, `.font-mono`): Model IDs, peer IDs, endpoints, ports, memory, latency, percentages, timestamps, generated config, and checksums.

**The Machine Values Rule.** If a string is a model name, endpoint, ID, memory value, route, timestamp, generated config, or numeric telemetry, set it in mono with tabular numbers. If it is instruction, explanation, or navigation copy, keep it sans.

## 4. Elevation

MeshLLM is flat by default. Depth comes from borders, tonal surface changes, translucency, and compact grouping. Shadows are reserved for floating UI, live visual feedback, drag state, focus, and mesh visualization so ordinary panels remain stable and operational.

### Surface vocabulary

- **Panel shell** (`panel-shell`): Flat bordered panel with `panel` background and 10px radius. Use for cards, chat surfaces, settings sections, and model catalogs.
- **Chrome** (`surface-chrome`): Sticky translucent top navigation using `--surface-chrome` and 10px backdrop blur.
- **Overlay** (`surface-overlay`, `surface-menu-panel`, `surface-popover-panel`, `surface-floating-panel`): Menus, hover cards, drawers, command surfaces, and floating panels. These may use blur and popover/modal shadows.
- **Inset / selected / error shadows** (`--shadow-surface-inset`, `--shadow-surface-selected`, `--shadow-surface-error-inset`): Internal state emphasis, not generic elevation.

### Shadow vocabulary

- **Low surface** (`--shadow-surface-low`): Subtle stacked panel depth only when a component is visually separated from the shell.
- **Popover / modal / drawer** (`--shadow-surface-popover`, `--shadow-surface-modal`, `--shadow-surface-drawer`): Floating layers and blocking UI.
- **Focus accent** (`--shadow-focus-accent`): Complements the required 2px accent focus outline.
- **Hover / drag** (`--shadow-surface-hover`, `--shadow-surface-drag`): Temporary input feedback.
- **Status good** (`--shadow-status-good`): Reserved for positive live state; do not reuse for decorative glow.
- **Mesh signal glow** (`mesh-glow`, `mesh-packet`, `mesh-live-pulse`, `mesh-radar-ping`): Live network visualization only.

**The Flat Until Floating Rule.** Panels, rows, cards, and tables do not receive elevation at rest. Shadows mean an element is floating, selected, live, dragged, focused, or responding to input.

## 5. Components

### Buttons and controls

- **Base control** (`.ui-control`): 6px radius, hairline border, panel background, dim text. Hover shifts toward panel-strong with accent-mixed border and subtle glow. Active controls translate down 1px and receive accent tint.
- **Primary control** (`.ui-control-primary`): Accent fill with `accent-ink`; use for decisive actions and active primary navigation. Hover mixes accent with foreground and adds primary glow.
- **Ghost control** (`.ui-control-ghost`): Transparent at rest, panel-strong on hover, accent-tinted when active or selected.
- **Destructive control** (`.ui-control-destructive`): Fault red tint and border at rest; destructive fill with destructive foreground on hover.
- **Focus and disabled states:** All interactive controls use a visible 2px accent outline. Disabled controls retain layout and reduce opacity instead of changing shape.

### Status badges, chips, and segmented controls

- **StatusBadge:** Full pill, tiny dot optional, text label required. Tone backgrounds use `color-mix(... 18%)`, borders use `30%`, and muted tone uses `fg-faint` plus `border`.
- **SegmentedControl:** `pill` variant uses `.segmented-control` with 28px height and 2px padding; selected items use foreground or accent fill through `data-selected-tone='accent'`. `buttons` variant uses individual `.ui-control` buttons.
- **Preference chips:** Use density-aware sizing and selected-state shadows. They are controls, not badges.

### Cards, panels, and banners

- **Panel shell:** 10px radius, one-pixel border, panel background, soft dividers, no rest shadow.
- **InfoBanner:** Panel shell with 10px radius, 19px horizontal and 15px vertical padding, optional accent icon frame, status, and action. Its gradient is a functional accent-to-panel tint, not decoration.
- **ErrorBoundaryPanel:** Bad-tinted border/background with a retry control; pair failure color with text and action.
- **Empty states:** Compact operational guidance with a clear next action; avoid mascot or assistant personality.

### Inputs, selects, sliders, and forms

- **NativeSelect:** 32px high, 240px minimum width, 6px radius, `ui-control` treatment, monospace value text, and explicit focus outline.
- **Slider:** 8px track, 16px thumb, accent progress, hover/focus glow, active thumb scale, visible min/max labels, and `aria-valuetext`.
- **Text fields / text areas:** Background field, hairline border, compact body type, clear placeholder color, stable disabled state, and destructive state with text plus red tint.

### Navigation and layout

- **TopNav:** Sticky `surface-chrome` layer with soft bottom border, compact brand cluster, primary tabs, API status chip, and utility controls. Active tabs are primary controls, not underline-only tabs.
- **RootLayout:** Wrap app content in `.density-shell`; preserve density preferences and keep footer/top nav aligned to shell scale.
- **Mobile:** Preserve route access and API/status controls first. Truncate endpoints and metadata before hiding core navigation.

### Signature component: Mesh Canvas

The mesh canvas uses a 22px technical grid, sparse route glow, live packets, radar pings, and node/link motion. It is the only place where continuous glow is part of meaning. Motion constants are short and stateful: reclamp, join, leave, packet, and radar behavior. Reduced motion disables continuous pulses, pings, palette transitions, and will-change effects.

## 6. Do's and Don'ts

### Do:

- **Do** use semantic tokens from `src/styles/globals.css` as the source of truth.
- **Do** keep light and dark equally intentional; neither theme is a secondary skin.
- **Do** separate surfaces with background, panel, inset panel, borders, and dividers before adding shadows.
- **Do** reserve accent for active tabs, selected states, focus, route paths, live packets, sliders, and decisive actions.
- **Do** use mono type for model names, peer IDs, endpoints, ports, VRAM, latency, percentages, timestamps, and generated config.
- **Do** pair status color with text labels, icons, or dots so state is not communicated by color alone.
- **Do** respect `prefers-reduced-motion`; motion must clarify state or routing, not perform personality.
- **Do** build from existing primitives (`ui-control`, `panel-shell`, `StatusBadge`, `SegmentedControl`, `NativeSelect`, `Slider`) before inventing variants.

### Don't:

- **Don't** make the product feel like a generic SaaS dashboard with oversized whitespace and interchangeable cards.
- **Don't** drift into gaming-neon cyberpunk, holographic sci-fi, loud AI magic, or decorative glow.
- **Don't** use consumer chat tropes: oversized rounded bubbles, giant composers, assistant personality, or conversation-first mobile layouts.
- **Don't** use decorative gradients, glassmorphism, multicolor dashboards, or gradient text as brand expression.
- **Don't** make dark mode the only intentional theme or let light mode become a washed-out afterthought.
- **Don't** collapse into dull enterprise gray sameness; hierarchy must stay sharp and useful.
- **Don't** add side-stripe borders, repeated identical card grids, or modal-first flows when inline controls would be clearer.
- **Don't** invent local colors, font sizes, animation timings, or shadows when the global tokens already provide the contract.
