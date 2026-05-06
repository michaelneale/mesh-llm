---
name: MeshLLM App
description: A compact operations console for distributed local AI inference.
colors:
  background-dark: 'oklch(0.17 0.015 250)'
  foreground-dark: 'oklch(0.96 0.005 80)'
  panel-dark: 'oklch(0.20 0.018 250)'
  panel-strong-dark: 'oklch(0.23 0.020 250)'
  muted-dark: 'oklch(0.23 0.020 250)'
  border-dark: 'oklch(0.30 0.020 250 / 0.9)'
  border-soft-dark: 'oklch(0.30 0.020 250 / 0.45)'
  accent-dark: 'oklch(0.80 0.14 200)'
  accent-soft-dark: 'oklch(0.30 0.06 200)'
  good-dark: 'oklch(0.78 0.14 150)'
  warn-dark: 'oklch(0.80 0.12 80)'
  bad-dark: 'oklch(0.70 0.18 25)'
  background-light: 'oklch(0.985 0.003 80)'
  foreground-light: 'oklch(0.22 0.020 250)'
  panel-light: 'oklch(0.975 0.004 80)'
  panel-strong-light: 'oklch(0.955 0.005 80)'
  muted-light: 'oklch(0.955 0.005 80)'
  border-light: 'oklch(0.88 0.005 80)'
  border-soft-light: 'oklch(0.93 0.005 80)'
  accent-light: 'oklch(0.62 0.14 220)'
  accent-soft-light: 'oklch(0.92 0.05 198)'
  good-light: 'oklch(0.58 0.13 150)'
  warn-light: 'oklch(0.62 0.14 55)'
  bad-light: 'oklch(0.62 0.16 28)'
typography:
  headline:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-text-15)'
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: '-0.02em'
  title:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-text-12-5)'
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: '0.012em'
  body:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-font-size)'
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: '-0.005em'
  label:
    fontFamily: 'Inter Tight, -apple-system, system-ui, sans-serif'
    fontSize: 'var(--density-text-10-5)'
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: '0.06em'
  mono:
    fontFamily: 'JetBrains Mono, ui-monospace, Menlo, monospace'
    fontSize: '0.92em'
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: '0'
rounded:
  control: '6px'
  panel: '10px'
spacing:
  shell-x: '18px'
  shell-top: '18px'
  panel-x: '14px'
  panel-y: '10px'
  row-y: '11px'
  control-x: '12px'
components:
  button-primary-dark:
    backgroundColor: '{colors.accent-dark}'
    textColor: 'oklch(0.20 0.04 220)'
    rounded: '{rounded.control}'
    padding: '7px 12px'
  button-primary-light:
    backgroundColor: '{colors.accent-light}'
    textColor: 'oklch(0.98 0.01 220)'
    rounded: '{rounded.control}'
    padding: '7px 12px'
  button-ghost-dark:
    backgroundColor: 'transparent'
    textColor: '{colors.foreground-dark}'
    rounded: '{rounded.control}'
    padding: '6px 8px'
  button-ghost-light:
    backgroundColor: 'transparent'
    textColor: '{colors.foreground-light}'
    rounded: '{rounded.control}'
    padding: '6px 8px'
  status-pill-dark:
    backgroundColor: '{colors.panel-dark}'
    textColor: '{colors.good-dark}'
    rounded: '999px'
    padding: '1px 8px'
  status-pill-light:
    backgroundColor: '{colors.panel-light}'
    textColor: '{colors.good-light}'
    rounded: '999px'
    padding: '1px 8px'
  panel-dark:
    backgroundColor: '{colors.panel-dark}'
    textColor: '{colors.foreground-dark}'
    rounded: '{rounded.panel}'
    padding: '10px 14px'
  panel-light:
    backgroundColor: '{colors.panel-light}'
    textColor: '{colors.foreground-light}'
    rounded: '{rounded.panel}'
    padding: '10px 14px'
  input-dark:
    backgroundColor: '{colors.background-dark}'
    textColor: '{colors.foreground-dark}'
    rounded: '{rounded.control}'
    padding: '8px 12px'
  input-light:
    backgroundColor: '{colors.background-light}'
    textColor: '{colors.foreground-light}'
    rounded: '{rounded.control}'
    padding: '8px 12px'
---

# Design System: MeshLLM App

## 1. Overview

**Creative North Star: "The Control Room Ledger"**

MeshLLM should feel like a private operations room for local AI infrastructure: compact, exact, and built for repeated use under pressure. The product is not trying to charm users with assistant personality; it earns trust by making peers, models, routes, memory, and configuration visibly governable.

The physical scene is an operator using a 27-inch display in a home lab, office, or server room, switching between live mesh status and configuration changes while keeping enough context on screen to avoid mistakes. That scene requires both dark and light themes to be real operating modes: dark for low-light monitoring, light for daytime configuration and review.

The system rejects generic SaaS dashboards, gaming-neon cyberpunk, consumer chat softness, and theatrical AI magic. It should be minimal but not empty, dense but not cluttered, technical but not hostile.

**Key Characteristics:**

- Compact app density with readable hierarchy.
- Flat bordered panels separated by tonal layers, not heavy shadows.
- Sparse, meaningful accent use for selection, focus, routes, and live signals.
- Mono metadata for IDs, model names, endpoints, timestamps, and capacities.
- Light and dark themes with equal visual authority.

## 2. Colors

The palette is a restrained operational neutral system with one active accent at a time. The accent can vary by user preference, but it must remain a control signal, not decoration.

### Primary

- **Live Circuit Cyan** (`accent-dark`, `accent-light`): Used for active tabs, primary actions, selected rows, route paths, focus outlines, and live mesh signals. It is sparse by design.
- **Soft Route Tint** (`accent-soft-dark`, `accent-soft-light`): Used for selected row backgrounds, quiet active fills, and low-intensity route context.

### Secondary

- **Capacity Green** (`good-dark`, `good-light`): Used for serving, ready, online, healthy capacity, and successful state markers.
- **Thermal Amber** (`warn-dark`, `warn-light`): Used for warning states that need operator attention without implying failure.
- **Fault Red** (`bad-dark`, `bad-light`): Used for destructive actions, offline states, and failed health conditions.

### Neutral

- **Console Field** (`background-dark`, `background-light`): The app background and deepest canvas layer.
- **Panel Deck** (`panel-dark`, `panel-light`): Primary panels, sidebars, and app sections.
- **Inset Deck** (`panel-strong-dark`, `panel-strong-light`): Inset controls, denser surfaces, and nested operational rows.
- **Hairline Border** (`border-dark`, `border-light`): Primary separation for panels and controls.
- **Soft Divider** (`border-soft-dark`, `border-soft-light`): Row separators, table dividers, and low-emphasis boundaries.
- **Primary Ink** (`foreground-dark`, `foreground-light`): Main UI text.

### Named Rules

**The One Live Signal Rule.** Accent color marks current action, focus, route, or live state. Do not spend it on decoration.

**The Three Surface Rule.** Use only app background, panel, and inset panel for ordinary depth. If a fourth neutral appears, simplify the hierarchy first.

## 3. Typography

**Display Font:** Inter Tight, system fallback  
**Body Font:** Inter Tight, system fallback  
**Label/Mono Font:** JetBrains Mono, ui-monospace fallback

**Character:** The type system is sharp, compressed, and technical. Inter Tight keeps dense controls legible without becoming soft; JetBrains Mono separates machine values from prose.

### Hierarchy

- **Display** (700, 20 to 22px, 1.2): Rare route headings, hero labels, and configuration section anchors. Keep it compact.
- **Headline** (700, 15 to 16px, 1.2): Node names, major shell labels, and high-value operational titles.
- **Title** (600, 12 to 13.5px, 1.35): Panel headers, drawer titles, section labels, and control groups.
- **Body** (400 to 500, 12 to 14px by density, 1.45): General UI copy, row descriptions, and status text. Cap explanatory passages at 65 to 75 characters.
- **Label** (500 to 600, 9.5 to 11.5px, tracked): Table headers, pills, metadata labels, and compact state labels. Uppercase is allowed for operational metadata.
- **Mono** (500, 0.92em, 1.4): Model IDs, peer IDs, endpoints, VRAM values, latency, percentages, and generated config snippets.

### Named Rules

**The Machine Values Rule.** If a string is a model name, endpoint, ID, memory value, route, or timestamp, set it in mono. If it is instruction or explanation, keep it sans.

## 4. Elevation

MeshLLM is flat by default. Depth comes from borders, tonal surface changes, and compact grouping. Shadows are reserved for overlays and live visualization effects so that ordinary panels stay stable and operational.

### Shadow Vocabulary

- **Control Glow** (`0 0 12px color-mix(in oklab, var(--color-accent) 10%, transparent)`): A hover response for controls that can act immediately.
- **Primary Action Glow** (`0 0 14px color-mix(in oklab, var(--color-accent) 18%, transparent)`): A stronger hover response for primary actions.
- **Overlay Lift** (`0 30px 90px rgba(0,0,0,0.55)`): Command palette and modal-level surfaces only.
- **Mesh Signal Glow** (`drop-shadow(0 0 6px ...)` plus `drop-shadow(0 0 14px ...)`): Live network paths and packet markers only.

### Named Rules

**The Flat Until Floating Rule.** Panels, rows, cards, and tables do not receive elevation at rest. Shadows mean an element is floating, live, or responding to input.

## 5. Components

### Buttons

- **Shape:** Tight technical radius (6px), or full pills only for compact action chips like “Add model.”
- **Primary:** Accent-filled with dark accent ink, used for current tabs and decisive actions. Keep padding compact, usually 7px to 12px horizontally.
- **Hover / Focus:** Hover uses a subtle glow and stronger accent mix. Focus uses a 2px accent outline with 3px offset.
- **Secondary / Ghost:** Transparent at rest, panel-strong fill on hover, and accent-tinted fill for pressed or active state.

### Chips

- **Style:** Rounded full, small type, thin border, and either neutral background or state-tinted background.
- **State:** Ready and serving use Capacity Green with a dot plus label. Selected role chips use accent tint and explicit text.

### Cards / Containers

- **Corner Style:** Panels use restrained rounded corners (10px) and controls use 6px.
- **Background:** Primary panels use Panel Deck; nested rows and compact controls use Inset Deck or Console Field.
- **Shadow Strategy:** No shadow at rest. Use borders and dividers for structure.
- **Border:** One-pixel Hairline Border for panel shells; Soft Divider for rows.
- **Internal Padding:** 14px horizontal and 10px vertical for headers; rows use 11px vertical for dense scanability.

### Inputs / Fields

- **Style:** Console Field background, Hairline Border, 6px radius, 12px horizontal padding, compact body type.
- **Focus:** 2px accent outline with 3px offset. Do not rely on border color alone.
- **Error / Disabled:** Destructive states use Fault Red tint plus text. Disabled controls reduce opacity and keep shape stable.

### Navigation

- **Style:** Sticky top bar with soft border, translucent panel background, compact brand cluster, primary tabs, API status chip, and utility icon controls.
- **Active State:** Active tabs are primary controls, not underlines. Inactive tabs are ghost controls.
- **Mobile Treatment:** Preserve route access and core utility actions first; truncate endpoint and metadata before hiding navigation.

### Signature Component: Mesh Canvas

The mesh canvas uses a faint technical grid, sparse route glow, and live packet markers. It is the one place where glow is part of meaning. Keep the effect subtle enough that tables and controls remain the primary operating surface.

## 6. Do's and Don'ts

### Do:

- **Do** use OKLCH tokens from `src/styles/globals.css` as the source of truth.
- **Do** separate surfaces with borders, dividers, and tonal shifts before adding shadows.
- **Do** keep accent use sparse and meaningful: active tabs, selected rows, focus states, routing paths, live indicators, and primary actions.
- **Do** use mono type for model names, peer IDs, endpoints, VRAM values, latency, percentages, and generated config.
- **Do** support light and dark as equal themes. Both must preserve precision, density, and composure.
- **Do** pair status color with text labels or dots so meaning survives color-blind and low-contrast conditions.

### Don't:

- **Don't** make the product feel like a generic SaaS dashboard.
- **Don't** drift into gaming-neon cyberpunk, holographic sci-fi, or loud AI magic styling.
- **Don't** use oversized rounded chat bubbles, giant composers, or a conversation-first mobile messenger feel.
- **Don't** rely on bright gradients, multicolor accents, oversized spacing, or decorative glassmorphism.
- **Don't** make dark mode feel like the only intentional theme.
- **Don't** make capability disappear into dull enterprise software with gray sameness and weak hierarchy.
- **Don't** use theatrical motion to imply intelligence; motion must clarify state, routing, or feedback.
- **Don't** add side-stripe borders, gradient text, repeated identical card grids, or modal-first flows.
