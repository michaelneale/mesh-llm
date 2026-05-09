# mesh-llm-ui

React/Vite UI crate for the Mesh LLM console.

## Development Process

Agressively use TODO lists to track the work you need to do. After each item is completed, sync the list so that it stays up to date.

## Project setup

Requires Node >= 24.

From repo root:

- `just ui-dev`
- `just ui-test`
- `just clean-ui`
- `scripts/build-ui.sh crates/mesh-llm-ui`

Package-local scripts:

- `npm run dev`
- `npm run typecheck`
- `npm run build`
- `npm run test`
- `npm run test:watch`

## Stack

- React 18
- Vite
- TypeScript
- Tailwind
- Radix UI primitives
- lucide-react icons
- Vitest + Testing Library + jsdom

## Source layout

- `src/components/ui/` contains reusable UI primitives.
- `src/features/app-shell/` owns shell, routing, command bar, status stream, status helpers, and topology types.
- `src/features/dashboard/` owns mesh/network dashboard views.
- `src/features/chat/` owns chat UI, composer, message rendering, attachments, and chat persistence.
- Keep feature-specific logic inside the relevant `features/*` folder.
- Put reusable pure helpers in `lib` files and test them directly.

## Imports

- Do not use deep relative parent imports such as:
  - `../../thing`
  - `../../../components/foo`
  - `../../../../lib/bar`
- Prefer configured path aliases:
  - `@/components/ui/button`
  - `@/features/chat/lib/storage`
  - `@/features/app-shell/lib/status`
- Same-directory relative imports are fine:
  - `./types`
  - `./constants`
  - `./helpers`
- Avoid crossing feature boundaries with relative imports.
- When moving code between features, update imports to aliases instead of increasing `../` depth.
- Shared code must live behind stable alias paths instead of feature-crossing relative traversals.

## TSX style

- Prefer function components.
- Use explicit prop object types for exported components.
- Keep derived data in `useMemo` when it is non-trivial or computed from status/model lists.
- Keep callbacks stable with `useCallback` when passed into stateful child components.
- Prefer small pure helper functions near the component when they only serve that file.
- Use `cn(...)` for conditional Tailwind class composition.
- Preserve dark-mode classes when changing UI.
- Avoid introducing new global state unless the app shell truly owns it.
- Keep dev-only features behind `import.meta.env.DEV`; do not ship playground-only code in production bundles.

## UI conventions

- Reuse existing `components/ui` primitives before creating new widgets.
- Use cards, badges, sheets, selects, tables, scroll areas, tooltips, and alerts consistently with existing dashboard/chat code.
- Prefer compact status labels and useful tooltip text over long inline explanations.
- Keep public/demo warnings visible but not noisy.
- Use accessible labels for icon-only buttons.
- Preserve keyboard behavior such as Escape-to-close for fullscreen or modal-like states.
- Build components for elements that can or will be shared across areas of the site, and refactor existing things when they can use a newly built component.

## Icons

Use `lucide-react` for generic UI/status icons. When looking for new icons, analyze the keywords of what you are adding an icon for, and search for a similarly matched icon.

Do not use lucide for brand icons. GitHub icons are local SVG assets.

## Testing

Use Vitest.

Good tests should:

- Prefer pure helper tests for routing, status normalization, model labels, attachment parsing, and storage behavior.
- Test user-visible behavior rather than implementation details for React components.
- Avoid snapshots for complex UI unless the snapshot is intentionally small.
- Cover edge cases: missing status, empty peers, client nodes, warm/cold models, malformed attachment data, and localStorage failures.
- Mock browser APIs explicitly when needed: `localStorage`, `matchMedia`, `FileReader`, canvas/image APIs, clipboard, and PDF/image extraction paths.
- Keep tests deterministic; avoid relying on real timers unless using fake timers.
- Add regression tests for any bug fix before changing behavior.

Before finishing UI changes, run:

- `npm run typecheck`
- `npm run test`
- `npm run build`

Or from repo root:

- `just ui-test`