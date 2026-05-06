type KeyboardLegendKeyToken = { type: 'key'; label: string; ariaLabel?: string; key?: string }
type KeyboardLegendSeparatorToken = { type: 'separator'; label: '+' | '/'; key?: string }
type KeyboardLegendToken = KeyboardLegendKeyToken | KeyboardLegendSeparatorToken
type KeyboardLegendItem = { label: string; text?: string; shortcut: readonly KeyboardLegendToken[] }
type KeyboardLegendGroup = { title: string; items: readonly KeyboardLegendItem[] }

const KEYBOARD_LEGEND_KBD_CLASS_NAME =
  'inline-flex h-5 min-w-[20px] items-center justify-center rounded-[var(--radius)] border border-border-soft bg-background px-1.5 font-mono text-[length:var(--density-type-label)] leading-none text-foreground'
const KEYBOARD_LEGEND_SYMBOLS: Record<string, { label: string; ariaLabel: string }> = {
  Alt: { label: '⌥', ariaLabel: 'Alt' },
  Ctrl: { label: '⌃', ariaLabel: 'Control' },
  Delete: { label: '⌫', ariaLabel: 'Delete' },
  Shift: { label: '⇧', ariaLabel: 'Shift' },
  Space: { label: '␣', ariaLabel: 'Space' },
  Tab: { label: '⇥', ariaLabel: 'Tab' }
}
const KEYBOARD_LEGEND_GROUPS: readonly KeyboardLegendGroup[] = [
  {
    title: 'Navigate',
    items: [
      { label: 'Nodes', shortcut: [{ type: 'key', label: 'Tab' }] },
      {
        label: 'GPU slots',
        text: '↑/↓',
        shortcut: [
          { type: 'key', label: '↑', ariaLabel: 'Up arrow' },
          { type: 'separator', label: '/' },
          { type: 'key', label: '↓', ariaLabel: 'Down arrow' }
        ]
      },
      {
        label: 'Select Model',
        text: '←/→',
        shortcut: [
          { type: 'key', label: '←', ariaLabel: 'Left arrow' },
          { type: 'separator', label: '/' },
          { type: 'key', label: '→', ariaLabel: 'Right arrow' }
        ]
      },
      {
        label: 'First/Last Model',
        text: 'Shift+←/→',
        shortcut: [
          { type: 'key', label: 'Shift' },
          { type: 'separator', label: '+' },
          { type: 'key', label: '←', ariaLabel: 'Left arrow' },
          { type: 'separator', label: '/' },
          { type: 'key', label: '→', ariaLabel: 'Right arrow' }
        ]
      },
      { label: 'Toggle Section', shortcut: [{ type: 'key', label: 'Space' }] }
    ]
  },
  {
    title: 'Selected Model',
    items: [
      {
        label: 'Adjust Context',
        text: 'Alt+←/→',
        shortcut: [
          { type: 'key', label: 'Alt' },
          { type: 'separator', label: '+' },
          { type: 'key', label: '←', ariaLabel: 'Left arrow' },
          { type: 'separator', label: '/' },
          { type: 'key', label: '→', ariaLabel: 'Right arrow' }
        ]
      },
      {
        label: 'Jump Context',
        text: 'Alt+Shift+←/→',
        shortcut: [
          { type: 'key', label: 'Alt' },
          { type: 'separator', label: '+', key: 'alt-shift-plus' },
          { type: 'key', label: 'Shift' },
          { type: 'separator', label: '+', key: 'shift-arrow-plus' },
          { type: 'key', label: '←', ariaLabel: 'Left arrow' },
          { type: 'separator', label: '/' },
          { type: 'key', label: '→', ariaLabel: 'Right arrow' }
        ]
      },
      {
        label: 'Move GPU',
        text: 'Shift+↑/↓',
        shortcut: [
          { type: 'key', label: 'Shift' },
          { type: 'separator', label: '+' },
          { type: 'key', label: '↑', ariaLabel: 'Up arrow' },
          { type: 'separator', label: '/' },
          { type: 'key', label: '↓', ariaLabel: 'Down arrow' }
        ]
      },
      {
        label: 'Toggle Placement',
        text: 'P/S',
        shortcut: [
          { type: 'key', label: 'P' },
          { type: 'separator', label: '/' },
          { type: 'key', label: 'S' }
        ]
      },
      { label: 'Selected model', shortcut: [{ type: 'key', label: 'Delete' }] }
    ]
  },
  {
    title: 'Actions',
    items: [
      { label: 'Add model', shortcut: [{ type: 'key', label: 'A' }] },
      {
        label: 'Undo',
        text: 'Ctrl+Z',
        shortcut: [
          { type: 'key', label: 'Ctrl' },
          { type: 'separator', label: '+' },
          { type: 'key', label: 'Z' }
        ]
      },
      {
        label: 'Redo',
        text: 'Ctrl+R',
        shortcut: [
          { type: 'key', label: 'Ctrl' },
          { type: 'separator', label: '+' },
          { type: 'key', label: 'R' }
        ]
      },
      {
        label: 'Save config',
        text: 'Ctrl+S',
        shortcut: [
          { type: 'key', label: 'Ctrl' },
          { type: 'separator', label: '+' },
          { type: 'key', label: 'S' }
        ]
      },
      {
        label: 'Revert',
        text: 'Ctrl+X',
        shortcut: [
          { type: 'key', label: 'Ctrl' },
          { type: 'separator', label: '+' },
          { type: 'key', label: 'X' }
        ]
      }
    ]
  }
] as const

function getKeyboardLegendKey(token: KeyboardLegendKeyToken) {
  const symbol = KEYBOARD_LEGEND_SYMBOLS[token.label]

  return {
    ariaLabel: token.ariaLabel ?? symbol?.ariaLabel,
    label: symbol?.label ?? token.label
  }
}

function KeyboardShortcutCluster({ shortcut }: { shortcut: readonly KeyboardLegendToken[] }) {
  return (
    <span className="inline-flex items-center justify-end gap-1 whitespace-nowrap text-fg-faint">
      {shortcut.map((token) => {
        if (token.type === 'key') {
          const key = getKeyboardLegendKey(token)

          return (
            <kbd
              key={token.key ?? `${token.type}-${token.label}-${token.ariaLabel}`}
              aria-label={key.ariaLabel}
              className={KEYBOARD_LEGEND_KBD_CLASS_NAME}
            >
              {key.label}
            </kbd>
          )
        }

        return (
          <span
            key={token.key ?? `${token.type}-${token.label}`}
            aria-hidden="true"
            className="font-mono text-[length:var(--density-type-micro)] leading-none text-fg-faint"
          >
            {token.label}
          </span>
        )
      })}
    </span>
  )
}

export function KeyboardLegend() {
  return (
    <section aria-label="Keyboard shortcuts" className="space-y-2.5">
      {KEYBOARD_LEGEND_GROUPS.map((group) => (
        <div className="space-y-1.5" key={group.title}>
          <div className="text-[length:var(--density-type-annotation)] font-semibold uppercase tracking-[0.06em] text-foreground">
            {group.title}
          </div>
          <dl className="space-y-1">
            {group.items.map((item) => (
              <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-2" key={item.label}>
                <dt className="min-w-0 truncate text-[length:var(--density-type-caption)] text-fg-dim">{item.label}</dt>
                <dd className="m-0 flex justify-end">
                  {item.text ? <span className="sr-only">{item.text}</span> : null}
                  <KeyboardShortcutCluster shortcut={item.shortcut} />
                </dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </section>
  )
}
