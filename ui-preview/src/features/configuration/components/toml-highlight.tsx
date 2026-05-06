import type { ReactNode } from 'react'

function highlightTomlLine(line: string): ReactNode {
  if (/^\[\[.+\]\]$/.test(line.trim())) return <span className="text-accent">{line}</span>
  const keyValue = line.match(/^(\w+)\s*=\s*(.+)$/)
  if (keyValue) {
    return (
      <>
        <span className="text-foreground">{keyValue[1]}</span>
        <span className="text-fg-dim">{' = '}</span>
        <span className="text-warn">{keyValue[2]}</span>
      </>
    )
  }
  return <span className="text-fg-dim">{line || ' '}</span>
}

export function HighlightedTomlLines({ toml }: { toml: string }) {
  const lineOccurrences = new Map<string, number>()

  return toml.split('\n').map((line) => {
    const occurrence = lineOccurrences.get(line) ?? 0
    lineOccurrences.set(line, occurrence + 1)

    return (
      <div className="relative text-transparent" key={`${line}-${occurrence}`}>
        {line || ' '}
        <span aria-hidden="true" className="pointer-events-none absolute left-0 top-0 whitespace-pre">
          {highlightTomlLine(line)}
        </span>
      </div>
    )
  })
}
