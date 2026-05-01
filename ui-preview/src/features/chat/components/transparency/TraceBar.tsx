import type { TraceSegment } from '@/features/app-tabs/types'

function segColor(tone: TraceSegment['tone']): string {
  if (tone === 'good') return 'var(--color-accent)'
  if (tone === 'warn') return 'var(--color-warn)'
  if (tone === 'bad') return 'var(--color-bad)'
  return 'color-mix(in oklab, var(--color-fg-faint) 50%, var(--color-panel-strong))'
}

export function TraceBar({ segs }: { segs: TraceSegment[] }) {
  const total = segs.reduce((sum, seg) => sum + seg.ms, 0)
  if (segs.length === 0 || total === 0) return <div className="rounded-[var(--radius)] border border-border p-3 text-[length:var(--density-type-annotation)] text-fg-faint">No trace segments recorded.</div>
  return (
    <div>
      <div className="flex h-2 overflow-hidden rounded-[var(--radius)] border border-border-soft bg-panel-strong">
        {segs.map((seg) => (
          <div key={seg.id} style={{ width: `${(seg.ms / total) * 100}%`, background: segColor(seg.tone), opacity: 0.85 }} />
        ))}
      </div>
      <div className="mt-2.5 grid gap-y-[5px]" style={{ gridTemplateColumns: '1fr auto', columnGap: 8 }}>
        {segs.map((seg) => (
          <div key={seg.id} className="contents">
            <div className="flex items-center gap-1.5 text-[length:var(--density-type-caption-lg)] text-fg-dim">
              <span className="size-2 rounded-sm" style={{ background: segColor(seg.tone) }} />
              {seg.label}
            </div>
            <span className="text-right font-mono text-[length:var(--density-type-caption)] text-fg-dim">{seg.ms} ms</span>
          </div>
        ))}
      </div>
    </div>
  )
}
