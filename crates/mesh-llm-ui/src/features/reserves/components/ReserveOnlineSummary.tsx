import type { ReserveNode } from '@/features/reserves/lib/reserve-types'
import { formatReserveVram } from '@/features/reserves/lib/reserve-formatters'

type ReserveOnlineSummaryProps = {
  nodes: ReserveNode[]
}

export function ReserveOnlineSummary({ nodes }: ReserveOnlineSummaryProps) {
  if (nodes.length === 0) return null

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="mr-1 inline-flex items-center gap-1.5 py-[3px] text-[10px] font-semibold uppercase leading-none tracking-[0.06em] text-[color:var(--color-good)]">
        <span
          className="size-[5px] rounded-full bg-[color:var(--color-good)] shadow-[0_0_6px_var(--color-good)]"
          aria-hidden="true"
        />
        {nodes.length} online
      </span>
      {nodes.map((node) => (
        <span
          key={node.id}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-[color:color-mix(in_oklab,var(--color-good)_25%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-good)_6%,var(--color-panel))] px-2 py-[3px] font-mono text-[10.5px] leading-none text-fg-dim"
        >
          <span
            className="size-[5px] rounded-full bg-[color:var(--color-good)] shadow-[0_0_6px_var(--color-good)]"
            aria-hidden="true"
          />
          {node.id}
          <span className="text-fg-faint">·</span>
          <span>{formatReserveVram(node.vram)}</span>
          {node.since ? (
            <>
              <span className="text-fg-faint">·</span>
              <span className="text-fg-faint">up {node.since}</span>
            </>
          ) : null}
        </span>
      ))}
    </div>
  )
}
