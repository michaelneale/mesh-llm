import { ShieldAlert } from 'lucide-react'
import { MetaPill } from '@/features/configuration/components/MetaPill'
import { formatGB } from '@/features/configuration/lib/config-display'

type ReservedConfigCardProps = {
  locationLabel: string
  reservedGB: number
}

export function ReservedConfigCard({ locationLabel, reservedGB }: ReservedConfigCardProps) {
  return (
    <article
      className="shadow-surface-inset mt-2 select-none rounded-[var(--radius-lg)] border border-border-soft bg-panel px-5 py-4"
      data-config-selection-area="true"
    >
      <div className="flex flex-wrap items-start gap-3">
        <span
          className="grid size-9 shrink-0 place-items-center rounded-[var(--radius)] border border-border-soft bg-background text-fg-dim"
          aria-hidden="true"
        >
          <ShieldAlert className="size-4" strokeWidth={1.8} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[length:var(--density-type-body)] font-semibold">System reserved space</h3>
            <MetaPill size="annotation">{formatGB(reservedGB)} GB</MetaPill>
            <span className="ml-auto text-[length:var(--density-type-caption)] text-fg-dim">
              on <span className="font-mono text-fg">{locationLabel}</span>
            </span>
          </div>
          <p className="mt-2 max-w-[72ch] text-[length:var(--density-type-control)] leading-relaxed text-fg-dim">
            This VRAM is held back for drivers, display overhead, and runtime safety margin. It is invariant system
            reserved space and has no configurable settings.
          </p>
        </div>
      </div>
    </article>
  )
}
