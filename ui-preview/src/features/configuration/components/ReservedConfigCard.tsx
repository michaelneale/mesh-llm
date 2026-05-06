import { ShieldAlert } from 'lucide-react'

type ReservedConfigCardProps = {
  locationLabel: string
  reservedGB: number
}

function formatGB(value: number) {
  return Number.isInteger(value) ? value.toString() : value.toFixed(1)
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
            <span className="rounded-full border border-border-soft bg-background px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] text-fg-dim">
              {formatGB(reservedGB)} GB
            </span>
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
