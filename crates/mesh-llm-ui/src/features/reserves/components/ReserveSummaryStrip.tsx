import { ReserveCapacityBar } from '@/features/reserves/components/ReserveCapacityBar'
import { ReserveSummaryCell } from '@/features/reserves/components/ReserveSummaryCell'
import { formatReserveEta, formatReserveVram } from '@/features/reserves/lib/reserve-formatters'
import type { ReserveFleetTotals } from '@/features/reserves/lib/reserve-totals'

type ReserveSummaryStripProps = {
  totals: ReserveFleetTotals
  liveMeshVramGB?: number
  autoWakeEnabled?: boolean
  policyLabel?: string
}

export function ReserveSummaryStrip({
  totals,
  liveMeshVramGB,
  autoWakeEnabled = true,
  policyLabel
}: ReserveSummaryStripProps) {
  const etaLabel = totals.longestEta == null ? '-' : formatReserveEta(totals.longestEta)
  const liveMeshLabel =
    liveMeshVramGB == null ? 'Live mesh VRAM pending' : `vs ${Math.round(liveMeshVramGB)} GB on the live mesh`

  return (
    <div className="border-b border-border/60 bg-[color-mix(in_oklab,var(--color-panel-strong)_50%,var(--color-panel))]">
      <div className="grid grid-cols-[repeat(auto-fit,minmax(148px,1fr))] gap-0 lg:grid-cols-[auto_auto_auto_auto_minmax(220px,1fr)]">
        <ReserveSummaryCell
          label="Managed nodes"
          value={totals.totalNodes}
          subLabel={`${totals.onlineNodes} online · ${totals.reserveNodes} reserve`}
          mono
        />
        <ReserveSummaryCell
          label="Combined VRAM"
          value={formatReserveVram(totals.totalVram)}
          subLabel={liveMeshLabel}
          mono
        />
        <ReserveSummaryCell label="Until ready" value={etaLabel} subLabel="longest in-flight wake" mono />
        <div className="hidden min-w-0 flex-col justify-center whitespace-nowrap border-b border-r border-border/60 px-[20px] py-3 sm:flex sm:px-[20px] lg:border-b-0 lg:px-[20px]">
          <div className="text-[10.5px] font-semibold uppercase leading-none tracking-[0.08em] text-fg-faint">
            Status
          </div>
          <div className="mt-1 flex flex-col gap-1 text-[11px] font-normal leading-none tracking-normal text-fg-faint">
            <span className="inline-flex items-center gap-1.5">
              <span className="size-1.5 rounded-full bg-[color:var(--color-accent)]" aria-hidden="true" />
              <span className="font-mono text-[13px] font-medium text-foreground">{totals.counts.joining}</span>
              joining
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="size-1.5 rounded-full bg-[color:var(--color-warn)]" aria-hidden="true" />
              <span className="font-mono text-[13px] font-medium text-foreground">{totals.counts.waking}</span>
              waking
            </span>
            <span className="inline-flex items-center gap-1.5 text-[color:var(--color-bad)]">
              <span className="size-1.5 rounded-full bg-current" aria-hidden="true" />
              <span className="font-mono text-[13px] font-medium">{totals.errorNodes}</span>
              <span className="text-[color:var(--color-bad)]">errors</span>
            </span>
          </div>
        </div>
        <div className="col-span-full flex min-w-0 flex-col justify-center gap-1.5 px-[20px] py-3 sm:px-[20px] lg:col-span-1 lg:min-w-[220px] lg:px-[20px]">
          <div className="flex items-center justify-between gap-4 text-[10px] font-semibold uppercase leading-none tracking-[0.08em] text-fg-faint">
            <span>Capacity by state</span>
            <span className="font-mono font-medium normal-case tracking-normal text-fg-dim">
              {formatReserveVram(totals.totalVram)}
            </span>
          </div>
          <ReserveCapacityBar allocation={totals.vramByState} />
          <div className="flex items-center justify-between gap-4 text-[10.5px] leading-none text-fg-faint">
            <span className="inline-flex items-center gap-1.5 font-mono">
              auto-wake
              {autoWakeEnabled ? (
                <span className="inline-flex items-center gap-1 rounded-full border border-[color:color-mix(in_oklab,var(--color-good)_38%,transparent)] bg-[color:color-mix(in_oklab,var(--color-good)_18%,transparent)] px-1.5 py-0.5 text-[10px] font-medium leading-none text-[color:var(--color-good)]">
                  <span className="size-1 rounded-full bg-current" aria-hidden="true" />
                  on
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 rounded-full border border-[color:color-mix(in_oklab,var(--color-fg-faint)_30%,transparent)] bg-[color:color-mix(in_oklab,var(--color-fg-faint)_10%,transparent)] px-1.5 py-0.5 text-[10px] font-medium leading-none text-fg-faint">
                  <span className="size-1 rounded-full bg-current" aria-hidden="true" />
                  paused
                </span>
              )}
            </span>
            {policyLabel ? <span className="font-mono">{policyLabel}</span> : null}
          </div>
        </div>
      </div>
    </div>
  )
}
