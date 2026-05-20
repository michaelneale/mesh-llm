import { Activity, ChevronDown, ChevronRight, Cloud, Network, Server, type LucideProps } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { StatusPill } from '@/components/ui/status-pill'
import { ReserveWakeStat } from '@/features/reserves/components/ReserveWakeStat'
import { formatReserveVram } from '@/features/reserves/lib/reserve-formatters'
import { getReserveStateMeta } from '@/features/reserves/lib/reserve-state'
import type { ReserveNodeState, ReserveProvider } from '@/features/reserves/lib/reserve-types'
import type { ReserveProviderTotals } from '@/features/reserves/lib/reserve-totals'
import { cn } from '@/lib/utils'

const PROVIDER_ROW_STATE_ORDER: ReserveNodeState[] = ['failed', 'unreachable', 'joining', 'waking', 'online', 'standby']

const PROVIDER_ROW_STATE_LABELS: Record<ReserveNodeState, string> = {
  failed: 'failed',
  unreachable: 'unreachable',
  joining: 'joining',
  waking: 'waking',
  online: 'online',
  standby: 'standby'
}

type ReserveProviderHeaderProps = {
  provider: ReserveProvider
  totals: ReserveProviderTotals
  expanded: boolean
  onToggle: () => void
  onWakeProvider?: () => void
}

export function ReserveProviderHeader({
  provider,
  totals,
  expanded,
  onToggle,
  onWakeProvider
}: ReserveProviderHeaderProps) {
  const standbyCount = totals.counts.standby
  const ToggleIcon = expanded ? ChevronDown : ChevronRight
  const allOnline = totals.totalNodes > 0 && totals.counts.online === totals.totalNodes
  const statusPills = PROVIDER_ROW_STATE_ORDER.map((state) => {
    const count = totals.counts[state]
    if (count === 0) return null
    const meta = getReserveStateMeta(state)

    return (
      <StatusPill
        key={state}
        className="h-[19px] px-1.5 text-[10.5px] font-medium"
        dot
        label={`${count} ${PROVIDER_ROW_STATE_LABELS[state]}`}
        tone={meta.tone}
      />
    )
  })

  return (
    <div className="px-[14px] py-[12px]">
      <div className="grid w-full grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-x-3 gap-y-2 sm:grid-cols-[minmax(0,1fr)_auto_auto_auto] lg:grid-cols-[minmax(0,1fr)_auto_auto_112px_auto_auto]">
        <button
          aria-expanded={expanded}
          className="flex min-w-0 items-center gap-3 text-left"
          data-testid="reserve-provider-toggle"
          onClick={onToggle}
          type="button"
        >
          <span
            className={cn(
              'inline-flex size-[28px] shrink-0 items-center justify-center rounded-[var(--radius)] border text-[11px] font-semibold',
              allOnline
                ? 'border-[color:color-mix(in_oklab,var(--color-good)_42%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-good)_15%,transparent)] text-[color:var(--color-good)]'
                : 'border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_10%,transparent)] text-[color:var(--color-accent)]'
            )}
          >
            <ReserveProviderIcon
              icon={provider.icon}
              fallback={provider.name.slice(0, 1)}
              className="size-3.5"
              aria-hidden="true"
            />
          </span>
          <span className="min-w-0 text-left">
            <span className="flex items-center gap-2 text-[13px] font-semibold leading-[1.05] text-foreground">
              <span className="truncate">{provider.name}</span>
              <Badge className="h-[19px] rounded-full px-[7px] py-0 text-[10.5px] font-medium leading-none text-fg-faint">
                {provider.kind}
              </Badge>
            </span>
            <span className="mt-[6px] block truncate text-[11px] leading-[1.2] tracking-[0.015em] text-fg-faint">
              {provider.region} · {provider.billing}
            </span>
          </span>
        </button>

        <div className="col-start-2 row-start-1 flex flex-col items-end gap-0.5 font-mono text-[10.5px] leading-none text-fg-dim sm:hidden">
          <span className="whitespace-nowrap text-foreground">{totals.totalNodes} nodes</span>
          <span className="whitespace-nowrap">{formatReserveVram(totals.totalVram)}</span>
        </div>
        <div className="col-start-1 row-start-2 flex min-w-0 flex-wrap items-center justify-start gap-1 overflow-hidden sm:col-span-4 lg:hidden">
          <div className="flex min-w-0 flex-wrap items-center justify-start gap-1 overflow-hidden">{statusPills}</div>
        </div>
        <div className="hidden min-w-0 flex-nowrap items-center justify-end gap-1 overflow-hidden pr-2 lg:col-start-2 lg:row-start-1 lg:flex">
          {statusPills}
        </div>
        <span
          aria-hidden="true"
          className="hidden h-[90%] w-px self-center justify-self-center bg-border lg:col-start-3 lg:row-start-1 lg:block"
        />
        <div className="hidden min-w-0 items-center justify-start gap-3 px-2 sm:col-start-2 sm:row-start-1 sm:flex lg:col-start-4">
          <ReserveWakeStat label="Nodes" value={totals.totalNodes} />
          <ReserveWakeStat label="VRAM" value={formatReserveVram(totals.totalVram)} />
        </div>
        <Button
          className="ui-control-primary col-span-2 col-start-2 row-start-2 ml-auto h-[24px] w-[102px] shrink-0 justify-center rounded-[var(--radius)] px-[10px] text-[11.5px] whitespace-nowrap sm:col-span-1 sm:col-start-3 sm:row-start-1 sm:ml-0 lg:col-start-5 lg:w-[96px] xl:w-[102px]"
          disabled={standbyCount === 0}
          onClick={onWakeProvider}
          size="sm"
          title={standbyCount === 0 ? 'No standby nodes are available to wake.' : undefined}
          type="button"
          variant="default"
        >
          <Activity className="size-[11px]" aria-hidden="true" />
          {standbyCount > 0 ? `Wake ${standbyCount}` : 'Wake provider'}
        </Button>
        <button
          aria-expanded={expanded}
          aria-label={expanded ? 'Collapse row' : 'Expand row'}
          className="col-start-3 row-start-1 inline-flex size-6 shrink-0 items-center justify-center justify-self-end rounded-[var(--radius)] text-fg-faint transition hover:bg-panel hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:col-start-4 lg:col-start-6"
          data-testid="reserve-provider-chevron-toggle"
          onClick={onToggle}
          type="button"
        >
          <ToggleIcon className="size-4" aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}

type ReserveProviderIconProps = LucideProps & {
  fallback: string
  icon: 'server' | 'cloud' | 'lan' | undefined
}

function ReserveProviderIcon({ fallback, icon, ...props }: ReserveProviderIconProps) {
  if (icon === 'cloud') return <Cloud {...props} />
  if (icon === 'server') return <Server {...props} />
  if (icon === 'lan') return <Network {...props} />
  return <>{fallback}</>
}
