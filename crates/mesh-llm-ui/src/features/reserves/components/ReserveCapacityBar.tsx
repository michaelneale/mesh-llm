import { cn } from '@/lib/utils'
import { RESERVE_STATE_ORDER, getReserveStateMeta } from '@/features/reserves/lib/reserve-state'
import { formatReserveVram } from '@/features/reserves/lib/reserve-formatters'
import type { ReserveNodeState } from '@/features/reserves/lib/reserve-types'
import type { ReserveStateVram } from '@/features/reserves/lib/reserve-totals'

type ReserveCapacityBarProps = {
  allocation: ReserveStateVram
  className?: string
}

const CAPACITY_STATE_ORDER: ReserveNodeState[] = ['online', 'joining', 'waking', 'failed', 'unreachable', 'standby']

export function ReserveCapacityBar({ allocation, className }: ReserveCapacityBarProps) {
  const total = RESERVE_STATE_ORDER.reduce((sum, state) => sum + allocation[state], 0)

  if (total === 0) {
    return (
      <div
        aria-label="No reserve capacity by state"
        className={cn('h-[6px] rounded-full bg-panel-strong', className)}
        role="img"
      />
    )
  }

  return (
    <div
      aria-label="Reserve capacity by node state"
      className={cn('flex h-[6px] overflow-hidden rounded-full border border-border/60 bg-panel-strong', className)}
      role="img"
    >
      {CAPACITY_STATE_ORDER.map((state) => {
        const stateVram = allocation[state]
        if (stateVram === 0) return null
        const meta = getReserveStateMeta(state)
        return (
          <div
            key={state}
            className={cn(
              'h-full border-r border-background last:border-r-0',
              reserveCapacitySegmentClass(state, meta.hatched)
            )}
            style={{ width: `${(stateVram / total) * 100}%` }}
            title={`${meta.label}: ${formatReserveVram(stateVram)}`}
          />
        )
      })}
    </div>
  )
}

function reserveCapacitySegmentClass(state: ReserveNodeState, hatched: boolean | undefined) {
  if (hatched) {
    return 'bg-[repeating-linear-gradient(135deg,color-mix(in_oklab,var(--color-bad)_55%,var(--color-panel-strong))_0_2px,color-mix(in_oklab,var(--color-bad)_30%,var(--color-panel-strong))_2px_4px)]'
  }
  if (state === 'online') return 'bg-[color:color-mix(in_oklab,var(--color-good)_55%,var(--color-panel-strong))]'
  if (state === 'joining') return 'bg-[color:color-mix(in_oklab,var(--color-accent)_75%,var(--color-panel-strong))]'
  if (state === 'waking') return 'bg-[color:color-mix(in_oklab,var(--color-warn)_60%,var(--color-panel-strong))]'
  if (state === 'failed') return 'bg-[color:color-mix(in_oklab,var(--color-bad)_65%,var(--color-panel-strong))]'
  return 'bg-[color:color-mix(in_oklab,var(--color-fg-faint)_35%,var(--color-panel-strong))]'
}
