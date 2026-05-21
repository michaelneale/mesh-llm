import { RESERVE_STATE_ORDER, getReserveStateMeta } from '@/features/reserves/lib/reserve-state'
import type { ReserveNodeState } from '@/features/reserves/lib/reserve-types'
import { cn } from '@/lib/utils'

export function ReserveStateLegend() {
  return (
    <div
      aria-label="Reserve node state legend"
      className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1"
      role="img"
    >
      {RESERVE_STATE_ORDER.map((state) => {
        const meta = getReserveStateMeta(state)
        return (
          <span
            key={state}
            className={cn(
              'inline-flex items-center gap-1.5 text-[11px] leading-none text-fg-dim',
              meta.hatched && 'text-fg-dim'
            )}
          >
            <span
              className={cn('size-[9px] rounded-[2px] border', reserveLegendSwatchClass(state, meta.hatched))}
              aria-hidden="true"
            />
            {meta.label}
          </span>
        )
      })}
    </div>
  )
}

function reserveLegendSwatchClass(state: ReserveNodeState, hatched: boolean | undefined) {
  if (hatched) {
    return 'border-[color:color-mix(in_oklab,var(--color-bad)_70%,transparent)] bg-[repeating-linear-gradient(135deg,color-mix(in_oklab,var(--color-bad)_55%,var(--color-panel-strong))_0_2px,color-mix(in_oklab,var(--color-bad)_18%,var(--color-panel-strong))_2px_4px)] text-[color:var(--color-bad)]'
  }
  if (state === 'standby') {
    return 'border-[color:color-mix(in_oklab,var(--color-fg-faint)_30%,transparent)] bg-[color:color-mix(in_oklab,var(--color-fg-faint)_14%,var(--color-panel-strong))] text-fg-faint'
  }
  if (state === 'waking') {
    return 'border-[color:color-mix(in_oklab,var(--color-warn)_70%,transparent)] bg-[color:color-mix(in_oklab,var(--color-warn)_60%,var(--color-panel-strong))] text-[color:var(--color-warn)] shadow-[0_0_0_2px_color-mix(in_oklab,var(--color-warn)_25%,transparent)]'
  }
  if (state === 'joining') {
    return 'border-[color:color-mix(in_oklab,var(--color-accent)_70%,transparent)] bg-[color:color-mix(in_oklab,var(--color-accent)_75%,var(--color-panel-strong))] text-[color:var(--color-accent)] shadow-[0_0_0_2px_color-mix(in_oklab,var(--color-accent)_25%,transparent)]'
  }
  if (state === 'online') {
    return 'border-[color:color-mix(in_oklab,var(--color-good)_70%,transparent)] bg-[color:color-mix(in_oklab,var(--color-good)_55%,var(--color-panel-strong))] text-[color:var(--color-good)]'
  }
  return 'border-[color:color-mix(in_oklab,var(--color-bad)_70%,transparent)] bg-[color:color-mix(in_oklab,var(--color-bad)_65%,var(--color-panel-strong))] text-[color:var(--color-bad)]'
}
