import type { StatusPillTone } from '@/components/ui/status-pill'
import type { ReserveNodeState } from '@/features/reserves/lib/reserve-types'

export type ReserveStateMeta = {
  label: string
  tone: StatusPillTone
  dotClassName: string
  hatched?: boolean
  pulse?: boolean
}

export const RESERVE_STATE_ORDER: ReserveNodeState[] = [
  'standby',
  'waking',
  'joining',
  'online',
  'failed',
  'unreachable'
]

const RESERVE_STATE_META: Record<ReserveNodeState, ReserveStateMeta> = {
  standby: {
    label: 'Standby',
    tone: 'neutral',
    dotClassName: 'border-border/70 bg-panel text-fg-faint'
  },
  waking: {
    label: 'Waking',
    tone: 'warn',
    dotClassName:
      'border-[color:color-mix(in_oklab,var(--color-warn)_42%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-warn)_16%,transparent)] text-[color:var(--color-warn)]',
    pulse: true
  },
  joining: {
    label: 'Joining',
    tone: 'info',
    dotClassName:
      'border-[color:color-mix(in_oklab,var(--color-accent)_40%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_14%,transparent)] text-[color:var(--color-accent)]',
    pulse: true
  },
  online: {
    label: 'Online',
    tone: 'good',
    dotClassName:
      'border-[color:color-mix(in_oklab,var(--color-good)_40%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-good)_14%,transparent)] text-[color:var(--color-good)]'
  },
  failed: {
    label: 'Wake failed',
    tone: 'bad',
    dotClassName:
      'border-[color:color-mix(in_oklab,var(--color-bad)_42%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_14%,transparent)] text-[color:var(--color-bad)]'
  },
  unreachable: {
    label: 'Unreachable',
    tone: 'bad',
    dotClassName:
      'border-[color:color-mix(in_oklab,var(--color-bad)_44%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_10%,transparent)] text-[color:var(--color-bad)]',
    hatched: true
  }
}

export function getReserveStateMeta(state: ReserveNodeState) {
  return RESERVE_STATE_META[state]
}
