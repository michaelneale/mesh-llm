import type { ReserveNode, ReserveNodeState, ReserveProvider } from '@/features/reserves/lib/reserve-types'
import { RESERVE_STATE_ORDER } from '@/features/reserves/lib/reserve-state'

export type ReserveStateCounts = Record<ReserveNodeState, number>
export type ReserveStateVram = Record<ReserveNodeState, number>

export type ReserveProviderTotals = {
  counts: ReserveStateCounts
  vramByState: ReserveStateVram
  totalNodes: number
  totalVram: number
  activeNodes: number
  errorNodes: number
  longestEta?: number
}

export type ReserveFleetTotals = ReserveProviderTotals & {
  onlineNodes: number
  reserveNodes: number
}

export function getReserveProviderTotals(provider: ReserveProvider): ReserveProviderTotals {
  return getReserveNodeTotals(provider.nodes)
}

export function getReserveFleetTotals(providers: ReserveProvider[]): ReserveFleetTotals {
  const totals = getReserveNodeTotals(providers.flatMap((provider) => provider.nodes))

  return {
    ...totals,
    onlineNodes: totals.counts.online,
    reserveNodes: totals.totalNodes - totals.counts.online
  }
}

function getReserveNodeTotals(nodes: ReserveNode[]): ReserveProviderTotals {
  const counts = createEmptyCounts()
  const vramByState = createEmptyVramTotals()
  let totalVram = 0
  let longestEta: number | undefined

  for (const node of nodes) {
    counts[node.state] += 1
    vramByState[node.state] += node.vram
    totalVram += node.vram
    if (node.eta != null) {
      longestEta = longestEta == null ? node.eta : Math.max(longestEta, node.eta)
    }
  }

  return {
    counts,
    vramByState,
    totalNodes: nodes.length,
    totalVram,
    activeNodes: counts.waking + counts.joining,
    errorNodes: counts.failed + counts.unreachable,
    longestEta
  }
}

function createEmptyCounts(): ReserveStateCounts {
  return RESERVE_STATE_ORDER.reduce((counts, state) => ({ ...counts, [state]: 0 }), {} as ReserveStateCounts)
}

function createEmptyVramTotals(): ReserveStateVram {
  return RESERVE_STATE_ORDER.reduce((totals, state) => ({ ...totals, [state]: 0 }), {} as ReserveStateVram)
}
