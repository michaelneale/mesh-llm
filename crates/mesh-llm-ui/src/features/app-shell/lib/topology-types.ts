import type { LiveNodeState } from '@/features/app-shell/lib/status-types'
import type { LatencySource } from '@/lib/api/types'

export type TopologyNode = {
  id: string
  vram: number
  state: LiveNodeState
  self: boolean
  host: boolean
  client: boolean
  serving: string
  servingModels: string[]
  statusLabel?: string
  ageSeconds?: number | null

  latencyMs?: number | null
  latencySource?: LatencySource
  latencyAgeMs?: number | null
  latencyObserverId?: string | null
  hostname?: string
  isSoc?: boolean
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[]
}
