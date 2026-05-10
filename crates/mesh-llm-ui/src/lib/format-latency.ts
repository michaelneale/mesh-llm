import { LatencySource } from '@/lib/api/types'

export type LatencyDescriptor = {
  text: string
  tooltip: string
}

export type PeerLatencyState = {
  latencyMs: number | null
  source: LatencySource
  ageMs: number | null
  observerId: string | null
}

/**
 * Determine the latency state for a peer based on displayed latency data
 */
export function peerLatencyState(latency: PeerLatencyState): PeerLatencyState {
  return {
    latencyMs: latency.latencyMs,
    source: latency.source,
    ageMs: latency.ageMs,
    observerId: latency.observerId
  }
}

/**
 * Get a hint/tooltip for the latency state
 */
export function peerLatencyHint(latency: PeerLatencyState): string {
  if (latency.latencyMs == null) {
    return 'unknown'
  }

  const ageSecs = Math.round((latency.ageMs ?? 0) / 1000)
  const ageText = ageSecs > 0 ? ` (${ageSecs}s old)` : ''

  switch (latency.source) {
    case LatencySource.DIRECT:
      return `direct${ageText}`
    case LatencySource.ESTIMATED:
      return `estimated${ageText}`
    case LatencySource.UNKNOWN:
      return `unknown${ageText}`
    default:
      return `unspecified${ageText}`
  }
}

/**
 * Format latency for display in UI
 */
export function formatPeerLatency(latencyMs: number | null): string {
  if (latencyMs == null) return '?'
  if (latencyMs > 0 && latencyMs < 1) return '<1ms'
  return `${Math.max(0, Math.round(latencyMs))}ms`
}

/**
 * Format latency with source info for detailed display
 */
export function formatPeerLatencySummary(latency: PeerLatencyState): string {
  if (latency.latencyMs == null) return '?'

  const base = formatPeerLatency(latency.latencyMs)

  switch (latency.source) {
    case LatencySource.DIRECT:
      return base
    case LatencySource.ESTIMATED:
      return `${base}~`
    case LatencySource.UNKNOWN:
      return `${base}!`
    default:
      return base
  }
}
