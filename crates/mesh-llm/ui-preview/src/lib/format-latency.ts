export type LatencySource = 'direct' | 'estimated' | 'unknown'

export type LatencyDescriptor = {
  latencyMs: number | null | undefined
  latencySource?: LatencySource | null
  latencyAgeMs?: number | null
  latencyObserverId?: string | null
}

export type PeerLatencyState =
  | { status: 'unknown' }
  | { status: 'direct'; latencyMs: number }
  | { status: 'estimated'; latencyMs: number }
  | { status: 'stale'; latencyMs: number }

const STALE_THRESHOLD_MS = 60_000

export function peerLatencyState(descriptor: LatencyDescriptor): PeerLatencyState {
  const { latencyMs, latencySource, latencyAgeMs } = descriptor
  if (latencyMs == null || latencySource === 'unknown') return { status: 'unknown' }
  if (latencySource === 'estimated') return { status: 'estimated', latencyMs }
  if (latencySource === 'direct' && typeof latencyAgeMs === 'number' && latencyAgeMs > STALE_THRESHOLD_MS) {
    return { status: 'stale', latencyMs }
  }
  return { status: 'direct', latencyMs }
}

export function peerLatencyHint(descriptor: LatencyDescriptor): string | null {
  const state = peerLatencyState(descriptor)
  if (state.status === 'estimated') return 'Estimated'
  if (state.status === 'stale') return 'Stale'
  return null
}

export function formatPeerLatency(descriptor: LatencyDescriptor): string {
  const state = peerLatencyState(descriptor)
  if (state.status === 'unknown') return 'Unknown'
  return `${state.latencyMs.toFixed(1)} ms`
}

export function formatPeerLatencySummary(descriptor: LatencyDescriptor): string {
  const state = peerLatencyState(descriptor)
  if (state.status === 'unknown') return 'Unknown'
  if (state.status === 'estimated') return `${state.latencyMs.toFixed(1)} ms est.`
  if (state.status === 'stale') return `${state.latencyMs.toFixed(1)} ms stale`
  return `${state.latencyMs.toFixed(1)} ms`
}

/** Legacy formatter kept for backward compat. */
export function formatLatency(latencyMs: number): string {
  if (latencyMs > 0 && latencyMs < 1) return '<1'
  return Math.max(0, Math.round(latencyMs)).toString()
}
