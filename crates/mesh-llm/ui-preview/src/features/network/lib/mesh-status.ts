import type { StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { MeshNode, Peer } from '@/features/app-tabs/types'

type MeshStatusSource = {
  nodeState?: Peer['nodeState']
  status?: Peer['status'] | MeshNode['status']
}

type MeshStatusLabels = {
  online: string
  degraded: string
  offline: string
}

const defaultLabels: MeshStatusLabels = {
  online: 'Connected',
  degraded: 'Loading',
  offline: 'Offline'
}

export function meshStatusLabel(source: MeshStatusSource, labels: Partial<MeshStatusLabels> = {}): string {
  const resolvedLabels = { ...defaultLabels, ...labels }
  if (source.nodeState === 'client') return 'API-only'
  if (source.nodeState === 'standby') return 'Standby'
  if (source.nodeState === 'loading') return 'Loading'
  if (source.nodeState === 'serving') return 'Serving'
  if (source.status === 'online') return resolvedLabels.online
  if (source.status === 'degraded') return resolvedLabels.degraded
  return resolvedLabels.offline
}

export function meshStatusTone(source: MeshStatusSource): StatusBadgeTone {
  if (source.nodeState === 'serving') return 'good'
  if (source.nodeState === 'loading') return 'warn'
  if (source.nodeState === 'client' || source.nodeState === 'standby') return 'muted'
  if (source.status === 'online') return 'good'
  if (source.status === 'degraded') return 'warn'
  return 'bad'
}

export function meshNodeStatusSource(peer: Peer | undefined, node: MeshNode): MeshStatusSource {
  return {
    nodeState: peer?.nodeState ?? node.meshState,
    status: peer?.status ?? node.status
  }
}
