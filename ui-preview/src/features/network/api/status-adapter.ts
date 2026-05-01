import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import type { StatusPayload, PeerInfo } from '@/lib/api/types'
import type { DashboardHarnessData, Peer, PeerSummary, StatusMetric, MeshNode, ModelSummary } from '@/features/app-tabs/types'

function mapNodeState(state: 'client' | 'standby' | 'loading' | 'serving'): 'online' | 'degraded' | 'offline' {
  if (state === 'serving') return 'online'
  if (state === 'loading') return 'degraded'
  return 'offline'
}

function adaptPeer(peer: PeerInfo): Peer {
  return {
    id: peer.node_id,
    hostname: peer.hostname,
    region: peer.region ?? '',
    status: mapNodeState(peer.node_state),
    hostedModels: peer.serving_models,
    sharePct: peer.share_pct ?? 0,
    latencyMs: peer.latency_ms ?? 0,
    loadPct: peer.load_pct ?? 0,
    shortId: peer.node_id.slice(0, 8),
    version: peer.version,
    vramGB: peer.my_vram_gb,
    toksPerSec: peer.tok_per_sec,
    hardwareLabel: peer.hardware_label,
    owner: peer.owner,
  }
}

function adaptSelfPeer(payload: StatusPayload): Peer {
  return {
    id: payload.node_id,
    hostname: payload.hostname ?? 'localhost',
    region: payload.region ?? '',
    status: mapNodeState(payload.node_state),
    hostedModels: payload.serving_models.map(m => m.name),
    sharePct: 0,
    latencyMs: 0,
    loadPct: payload.load_pct ?? 0,
    shortId: payload.node_id.slice(0, 8),
    role: 'you' as const,
    version: payload.version,
    vramGB: payload.my_vram_gb,
    toksPerSec: payload.tok_per_sec,
  }
}

function adaptPeerSummary(peers: Peer[]): PeerSummary {
  const online = peers.filter(p => p.status === 'online').length
  const totalVram = peers.reduce((sum, p) => sum + (p.vramGB ?? 0), 0)
  return { total: peers.length, online, capacity: `${totalVram.toFixed(0)} GB` }
}

function adaptStatusMetrics(payload: StatusPayload): StatusMetric[] {
  return [
    {
      id: 'node-state',
      label: 'Node State',
      value: payload.node_state,
      badge: {
        label: payload.node_state,
        tone: payload.node_state === 'serving' ? 'good' : payload.node_state === 'loading' ? 'warn' : 'muted',
      },
    },
    {
      id: 'active-requests',
      label: 'Active Requests',
      value: payload.active_requests ?? 0,
    },
    {
      id: 'tok-per-sec',
      label: 'Tokens/sec',
      value: payload.tok_per_sec?.toFixed(1) ?? '0',
      unit: 'tok/s',
    },
    {
      id: 'vram',
      label: 'VRAM',
      value: payload.my_vram_gb?.toFixed(1) ?? '0',
      unit: 'GB',
    },
  ]
}

function adaptMeshNodes(payload: StatusPayload, peers: Peer[]): MeshNode[] {
  const selfNode: MeshNode = {
    id: payload.node_id,
    label: payload.hostname ?? 'localhost',
    x: 0,
    y: 0,
    status: mapNodeState(payload.node_state),
    role: 'self',
    meshState: payload.node_state,
    servingModels: payload.serving_models.map(m => m.name),
    hostname: payload.hostname,
    vramGB: payload.my_vram_gb,
  }

  const remotePeers = peers.filter(p => p.role !== 'you')
  const peerCount = remotePeers.length

  const peerNodes: MeshNode[] = remotePeers.map((peer, i) => ({
    id: peer.id,
    label: peer.hostname,
    x: Math.cos((2 * Math.PI * i) / peerCount) * 200,
    y: Math.sin((2 * Math.PI * i) / peerCount) * 200,
    status: peer.status,
    role: 'peer' as const,
    meshState: peer.status === 'online' ? 'serving' : peer.status === 'degraded' ? 'loading' : 'standby',
    servingModels: peer.hostedModels,
    hostname: peer.hostname,
    vramGB: peer.vramGB,
  }))

  return [selfNode, ...peerNodes]
}

export function adaptStatusToDashboard(payload: StatusPayload, models: ModelSummary[] = []): DashboardHarnessData {
  const selfPeer = adaptSelfPeer(payload)
  const remotePeers = payload.peers.map(adaptPeer)
  const allPeers = [selfPeer, ...remotePeers]
  return {
    ...DASHBOARD_HARNESS,
    peers: allPeers,
    peerSummary: adaptPeerSummary(allPeers),
    statusMetrics: adaptStatusMetrics(payload),
    meshNodeSeeds: adaptMeshNodes(payload, allPeers),
    meshId: payload.mesh_id ?? '',
    models,
  }
}
