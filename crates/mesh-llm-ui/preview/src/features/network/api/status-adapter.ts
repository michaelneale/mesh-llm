import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import type { StatusPayload, PeerInfo, GpuInfo, ServingModelEntry } from '@/lib/api/types'
import { isPublicMesh } from '@/lib/api/mesh-visibility'
import type {
  DashboardHarnessData,
  DashboardConnectData,
  HeroAction,
  Peer,
  PeerSummary,
  StatusMetric,
  MeshNode,
  ModelSummary
} from '@/features/app-tabs/types'

const DASHBOARD_HERO_ACTIONS: HeroAction[] = DASHBOARD_HARNESS.hero.actions

const PUBLIC_MESH_HERO: DashboardHarnessData['hero'] = {
  title: 'Welcome to the public mesh',
  description:
    'Run open models with shared community capacity. Start a local node to contribute compute or use the OpenAI-compatible endpoint.',
  actions: DASHBOARD_HERO_ACTIONS
}

const PUBLIC_MESH_CONNECT: DashboardConnectData = {
  ...DASHBOARD_HARNESS.connect,
  runCommand: 'mesh-llm --auto',
  description: 'join the public mesh'
}

function adaptHero(payload: StatusPayload): DashboardHarnessData['hero'] {
  return isPublicMesh(payload) ? PUBLIC_MESH_HERO : DASHBOARD_HARNESS.hero
}

function adaptConnect(payload: StatusPayload): DashboardConnectData {
  return isPublicMesh(payload) ? PUBLIC_MESH_CONNECT : DASHBOARD_HARNESS.connect
}

type NodeState = NonNullable<Peer['nodeState']>

function isNodeState(state: string | undefined): state is NodeState {
  return state === 'client' || state === 'standby' || state === 'loading' || state === 'serving'
}

function mapNodeState(state: string | undefined): Peer['status'] {
  if (state === 'loading') return 'degraded'
  if (isNodeState(state)) return 'online'
  return 'offline'
}

function servingModelName(model: ServingModelEntry): string {
  return typeof model === 'string' ? model : model.name
}

function resolvePeerId(peer: PeerInfo, fallbackIndex: number): string {
  return peer.node_id ?? peer.id ?? peer.hostname ?? `peer-${fallbackIndex}`
}

function resolvePeerState(peer: PeerInfo): string | undefined {
  return peer.node_state ?? peer.state ?? peer.role?.toLowerCase()
}

function resolvePeerNodeState(peer: PeerInfo): Peer['nodeState'] | undefined {
  const state = resolvePeerState(peer)
  if (isNodeState(state)) return state
  return undefined
}

function resolvePeerRole(peer: PeerInfo): NonNullable<Peer['role']> {
  const role = peer.role?.toLowerCase()
  const state = resolvePeerNodeState(peer)

  if (role === 'host') return 'host'
  if (role === 'worker') return 'worker'
  if (role === 'client' || state === 'client') return 'client'
  return 'peer'
}

function normalizeModelList(models: (string | undefined)[]): string[] {
  const seen = new Set<string>()
  const normalized: string[] = []

  for (const model of models) {
    const trimmed = model?.trim()
    if (!trimmed || seen.has(trimmed)) continue
    seen.add(trimmed)
    normalized.push(trimmed)
  }

  return normalized
}

function resolveHostedModels(peer: PeerInfo): string[] {
  const modelLists = [peer.serving_models, peer.hosted_models, peer.models]

  for (const models of modelLists) {
    const normalized = normalizeModelList(models ?? [])
    if (normalized.length > 0) return normalized
  }

  return []
}

function normalizeSharePct(sharePct: number | undefined): number {
  if (typeof sharePct !== 'number' || !Number.isFinite(sharePct)) return 0
  return Math.min(Math.max(Math.round(sharePct), 0), 100)
}

function resolveOwner(owner: PeerInfo['owner']): string | undefined {
  if (typeof owner === 'string') return owner
  return owner?.display_name ?? owner?.name ?? owner?.status
}

function finiteMetric(value: number | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function gpuTotalVramGb(gpus?: GpuInfo[]): number | null {
  if (!gpus?.length) return null

  const total = gpus.reduce((sum, gpu) => {
    if (typeof gpu.total_vram_gb === 'number' && Number.isFinite(gpu.total_vram_gb) && gpu.total_vram_gb > 0) {
      return sum + gpu.total_vram_gb
    }

    if (typeof gpu.vram_bytes === 'number' && Number.isFinite(gpu.vram_bytes) && gpu.vram_bytes > 0) {
      return sum + gpu.vram_bytes / 1024 ** 3
    }

    return sum
  }, 0)

  return total > 0 ? total : null
}

function peerVramGb(peer: PeerInfo): number {
  return finiteMetric(peer.my_vram_gb ?? peer.vram_gb ?? gpuTotalVramGb(peer.gpus) ?? undefined)
}

function meshTotalVramGb(payload: StatusPayload): number {
  const localVram = finiteMetric(payload.my_vram_gb || gpuTotalVramGb(payload.gpus) || undefined)
  return payload.peers.reduce((sum, peer) => sum + peerVramGb(peer), localVram)
}

function resolveInflightRequests(payload: StatusPayload): number {
  return finiteMetric(payload.inflight_requests)
}

function adaptPeer(peer: PeerInfo, fallbackIndex: number): Peer {
  const id = resolvePeerId(peer, fallbackIndex)
  const nodeState = resolvePeerNodeState(peer)

  return {
    id,
    hostname: peer.hostname ?? id,
    region: peer.region ?? '',
    status: mapNodeState(resolvePeerState(peer)),
    hostedModels: resolveHostedModels(peer),
    sharePct: normalizeSharePct(peer.share_pct),
    latencyMs: peer.latency_ms ?? peer.rtt_ms ?? 0,
    loadPct: peer.load_pct ?? 0,
    shortId: id.slice(0, 8),
    version: peer.version,
    vramGB: peerVramGb(peer),
    role: resolvePeerRole(peer),
    nodeState,
    toksPerSec: peer.tok_per_sec,
    hardwareLabel: peer.hardware_label,
    owner: resolveOwner(peer.owner)
  }
}

function adaptSelfPeer(payload: StatusPayload): Peer {
  const servingModels = normalizeModelList([
    ...payload.serving_models.map(servingModelName),
    payload.node_state === 'serving' ? payload.model_name : undefined
  ])

  return {
    id: payload.node_id,
    hostname: payload.hostname ?? payload.my_hostname ?? 'localhost',
    region: payload.region ?? '',
    status: mapNodeState(payload.node_state),
    hostedModels: servingModels,
    sharePct: 0,
    latencyMs: 0,
    loadPct: payload.load_pct ?? 0,
    shortId: payload.node_id.slice(0, 8),
    role: 'you' as const,
    nodeState: payload.node_state,
    version: payload.version,
    vramGB: payload.my_vram_gb,
    toksPerSec: payload.tok_per_sec
  }
}

function normalizePeerShares(peers: Peer[]): Peer[] {
  if (peers.some((peer) => peer.sharePct > 0)) return peers

  const peersWithShareCapacity = peers.filter((peer) => peer.nodeState === 'serving' || peer.hostedModels.length > 0)
  const totalVram = peersWithShareCapacity.reduce((sum, peer) => sum + (peer.vramGB ?? 0), 0)
  if (totalVram <= 0) return peers

  return peers.map((peer) => ({
    ...peer,
    sharePct:
      peer.nodeState === 'serving' || peer.hostedModels.length > 0
        ? normalizeSharePct(((peer.vramGB ?? 0) / totalVram) * 100)
        : peer.sharePct
  }))
}

function adaptPeerSummary(peers: Peer[]): PeerSummary {
  const online = peers.filter((p) => p.status === 'online').length
  const totalVram = peers.reduce((sum, p) => sum + (p.vramGB ?? 0), 0)
  return { total: peers.length, online, capacity: `${totalVram.toFixed(0)} GB` }
}

function adaptStatusMetrics(payload: StatusPayload): StatusMetric[] {
  const localServingModelNames = normalizeModelList([
    ...payload.serving_models.map(servingModelName),
    payload.node_state === 'serving' ? payload.model_name : undefined
  ])
  const remoteServingModelNames = normalizeModelList(payload.peers.flatMap(resolveHostedModels))
  const activeModelNames = normalizeModelList([...localServingModelNames, ...remoteServingModelNames])
  const totalMeshVram = meshTotalVramGb(payload)
  const peerCount = payload.peers.length
  const inflightRequests = resolveInflightRequests(payload)
  const owner = resolveOwner(payload.owner) ?? 'Unsigned'

  return [
    {
      id: 'node-id',
      label: 'Node ID',
      value: payload.node_id || 'n/a',
      variant: 'identity',
      mono: true,
      badge: {
        label: payload.node_state,
        tone: payload.node_state === 'serving' ? 'good' : payload.node_state === 'loading' ? 'warn' : 'muted'
      }
    },
    {
      id: 'owner',
      label: 'Owner',
      value: owner,
      variant: 'identity',
      mono: true,
      badge: {
        label: owner.toLowerCase() === 'unsigned' ? 'not cryptographically bound' : 'verified identity',
        tone: 'muted'
      }
    },
    {
      id: 'nodes',
      label: 'Nodes',
      value: peerCount + 1,
      meta: `1 you · ${peerCount} peers`
    },
    {
      id: 'active-models',
      label: 'Active models',
      value: activeModelNames.length,
      meta: `${localServingModelNames.length} loaded locally · ${remoteServingModelNames.length} remote`
    },
    {
      id: 'mesh-vram',
      label: 'Mesh VRAM',
      value: totalMeshVram.toFixed(1),
      unit: 'GB'
    },
    {
      id: 'inflight',
      label: 'Inflight',
      value: inflightRequests
    }
  ]
}

function adaptMeshNodeSeeds(payload: StatusPayload): MeshNode[] {
  const selfNode: MeshNode = {
    id: payload.node_id,
    label: payload.hostname ?? 'localhost',
    x: 50,
    y: 50,
    status: mapNodeState(payload.node_state),
    role: 'self',
    meshState: payload.node_state,
    servingModels: payload.serving_models.map(servingModelName),
    hostname: payload.hostname,
    vramGB: payload.my_vram_gb
  }

  return [selfNode]
}

export function adaptStatusToDashboard(payload: StatusPayload, models: ModelSummary[] = []): DashboardHarnessData {
  const selfPeer = adaptSelfPeer(payload)
  const remotePeers = payload.peers.map(adaptPeer)
  const allPeers = normalizePeerShares([selfPeer, ...remotePeers])
  return {
    ...DASHBOARD_HARNESS,
    hero: adaptHero(payload),
    peers: allPeers,
    peerSummary: adaptPeerSummary(allPeers),
    statusMetrics: adaptStatusMetrics(payload),
    meshNodeSeeds: adaptMeshNodeSeeds(payload),
    meshId: payload.mesh_id ?? '',
    models,
    connect: adaptConnect(payload)
  }
}
