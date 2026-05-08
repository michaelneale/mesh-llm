import type { StatusPayload, MeshModelRaw, PeerInfo } from '@/lib/api/types'
import type { ConfigurationHarnessData, ConfigNode, ConfigModel } from '@/features/app-tabs/types'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'

function mapNodeState(state: string | undefined): 'online' | 'degraded' | 'offline' {
  if (state === 'client') return 'offline'
  return 'online'
}

function resolvePeerId(peer: PeerInfo, fallbackIndex: number): string {
  return peer.node_id ?? peer.id ?? peer.hostname ?? `peer-${fallbackIndex}`
}

function adaptPeerToConfigNode(peer: PeerInfo, fallbackIndex: number): ConfigNode {
  const id = resolvePeerId(peer, fallbackIndex)

  return {
    id,
    hostname: peer.hostname ?? id,
    region: peer.region ?? 'unknown',
    status: mapNodeState(peer.node_state ?? peer.state ?? peer.role?.toLowerCase()),
    cpu: peer.hardware_label ?? 'Unknown CPU',
    ramGB: 0,
    gpus: [],
    placement: 'separate'
  }
}

function adaptModelToConfigModel(model: MeshModelRaw): ConfigModel {
  return {
    id: model.name,
    name: model.name,
    family: model.family ?? model.name.split('/')[0] ?? 'unknown',
    paramsB: model.params_b ?? 0,
    quant: model.quantization ?? 'unknown',
    sizeGB: model.size_gb ?? 0,
    diskGB: model.disk_gb ?? 0,
    ctxMaxK: model.context_length == null ? 0 : Math.round(model.context_length / 1000),
    moe: model.capabilities?.moe ?? model.moe ?? false,
    vision: model.capabilities?.vision ?? model.vision ?? model.tags?.includes('vision') ?? false,
    tags: model.tags ?? []
  }
}

export function adaptStatusToConfiguration(payload: StatusPayload, models: MeshModelRaw[]): ConfigurationHarnessData {
  const nodes: ConfigNode[] = payload.peers.map(adaptPeerToConfigNode)
  const catalog: ConfigModel[] = models.map(adaptModelToConfigModel)

  return {
    ...CONFIGURATION_HARNESS,
    nodes,
    catalog,
    assigns: [],
    preferredAssignId: undefined
  }
}
