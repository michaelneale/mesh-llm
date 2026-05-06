import { MESH_NODES } from '@/features/app-tabs/data'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import { chooseClusteredMeshNodePosition, type MeshPlacementProfile } from './mesh-placement'

export const DASHBOARD_MESH_ID = 'dashboard-mesh'

function pinnedMeshNodeForPeer(peer: Peer, meshNodeSeeds: MeshNode[] = MESH_NODES) {
  const normalizedHostname = peer.hostname.toLowerCase()

  return (
    meshNodeSeeds.find((node) => node.peerId === peer.id || node.id === peer.id) ??
    meshNodeSeeds.find((node) => node.label.toLowerCase() === normalizedHostname)
  )
}

export function peerPlacementProfile(peer: Peer): MeshPlacementProfile {
  const isClient = peer.role === 'peer' && peer.hostedModels.length === 0 && !peer.vramGB

  return {
    renderKind:
      peer.role === 'you' ? 'self' : isClient ? 'client' : peer.hostedModels.length > 0 ? 'serving' : 'worker',
    host: peer.role === 'host',
    client: isClient
  }
}

export function peerToMeshNode(peer: Peer, position: Pick<MeshNode, 'x' | 'y'>): MeshNode {
  const profile = peerPlacementProfile(peer)

  return {
    id: peer.id,
    peerId: peer.id,
    label: peer.hostname.toUpperCase(),
    subLabel: profile.client ? peer.region : peer.hostedModels.length > 0 ? 'SERVING' : peer.region,
    status: peer.status,
    role: peer.role === 'you' ? 'self' : 'peer',
    renderKind: profile.renderKind,
    meshState: profile.client ? 'client' : peer.hostedModels.length > 0 ? 'serving' : 'standby',
    host: profile.host,
    client: profile.client,
    servingModels: peer.hostedModels,
    latencyMs: peer.latencyMs,
    hostname: peer.hostname,
    vramGB: peer.vramGB,
    x: position.x,
    y: position.y
  }
}

export function meshNodeForPeer(peer: Peer, meshNodes: MeshNode[], meshId = DASHBOARD_MESH_ID) {
  const normalizedHostname = peer.hostname.toLowerCase()

  return (
    meshNodes.find((node) => node.peerId === peer.id || node.id === peer.id) ??
    meshNodes.find((node) => node.label.toLowerCase() === normalizedHostname) ??
    peerToMeshNode(
      peer,
      chooseClusteredMeshNodePosition(meshId, meshNodes.length + 1, peerPlacementProfile(peer), meshNodes)
    )
  )
}

export function buildDashboardMeshNodes(
  peers: Peer[],
  meshId = DASHBOARD_MESH_ID,
  meshNodeSeeds: MeshNode[] = MESH_NODES
) {
  return peers.reduce<MeshNode[]>((meshNodes, peer, index) => {
    const pinnedNode = pinnedMeshNodeForPeer(peer, meshNodeSeeds)
    const profile = peerPlacementProfile(peer)

    if (pinnedNode) {
      meshNodes.push({
        ...pinnedNode,
        peerId: peer.id,
        status: peer.status,
        role: peer.role === 'you' ? 'self' : pinnedNode.role,
        renderKind: profile.renderKind,
        meshState: profile.client ? 'client' : peer.hostedModels.length > 0 ? 'serving' : pinnedNode.meshState,
        host: profile.host,
        client: profile.client,
        servingModels: peer.hostedModels,
        latencyMs: peer.latencyMs,
        hostname: peer.hostname,
        vramGB: peer.vramGB
      })
      return meshNodes
    }

    const position = chooseClusteredMeshNodePosition(meshId, index + 1, profile, meshNodes)
    meshNodes.push(peerToMeshNode(peer, position))
    return meshNodes
  }, [])
}
