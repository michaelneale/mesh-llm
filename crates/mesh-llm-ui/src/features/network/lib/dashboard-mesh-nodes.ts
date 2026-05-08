import { MESH_NODES } from '@/features/app-tabs/data'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import { meshPeerSecondaryLabel } from '@/features/network/lib/mesh-node-labels'
import { chooseClusteredMeshNodePosition, type MeshPlacementProfile } from '@/features/network/lib/mesh-placement'

export const DASHBOARD_MESH_ID = 'dashboard-mesh'

function pinnedMeshNodeForPeer(peer: Peer, meshNodeSeeds: MeshNode[] = MESH_NODES) {
  const normalizedHostname = peer.hostname.toLowerCase()

  return (
    meshNodeSeeds.find((node) => node.peerId === peer.id || node.id === peer.id) ??
    meshNodeSeeds.find((node) => node.label.toLowerCase() === normalizedHostname)
  )
}

export function peerPlacementProfile(peer: Peer): MeshPlacementProfile {
  const isLegacyClient =
    peer.nodeState == null && peer.role === 'peer' && peer.hostedModels.length === 0 && !peer.vramGB
  const isClient = peer.nodeState === 'client' || peer.role === 'client' || isLegacyClient

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
    subLabel: meshPeerSecondaryLabel(peer),
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

function meshNodeWithPeerState(node: MeshNode, peer: Peer): MeshNode {
  const profile = peerPlacementProfile(peer)
  const meshState = profile.client
    ? 'client'
    : peer.hostedModels.length > 0 || peer.nodeState === 'serving'
      ? 'serving'
      : peer.nodeState === 'loading'
        ? 'loading'
        : 'standby'

  return {
    ...node,
    peerId: peer.id,
    status: peer.status,
    role: peer.role === 'you' ? 'self' : node.role,
    renderKind: profile.renderKind,
    meshState,
    host: profile.host,
    client: profile.client,
    servingModels: peer.hostedModels,
    latencyMs: peer.latencyMs,
    subLabel: meshPeerSecondaryLabel(peer),
    hostname: peer.hostname,
    vramGB: peer.vramGB
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

function existingMeshNodeForPeer(peer: Peer, meshNodes: MeshNode[]) {
  return meshNodes.find((node) => node.peerId === peer.id || node.id === peer.id)
}

function meshNodeIdentity(node: MeshNode) {
  return node.peerId ?? node.id
}

function uniqueMeshNodesByIdentity(nodes: MeshNode[]) {
  const uniqueNodes = new Map<string, MeshNode>()

  for (const node of nodes) {
    uniqueNodes.set(meshNodeIdentity(node), node)
  }

  return [...uniqueNodes.values()]
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
      meshNodes.push(meshNodeWithPeerState(pinnedNode, peer))
      return meshNodes
    }

    const position = chooseClusteredMeshNodePosition(meshId, index + 1, profile, meshNodes)
    meshNodes.push(peerToMeshNode(peer, position))
    return meshNodes
  }, [])
}

export function reconcileDashboardMeshNodes(
  previousNodes: MeshNode[],
  peers: Peer[],
  meshId = DASHBOARD_MESH_ID,
  meshNodeSeeds: MeshNode[] = MESH_NODES
) {
  const livePeerIds = new Set(peers.map((peer) => peer.id))
  const survivorNodes = previousNodes.filter((node) => livePeerIds.has(node.peerId ?? node.id))

  return peers.reduce<MeshNode[]>((meshNodes, peer) => {
    const pinnedNode = pinnedMeshNodeForPeer(peer, meshNodeSeeds)
    const previousNode = existingMeshNodeForPeer(peer, previousNodes)

    if (pinnedNode) {
      meshNodes.push(meshNodeWithPeerState(pinnedNode, peer))
      return meshNodes
    }

    if (previousNode) {
      meshNodes.push(meshNodeWithPeerState(previousNode, peer))
      return meshNodes
    }

    const position = chooseClusteredMeshNodePosition(
      meshId,
      previousNodes.length + meshNodes.length + 1,
      peerPlacementProfile(peer),
      uniqueMeshNodesByIdentity([...survivorNodes, ...meshNodes])
    )
    meshNodes.push(peerToMeshNode(peer, position))
    return meshNodes
  }, [])
}
