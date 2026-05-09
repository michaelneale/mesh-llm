import type { MeshNode, Peer } from '@/features/app-tabs/types'

export function nonBlankText(value: string | null | undefined) {
  const trimmed = value?.trim()
  return trimmed ? trimmed : undefined
}

function isClientPeer(peer: Peer) {
  const isLegacyClient =
    peer.nodeState == null && peer.role === 'peer' && peer.hostedModels.length === 0 && !peer.vramGB
  return peer.nodeState === 'client' || peer.role === 'client' || isLegacyClient
}

export function meshPeerSecondaryLabel(peer: Peer) {
  const region = nonBlankText(peer.region)

  if (peer.role === 'you') return region ?? 'LOCAL'
  if (peer.role === 'host') return 'HOST'
  if (peer.hostedModels.length > 0 || peer.nodeState === 'serving') return 'SERVING'

  if (region) return region

  if (isClientPeer(peer)) return 'CLIENT'
  if (peer.nodeState === 'loading') return 'LOADING'
  if (peer.nodeState === 'standby' || peer.role === 'worker') return 'WORKER'
  return 'PEER'
}

function meshNodeFallbackLabel(node: MeshNode) {
  if (node.role === 'self') return 'LOCAL'
  if (node.host) return 'HOST'
  if (node.client || node.renderKind === 'client' || node.meshState === 'client') return 'CLIENT'
  if (node.renderKind === 'serving' || node.meshState === 'serving' || (node.servingModels?.length ?? 0) > 0)
    return 'SERVING'
  if (node.renderKind === 'active' || node.meshState === 'loading') return 'ACTIVE'
  if (node.renderKind === 'worker' || node.meshState === 'standby') return 'WORKER'
  return 'Mesh node'
}

export function meshNodeSecondaryLabel(node: MeshNode, peer?: Peer) {
  if (peer) return meshPeerSecondaryLabel(peer)
  return nonBlankText(node.subLabel) ?? meshNodeFallbackLabel(node)
}
