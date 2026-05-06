import type { MeshNode, MeshNodeRenderKind, Peer } from '@/features/app-tabs/types'

export type MeshLink = { id: string; source: MeshNode; target: MeshNode }

type MeshLinkCandidate = {
  source: MeshNode
  target: MeshNode
  distance: number
  key: string
}

const CLIENT_LINK_LIMIT = 1
const NON_CLIENT_LINK_LIMIT = 3

function meshNodeRenderKind(node: MeshNode, peer: Peer | undefined): MeshNodeRenderKind {
  if (peer?.role === 'you' || node.role === 'self') return 'self'
  if (node.renderKind) return node.renderKind
  if (node.client) return 'client'
  if (node.meshState === 'serving' || (node.servingModels?.length ?? 0) > 0) return 'serving'
  if (node.meshState === 'loading' || node.status === 'degraded') return 'active'
  return 'worker'
}

function nodeDistanceSquared(source: MeshNode, target: MeshNode) {
  return (source.x - target.x) ** 2 + (source.y - target.y) ** 2
}

function isClientMeshNode(node: MeshNode, getNodePeer?: (node: MeshNode) => Peer | undefined) {
  return meshNodeRenderKind(node, getNodePeer?.(node)) === 'client'
}

function meshLinkLimit(node: MeshNode, getNodePeer?: (node: MeshNode) => Peer | undefined) {
  return isClientMeshNode(node, getNodePeer) ? CLIENT_LINK_LIMIT : NON_CLIENT_LINK_LIMIT
}

function linkConsumesCapacity(
  node: MeshNode,
  linkedNode: MeshNode,
  getNodePeer?: (node: MeshNode) => Peer | undefined
) {
  return isClientMeshNode(node, getNodePeer) || !isClientMeshNode(linkedNode, getNodePeer)
}

function meshLinkKey(source: MeshNode, target: MeshNode) {
  return [source.id, target.id].sort().join('\u001f')
}

function closestNodes(source: MeshNode, nodes: MeshNode[]) {
  return nodes
    .filter((target) => target.id !== source.id)
    .sort((first, second) => {
      const distanceDelta = nodeDistanceSquared(source, first) - nodeDistanceSquared(source, second)

      if (distanceDelta !== 0) {
        return distanceDelta
      }

      return first.id.localeCompare(second.id)
    })
}

export function buildMeshLinks(nodes: MeshNode[], getNodePeer?: (node: MeshNode) => Peer | undefined): MeshLink[] {
  const links = new Map<string, MeshLink>()
  const linkCounts = new Map(nodes.map((node) => [node.id, 0]))
  const nonClientNodes = nodes.filter((node) => !isClientMeshNode(node, getNodePeer))
  const clientCandidates: MeshLinkCandidate[] = []
  const nonClientCandidates: MeshLinkCandidate[] = []

  const addLink = (candidate: MeshLinkCandidate) => {
    const sourceCount = linkCounts.get(candidate.source.id) ?? 0
    const targetCount = linkCounts.get(candidate.target.id) ?? 0
    const consumesSourceCapacity = linkConsumesCapacity(candidate.source, candidate.target, getNodePeer)
    const consumesTargetCapacity = linkConsumesCapacity(candidate.target, candidate.source, getNodePeer)

    if (
      (consumesSourceCapacity && sourceCount >= meshLinkLimit(candidate.source, getNodePeer)) ||
      (consumesTargetCapacity && targetCount >= meshLinkLimit(candidate.target, getNodePeer))
    ) {
      return
    }

    links.set(candidate.key, {
      id: `${candidate.source.id}-${candidate.target.id}`,
      source: candidate.source,
      target: candidate.target
    })

    if (consumesSourceCapacity) {
      linkCounts.set(candidate.source.id, sourceCount + 1)
    }

    if (consumesTargetCapacity) {
      linkCounts.set(candidate.target.id, targetCount + 1)
    }
  }

  for (const clientNode of nodes.filter((node) => isClientMeshNode(node, getNodePeer))) {
    const nearestNonClient = closestNodes(clientNode, nonClientNodes)[0]

    if (!nearestNonClient) continue

    const key = meshLinkKey(clientNode, nearestNonClient)
    clientCandidates.push({
      source: clientNode,
      target: nearestNonClient,
      distance: nodeDistanceSquared(clientNode, nearestNonClient),
      key
    })
  }

  for (const source of nonClientNodes) {
    for (const target of nonClientNodes) {
      if (source.id >= target.id) continue

      const key = meshLinkKey(source, target)
      nonClientCandidates.push({
        source,
        target,
        distance: nodeDistanceSquared(source, target),
        key
      })
    }
  }

  const sortCandidates = (candidates: MeshLinkCandidate[]) =>
    candidates.sort((first, second) => {
      const distanceDelta = first.distance - second.distance

      if (distanceDelta !== 0) return distanceDelta

      return first.key.localeCompare(second.key)
    })

  for (const candidate of sortCandidates(clientCandidates)) {
    addLink(candidate)
  }

  for (const candidate of sortCandidates(nonClientCandidates)) {
    if (links.has(candidate.key)) continue
    addLink(candidate)
  }

  return Array.from(links.values())
}
