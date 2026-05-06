import type { MeshNode, MeshNodeRenderKind } from '@/features/app-tabs/types'

export type MeshPlacementProfile = {
  renderKind?: MeshNodeRenderKind
  host?: boolean
  client?: boolean
}

type MeshNodePosition = { x: number; y: number }

const DEBUG_PLACEMENT_ATTEMPTS = 72
const DEBUG_PLACEMENT_GAP_PERCENT = 10
const DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT = 20
const DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT = 7
const DEBUG_PLACEMENT_CLUSTER_PADDING_PERCENT = 24
const DEBUG_PLACEMENT_CLUSTER_GROWTH_PERCENT = 4

function hashStringToUint32(value: string) {
  let hash = 2166136261

  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(hash ^ value.charCodeAt(index), 16777619)
  }

  return hash >>> 0
}

function createSeededRandom(seed: string) {
  let state = hashStringToUint32(seed) || 0x9e3779b9

  return () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function roundMeshCoordinate(value: number) {
  return Math.round(value * 100) / 100
}

function placementRenderKind(node: MeshNode) {
  if (node.role === 'self') return 'self'
  if (node.client) return 'client'
  if (node.renderKind) return node.renderKind
  if (node.host) return 'worker'
  if (node.meshState === 'serving' || (node.servingModels?.length ?? 0) > 0) return 'serving'
  if (node.meshState === 'loading' || node.status === 'degraded') return 'active'
  return 'worker'
}

function isPlacementDebugNode(node: MeshNode) {
  return 'debug' in node && node.debug === true
}

function positionDistanceSquared(first: MeshNodePosition, second: MeshNodePosition) {
  return (first.x - second.x) ** 2 + (first.y - second.y) ** 2
}

function unboundedRoamingPosition(meshSeed: string, index: number): MeshNodePosition {
  const random = createSeededRandom(`${meshSeed}:${index}:empty`)
  const span = 100 - DEBUG_PLACEMENT_GAP_PERCENT * 2

  return {
    x: roundMeshCoordinate(DEBUG_PLACEMENT_GAP_PERCENT + random() * span),
    y: roundMeshCoordinate(DEBUG_PLACEMENT_GAP_PERCENT + random() * span)
  }
}

function candidatePosition(
  meshSeed: string,
  index: number,
  attempt: number,
  profile: MeshPlacementProfile,
  anchors: MeshNode[],
  centroid: MeshNodePosition
): MeshNodePosition {
  const random = createSeededRandom(`${meshSeed}:${index}:${attempt}`)

  if (anchors.length === 0) {
    return unboundedRoamingPosition(meshSeed, index)
  }

  const anchor = anchors[Math.floor(random() * anchors.length)]
  const isClientProfile = profile.client || profile.renderKind === 'client'
  const isHostProfile = Boolean(profile.host)
  const minRadius = isHostProfile ? DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT : DEBUG_PLACEMENT_GAP_PERCENT
  const maxRadius = isHostProfile
    ? Math.max(minRadius + 1, DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT - 3)
    : DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT
  const radius = minRadius + random() * (maxRadius - minRadius)
  const outwardAngle = Math.atan2(anchor.y - centroid.y, anchor.x - centroid.x)
  const angle = isClientProfile ? outwardAngle + (random() - 0.5) * Math.PI * 1.35 : random() * Math.PI * 2

  return {
    x: roundMeshCoordinate(anchor.x + Math.cos(angle) * radius),
    y: roundMeshCoordinate(anchor.y + Math.sin(angle) * radius)
  }
}

function sparsePlacementScore(position: MeshNodePosition, nodes: MeshNode[]) {
  if (nodes.length === 0) {
    return Number.POSITIVE_INFINITY
  }

  return Math.min(...nodes.map((node) => positionDistanceSquared(position, node)))
}

function isWithinPlacementDistance(position: MeshNodePosition, nodes: MeshNode[]) {
  if (nodes.length === 0) {
    return true
  }

  return sparsePlacementScore(position, nodes) <= DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT ** 2
}

function placementBaseNodes(existingNodes: MeshNode[]) {
  const realNodes = existingNodes.filter((node) => !isPlacementDebugNode(node))

  return realNodes.length > 0 ? realNodes : existingNodes
}

function placementCentroid(nodes: Array<Pick<MeshNode, 'x' | 'y'>>): MeshNodePosition {
  if (nodes.length === 0) {
    return { x: 50, y: 50 }
  }

  return {
    x: nodes.reduce((sum, node) => sum + node.x, 0) / nodes.length,
    y: nodes.reduce((sum, node) => sum + node.y, 0) / nodes.length
  }
}

function placementRadius(nodes: Array<Pick<MeshNode, 'x' | 'y'>>, centroid: MeshNodePosition) {
  if (nodes.length === 0) {
    return 0
  }

  return Math.max(...nodes.map((node) => Math.hypot(node.x - centroid.x, node.y - centroid.y)))
}

function compatiblePlacementAnchors(
  profile: MeshPlacementProfile,
  existingNodes: MeshNode[],
  centroid: MeshNodePosition
) {
  const nonClientNodes = existingNodes.filter((node) => placementRenderKind(node) !== 'client')
  const anchors = nonClientNodes.length > 0 ? nonClientNodes : existingNodes

  if (!profile.host) {
    return anchors
  }

  return [...anchors]
    .sort((first, second) => {
      const distanceDelta = positionDistanceSquared(first, centroid) - positionDistanceSquared(second, centroid)

      if (distanceDelta !== 0) return distanceDelta
      return first.id.localeCompare(second.id)
    })
    .slice(0, Math.max(1, Math.min(4, anchors.length)))
}

function clusterEnvelopeRadius(existingNodes: MeshNode[], baseNodes: MeshNode[], centroid: MeshNodePosition) {
  const debugCount = Math.max(0, existingNodes.length - baseNodes.length)

  return Math.max(
    DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT * 2,
    placementRadius(baseNodes, centroid) +
      DEBUG_PLACEMENT_CLUSTER_PADDING_PERCENT +
      Math.sqrt(debugCount + 1) * DEBUG_PLACEMENT_CLUSTER_GROWTH_PERCENT
  )
}

function isWithinClusterEnvelope(position: MeshNodePosition, centroid: MeshNodePosition, radius: number) {
  return positionDistanceSquared(position, centroid) <= radius ** 2
}

function clampPositionToCluster(
  position: MeshNodePosition,
  centroid: MeshNodePosition,
  radius: number
): MeshNodePosition {
  const distance = Math.hypot(position.x - centroid.x, position.y - centroid.y)

  if (distance <= radius || distance === 0) {
    return position
  }

  const scale = radius / distance

  return {
    x: roundMeshCoordinate(centroid.x + (position.x - centroid.x) * scale),
    y: roundMeshCoordinate(centroid.y + (position.y - centroid.y) * scale)
  }
}

function clusteredPlacementScore(position: MeshNodePosition, nodes: MeshNode[], centroid: MeshNodePosition) {
  const nearestDistance = Math.sqrt(sparsePlacementScore(position, nodes))
  const centroidDistance = Math.hypot(position.x - centroid.x, position.y - centroid.y)

  return nearestDistance - centroidDistance * 0.08
}

function fallbackPosition(
  meshSeed: string,
  index: number,
  profile: MeshPlacementProfile,
  centroid: MeshNodePosition,
  clusterRadius: number,
  anchors: MeshNode[]
): MeshNodePosition {
  if (anchors.length === 0) {
    return unboundedRoamingPosition(meshSeed, index)
  }

  return clampPositionToCluster(
    candidatePosition(meshSeed, index, DEBUG_PLACEMENT_ATTEMPTS, profile, anchors, centroid),
    centroid,
    clusterRadius
  )
}

export function chooseClusteredMeshNodePosition(
  meshSeed: string,
  index: number,
  profile: MeshPlacementProfile,
  existingNodes: MeshNode[]
): MeshNodePosition {
  const baseNodes = placementBaseNodes(existingNodes)
  const centroid = placementCentroid(baseNodes)
  const clusterRadius = clusterEnvelopeRadius(existingNodes, baseNodes, centroid)
  const anchors = compatiblePlacementAnchors(profile, existingNodes, centroid)
  const candidates = Array.from({ length: DEBUG_PLACEMENT_ATTEMPTS }, (_, attempt) =>
    candidatePosition(meshSeed, index, attempt, profile, anchors, centroid)
  ).filter(
    (position) =>
      isWithinClusterEnvelope(position, centroid, clusterRadius) && isWithinPlacementDistance(position, existingNodes)
  )
  const sparseCandidates = candidates.filter(
    (position) => sparsePlacementScore(position, existingNodes) >= DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT ** 2
  )
  const candidatePool = sparseCandidates.length > 0 ? sparseCandidates : candidates

  if (candidatePool.length === 0) {
    return fallbackPosition(meshSeed, index, profile, centroid, clusterRadius, anchors)
  }

  return candidatePool.sort((first, second) => {
    const scoreDelta =
      clusteredPlacementScore(second, existingNodes, centroid) - clusteredPlacementScore(first, existingNodes, centroid)

    if (scoreDelta !== 0) return scoreDelta
    if (first.x !== second.x) return first.x - second.x
    return first.y - second.y
  })[0]
}
