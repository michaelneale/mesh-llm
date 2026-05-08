import { describe, expect, it } from 'vitest'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import { peerPlacementProfile, reconcileDashboardMeshNodes } from './dashboard-mesh-nodes'
import { chooseClusteredMeshNodePosition } from './mesh-placement'

const selfSeed: MeshNode = {
  id: 'self-node',
  peerId: 'self-node',
  label: 'SELF',
  x: 50,
  y: 50,
  status: 'online',
  role: 'self',
  renderKind: 'self',
  meshState: 'serving',
  servingModels: ['Self-Model']
}

function peer(overrides: Partial<Peer> & Pick<Peer, 'id' | 'hostname'>): Peer {
  return {
    id: overrides.id,
    hostname: overrides.hostname,
    region: overrides.region ?? 'iad-1',
    status: overrides.status ?? 'online',
    hostedModels: overrides.hostedModels ?? [],
    sharePct: overrides.sharePct ?? 0,
    latencyMs: overrides.latencyMs ?? 1,
    loadPct: overrides.loadPct ?? 0,
    role: overrides.role ?? 'peer',
    nodeState: overrides.nodeState,
    version: overrides.version,
    vramGB: overrides.vramGB,
    toksPerSec: overrides.toksPerSec,
    hardwareLabel: overrides.hardwareLabel,
    ownership: overrides.ownership,
    owner: overrides.owner,
    shortId: overrides.shortId
  }
}

function positionByPeerId(nodes: MeshNode[], peerId: string) {
  const node = nodes.find((candidate) => candidate.peerId === peerId)
  expect(node).toBeDefined()
  return { x: node?.x, y: node?.y }
}

function distanceBetweenPeers(nodes: MeshNode[], firstPeerId: string, secondPeerId: string) {
  const first = nodes.find((node) => node.peerId === firstPeerId)
  const second = nodes.find((node) => node.peerId === secondPeerId)

  expect(first).toBeDefined()
  expect(second).toBeDefined()

  return Math.hypot((first?.x ?? 0) - (second?.x ?? 0), (first?.y ?? 0) - (second?.y ?? 0))
}

describe('reconcileDashboardMeshNodes', () => {
  it('preserves survivor positions across API peer reorders while updating peer state', () => {
    const initialPeers = [
      peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
      peer({ id: 'alpha', hostname: 'alpha', hostedModels: ['Alpha-Model'], nodeState: 'serving', latencyMs: 4 }),
      peer({ id: 'beta', hostname: 'beta', role: 'client', nodeState: 'client', vramGB: 0, latencyMs: 8 })
    ]
    const initialNodes = reconcileDashboardMeshNodes([], initialPeers, 'mesh-a', [selfSeed])
    const alphaPosition = positionByPeerId(initialNodes, 'alpha')
    const betaPosition = positionByPeerId(initialNodes, 'beta')

    const refreshedNodes = reconcileDashboardMeshNodes(
      initialNodes,
      [
        peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
        peer({ id: 'beta', hostname: 'beta', role: 'client', nodeState: 'client', vramGB: 0, latencyMs: 3 }),
        peer({ id: 'alpha', hostname: 'alpha', hostedModels: ['Alpha-Model'], nodeState: 'serving', latencyMs: 2 })
      ],
      'mesh-a',
      [selfSeed]
    )

    expect(positionByPeerId(refreshedNodes, 'alpha')).toEqual(alphaPosition)
    expect(positionByPeerId(refreshedNodes, 'beta')).toEqual(betaPosition)
    expect(refreshedNodes.find((node) => node.peerId === 'alpha')).toEqual(
      expect.objectContaining({ latencyMs: 2, renderKind: 'serving', meshState: 'serving' })
    )
    expect(refreshedNodes.find((node) => node.peerId === 'beta')).toEqual(
      expect.objectContaining({ latencyMs: 3, client: true, renderKind: 'client', meshState: 'client' })
    )
  })

  it('adds and removes peers by id so MeshViz lifecycle can animate joins and leaves', () => {
    const initialPeers = [
      peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
      peer({ id: 'alpha', hostname: 'alpha', hostedModels: ['Alpha-Model'], nodeState: 'serving' }),
      peer({ id: 'beta', hostname: 'beta', role: 'client', nodeState: 'client', vramGB: 0 })
    ]
    const initialNodes = reconcileDashboardMeshNodes([], initialPeers, 'mesh-b', [selfSeed])
    const alphaPosition = positionByPeerId(initialNodes, 'alpha')

    const nextNodes = reconcileDashboardMeshNodes(
      initialNodes,
      [
        peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
        peer({ id: 'alpha', hostname: 'alpha', hostedModels: ['Alpha-Model'], nodeState: 'serving' }),
        peer({ id: 'gamma', hostname: 'gamma', role: 'worker', nodeState: 'standby', vramGB: 12 })
      ],
      'mesh-b',
      [selfSeed]
    )

    expect(nextNodes.map((node) => node.peerId)).toEqual(['self-node', 'alpha', 'gamma'])
    expect(positionByPeerId(nextNodes, 'alpha')).toEqual(alphaPosition)
    expect(nextNodes.find((node) => node.peerId === 'beta')).toBeUndefined()
    expect(nextNodes.find((node) => node.peerId === 'gamma')).toEqual(
      expect.objectContaining({ renderKind: 'worker', meshState: 'standby' })
    )
  })

  it('uses classification labels for generated peers with blank regions', () => {
    const nodes = reconcileDashboardMeshNodes(
      [],
      [
        peer({ id: 'api-client', hostname: 'api-client', region: '', role: 'client', nodeState: 'client', vramGB: 0 }),
        peer({ id: 'idle-worker', hostname: 'idle-worker', region: '', role: 'worker', nodeState: 'standby' })
      ],
      'mesh-labels',
      []
    )

    expect(nodes.find((node) => node.peerId === 'api-client')).toEqual(
      expect.objectContaining({ client: true, renderKind: 'client', meshState: 'client', subLabel: 'CLIENT' })
    )
    expect(nodes.find((node) => node.peerId === 'idle-worker')).toEqual(
      expect.objectContaining({ client: false, renderKind: 'worker', meshState: 'standby', subLabel: 'WORKER' })
    )
  })

  it('refreshes reused node labels from current peer classification', () => {
    const previousNodes: MeshNode[] = [
      {
        id: 'reused-worker',
        peerId: 'reused-worker',
        label: 'REUSED-WORKER',
        subLabel: '',
        x: 45,
        y: 55,
        status: 'online',
        renderKind: 'serving',
        meshState: 'serving'
      }
    ]

    const nodes = reconcileDashboardMeshNodes(
      previousNodes,
      [peer({ id: 'reused-worker', hostname: 'reused-worker', region: '', role: 'worker', nodeState: 'standby' })],
      'mesh-reused-labels',
      []
    )

    expect(nodes.find((node) => node.peerId === 'reused-worker')).toEqual(
      expect.objectContaining({ renderKind: 'worker', meshState: 'standby', subLabel: 'WORKER' })
    )
  })

  it('does not reuse a previous node when a new peer id appears on the same hostname', () => {
    const initialNodes = reconcileDashboardMeshNodes(
      [],
      [
        peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
        peer({ id: 'old-peer-id', hostname: 'shared-host', hostedModels: ['Old-Model'], nodeState: 'serving' })
      ],
      'mesh-c',
      [selfSeed]
    )
    const oldNode = initialNodes.find((node) => node.peerId === 'old-peer-id')

    const nextNodes = reconcileDashboardMeshNodes(
      initialNodes,
      [
        peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
        peer({ id: 'new-peer-id', hostname: 'shared-host', hostedModels: ['New-Model'], nodeState: 'serving' })
      ],
      'mesh-c',
      [selfSeed]
    )
    const newNode = nextNodes.find((node) => node.peerId === 'new-peer-id')

    expect(oldNode).toBeDefined()
    expect(newNode).toBeDefined()
    expect(nextNodes.find((node) => node.peerId === 'old-peer-id')).toBeUndefined()
    expect(newNode?.id).toBe('new-peer-id')
    expect(newNode?.id).not.toBe(oldNode?.id)
  })

  it('recomputes mesh state when a reused peer stops serving', () => {
    const initialNodes = reconcileDashboardMeshNodes(
      [],
      [
        peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
        peer({ id: 'worker-a', hostname: 'worker-a', hostedModels: ['Worker-Model'], nodeState: 'serving' })
      ],
      'mesh-d',
      [selfSeed]
    )

    const nextNodes = reconcileDashboardMeshNodes(
      initialNodes,
      [
        peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
        peer({ id: 'worker-a', hostname: 'worker-a', hostedModels: [], nodeState: 'standby' })
      ],
      'mesh-d',
      [selfSeed]
    )

    expect(initialNodes.find((node) => node.peerId === 'worker-a')).toEqual(
      expect.objectContaining({ renderKind: 'serving', meshState: 'serving' })
    )
    expect(nextNodes.find((node) => node.peerId === 'worker-a')).toEqual(
      expect.objectContaining({ renderKind: 'worker', meshState: 'standby' })
    )
  })

  it('places new peers against later survivor positions from the previous graph', () => {
    const newPeer = peer({ id: 'new-worker', hostname: 'new-worker', role: 'worker', nodeState: 'standby' })
    const collidingPosition = chooseClusteredMeshNodePosition('mesh-order', 4, peerPlacementProfile(newPeer), [
      selfSeed
    ])
    const previousNodes: MeshNode[] = [
      selfSeed,
      {
        id: 'later-survivor',
        peerId: 'later-survivor',
        label: 'LATER-SURVIVOR',
        x: collidingPosition.x,
        y: collidingPosition.y,
        status: 'online',
        renderKind: 'worker',
        meshState: 'standby'
      }
    ]

    const nextNodes = reconcileDashboardMeshNodes(
      previousNodes,
      [
        peer({ id: 'self-node', hostname: 'self', role: 'you', hostedModels: ['Self-Model'], nodeState: 'serving' }),
        newPeer,
        peer({ id: 'later-survivor', hostname: 'later-survivor', role: 'worker', nodeState: 'standby' })
      ],
      'mesh-order',
      [selfSeed]
    )

    expect(distanceBetweenPeers(nextNodes, 'new-worker', 'later-survivor')).toBeGreaterThanOrEqual(10)
  })
})
