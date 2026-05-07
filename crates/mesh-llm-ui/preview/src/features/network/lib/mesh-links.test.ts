import { describe, expect, it } from 'vitest'
import { buildMeshLinks } from '@/features/network/lib/mesh-links'
import type { MeshNode } from '@/features/app-tabs/types'

const workerA: MeshNode = {
  id: 'worker-a',
  label: 'Worker A',
  x: 10,
  y: 10,
  status: 'online',
  renderKind: 'worker'
}
const workerB: MeshNode = {
  id: 'worker-b',
  label: 'Worker B',
  x: 90,
  y: 10,
  status: 'online',
  renderKind: 'worker'
}
const client: MeshNode = {
  id: 'client-a',
  label: 'Client A',
  x: 12,
  y: 12,
  status: 'online',
  renderKind: 'client',
  client: true
}
const farClient: MeshNode = {
  id: 'client-far',
  label: 'Far Client',
  x: 4,
  y: 50,
  status: 'online',
  renderKind: 'client',
  client: true
}
const farWorker: MeshNode = {
  id: 'worker-far',
  label: 'Far Worker',
  x: 96,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const hub: MeshNode = {
  id: 'hub',
  label: 'Hub',
  x: 50,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const localWorkerA: MeshNode = {
  id: 'local-a',
  label: 'Local A',
  x: 54,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const localWorkerB: MeshNode = {
  id: 'local-b',
  label: 'Local B',
  x: 50,
  y: 55,
  status: 'online',
  renderKind: 'worker'
}
const distantSecondaryWorker: MeshNode = {
  id: 'distant-secondary',
  label: 'Distant Secondary',
  x: 94,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const bridgeClusterA: MeshNode = {
  id: 'bridge-a',
  label: 'Bridge A',
  x: 20,
  y: 40,
  status: 'online',
  renderKind: 'worker'
}
const bridgeClusterB: MeshNode = {
  id: 'bridge-b',
  label: 'Bridge B',
  x: 24,
  y: 42,
  status: 'online',
  renderKind: 'worker'
}
const bridgeClusterC: MeshNode = {
  id: 'bridge-c',
  label: 'Bridge C',
  x: 46,
  y: 42,
  status: 'online',
  renderKind: 'worker'
}
const bridgeClusterD: MeshNode = {
  id: 'bridge-d',
  label: 'Bridge D',
  x: 50,
  y: 40,
  status: 'online',
  renderKind: 'worker'
}
const denseCoreA: MeshNode = {
  id: 'core-a',
  label: 'Core A',
  x: 50,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const denseCoreB: MeshNode = {
  id: 'core-b',
  label: 'Core B',
  x: 54,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const denseCoreC: MeshNode = {
  id: 'core-c',
  label: 'Core C',
  x: 52,
  y: 54,
  status: 'online',
  renderKind: 'worker'
}
const fringeNode: MeshNode = {
  id: 'fringe',
  label: 'Fringe',
  x: 90,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const fringeNeighbor: MeshNode = {
  id: 'fringe-neighbor',
  label: 'Fringe Neighbor',
  x: 92,
  y: 56,
  status: 'online',
  renderKind: 'worker'
}
const centerCoreA: MeshNode = {
  id: 'center-a',
  label: 'Center A',
  x: 50,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const centerCoreB: MeshNode = {
  id: 'center-b',
  label: 'Center B',
  x: 62,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}
const centerCoreC: MeshNode = {
  id: 'center-c',
  label: 'Center C',
  x: 56,
  y: 60,
  status: 'online',
  renderKind: 'worker'
}
const centerCoreD: MeshNode = {
  id: 'center-d',
  label: 'Center D',
  x: 72,
  y: 50,
  status: 'online',
  renderKind: 'worker'
}

function linkKey(sourceId: string | undefined, targetId: string | undefined) {
  return [sourceId, targetId].sort().join('::')
}

function linkDegree(links: ReturnType<typeof buildMeshLinks>, nodeId: string) {
  return links.filter((link) => link.source.id === nodeId || link.target.id === nodeId).length
}

describe('buildMeshLinks', () => {
  it('connects clients to their nearest non-client node', () => {
    const links = buildMeshLinks([workerB, client, workerA])
    const clientLink = links.find((link) => link.source.id === client.id || link.target.id === client.id)

    expect(clientLink).toBeDefined()
    expect([clientLink?.source.id, clientLink?.target.id]).toContain(workerA.id)
  })

  it('keeps client nodes to a single capacity-consuming link', () => {
    const links = buildMeshLinks([workerA, workerB, client])
    const clientDegree = links.filter((link) => link.source.id === client.id || link.target.id === client.id).length

    expect(clientDegree).toBe(1)
  })

  it('does not connect visually distant nodes just to satisfy nearest-neighbor linking', () => {
    const links = buildMeshLinks([farClient, farWorker])

    expect(links).toHaveLength(0)
  })

  it('does not connect distant non-client islands when no local edge exists', () => {
    const links = buildMeshLinks([workerA, farWorker])

    expect(links).toHaveLength(0)
  })

  it('prunes distant secondary links after a node already has local connectivity', () => {
    const links = buildMeshLinks([hub, localWorkerA, localWorkerB, distantSecondaryWorker])
    const linkKeys = links.map((link) => linkKey(link.source.id, link.target.id))

    expect(linkKeys).toContain('hub::local-a')
    expect(linkKeys).toContain('hub::local-b')
    expect(linkKeys).toContain('local-a::local-b')
    expect(linkKeys).not.toContain('distant-secondary::local-a')
    expect(linkKeys).not.toContain('distant-secondary::hub')
  })

  it('keeps the nearest valid bridge when reconnecting local clusters', () => {
    const links = buildMeshLinks([bridgeClusterA, bridgeClusterB, bridgeClusterC, bridgeClusterD])
    const linkKeys = links.map((link) => linkKey(link.source.id, link.target.id))

    expect(linkKeys).toContain('bridge-a::bridge-b')
    expect(linkKeys).toContain('bridge-c::bridge-d')
    expect(linkKeys).toContain('bridge-b::bridge-c')
    expect(linkKeys).not.toContain('bridge-a::bridge-c')
    expect(linkKeys).not.toContain('bridge-b::bridge-d')
  })

  it('connects a nearby fringe pair with one short bridge without forming spokes', () => {
    const links = buildMeshLinks([denseCoreA, denseCoreB, denseCoreC, fringeNode, fringeNeighbor])
    const linkKeys = links.map((link) => linkKey(link.source.id, link.target.id))

    expect(linkKeys).toContain('core-a::core-b')
    expect(linkKeys).toContain('core-a::core-c')
    expect(linkKeys).toContain('core-b::core-c')
    expect(linkKeys).toContain('fringe::fringe-neighbor')
    expect(linkKeys).toContain('core-b::fringe')
    expect(linkKeys).not.toContain('core-c::fringe')
  })

  it('adds short redundant links around center clusters without exceeding node capacity', () => {
    const nodes = [centerCoreA, centerCoreB, centerCoreC, centerCoreD]
    const links = buildMeshLinks(nodes)
    const linkKeys = links.map((link) => linkKey(link.source.id, link.target.id))

    expect(linkKeys).toContain('center-a::center-d')

    for (const node of nodes) {
      expect(linkDegree(links, node.id)).toBeLessThanOrEqual(3)
    }
  })
})
