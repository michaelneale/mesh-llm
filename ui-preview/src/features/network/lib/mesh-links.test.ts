import { describe, expect, it } from 'vitest'
import { buildMeshLinks } from '@/features/network/lib/mesh-links'
import type { MeshNode } from '@/features/app-tabs/types'

const workerA: MeshNode = {
  id: 'worker-a',
  label: 'Worker A',
  x: 10,
  y: 10,
  status: 'online',
  renderKind: 'worker',
}
const workerB: MeshNode = {
  id: 'worker-b',
  label: 'Worker B',
  x: 90,
  y: 10,
  status: 'online',
  renderKind: 'worker',
}
const client: MeshNode = {
  id: 'client-a',
  label: 'Client A',
  x: 12,
  y: 12,
  status: 'online',
  renderKind: 'client',
  client: true,
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
})
