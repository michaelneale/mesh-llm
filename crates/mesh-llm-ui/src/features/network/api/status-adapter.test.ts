import { describe, expect, it } from 'vitest'
import { adaptStatusToDashboard } from '@/features/network/api/status-adapter'
import { buildDashboardMeshNodes } from '@/features/network/lib/dashboard-mesh-nodes'
import type { StatusPayload } from '@/lib/api/types'

const PUBLIC_STATUS_PAYLOAD: StatusPayload = {
  node_id: '16ce0bb4de',
  node_state: 'serving',
  model_name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
  peers: [
    {
      id: 'aeac0d8e53',
      owner: { status: 'unsigned', verified: false },
      role: 'Client',
      state: 'client',
      models: [],
      vram_gb: 0,
      serving_models: [],
      hosted_models: [],
      version: '0.65.0',
      rtt_ms: 7,
      hostname: '1266a345aeb9'
    }
  ],
  models: [],
  my_vram_gb: 12.5,
  gpus: [],
  serving_models: [{ name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M', node_id: '16ce0bb4de', status: 'warm' }],
  my_hostname: 'public-host',
  version: '0.65.0-rc2'
}

describe('adaptStatusToDashboard', () => {
  it('renders the public mesh welcome and connect command when publication_state is public', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      publication_state: 'public',
      nostr_discovery: false
    })

    expect(dashboard.hero).toEqual(
      expect.objectContaining({
        title: 'Welcome to the public mesh',
        description: expect.stringContaining('shared community capacity')
      })
    )
    expect(dashboard.connect).toEqual(
      expect.objectContaining({
        runCommand: 'mesh-llm --auto',
        description: 'join the public mesh'
      })
    )
  })

  it('keeps publication_state precedence over nostr_discovery fallback', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      publication_state: 'private',
      nostr_discovery: true
    })

    expect(dashboard.hero.title).toBe('Your private mesh')
    expect(dashboard.connect.runCommand).toBe('mesh-llm --auto --join <mesh-invite-token>')
  })

  it('falls back to nostr_discovery when publication_state is absent', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      nostr_discovery: true
    })

    expect(dashboard.hero.title).toBe('Welcome to the public mesh')
    expect(dashboard.connect.runCommand).toBe('mesh-llm --auto')
  })

  it('accepts public status peers without node_id', () => {
    const dashboard = adaptStatusToDashboard(PUBLIC_STATUS_PAYLOAD)

    expect(dashboard.peers).toEqual([
      expect.objectContaining({
        id: '16ce0bb4de',
        hostname: 'public-host',
        shortId: '16ce0bb4',
        role: 'you'
      }),
      expect.objectContaining({
        id: 'aeac0d8e53',
        hostname: '1266a345aeb9',
        shortId: 'aeac0d8e',
        status: 'online',
        role: 'client',
        nodeState: 'client',
        latencyMs: 7,
        vramGB: 0,
        owner: 'unsigned'
      })
    ])
  })

  it('adapts live status into the six-cell dashboard metrics bar', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      inflight_requests: 3,
      gpus: [{ idx: 0, name: 'local-gpu', total_vram_gb: 12.5, used_vram_gb: 5 }],
      peers: [
        {
          id: 'remote-serving',
          role: 'Host',
          state: 'serving',
          serving_models: ['Remote-Model-Q4_K_M'],
          hosted_models: [],
          models: [],
          vram_gb: 20,
          gpus: [{ idx: 0, name: 'remote-gpu', total_vram_gb: 20, used_vram_gb: 4 }],
          hostname: 'remote-serving'
        }
      ]
    })

    expect(dashboard.statusMetrics.map((metric) => metric.id)).toEqual([
      'node-id',
      'owner',
      'nodes',
      'active-models',
      'mesh-vram',
      'inflight'
    ])
    expect(dashboard.statusMetrics.find((metric) => metric.id === 'mesh-vram')).toEqual(
      expect.objectContaining({ value: '32.5', unit: 'GB' })
    )
    expect(dashboard.statusMetrics.find((metric) => metric.id === 'mesh-vram')).not.toHaveProperty('meta')
    expect(dashboard.statusMetrics.find((metric) => metric.id === 'inflight')).toEqual(
      expect.objectContaining({ value: 3 })
    )
    expect(dashboard.statusMetrics.find((metric) => metric.id === 'inflight')).not.toHaveProperty('meta')
    expect(dashboard.statusMetrics.find((metric) => metric.id === 'active-models')).toEqual(
      expect.objectContaining({ value: 2, meta: '1 loaded locally · 1 remote' })
    )
  })

  it('keeps live client and standby nodes connected instead of offline', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      node_state: 'standby',
      serving_models: ['Local-String-Model-Q4_K_M'],
      peers: [
        {
          id: 'idle-worker',
          role: 'Worker',
          state: 'standby',
          models: [],
          serving_models: [],
          hosted_models: [],
          vram_gb: 0,
          hostname: 'idle-worker'
        },
        {
          id: 'api-client',
          role: 'Client',
          state: 'client',
          models: [],
          serving_models: [],
          hosted_models: [],
          vram_gb: 0,
          hostname: 'api-client'
        }
      ]
    })

    expect(dashboard.peers.find((peer) => peer.id === '16ce0bb4de')).toEqual(
      expect.objectContaining({ status: 'online', nodeState: 'standby', hostedModels: ['Local-String-Model-Q4_K_M'] })
    )
    expect(dashboard.peers.find((peer) => peer.id === 'idle-worker')).toEqual(
      expect.objectContaining({ status: 'online', nodeState: 'standby' })
    )
    expect(dashboard.peers.find((peer) => peer.id === 'api-client')).toEqual(
      expect.objectContaining({ status: 'online', nodeState: 'client' })
    )
    expect(dashboard.peerSummary.online).toBe(3)
    expect(dashboard.meshNodeSeeds[0]).toEqual(
      expect.objectContaining({ status: 'online', meshState: 'standby', servingModels: ['Local-String-Model-Q4_K_M'] })
    )
  })

  it('defaults the in-flight request count to zero when the backend field is absent', () => {
    const dashboard = adaptStatusToDashboard(PUBLIC_STATUS_PAYLOAD)

    expect(dashboard.statusMetrics.find((metric) => metric.id === 'inflight')).toEqual(
      expect.objectContaining({ value: 0 })
    )
    expect(dashboard.statusMetrics.find((metric) => metric.id === 'inflight')).not.toHaveProperty('meta')
  })

  it('only seeds the self mesh node so live peers use clustered placement', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      peers: [
        {
          id: 'peer-a',
          role: 'Host',
          state: 'serving',
          models: ['Hermes-2-Pro-Mistral-7B-Q4_K_M'],
          serving_models: ['Hermes-2-Pro-Mistral-7B-Q4_K_M'],
          hosted_models: ['Hermes-2-Pro-Mistral-7B-Q4_K_M'],
          vram_gb: 24,
          hostname: 'worker-a'
        },
        {
          id: 'peer-b',
          role: 'Worker',
          state: 'serving',
          models: ['Qwen3.5-4B-UD'],
          serving_models: ['Qwen3.5-4B-UD'],
          hosted_models: ['Qwen3.5-4B-UD'],
          vram_gb: 16,
          hostname: 'worker-b'
        },
        {
          id: 'peer-c',
          role: 'Client',
          state: 'client',
          models: [],
          serving_models: [],
          hosted_models: [],
          vram_gb: 0,
          hostname: 'client-c'
        }
      ]
    })

    expect(dashboard.meshNodeSeeds).toEqual([
      expect.objectContaining({
        id: '16ce0bb4de',
        role: 'self',
        x: 50,
        y: 50
      })
    ])

    const meshNodes = buildDashboardMeshNodes(dashboard.peers, dashboard.meshId, dashboard.meshNodeSeeds)
    const selfNode = meshNodes.find((node) => node.role === 'self')
    const remoteNodes = meshNodes.filter((node) => node.role !== 'self')

    expect(selfNode).toEqual(expect.objectContaining({ id: '16ce0bb4de', x: 50, y: 50 }))
    expect(remoteNodes).toHaveLength(3)
    expect(remoteNodes.map((node) => node.peerId)).toEqual(['peer-a', 'peer-b', 'peer-c'])
    expect(remoteNodes.find((node) => node.peerId === 'peer-a')).toEqual(
      expect.objectContaining({ host: true, renderKind: 'serving', meshState: 'serving' })
    )
    expect(remoteNodes.find((node) => node.peerId === 'peer-b')).toEqual(
      expect.objectContaining({ client: false, host: false, renderKind: 'serving', meshState: 'serving' })
    )
    expect(remoteNodes.find((node) => node.peerId === 'peer-c')).toEqual(
      expect.objectContaining({ client: true, host: false, renderKind: 'client', meshState: 'client' })
    )
    for (const node of remoteNodes) {
      expect(node.x).toBeGreaterThanOrEqual(0)
      expect(node.x).toBeLessThanOrEqual(100)
      expect(node.y).toBeGreaterThanOrEqual(0)
      expect(node.y).toBeLessThanOrEqual(100)
    }
  })

  it('keeps idle live workers distinct from live clients', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      peers: [
        {
          id: 'idle-worker',
          role: 'Worker',
          state: 'standby',
          models: [],
          serving_models: [],
          hosted_models: [],
          vram_gb: 0,
          hostname: 'idle-worker'
        },
        {
          id: 'api-client',
          role: 'Client',
          state: 'client',
          models: [],
          serving_models: [],
          hosted_models: [],
          vram_gb: 0,
          hostname: 'api-client'
        }
      ]
    })

    const meshNodes = buildDashboardMeshNodes(dashboard.peers, dashboard.meshId, dashboard.meshNodeSeeds)

    expect(dashboard.peers.find((peer) => peer.id === 'idle-worker')).toEqual(
      expect.objectContaining({ role: 'worker', status: 'online', nodeState: 'standby' })
    )
    expect(dashboard.peers.find((peer) => peer.id === 'api-client')).toEqual(
      expect.objectContaining({ role: 'client', status: 'online', nodeState: 'client' })
    )
    expect(meshNodes.find((node) => node.peerId === 'idle-worker')).toEqual(
      expect.objectContaining({ client: false, renderKind: 'worker', meshState: 'standby' })
    )
    expect(meshNodes.find((node) => node.peerId === 'api-client')).toEqual(
      expect.objectContaining({ client: true, renderKind: 'client', meshState: 'client' })
    )
  })

  it('normalizes live hosted model fallbacks and infers share from VRAM when missing', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      my_vram_gb: 10,
      serving_models: [],
      model_name: 'Self-Model-Q4_K_M',
      peers: [
        {
          id: 'hosted-fallback',
          role: 'Host',
          state: 'serving',
          serving_models: [],
          hosted_models: ['Hosted-Model-Q4_K_M'],
          models: ['Ignored-Model-Q4_K_M'],
          vram_gb: 20,
          hostname: 'hosted-fallback'
        },
        {
          id: 'models-fallback',
          role: 'Worker',
          state: 'serving',
          serving_models: [],
          hosted_models: [],
          models: ['Models-Only-Q4_K_M'],
          vram_gb: 0,
          hostname: 'models-fallback'
        }
      ]
    })

    expect(dashboard.peers.find((peer) => peer.id === '16ce0bb4de')).toEqual(
      expect.objectContaining({ hostedModels: ['Self-Model-Q4_K_M'], sharePct: 33 })
    )
    expect(dashboard.peers.find((peer) => peer.id === 'hosted-fallback')).toEqual(
      expect.objectContaining({ hostedModels: ['Hosted-Model-Q4_K_M'], sharePct: 67 })
    )
    expect(dashboard.peers.find((peer) => peer.id === 'models-fallback')).toEqual(
      expect.objectContaining({ hostedModels: ['Models-Only-Q4_K_M'], sharePct: 0 })
    )
  })

  it('does not infer mesh share for non-serving peers without hosted models', () => {
    const dashboard = adaptStatusToDashboard({
      ...PUBLIC_STATUS_PAYLOAD,
      my_vram_gb: 10,
      peers: [
        {
          id: 'serving-worker',
          role: 'Worker',
          state: 'serving',
          serving_models: ['Serving-Model-Q4_K_M'],
          hosted_models: [],
          models: [],
          vram_gb: 20,
          hostname: 'serving-worker'
        },
        {
          id: 'standby-worker',
          role: 'Worker',
          state: 'standby',
          serving_models: [],
          hosted_models: [],
          models: [],
          vram_gb: 30,
          hostname: 'standby-worker'
        },
        {
          id: 'api-client',
          role: 'Client',
          state: 'client',
          serving_models: [],
          hosted_models: [],
          models: [],
          vram_gb: 40,
          hostname: 'api-client'
        }
      ]
    })

    expect(dashboard.peers.find((peer) => peer.id === '16ce0bb4de')).toEqual(expect.objectContaining({ sharePct: 33 }))
    expect(dashboard.peers.find((peer) => peer.id === 'serving-worker')).toEqual(
      expect.objectContaining({ sharePct: 67 })
    )
    expect(dashboard.peers.find((peer) => peer.id === 'standby-worker')).toEqual(
      expect.objectContaining({ sharePct: 0 })
    )
    expect(dashboard.peers.find((peer) => peer.id === 'api-client')).toEqual(expect.objectContaining({ sharePct: 0 }))
  })
})
