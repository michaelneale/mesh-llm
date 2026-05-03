import { useMemo } from 'react'
import { MeshViz } from '@/features/network/components/MeshViz'
import type { MeshNode, MeshNodeRenderKind } from '@/features/app-tabs/types'

const PERF_NODE_COUNT = 200
const PERF_SELF_ID = 'perf-self'
const PERF_MESH_ID = 'meshviz-perf-200'
const GOLDEN_ANGLE_RADIANS = Math.PI * (3 - Math.sqrt(5))

type MeshVizPerfHarnessProps = {
  height?: number
}

function buildMeshVizPerfNodes(count: number): MeshNode[] {
  return Array.from({ length: count }, (_, index) => {
    if (index === 0) {
      return {
        id: PERF_SELF_ID,
        peerId: 'perf-peer-self',
        label: 'PERF SELF',
        subLabel: 'PERF · YOU',
        x: 50,
        y: 50,
        status: 'online',
        role: 'self',
        host: true,
        renderKind: 'self',
        meshState: 'serving',
        latencyMs: 1,
        vramGB: 64,
      }
    }

    const angle = index * GOLDEN_ANGLE_RADIANS
    const radius = Math.sqrt(index / Math.max(count - 1, 1)) * 86
    const renderKind = renderKindForIndex(index)
    const isClient = renderKind === 'client'
    const isHost = !isClient && index % 17 === 0

    return {
      id: `perf-node-${index.toString().padStart(3, '0')}`,
      peerId: `perf-peer-${index.toString().padStart(3, '0')}`,
      label: `PERF ${index.toString().padStart(3, '0')}`,
      subLabel: isClient ? 'CLIENT' : isHost ? 'HOST' : 'SERVING',
      x: 50 + Math.cos(angle) * radius,
      y: 50 + Math.sin(angle) * radius * 0.72,
      status: index % 43 === 0 ? 'degraded' : 'online',
      renderKind,
      meshState: isClient ? 'client' : index % 11 === 0 ? 'standby' : 'serving',
      host: isHost,
      client: isClient,
      servingModels: isClient ? undefined : [`perf-model-${index % 8}`],
      latencyMs: isClient ? null : 1 + (index % 19),
      hostname: `perf-node-${index.toString().padStart(3, '0')}.local`,
      vramGB: isClient ? undefined : 16 + (index % 6) * 8,
    }
  })
}

function renderKindForIndex(index: number): MeshNodeRenderKind {
  if (index % 5 === 0) return 'client'
  if (index % 7 === 0) return 'active'
  if (index % 11 === 0) return 'worker'
  return 'serving'
}

export function MeshVizPerfHarness({ height = 680 }: MeshVizPerfHarnessProps) {
  const nodes = useMemo(() => buildMeshVizPerfNodes(PERF_NODE_COUNT), [])

  return (
    <MeshViz
      animateTopology={false}
      compact
      enableDebugShortcuts
      height={height}
      meshId={PERF_MESH_ID}
      nodes={nodes}
      selfId={PERF_SELF_ID}
    />
  )
}
