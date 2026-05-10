import { describe, expect, it } from 'vitest'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import {
  createDebugNode,
  debugNodeMatchesShortcut,
  debugNodeShortcutCount,
  getDebugNodeShortcutBlueprint,
  hoverCardPlacement,
  isTextEditingTarget,
  latencyLabel,
  nodeMetrics,
  nodeVisuals,
  renderKind,
  roleLabel,
  statusFill,
  statusTone
} from '@/features/network/components/MeshViz.helpers'

const servingNode: MeshNode = {
  id: 'serving-node',
  label: 'SERVING',
  x: 76,
  y: 62,
  status: 'online',
  meshState: 'serving',
  servingModels: ['llama-mesh-q4'],
  subLabel: 'EU West',
  vramGB: 24,
  latencyMs: 3.4
}

const hostPeer: Peer = {
  id: 'peer-1',
  hostname: 'peer-1.mesh',
  region: 'us-east',
  status: 'degraded',
  hostedModels: ['peer-model-q8'],
  sharePct: 37,
  latencyMs: 1.2,
  loadPct: 41,
  shortId: 'peer-1',
  role: 'host',
  vramGB: 48
}

describe('MeshViz helpers', () => {
  it('derives render labels and hover metrics from node and peer data', () => {
    expect(renderKind(servingNode, undefined)).toBe('serving')
    expect(roleLabel(servingNode, hostPeer)).toBe('Host')
    expect(latencyLabel(servingNode, hostPeer)).toBe('1ms')

    expect(nodeMetrics(servingNode, hostPeer).map(({ id, label, value }) => ({ id, label, value }))).toEqual([
      { id: 'model', label: 'Model', value: 'peer-model-q8' },
      { id: 'latency', label: 'Latency', value: '1ms' },
      { id: 'compute', label: 'Compute', value: 'us-east' },
      { id: 'vram', label: 'VRAM', value: '48.0 GB' }
    ])
  })

  it('keeps status and visual tokens aligned for debug and non-debug nodes', () => {
    expect(statusFill('online')).toBe('var(--color-accent)')
    expect(statusFill('degraded')).toBe('var(--color-warn)')
    expect(statusTone('offline')).toEqual({
      background: 'color-mix(in oklab, var(--color-bad) 14%, var(--color-background))',
      borderColor: 'color-mix(in oklab, var(--color-bad) 24%, var(--color-background))',
      color: 'var(--color-bad)'
    })

    const debugHost = createDebugNode(3, getDebugNodeShortcutBlueprint(3), { x: 20, y: 40 })

    expect(nodeVisuals(servingNode, undefined, false, false)).toEqual({
      fill: 'var(--color-accent)',
      haloSize: 36,
      coreSize: 16,
      labelColor: 'var(--color-foreground)'
    })
    expect(nodeVisuals(debugHost, hostPeer, false, false)).toEqual({
      fill: 'var(--color-warn)',
      haloSize: 48,
      coreSize: 17,
      labelColor: 'var(--color-warn)'
    })

    const schemeNodeColors = ['muted-node', 'default-node', 'tertiary-node', 'self-node'] as const

    expect(nodeVisuals(servingNode, undefined, true, false, schemeNodeColors).fill).toBe('self-node')
    expect(nodeVisuals(servingNode, undefined, false, false, schemeNodeColors).fill).toBe('default-node')
    expect(hoverCardPlacement(servingNode)).toEqual({ side: 'top', align: 'end' })
  })

  it('creates deterministic debug nodes and matches them back to shortcut groups', () => {
    const clientNode = createDebugNode(1, getDebugNodeShortcutBlueprint(1), { x: 10, y: 20 })
    const workerNode = createDebugNode(2, getDebugNodeShortcutBlueprint(2), { x: 30, y: 40 })

    expect(clientNode).toMatchObject({
      id: 'debug-client-1',
      label: 'DEBUG-CLIENT-1',
      client: true,
      hostname: 'debug-client-1.mesh.test',
      latencyMs: null
    })
    expect(workerNode).toMatchObject({
      id: 'debug-worker-2',
      label: 'DEBUG-WORKER-2',
      renderKind: 'worker',
      hostname: 'debug-worker-2.mesh.test',
      vramGB: 24,
      latencyMs: 7.2
    })
    expect(debugNodeMatchesShortcut(clientNode, 1)).toBe(true)
    expect(debugNodeMatchesShortcut(workerNode, 2)).toBe(true)
    expect(debugNodeMatchesShortcut(workerNode, 3)).toBe(false)
  })

  it('detects text-editing targets and supported debug shortcut keys', () => {
    const input = document.createElement('input')
    const textarea = document.createElement('textarea')
    const editable = document.createElement('div')
    Object.defineProperty(editable, 'isContentEditable', { value: true })

    expect(isTextEditingTarget(input)).toBe(true)
    expect(isTextEditingTarget(textarea)).toBe(true)
    expect(isTextEditingTarget(editable)).toBe(true)
    expect(isTextEditingTarget(document.createElement('button'))).toBe(false)
    expect(isTextEditingTarget(null)).toBe(false)

    expect(debugNodeShortcutCount(new KeyboardEvent('keydown', { key: '1' }))).toBe(1)
    expect(debugNodeShortcutCount(new KeyboardEvent('keydown', { code: 'Digit2' }))).toBe(2)
    expect(debugNodeShortcutCount(new KeyboardEvent('keydown', { key: 'x' }))).toBeUndefined()
  })
})
