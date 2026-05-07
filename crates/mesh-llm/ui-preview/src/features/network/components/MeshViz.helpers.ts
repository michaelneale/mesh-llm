import { Activity, Cpu, HardDrive, Hash, type LucideIcon } from 'lucide-react'
import type { CSSProperties } from 'react'
import type { MeshVizNodeColors } from '@/features/network/lib/mesh-viz-dot-color-schemes'
import type { MeshNode, MeshNodeRenderKind, Peer } from '@/features/app-tabs/types'
import { nonBlankText } from '@/features/network/lib/mesh-node-labels'

export type HoverMetric = {
  id: string
  label: string
  value: string
  icon: LucideIcon
}

export type HoverCardPlacement = {
  side: 'top' | 'bottom'
  align: 'start' | 'center' | 'end'
}

export type DebugMeshNode = MeshNode & {
  debug: true
  renderKind: Exclude<MeshNodeRenderKind, 'self'>
}

export type DebugNodeBlueprint = {
  renderKind: Exclude<MeshNodeRenderKind, 'self'>
  label: string
  subLabel: string
  meshState: NonNullable<MeshNode['meshState']>
  status: MeshNode['status']
  host?: boolean
  client?: boolean
  servingModels?: string[]
  hostnamePrefix: string
  vramGB?: number
  latencyMs?: number | null
}

export type DebugNodeShortcut = 1 | 2 | 3

export type DebugNodePosition = { x: number; y: number }

export const NODE_LABEL_FADE_THRESHOLD = 8

const DEBUG_NODE_BLUEPRINTS: [DebugNodeBlueprint, ...DebugNodeBlueprint[]] = [
  {
    renderKind: 'client',
    label: 'CLIENT',
    subLabel: 'DEBUG · CLIENT',
    meshState: 'client',
    status: 'online',
    client: true,
    hostnamePrefix: 'debug-client',
    latencyMs: null
  },
  {
    renderKind: 'worker',
    label: 'WORKER',
    subLabel: 'DEBUG · WORKER',
    meshState: 'standby',
    status: 'online',
    hostnamePrefix: 'debug-worker',
    vramGB: 24,
    latencyMs: 7.2
  },
  {
    renderKind: 'serving',
    label: 'SERVING',
    subLabel: 'DEBUG · SERVING',
    meshState: 'serving',
    status: 'online',
    servingModels: ['debug-model-q4'],
    hostnamePrefix: 'debug-serving',
    vramGB: 48,
    latencyMs: 3.8
  },
  {
    renderKind: 'active',
    label: 'ACTIVE',
    subLabel: 'DEBUG · ACTIVE',
    meshState: 'loading',
    status: 'degraded',
    servingModels: ['debug-active-route'],
    hostnamePrefix: 'debug-active',
    vramGB: 32,
    latencyMs: 11.4
  },
  {
    renderKind: 'worker',
    label: 'HOST',
    subLabel: 'DEBUG · HOST',
    meshState: 'standby',
    status: 'online',
    host: true,
    hostnamePrefix: 'debug-host',
    vramGB: 64,
    latencyMs: 1.6
  }
]

const DEBUG_NODE_SHORTCUT_BLUEPRINTS = {
  1: DEBUG_NODE_BLUEPRINTS[0],
  2: DEBUG_NODE_BLUEPRINTS[1],
  3: DEBUG_NODE_BLUEPRINTS[4]
} satisfies Record<DebugNodeShortcut, DebugNodeBlueprint>

export function prefersReducedMotion() {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  )
}

export function isTextEditingTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false
  }

  const tagName = target.tagName.toLowerCase()

  return target.isContentEditable || tagName === 'input' || tagName === 'textarea' || tagName === 'select'
}

export function debugNodeShortcutCount(event: KeyboardEvent) {
  if (event.key === '1' || event.code === 'Digit1') return 1
  if (event.key === '2' || event.code === 'Digit2') return 2
  if (event.key === '3' || event.code === 'Digit3') return 3
  return undefined
}

export function statusFill(status: MeshNode['status']) {
  if (status === 'online') return 'var(--color-accent)'
  if (status === 'degraded') return 'var(--color-warn)'
  return 'var(--color-muted-foreground)'
}

export function statusTone(status: MeshNode['status']): CSSProperties {
  if (status === 'online') {
    return {
      background: 'color-mix(in oklab, var(--color-good) 18%, var(--color-background))',
      borderColor: 'color-mix(in oklab, var(--color-good) 30%, var(--color-background))',
      color: 'var(--color-good)'
    }
  }

  if (status === 'degraded') {
    return {
      background: 'color-mix(in oklab, var(--color-warn) 16%, var(--color-background))',
      borderColor: 'color-mix(in oklab, var(--color-warn) 28%, var(--color-background))',
      color: 'var(--color-warn)'
    }
  }

  return {
    background: 'color-mix(in oklab, var(--color-bad) 14%, var(--color-background))',
    borderColor: 'color-mix(in oklab, var(--color-bad) 24%, var(--color-background))',
    color: 'var(--color-bad)'
  }
}

export function isDebugNode(node: MeshNode): node is DebugMeshNode {
  return 'debug' in node && node.debug === true
}

export function isHostNode(node: MeshNode, peer: Peer | undefined) {
  return Boolean(node.host || peer?.role === 'host')
}

export function renderKind(node: MeshNode, peer: Peer | undefined): MeshNodeRenderKind {
  if (peer?.role === 'you' || node.role === 'self') return 'self'
  if (node.renderKind) return node.renderKind
  if (node.client) return 'client'
  if (node.meshState === 'serving' || (node.servingModels?.length ?? 0) > 0) return 'serving'
  if (node.meshState === 'loading' || node.status === 'degraded') return 'active'
  return 'worker'
}

export function debugFill(node: MeshNode, peer: Peer | undefined) {
  if (isHostNode(node, peer)) return 'var(--color-warn)'

  const kind = renderKind(node, peer)

  if (kind === 'client') return 'var(--color-muted-foreground)'
  if (kind === 'active') return 'var(--color-good)'
  if (kind === 'serving') return 'var(--color-foreground)'
  return 'var(--color-accent)'
}

export function schemeNodeFill(
  node: MeshNode,
  peer: Peer | undefined,
  isSelf: boolean,
  nodeColors: MeshVizNodeColors | undefined
) {
  if (!nodeColors) {
    return isDebugNode(node) ? debugFill(node, peer) : statusFill(node.status)
  }

  if (isSelf) return nodeColors[3]
  if (node.status === 'offline') return nodeColors[0]
  if (node.status === 'degraded') return nodeColors[2]
  if (isHostNode(node, peer)) return nodeColors[2]

  const kind = renderKind(node, peer)

  if (kind === 'client') return nodeColors[0]
  if (kind === 'active') return nodeColors[2]
  return nodeColors[1]
}

export function nodeVisuals(
  node: MeshNode,
  peer: Peer | undefined,
  isSelf: boolean,
  isSelected: boolean,
  nodeColors?: MeshVizNodeColors
) {
  const fill = schemeNodeFill(node, peer, isSelf, nodeColors)
  const isHost = isHostNode(node, peer)
  const kind = renderKind(node, peer)
  const hostBoost = isDebugNode(node) && isHost ? 8 : 0

  return {
    fill,
    haloSize: isSelf ? 52 : isDebugNode(node) ? 40 + hostBoost : isHost ? 42 : kind === 'client' ? 34 : 36,
    coreSize: isSelf ? 18 : isHost ? 17 : kind === 'client' ? 14 : 16,
    labelColor: isSelected || isSelf || isDebugNode(node) ? fill : 'var(--color-foreground)'
  }
}

export function getDebugNodeShortcutBlueprint(shortcut: DebugNodeShortcut) {
  return DEBUG_NODE_SHORTCUT_BLUEPRINTS[shortcut]
}

export function debugNodeMatchesShortcut(node: DebugMeshNode, shortcut: DebugNodeShortcut) {
  const blueprint = getDebugNodeShortcutBlueprint(shortcut)

  if (blueprint.host) return Boolean(node.host)
  if (blueprint.client) return Boolean(node.client)

  return node.renderKind === blueprint.renderKind && !node.host && !node.client
}

export function createDebugNode(
  index: number,
  blueprint: DebugNodeBlueprint,
  position: DebugNodePosition
): DebugMeshNode {
  const node: DebugMeshNode = {
    debug: true,
    id: `debug-${blueprint.label.toLowerCase()}-${index}`,
    label: `DEBUG-${blueprint.label}-${index}`,
    x: position.x,
    y: position.y,
    status: blueprint.status,
    role: 'peer',
    subLabel: blueprint.subLabel,
    renderKind: blueprint.renderKind,
    meshState: blueprint.meshState,
    hostname: `${blueprint.hostnamePrefix}-${index}.mesh.test`
  }

  if (blueprint.host) node.host = blueprint.host
  if (blueprint.client) node.client = blueprint.client
  if (blueprint.servingModels) node.servingModels = blueprint.servingModels
  if (blueprint.latencyMs !== undefined) node.latencyMs = blueprint.latencyMs
  if (blueprint.vramGB !== undefined) node.vramGB = blueprint.vramGB

  return node
}

export function roleLabel(node: MeshNode, peer: Peer | undefined) {
  if (peer?.role === 'you' || node.role === 'self') return 'You'
  if (isHostNode(node, peer)) return 'Host'
  if (peer?.role === 'client') return 'Client'
  if (node.client || node.renderKind === 'client') return 'Client'
  if (node.renderKind === 'serving') return 'Serving'
  if (node.renderKind === 'active') return 'Active'
  if (node.renderKind === 'worker') return 'Worker'
  return 'Peer'
}

export function latencyLabel(node: MeshNode, peer: Peer | undefined) {
  if (!peer) {
    const latencyMs = node.latencyMs

    if (latencyMs == null) return 'N/A'
    return latencyMs < 2 ? '<1 ms' : `${latencyMs.toFixed(1)} ms`
  }

  return peer.latencyMs < 2 ? '<1 ms' : `${peer.latencyMs.toFixed(1)} ms`
}

export function hoverCardPlacement(node: MeshNode): HoverCardPlacement {
  return {
    side: node.y > 58 ? 'top' : 'bottom',
    align: node.x < 30 ? 'start' : node.x > 70 ? 'end' : 'center'
  }
}

export function nodeMetrics(node: MeshNode, peer: Peer | undefined): HoverMetric[] {
  const computeLabel = peer
    ? peer.role === 'you'
      ? 'Local'
      : (nonBlankText(peer.region) ?? roleLabel(node, peer))
    : node.role === 'self'
      ? 'Local'
      : roleLabel(node, peer)

  return [
    {
      id: 'model',
      label: 'Model',
      value: peer?.hostedModels[0] ?? node.servingModels?.[0] ?? (node.client ? 'API-only' : 'No hosted model'),
      icon: Hash
    },
    {
      id: 'latency',
      label: 'Latency',
      value: latencyLabel(node, peer),
      icon: Activity
    },
    {
      id: 'compute',
      label: 'Compute',
      value: computeLabel,
      icon: Cpu
    },
    {
      id: 'vram',
      label: 'VRAM',
      value:
        peer?.vramGB != null
          ? `${peer.vramGB.toFixed(1)} GB`
          : node.vramGB != null
            ? `${node.vramGB.toFixed(1)} GB`
            : 'N/A',
      icon: HardDrive
    }
  ]
}
