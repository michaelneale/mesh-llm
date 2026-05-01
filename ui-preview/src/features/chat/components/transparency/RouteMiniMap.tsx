import { useEffect, useRef } from 'react'
import { animate } from 'animejs'

import type { TransparencyNode } from '@/features/app-tabs/types'

type RouteMiniMapProps = { nodes: TransparencyNode[]; pickId: string; direction: 'in' | 'out' }
type Point = { x: number; y: number }
type PositionedNode = { node: TransparencyNode; point: Point }

const WIDTH = 300
const HEIGHT = 90
const LOCAL_POINT: Point = { x: 36, y: HEIGHT / 2 }
const TOP_PEER_POINT: Point = { x: WIDTH - 36, y: 22 }
const BOTTOM_PEER_POINT: Point = { x: WIDTH - 36, y: HEIGHT - 22 }

function normalizeNodeId(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '')
}

function isDeskNode(node: TransparencyNode) {
  return node.id === 'desk' || normalizeNodeId(node.label) === 'you'
}

function findNode(nodes: TransparencyNode[], id: string) {
  const normalizedId = normalizeNodeId(id)
  return nodes.find((node) => node.id === id)
    ?? nodes.find((node) => normalizeNodeId(node.id) === normalizedId)
    ?? nodes.find((node) => normalizeNodeId(node.label) === normalizedId)
}

function nodeLabel(node: TransparencyNode, local: boolean) {
  return local ? `YOU · ${node.label}` : node.label
}

function packetTransform(point: Point) {
  return `translate(${point.x} ${point.y})`
}

export function RouteMiniMap({ nodes, pickId, direction }: RouteMiniMapProps) {
  const packetRef = useRef<SVGGElement>(null)
  const localNode = nodes.find((node) => node.isLocal === true)
    ?? nodes.find((node) => !isDeskNode(node))
    ?? nodes[0]

  const pickedNode = localNode ? findNode(nodes, pickId) ?? localNode : undefined
  const remoteNodes = localNode
    ? nodes.filter((node) => !isDeskNode(node) && node.id !== localNode.id)
    : []
  const orderedRemoteNodes = [
    localNode && pickedNode?.id !== localNode.id ? pickedNode : undefined,
    ...remoteNodes.filter((node) => node.id !== pickedNode?.id),
  ].filter((node): node is TransparencyNode => node != null)

  const positionedPeers: PositionedNode[] = orderedRemoteNodes.slice(0, 2).map((node, index) => ({
    node,
    point: index === 0 ? TOP_PEER_POINT : BOTTOM_PEER_POINT,
  }))
  const pickedPeer = positionedPeers.find(({ node }) => node.id === pickedNode?.id)
  const activeTarget = pickedPeer?.point ?? LOCAL_POINT
  const packetFrom = direction === 'out' ? LOCAL_POINT : activeTarget
  const packetTo = direction === 'out' ? activeTarget : LOCAL_POINT
  const hasActiveRoute = localNode != null && activeTarget !== LOCAL_POINT

  useEffect(() => {
    const packetElement = packetRef.current

    if (!packetElement || !hasActiveRoute) {
      return undefined
    }

    const packetPosition = { x: packetFrom.x, y: packetFrom.y }
    const placePacket = (point: Point) => {
      packetElement.setAttribute('transform', packetTransform(point))
    }

    placePacket(packetPosition)

    const animation = animate(packetPosition, {
      x: { from: packetFrom.x, to: packetTo.x },
      y: { from: packetFrom.y, to: packetTo.y },
      duration: 1500,
      ease: 'out(2)',
      loop: true,
      onUpdate: () => placePacket(packetPosition),
    })

    return () => {
      animation.revert()
      placePacket({ x: packetFrom.x, y: packetFrom.y })
    }
  }, [hasActiveRoute, packetFrom.x, packetFrom.y, packetTo.x, packetTo.y])

  if (!localNode || !pickedNode) {
    return null
  }

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="block h-[90px] w-full"
      role="img"
      aria-label={`${direction} route map`}
    >
      {positionedPeers.filter(({ node }) => node.id !== pickedNode.id).map(({ node, point }) => (
        <line
          key={node.id}
          x1={LOCAL_POINT.x}
          y1={LOCAL_POINT.y}
          x2={point.x}
          y2={point.y}
          stroke="var(--color-border)"
          strokeWidth="1"
          strokeDasharray="2 3"
        />
      ))}

      {activeTarget !== LOCAL_POINT && (
        <line
          x1={LOCAL_POINT.x}
          y1={LOCAL_POINT.y}
          x2={activeTarget.x}
          y2={activeTarget.y}
          stroke="var(--color-accent)"
          strokeWidth="1.5"
        />
      )}

      {hasActiveRoute && (
        <g ref={packetRef} transform={packetTransform(packetFrom)}>
          <circle r="8" fill="var(--color-accent)" opacity="0.2" />
          <circle r="4" fill="var(--color-accent)" />
        </g>
      )}

      <g>
        <circle
          cx={LOCAL_POINT.x}
          cy={LOCAL_POINT.y}
          r="9"
          fill="color-mix(in oklab, var(--color-accent) 40%, var(--color-panel-strong))"
          stroke="var(--color-accent)"
          strokeWidth="1.2"
        />
        <text
          x={LOCAL_POINT.x}
          y={LOCAL_POINT.y + 22}
          textAnchor="middle"
          fontSize="var(--density-type-micro)"
          fill="var(--color-fg-dim)"
          fontFamily="JetBrains Mono, ui-monospace, Menlo, monospace"
          letterSpacing="0.5"
        >
          {nodeLabel(localNode, true)}
        </text>
      </g>

      {positionedPeers.map(({ node, point }) => {
        const active = node.id === pickedNode.id
        return (
          <g key={node.id}>
            <circle
              cx={point.x}
              cy={point.y}
              r={active ? 9 : 6}
              fill={active ? 'var(--color-accent)' : 'var(--color-panel-strong)'}
              stroke={active ? 'var(--color-accent)' : 'var(--color-border)'}
              strokeWidth="1.2"
            />
            <text
              x={point.x}
              y={point.y < HEIGHT / 2 ? point.y - 13 : point.y + 21}
              textAnchor="middle"
              fontSize="var(--density-type-micro)"
              fill="var(--color-fg-dim)"
              fontFamily="JetBrains Mono, ui-monospace, Menlo, monospace"
              letterSpacing="0.5"
            >
              {nodeLabel(node, false)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
