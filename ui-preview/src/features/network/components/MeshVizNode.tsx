import { useLayoutEffect, useRef, type RefObject } from 'react'
import * as HoverCardPrimitive from '@radix-ui/react-hover-card'
import { animate } from 'animejs'
import { cn } from '@/lib/cn'
import type { MeshVizNodeColors } from '@/features/network/lib/mesh-viz-dot-color-schemes'
import { pointToScreen, type Viewport } from '@/features/network/lib/mesh-viewport'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import { isDebugNode, nodeVisuals } from './MeshViz.helpers'
import { MeshVizNodeHoverCard } from './MeshVizNodeHoverCard'

export type MeshVizNodeLifecycle = 'entering' | 'present' | 'leaving'

const meshPaletteFadeClassName = 'mesh-palette-fade'

type MeshVizNodeProps = {
  node: MeshNode
  peer?: Peer
  selfId: string
  selectedNodeId?: string
  openNodeId?: string
  hoveredNodeId?: string
  shouldFadeNodeLabels: boolean
  reduceMotion: boolean
  canvasWidth: number
  canvasHeight: number
  viewport: Viewport
  nodeColors?: MeshVizNodeColors
  lifecycle: MeshVizNodeLifecycle
  radarPingRef: RefObject<HTMLSpanElement | null>
  onHoverStart: (nodeId: string) => void
  onHoverEnd: (nodeId: string) => void
  onToggleOpen: (nodeId: string) => void
  onCloseOpen: () => void
}

type MeshVizNodeLabelProps = Pick<
  MeshVizNodeProps,
  | 'node'
  | 'peer'
  | 'selfId'
  | 'selectedNodeId'
  | 'openNodeId'
  | 'hoveredNodeId'
  | 'shouldFadeNodeLabels'
  | 'reduceMotion'
  | 'canvasWidth'
  | 'canvasHeight'
  | 'viewport'
  | 'nodeColors'
  | 'lifecycle'
>

export function MeshVizNode({
  node,
  peer,
  selfId,
  selectedNodeId,
  openNodeId,
  canvasWidth,
  canvasHeight,
  viewport,
  nodeColors,
  lifecycle,
  radarPingRef,
  onHoverStart,
  onHoverEnd,
  onToggleOpen,
  onCloseOpen,
}: MeshVizNodeProps) {
  const isDebug = isDebugNode(node)
  const isSelf = node.id === selfId
  const isSelected = !isDebug && node.id === selectedNodeId
  const isOpen = node.id === openNodeId
  const isLeaving = lifecycle === 'leaving'
  const { fill, haloSize, coreSize } = nodeVisuals(node, peer, isSelf, isSelected, nodeColors)
  const screenPoint = pointToScreen(node, canvasWidth, canvasHeight, viewport)

  return (
    <HoverCardPrimitive.Root
      open={isOpen}
      openDelay={0}
      closeDelay={0}
    >
      <HoverCardPrimitive.Trigger asChild>
        <button
          type="button"
          aria-label={`View ${node.label} node${isSelected ? ' (selected)' : ''}`}
          aria-describedby={isOpen ? `mesh-node-popover-${node.id}` : undefined}
          data-active={isSelected ? 'true' : undefined}
          data-context-open={isOpen ? 'true' : undefined}
          data-debug={isDebug ? 'true' : undefined}
          data-node-lifecycle={lifecycle}
          data-node-x={node.x}
          data-node-y={node.y}
          disabled={isLeaving}
          onPointerEnter={() => onHoverStart(node.id)}
          onPointerLeave={() => onHoverEnd(node.id)}
          onFocus={() => onHoverStart(node.id)}
          onBlur={() => onHoverEnd(node.id)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') onCloseOpen()
          }}
          onClick={() => {
            onToggleOpen(node.id)
          }}
          className={cn(
            'group absolute flex -translate-x-1/2 items-center bg-transparent p-0 outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
            isLeaving && 'pointer-events-none',
            isSelected && 'z-20',
            isOpen && 'z-30',
          )}
          style={{
            left: `${screenPoint.x}px`,
            top: `${screenPoint.y - haloSize / 2}px`,
          }}
        >
          <span
            className="relative grid place-items-center"
            style={{
              transform: 'scale(var(--mesh-node-live-scale, 1))',
              transformOrigin: `50% ${haloSize / 2}px`,
              willChange: 'transform',
            }}
          >
            <span
              data-node-lifecycle={lifecycle}
              className={cn(
                meshPaletteFadeClassName,
                'mesh-node-orb relative grid place-items-center rounded-full',
              )}
              style={{
                color: fill,
                width: haloSize,
                height: haloSize,
              }}
            >
              <span
                aria-hidden="true"
                data-node-id={node.id}
                data-testid="mesh-node-context-highlight"
                className={cn(
                  'pointer-events-none absolute rounded-full transition-opacity duration-150 ease-out',
                  isOpen ? 'opacity-100' : 'opacity-0',
                )}
                style={{
                  inset: -6,
                  background:
                    'radial-gradient(circle, color-mix(in oklab, currentColor 52%, transparent) 0%, color-mix(in oklab, currentColor 28%, transparent) 48%, transparent 74%)',
                  border:
                    '1px solid color-mix(in oklab, currentColor 52%, var(--color-background))',
                  boxShadow:
                    '0 0 20px color-mix(in oklab, currentColor 28%, transparent)',
                }}
              />
              {isSelf && (
                <span
                  ref={radarPingRef}
                  className={cn(meshPaletteFadeClassName, 'absolute inset-0 rounded-full mesh-radar-ping')}
                  style={{ color: fill }}
                />
              )}
              <span
                data-node-id={node.id}
                data-node-lifecycle={lifecycle}
                data-mesh-node-core={node.id}
                data-testid="mesh-node-core"
                className={cn(
                  meshPaletteFadeClassName,
                  'mesh-glow mesh-node-core relative overflow-hidden rounded-full',
                )}
                style={{
                  color: fill,
                  width: coreSize,
                  height: coreSize,
                  backgroundColor: `color-mix(in oklab, currentColor ${isSelf ? '18%' : '14%'}, var(--color-panel-strong))`,
                  border:
                    `1px solid color-mix(in oklab, currentColor ${isSelf ? '68%' : '58%'}, var(--color-border))`,
                  boxShadow: isSelected
                    ? `0 0 ${isSelf ? 16 : 13}px color-mix(in oklab, currentColor 24%, transparent), 0 0 0 3px color-mix(in oklab, currentColor 12%, transparent), inset 0 0 8px color-mix(in oklab, currentColor 12%, transparent)`
                    : `0 0 ${isSelf ? 12 : 10}px color-mix(in oklab, currentColor 16%, transparent), inset 0 0 8px color-mix(in oklab, currentColor 8%, transparent)`,
                }}
              >
                <span
                  aria-hidden="true"
                  data-node-id={node.id}
                  data-testid="mesh-node-core-overlay"
                  className={cn(
                    'pointer-events-none absolute inset-0 rounded-full transition-opacity duration-150 ease-out',
                    isOpen ? 'opacity-45' : 'opacity-0',
                  )}
                  style={{ backgroundColor: 'color-mix(in oklab, currentColor 56%, transparent)' }}
                />
              </span>
            </span>
          </span>
        </button>
      </HoverCardPrimitive.Trigger>
      <MeshVizNodeHoverCard node={node} peer={peer} />
    </HoverCardPrimitive.Root>
  )
}

export function MeshVizNodeLabel({
  node,
  peer,
  selfId,
  selectedNodeId,
  openNodeId,
  hoveredNodeId,
  shouldFadeNodeLabels,
  reduceMotion,
  canvasWidth,
  canvasHeight,
  viewport,
  nodeColors,
  lifecycle,
}: MeshVizNodeLabelProps) {
  const isDebug = isDebugNode(node)
  const isSelf = node.id === selfId
  const isSelected = !isDebug && node.id === selectedNodeId
  const isOpen = node.id === openNodeId
  const isHovered = node.id === hoveredNodeId
  const shouldRevealNodeLabel = lifecycle !== 'leaving' && (!shouldFadeNodeLabels || isOpen || isHovered)
  const labelFadeDuration = shouldRevealNodeLabel && shouldFadeNodeLabels && isHovered ? 300 : 500
  const { haloSize, labelColor } = nodeVisuals(node, peer, isSelf, isSelected, nodeColors)
  const screenPoint = pointToScreen(node, canvasWidth, canvasHeight, viewport)
  const labelRef = useRef<HTMLSpanElement>(null)
  const labelAnimationRef = useRef<ReturnType<typeof animate> | undefined>(undefined)
  const previousLabelOpacityRef = useRef<number | undefined>(undefined)

  useLayoutEffect(() => {
    const labelElement = labelRef.current
    const targetOpacity = shouldRevealNodeLabel ? 1 : 0

    if (!labelElement) return undefined

    labelAnimationRef.current?.pause()

    const currentOpacity = Number.parseFloat(labelElement.style.opacity)
    const fromOpacity = Number.isFinite(currentOpacity)
      ? currentOpacity
      : previousLabelOpacityRef.current ?? targetOpacity

    if (reduceMotion || fromOpacity === targetOpacity) {
      labelElement.style.opacity = `${targetOpacity}`
      previousLabelOpacityRef.current = targetOpacity
      return undefined
    }

    labelElement.style.opacity = `${fromOpacity}`
    previousLabelOpacityRef.current = targetOpacity

    const animation = animate(labelElement, {
      opacity: { from: fromOpacity, to: targetOpacity },
      duration: labelFadeDuration,
      ease: 'out(3)',
    })

    labelAnimationRef.current = animation

    return () => {
      animation.pause()
    }
  }, [labelFadeDuration, reduceMotion, shouldRevealNodeLabel])

  return (
    <span
      ref={labelRef}
      data-node-id={node.id}
      data-testid="mesh-node-label"
      data-node-lifecycle={lifecycle}
      className={cn(
        'pointer-events-none absolute left-1/2 top-full z-[60] mt-1.5 flex min-w-max flex-col items-center gap-1 transition-opacity ease-out',
        shouldRevealNodeLabel
          ? labelFadeDuration === 300
            ? 'opacity-100 duration-[300ms]'
            : 'opacity-100 duration-[500ms]'
          : 'opacity-0 duration-[500ms]',
      )}
      style={{
        left: `${screenPoint.x}px`,
        top: `${screenPoint.y + haloSize / 2}px`,
        transform: 'translateX(-50%) scale(var(--mesh-node-live-scale, 1))',
        transformOrigin: '50% 0',
        willChange: 'transform, opacity',
      }}
    >
      <span
        className={cn(
          meshPaletteFadeClassName,
          'font-mono text-[length:var(--density-type-label)] font-semibold uppercase leading-none tracking-[0.06em]',
        )}
        style={{
          color:
            labelColor,
        }}
      >
        {node.label}
      </span>
      {node.subLabel && (
        <span className="font-mono text-[length:var(--density-type-micro)] uppercase tracking-[0.04em] text-muted-foreground">
          {node.subLabel}
        </span>
      )}
    </span>
  )
}
