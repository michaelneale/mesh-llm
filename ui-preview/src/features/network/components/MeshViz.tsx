import {
  forwardRef,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { animate, createTimeline, type JSAnimation, type Timeline } from 'animejs'
import { Maximize2, Minus, Plus, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/cn'
import { isDevelopmentMode } from '@/lib/env'
import { buildMeshLinks } from '@/features/network/lib/mesh-links'
import { chooseClusteredMeshNodePosition } from '@/features/network/lib/mesh-placement'
import {
  MESH_VIZ_DOT_COLOR_SCHEMES,
  meshVizDotColorSchemeAtIndex,
  nextMeshVizDotColorSchemeIndex,
  themeFromDocument
} from '@/features/network/lib/mesh-viz-dot-color-schemes'
import {
  calculateMaxZoomOut,
  calculateNodeBounds,
  centerScreenRect,
  clamp,
  clampViewportToPanBounds,
  DEFAULT_VIEWPORT,
  FIT_PADDING_PX,
  focusPointWithinNodeBounds,
  GRID_SIZE_PX,
  gridPatternTransform,
  isIdentityLayerTransform,
  MAX_ZOOM,
  NODE_VISUAL_BOUNDS_PADDING_PX,
  nodeBoundsToScreenRect,
  nodeFitsInsideViewport,
  PAN_DEAD_ZONE_PX,
  pointToScreen,
  type LayerTransform,
  type Point,
  type Viewport,
  type WorldPoint,
  viewportLayerTransform,
  viewportsMatch
} from '@/features/network/lib/mesh-viewport'
import type { MeshNode, Peer, ResolvedTheme } from '@/features/app-tabs/types'
import { MeshVizDebugControls, type MeshVizGridMode } from './MeshVizDebugControls'
import {
  createDebugNode,
  debugNodeMatchesShortcut,
  debugNodeShortcutCount,
  getDebugNodeShortcutBlueprint,
  isTextEditingTarget,
  NODE_LABEL_FADE_THRESHOLD,
  nodeVisuals,
  prefersReducedMotion,
  type DebugMeshNode,
  type DebugNodeShortcut
} from './MeshViz.helpers'
import { MeshVizNode, MeshVizNodeLabel, type MeshVizNodeLifecycle } from './MeshVizNode'
import { useMeshVizTraffic } from './useMeshVizTraffic'

type MeshVizProps = {
  nodes: MeshNode[]
  selfId: string
  meshId?: string
  selectedNodeId?: string
  onPick?: (node: MeshNode) => void
  height?: number
  accent?: string
  compact?: boolean
  enableDebugShortcuts?: boolean
  animateTopology?: boolean
  onFullscreen?: () => void
  getNodePeer?: (node: MeshNode) => Peer | undefined
}
export type MeshVizHandle = {
  playTraffic: (sourceNodeId: string, targetNodeId: string) => boolean
}
type DragState = {
  active: boolean
  pointerId: number | null
  originX: number
  originY: number
  panX: number
  panY: number
}
type TouchPointState = {
  pointerId: number
  clientX: number
  clientY: number
}
type PinchZoomState = {
  active: boolean
  initialDistance: number
  initialZoom: number
}
type MeshLifecycleTimelineRecord = {
  keys: Set<string>
  timeline: Timeline
}
type LinkRestoreTimelineRecord = {
  linkIds: Set<string>
  timeline: Timeline
}

const VIEWPORT_RECLAMP_DURATION = 220
const VIEWPORT_RECLAMP_EASE = 'outExpo'
const RADAR_PULSE_DURATION = 2000
const RADAR_PULSE_LOOP_DELAY = 1000
const RADAR_PULSE_EASE = 'linear'
const NODE_JOIN_STAGGER_MS = 720
const NODE_JOIN_DURATION_MS = 380
const LINK_JOIN_DURATION_MS = 420
const CONNECTED_NODE_PULSE_DELAY_MS = 500
const CONNECTED_NODE_PULSE_DURATION_MS = 360
const NODE_JOIN_SETTLE_BUFFER_MS = 60
const LINK_LEAVE_DURATION_MS = 260
const NODE_LEAVE_DELAY_MS = 90
const NODE_LEAVE_DURATION_MS = 280
const NODE_LEAVE_STAGGER_MS = 260
const NODE_LEAVE_SETTLE_BUFFER_MS = 120
const WHEEL_ZOOM_IN = 1.08
const WHEEL_ZOOM_OUT = 0.92
const WHEEL_ZOOM_COMMIT_DELAY_MS = 90
const BUTTON_ZOOM_IN = 1.12
const BUTTON_ZOOM_OUT = 0.88

function lifecycleTransitionKey(nodeId: string, phase: MeshVizNodeLifecycle) {
  return `${nodeId}\u001f${phase}`
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined
}

function numericCssValue(value: string | null | undefined, fallback: number) {
  if (!value || value === 'none') {
    return fallback
  }

  const parsedValue = Number.parseFloat(value)
  return Number.isFinite(parsedValue) ? parsedValue : fallback
}

function currentElementOpacity(element: Element, fallback: number) {
  return numericCssValue(
    element instanceof HTMLElement || element instanceof SVGElement
      ? element.style.opacity || element.getAttribute('opacity') || window.getComputedStyle(element).opacity
      : element.getAttribute('opacity'),
    fallback
  )
}

function currentElementScale(element: HTMLElement, fallback: number) {
  return numericCssValue(element.style.scale || window.getComputedStyle(element).scale, fallback)
}

function currentStrokeDashOffset(element: SVGElement, fallback: number) {
  return numericCssValue(
    element.style.getPropertyValue('stroke-dashoffset') ||
      element.getAttribute('stroke-dashoffset') ||
      window.getComputedStyle(element).getPropertyValue('stroke-dashoffset'),
    fallback
  )
}

function nodeJoinSettleDelay(index: number) {
  return (
    index * NODE_JOIN_STAGGER_MS +
    CONNECTED_NODE_PULSE_DELAY_MS +
    CONNECTED_NODE_PULSE_DURATION_MS +
    NODE_JOIN_SETTLE_BUFFER_MS
  )
}

function nodeLeaveRemovalDelay(index: number) {
  return index * NODE_LEAVE_STAGGER_MS + NODE_LEAVE_DELAY_MS + NODE_LEAVE_DURATION_MS + NODE_LEAVE_SETTLE_BUFFER_MS
}

export const MeshViz = forwardRef<MeshVizHandle, MeshVizProps>(function MeshViz(
  {
    nodes,
    selfId,
    meshId,
    selectedNodeId,
    height = 460,
    compact = false,
    enableDebugShortcuts = false,
    animateTopology = true,
    onFullscreen,
    getNodePeer
  }: MeshVizProps,
  ref
) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const gridPatternRef = useRef<SVGPatternElement>(null)
  const gridPathRef = useRef<SVGPathElement>(null)
  const gridDotRef = useRef<SVGCircleElement>(null)
  const gridAccentDotRef = useRef<SVGCircleElement>(null)
  const gridTertiaryDotRef = useRef<SVGCircleElement>(null)
  const svgPanLayerRef = useRef<SVGGElement>(null)
  const nodeLayerRef = useRef<HTMLDivElement>(null)
  const labelLayerRef = useRef<HTMLDivElement>(null)
  const packetLayerRef = useRef<HTMLDivElement>(null)
  const panTransformFrameRef = useRef<number | null>(null)
  const liveLayerTransformRef = useRef<LayerTransform>({ x: 0, y: 0, scale: 1 })
  const renderedViewportRef = useRef<Viewport>(DEFAULT_VIEWPORT)
  const liveLayerBaseViewportRef = useRef<Viewport>(DEFAULT_VIEWPORT)
  const liveLayerTransformActiveRef = useRef(false)
  const wheelZoomCommitTimeoutRef = useRef<number | null>(null)
  // During drag, viewportRef is live while React state stays committed; this flag clears the transient layer
  // transform after the committed viewport has rendered so there is no visible snap-back frame.
  const pendingPanTransformResetRef = useRef(false)
  const viewportAnimationRef = useRef<JSAnimation | undefined>(undefined)
  const canvasSizeRef = useRef({ width: 0, height: 0 })
  const viewportRef = useRef<Viewport>(DEFAULT_VIEWPORT)
  const zoomFocusRef = useRef<WorldPoint | undefined>(undefined)
  const zoomAnchorRef = useRef<Point | undefined>(undefined)
  const debugNodeCounterRef = useRef(0)
  const fittedNodesSignatureRef = useRef('')
  const fittedNodeIdsRef = useRef<Set<string>>(new Set())
  const topologyNodeSnapshotsRef = useRef<Map<string, MeshNode>>(new Map())
  const nodeLifecycleTimeoutsRef = useRef<Map<string, number>>(new Map())
  const nodeLifecycleAnimationKeysRef = useRef<Set<string>>(new Set())
  const meshLifecycleTimelineRecordsRef = useRef<Map<number, MeshLifecycleTimelineRecord>>(new Map())
  const meshLifecycleTimelineIdRef = useRef(0)
  const previousLinkIdsRef = useRef<Set<string>>(new Set())
  const linkRestoreAnimationIdsRef = useRef<Set<string>>(new Set())
  const linkRestoreTimelineRecordsRef = useRef<Map<number, LinkRestoreTimelineRecord>>(new Map())
  const linkRestoreTimelineIdRef = useRef(0)
  const hasUserControlledViewportRef = useRef(false)
  const wasFullscreenRef = useRef(false)
  const dragRef = useRef<DragState>({
    active: false,
    pointerId: null,
    originX: 0,
    originY: 0,
    panX: 0,
    panY: 0
  })
  const touchPointersRef = useRef<Map<number, TouchPointState>>(new Map())
  const pinchZoomRef = useRef<PinchZoomState>({
    active: false,
    initialDistance: 1,
    initialZoom: DEFAULT_VIEWPORT.zoom
  })
  const radarPingRef = useRef<HTMLSpanElement>(null)
  const [openNodeId, setOpenNodeId] = useState<string | undefined>()
  const [hoveredNodeId, setHoveredNodeId] = useState<string | undefined>()
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })
  const [viewport, setViewportState] = useState<Viewport>(DEFAULT_VIEWPORT)
  const [isPanning, setIsPanning] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showPanBounds, setShowPanBounds] = useState(false)
  const [gridMode, setGridMode] = useState<MeshVizGridMode>('line')
  const [dotColorSchemeIndex, setDotColorSchemeIndex] = useState(0)
  const [dotColorSchemeTheme, setDotColorSchemeTheme] = useState<ResolvedTheme>(() =>
    typeof document === 'undefined' ? 'dark' : themeFromDocument()
  )
  const [debugNodes, setDebugNodes] = useState<DebugMeshNode[]>([])
  const [exitingNodes, setExitingNodes] = useState<MeshNode[]>([])
  const [nodeLifecyclePhases, setNodeLifecyclePhases] = useState<Record<string, MeshVizNodeLifecycle>>({})
  const reduceMotion = prefersReducedMotion()
  const isDevelopment = isDevelopmentMode()
  const debugShortcutsEnabled = isDevelopment || enableDebugShortcuts
  const currentRenderNodes = useMemo(() => [...nodes, ...debugNodes], [debugNodes, nodes])
  const currentRenderNodeIds = useMemo(() => new Set(currentRenderNodes.map((node) => node.id)), [currentRenderNodes])
  const pendingExitingNodes = useMemo(() => {
    if (reduceMotion) {
      return []
    }

    return [...topologyNodeSnapshotsRef.current.entries()]
      .filter(([nodeId]) => !currentRenderNodeIds.has(nodeId))
      .filter(([nodeId]) => nodeLifecyclePhases[nodeId] !== undefined || nodeLifecycleTimeoutsRef.current.has(nodeId))
      .map(([, node]) => node)
  }, [currentRenderNodeIds, nodeLifecyclePhases, reduceMotion])
  const pendingExitingNodeIds = useMemo(
    () => new Set(pendingExitingNodes.map((node) => node.id)),
    [pendingExitingNodes]
  )
  const renderNodes = useMemo(
    () => [
      ...currentRenderNodes,
      ...exitingNodes.filter((node) => !currentRenderNodeIds.has(node.id)),
      ...pendingExitingNodes.filter(
        (node) => !currentRenderNodeIds.has(node.id) && !exitingNodes.some((exitingNode) => exitingNode.id === node.id)
      )
    ],
    [currentRenderNodeIds, currentRenderNodes, exitingNodes, pendingExitingNodes]
  )
  const meshSeed = useMemo(
    () =>
      meshId ??
      `${selfId}:${nodes
        .map((node) => node.id)
        .sort()
        .join('|')}`,
    [meshId, nodes, selfId]
  )
  const links = useMemo(() => buildMeshLinks(renderNodes, getNodePeer), [getNodePeer, renderNodes])
  const nodesFitSignature = useMemo(
    () => renderNodes.map((node) => `${node.id}:${node.x}:${node.y}`).join('|'),
    [renderNodes]
  )
  const linkCount = links.length
  const shouldFadeNodeLabels = renderNodes.length >= NODE_LABEL_FADE_THRESHOLD
  const safeCanvasWidth = Math.max(canvasSize.width, 1)
  const safeCanvasHeight = Math.max(canvasSize.height, 1)
  const gridSize = Math.max(18, GRID_SIZE_PX * viewport.zoom)
  const gridTransform = gridPatternTransform(viewport, gridSize)
  const dotColorSchemes = MESH_VIZ_DOT_COLOR_SCHEMES[dotColorSchemeTheme]
  const dotColorScheme = meshVizDotColorSchemeAtIndex(dotColorSchemeTheme, dotColorSchemeIndex)
  const nodeBounds = useMemo(
    () => calculateNodeBounds(currentRenderNodes, { width: safeCanvasWidth, height: safeCanvasHeight }),
    [currentRenderNodes, safeCanvasHeight, safeCanvasWidth]
  )
  const nodeBoundsRect = nodeBoundsToScreenRect(nodeBounds, viewport)
  const deadZoneRect = nodeBoundsRect
    ? {
        x: nodeBoundsRect.x - PAN_DEAD_ZONE_PX,
        y: nodeBoundsRect.y - PAN_DEAD_ZONE_PX,
        width: nodeBoundsRect.width + PAN_DEAD_ZONE_PX * 2,
        height: nodeBoundsRect.height + PAN_DEAD_ZONE_PX * 2
      }
    : undefined
  const centeredBoundsRect = nodeBoundsRect ? centerScreenRect(nodeBoundsRect) : undefined

  const setViewport = useCallback((nextViewport: Viewport, options?: { userControlled?: boolean }) => {
    if (options?.userControlled) {
      viewportAnimationRef.current?.revert()
      viewportAnimationRef.current = undefined
      hasUserControlledViewportRef.current = true
    }

    viewportRef.current = nextViewport
    setViewportState(nextViewport)
  }, [])

  const nodeLifecyclePhase = useCallback(
    (nodeId: string): MeshVizNodeLifecycle => {
      if (!animateTopology) {
        return 'present'
      }

      if (pendingExitingNodeIds.has(nodeId)) {
        return 'leaving'
      }

      if (nodeLifecyclePhases[nodeId]) {
        return nodeLifecyclePhases[nodeId]
      }

      return 'entering'
    },
    [animateTopology, nodeLifecyclePhases, pendingExitingNodeIds]
  )

  const linkLifecyclePhase = useCallback(
    (sourceNodeId: string, targetNodeId: string): MeshVizNodeLifecycle => {
      const sourcePhase = nodeLifecyclePhase(sourceNodeId)
      const targetPhase = nodeLifecyclePhase(targetNodeId)

      if (sourcePhase === 'leaving' || targetPhase === 'leaving') return 'leaving'
      if (sourcePhase === 'entering' || targetPhase === 'entering') return 'entering'
      return 'present'
    },
    [nodeLifecyclePhase]
  )

  const applyGridPattern = useCallback((nextViewport: Viewport) => {
    const nextGridSize = Math.max(18, GRID_SIZE_PX * nextViewport.zoom)

    if (gridPatternRef.current) {
      gridPatternRef.current.setAttribute('width', `${nextGridSize}`)
      gridPatternRef.current.setAttribute('height', `${nextGridSize}`)
      gridPatternRef.current.setAttribute('patternTransform', gridPatternTransform(nextViewport, nextGridSize))
    }

    if (gridPathRef.current) {
      gridPathRef.current.setAttribute('d', `M ${nextGridSize} 0 L 0 0 0 ${nextGridSize}`)
    }

    if (gridDotRef.current) {
      gridDotRef.current.setAttribute('cx', '0')
      gridDotRef.current.setAttribute('cy', '0')
    }

    if (gridAccentDotRef.current) {
      const accentDotOffset = nextGridSize / 2

      gridAccentDotRef.current.setAttribute('cx', `${accentDotOffset}`)
      gridAccentDotRef.current.setAttribute('cy', `${accentDotOffset}`)
    }

    if (gridTertiaryDotRef.current) {
      gridTertiaryDotRef.current.setAttribute('cx', '0')
      gridTertiaryDotRef.current.setAttribute('cy', `${nextGridSize / 2}`)
    }
  }, [])

  const applyLayerTransform = useCallback(
    (transform: LayerTransform) => {
      const isIdentity = isIdentityLayerTransform(transform)
      const htmlTransform = isIdentity
        ? ''
        : `translate3d(${transform.x}px, ${transform.y}px, 0) scale(${transform.scale})`
      const svgTransform = isIdentity ? '' : `translate(${transform.x} ${transform.y}) scale(${transform.scale})`

      if (svgPanLayerRef.current) {
        if (svgTransform) {
          svgPanLayerRef.current.setAttribute('transform', svgTransform)
        } else {
          svgPanLayerRef.current.removeAttribute('transform')
        }
      }

      if (nodeLayerRef.current) {
        nodeLayerRef.current.style.transform = htmlTransform
        nodeLayerRef.current.style.setProperty('--mesh-node-live-scale', isIdentity ? '1' : `${1 / transform.scale}`)
      }

      if (labelLayerRef.current) {
        labelLayerRef.current.style.transform = htmlTransform
        labelLayerRef.current.style.setProperty('--mesh-node-live-scale', isIdentity ? '1' : `${1 / transform.scale}`)
      }

      if (packetLayerRef.current) {
        packetLayerRef.current.style.transform = htmlTransform
      }

      applyGridPattern(viewportRef.current)
    },
    [applyGridPattern]
  )

  const activateLiveLayerTransform = useCallback(() => {
    if (liveLayerTransformActiveRef.current) {
      return
    }

    liveLayerBaseViewportRef.current = renderedViewportRef.current
    liveLayerTransformActiveRef.current = true
  }, [])

  const scheduleViewportLayerTransform = useCallback(
    (nextViewport: Viewport) => {
      activateLiveLayerTransform()
      liveLayerTransformRef.current = viewportLayerTransform(liveLayerBaseViewportRef.current, nextViewport)

      if (panTransformFrameRef.current !== null) {
        return
      }

      panTransformFrameRef.current = window.requestAnimationFrame(() => {
        panTransformFrameRef.current = null
        applyLayerTransform(liveLayerTransformRef.current)
      })
    },
    [activateLiveLayerTransform, applyLayerTransform]
  )

  const clearViewportLayerTransform = useCallback(() => {
    liveLayerTransformActiveRef.current = false
    liveLayerBaseViewportRef.current = viewportRef.current
    liveLayerTransformRef.current = { x: 0, y: 0, scale: 1 }

    if (panTransformFrameRef.current !== null) {
      window.cancelAnimationFrame(panTransformFrameRef.current)
      panTransformFrameRef.current = null
    }

    applyLayerTransform(liveLayerTransformRef.current)
  }, [applyLayerTransform])

  const scheduleWheelZoomCommit = useCallback(() => {
    if (wheelZoomCommitTimeoutRef.current !== null) {
      window.clearTimeout(wheelZoomCommitTimeoutRef.current)
    }

    wheelZoomCommitTimeoutRef.current = window.setTimeout(() => {
      wheelZoomCommitTimeoutRef.current = null

      if (dragRef.current.active || pinchZoomRef.current.active) {
        return
      }

      pendingPanTransformResetRef.current = true
      setViewport(viewportRef.current, { userControlled: true })
    }, WHEEL_ZOOM_COMMIT_DELAY_MS)
  }, [setViewport])

  const flushLiveViewportTransform = useCallback(() => {
    if (wheelZoomCommitTimeoutRef.current !== null) {
      window.clearTimeout(wheelZoomCommitTimeoutRef.current)
      wheelZoomCommitTimeoutRef.current = null
    }

    if (!liveLayerTransformActiveRef.current) {
      return
    }

    pendingPanTransformResetRef.current = false
    clearViewportLayerTransform()
    viewportAnimationRef.current?.revert()
    viewportAnimationRef.current = undefined
    setViewportState(viewportRef.current)
  }, [clearViewportLayerTransform])

  const transitionViewportTo = useCallback(
    (targetViewport: Viewport) => {
      const startViewport = viewportRef.current

      flushLiveViewportTransform()

      viewportAnimationRef.current?.revert()
      viewportAnimationRef.current = undefined

      if (viewportsMatch(startViewport, targetViewport)) {
        return
      }

      if (reduceMotion) {
        setViewport(targetViewport)
        return
      }

      const animatedViewport = { ...startViewport }

      viewportAnimationRef.current = animate(animatedViewport, {
        zoom: { from: startViewport.zoom, to: targetViewport.zoom },
        panX: { from: startViewport.panX, to: targetViewport.panX },
        panY: { from: startViewport.panY, to: targetViewport.panY },
        duration: VIEWPORT_RECLAMP_DURATION,
        ease: VIEWPORT_RECLAMP_EASE,
        loop: false,
        onUpdate: () => setViewport({ ...animatedViewport }),
        onComplete: () => {
          setViewport(targetViewport)
          viewportAnimationRef.current = undefined
        }
      })
    },
    [flushLiveViewportTransform, reduceMotion, setViewport]
  )

  const calculateFitViewport = useCallback(
    (size: { width: number; height: number }) => {
      if (currentRenderNodes.length === 0 || size.width <= 0 || size.height <= 0) {
        return DEFAULT_VIEWPORT
      }

      const minX = Math.min(...currentRenderNodes.map((node) => node.x))
      const maxX = Math.max(...currentRenderNodes.map((node) => node.x))
      const minY = Math.min(...currentRenderNodes.map((node) => node.y))
      const maxY = Math.max(...currentRenderNodes.map((node) => node.y))
      const availableWidth = Math.max(1, size.width - FIT_PADDING_PX * 2)
      const availableHeight = Math.max(1, size.height - FIT_PADDING_PX * 2)
      const worldWidth = Math.max(1, ((maxX - minX) / 100) * size.width)
      const worldHeight = Math.max(1, ((maxY - minY) / 100) * size.height)
      const minZoom = calculateMaxZoomOut(currentRenderNodes, size)
      const zoom = clamp(Math.min(availableWidth / worldWidth, availableHeight / worldHeight), minZoom, MAX_ZOOM)
      const centerX = ((minX + maxX) / 2 / 100) * size.width
      const centerY = ((minY + maxY) / 2 / 100) * size.height

      return {
        zoom,
        panX: size.width * 0.5 - centerX * zoom,
        panY: size.height * 0.5 - centerY * zoom
      }
    },
    [currentRenderNodes]
  )

  const fitNodes = useCallback(() => {
    viewportAnimationRef.current?.revert()
    viewportAnimationRef.current = undefined
    zoomFocusRef.current = undefined
    zoomAnchorRef.current = undefined
    flushLiveViewportTransform()
    hasUserControlledViewportRef.current = false
    fittedNodesSignatureRef.current = nodesFitSignature
    setViewport(calculateFitViewport(canvasSizeRef.current))
  }, [calculateFitViewport, flushLiveViewportTransform, nodesFitSignature, setViewport])

  const zoomAroundPoint = useCallback(
    (nextZoom: number, anchorX: number, anchorY: number, options?: { live?: boolean }) => {
      const currentViewport = viewportRef.current
      const canvasSize = canvasSizeRef.current
      const minZoom = calculateMaxZoomOut(currentRenderNodes, canvasSize)
      const zoom = clamp(nextZoom, minZoom, MAX_ZOOM)
      const candidateFocus = {
        x: (anchorX - currentViewport.panX) / currentViewport.zoom,
        y: (anchorY - currentViewport.panY) / currentViewport.zoom
      }
      const currentFocus = zoomFocusRef.current
      const currentAnchor = zoomAnchorRef.current
      const sameZoomAnchor =
        currentAnchor &&
        Math.abs(currentAnchor.x - anchorX) <= NODE_VISUAL_BOUNDS_PADDING_PX &&
        Math.abs(currentAnchor.y - anchorY) <= NODE_VISUAL_BOUNDS_PADDING_PX
      const focusPoint = currentFocus && sameZoomAnchor ? currentFocus : candidateFocus

      zoomAnchorRef.current = { x: anchorX, y: anchorY }

      zoomFocusRef.current = focusPoint

      const nextViewport = clampViewportToPanBounds(
        currentRenderNodes,
        canvasSize,
        {
          zoom,
          panX: anchorX - focusPoint.x * zoom,
          panY: anchorY - focusPoint.y * zoom
        },
        focusPoint,
        minZoom
      )

      if (options?.live) {
        viewportAnimationRef.current?.revert()
        viewportAnimationRef.current = undefined
        hasUserControlledViewportRef.current = true
        viewportRef.current = nextViewport
        scheduleViewportLayerTransform(nextViewport)
        scheduleWheelZoomCommit()
        return
      }

      flushLiveViewportTransform()
      setViewport(nextViewport, { userControlled: true })
    },
    [
      currentRenderNodes,
      flushLiveViewportTransform,
      scheduleViewportLayerTransform,
      scheduleWheelZoomCommit,
      setViewport
    ]
  )

  const zoomAtCenter = useCallback(
    (factor: number) => {
      zoomAroundPoint(
        viewportRef.current.zoom * factor,
        canvasSizeRef.current.width * 0.5,
        canvasSizeRef.current.height * 0.5
      )
    },
    [zoomAroundPoint]
  )

  const nodeColorForTraffic = useCallback(
    (node: MeshNode) =>
      nodeVisuals(node, getNodePeer?.(node), node.id === selfId, false, dotColorScheme.nodeColors).fill,
    [dotColorScheme.nodeColors, getNodePeer, selfId]
  )

  const updateCanvasSize = useCallback(() => {
    const canvasElement = canvasRef.current

    if (!canvasElement) {
      return
    }

    const nextSize = {
      width: canvasElement.clientWidth,
      height: canvasElement.clientHeight
    }
    canvasSizeRef.current = nextSize
    setCanvasSize((current) =>
      current.width === nextSize.width && current.height === nextSize.height ? current : nextSize
    )
  }, [])

  const { clearTrafficPackets, placeTrafficPackets, playRandomTraffic, playSelfTraffic, playTraffic } =
    useMeshVizTraffic({
      canvasRef,
      canvasSizeRef,
      links,
      liveLayerBaseViewportRef,
      liveLayerTransformActiveRef,
      nodeColorForTraffic,
      packetLayerRef,
      reduceMotion,
      renderNodes,
      selfId,
      updateCanvasSize,
      viewportRef
    })

  useEffect(() => {
    const previousSnapshots = topologyNodeSnapshotsRef.current
    const currentIds = new Set(currentRenderNodes.map((node) => node.id))
    const addedNodes = currentRenderNodes.filter((node) => !previousSnapshots.has(node.id))
    const removedNodes = [...previousSnapshots.entries()]
      .filter(([nodeId]) => !currentIds.has(nodeId))
      .map(([, node]) => node)

    if (addedNodes.length > 0 || removedNodes.length > 0) {
      setNodeLifecyclePhases((current) => {
        const next = { ...current }

        for (const node of addedNodes) {
          next[node.id] = reduceMotion ? 'present' : 'entering'
        }

        for (const node of removedNodes) {
          if (reduceMotion) {
            delete next[node.id]
          } else {
            next[node.id] = 'leaving'
          }
        }

        return next
      })
    }

    if (reduceMotion) {
      for (const node of [...addedNodes, ...removedNodes]) {
        const existingTimeout = nodeLifecycleTimeoutsRef.current.get(node.id)

        if (existingTimeout !== undefined) {
          window.clearTimeout(existingTimeout)
          nodeLifecycleTimeoutsRef.current.delete(node.id)
        }
      }

      if (removedNodes.length > 0) {
        const removedNodeIds = new Set(removedNodes.map((node) => node.id))
        setExitingNodes((current) => current.filter((node) => !removedNodeIds.has(node.id)))
      }

      topologyNodeSnapshotsRef.current = new Map(currentRenderNodes.map((node) => [node.id, node]))
      return
    }

    if (addedNodes.length > 0) {
      setExitingNodes((current) => current.filter((node) => !currentIds.has(node.id)))

      addedNodes.forEach((node, index) => {
        const existingTimeout = nodeLifecycleTimeoutsRef.current.get(node.id)

        if (existingTimeout !== undefined) {
          window.clearTimeout(existingTimeout)
        }

        const timeout = window.setTimeout(() => {
          nodeLifecycleTimeoutsRef.current.delete(node.id)
          setNodeLifecyclePhases((current) => {
            if (current[node.id] !== 'entering') return current
            return { ...current, [node.id]: 'present' }
          })
        }, nodeJoinSettleDelay(index))

        nodeLifecycleTimeoutsRef.current.set(node.id, timeout)
      })
    }

    if (removedNodes.length > 0) {
      setExitingNodes((current) => {
        const activeExitingNodes = current.filter((node) => !currentIds.has(node.id))
        const activeExitingIds = new Set(activeExitingNodes.map((node) => node.id))
        const nextRemovedNodes = removedNodes.filter((node) => !activeExitingIds.has(node.id))

        return [...activeExitingNodes, ...nextRemovedNodes]
      })

      removedNodes.forEach((node, index) => {
        const existingTimeout = nodeLifecycleTimeoutsRef.current.get(node.id)

        if (existingTimeout !== undefined) {
          window.clearTimeout(existingTimeout)
        }

        const timeout = window.setTimeout(() => {
          nodeLifecycleTimeoutsRef.current.delete(node.id)
          setExitingNodes((current) => current.filter((exitingNode) => exitingNode.id !== node.id))
          setNodeLifecyclePhases((current) => {
            if (!(node.id in current)) return current

            const { [node.id]: removedPhase, ...next } = current
            void removedPhase
            return next
          })
        }, nodeLeaveRemovalDelay(index))

        nodeLifecycleTimeoutsRef.current.set(node.id, timeout)
      })
    }

    topologyNodeSnapshotsRef.current = new Map(currentRenderNodes.map((node) => [node.id, node]))
  }, [currentRenderNodes, reduceMotion])

  useLayoutEffect(() => {
    const activeTransitionKeys = new Set<string>()
    const transitioningNodes = renderNodes
      .map((node) => ({ node, phase: nodeLifecyclePhase(node.id) }))
      .filter(({ phase }) => phase === 'entering' || phase === 'leaving')

    for (const { node, phase } of transitioningNodes) {
      activeTransitionKeys.add(lifecycleTransitionKey(node.id, phase))
    }

    for (const key of [...nodeLifecycleAnimationKeysRef.current]) {
      if (!activeTransitionKeys.has(key)) {
        nodeLifecycleAnimationKeysRef.current.delete(key)
      }
    }

    for (const [recordId, record] of meshLifecycleTimelineRecordsRef.current) {
      const stillActive = [...record.keys].some((key) => activeTransitionKeys.has(key))

      if (!stillActive) {
        record.timeline.pause()
        meshLifecycleTimelineRecordsRef.current.delete(recordId)
      }
    }

    if (reduceMotion) {
      for (const record of meshLifecycleTimelineRecordsRef.current.values()) {
        record.timeline.revert()
      }

      meshLifecycleTimelineRecordsRef.current.clear()
      nodeLifecycleAnimationKeysRef.current.clear()
      return undefined
    }

    if (transitioningNodes.length === 0) {
      return undefined
    }

    const nodeLayerElement = nodeLayerRef.current
    const svgPanLayerElement = svgPanLayerRef.current

    if (!nodeLayerElement || !svgPanLayerElement) {
      return undefined
    }

    const nodeCoreElements = new Map<string, HTMLElement>()
    nodeLayerElement.querySelectorAll<HTMLElement>('[data-mesh-node-core]').forEach((element) => {
      const nodeId = element.dataset.meshNodeCore

      if (nodeId) {
        nodeCoreElements.set(nodeId, element)
      }
    })

    const linkElements = new Map<string, SVGLineElement>()
    svgPanLayerElement.querySelectorAll<SVGLineElement>('[data-mesh-link-id]').forEach((element) => {
      const linkId = element.dataset.meshLinkId

      if (linkId) {
        linkElements.set(linkId, element)
      }
    })

    const newTransitions = transitioningNodes.filter(({ node, phase }) => {
      const key = lifecycleTransitionKey(node.id, phase)

      return !nodeLifecycleAnimationKeysRef.current.has(key)
    })

    if (newTransitions.length === 0) {
      return undefined
    }

    const timelineId = meshLifecycleTimelineIdRef.current
    meshLifecycleTimelineIdRef.current += 1
    const timelineKeys = new Set(newTransitions.map(({ node, phase }) => lifecycleTransitionKey(node.id, phase)))
    const transitionIndexByNodeId = new Map(newTransitions.map(({ node }, index) => [node.id, index]))
    const animatedLinkIds = new Set<string>()
    const enteringCoreRestingShadows = new Map<HTMLElement, string>()
    const pulsedCoreElements = new Set<HTMLElement>()
    const enteringLinkElements = new Set<SVGLineElement>()

    for (const key of timelineKeys) {
      nodeLifecycleAnimationKeysRef.current.add(key)
    }

    const timeline = createTimeline({
      defaults: { ease: 'outQuart' },
      onComplete: () => {
        for (const [element, boxShadow] of enteringCoreRestingShadows) {
          element.style.removeProperty('opacity')
          element.style.removeProperty('scale')
          element.style.boxShadow = boxShadow
        }

        for (const element of pulsedCoreElements) {
          if (!enteringCoreRestingShadows.has(element)) {
            element.style.removeProperty('opacity')
            element.style.removeProperty('scale')
          }
        }

        for (const element of enteringLinkElements) {
          element.style.removeProperty('opacity')
          element.style.removeProperty('stroke-dashoffset')
        }

        meshLifecycleTimelineRecordsRef.current.delete(timelineId)
      }
    })

    newTransitions.forEach(({ node, phase }, index) => {
      const nodeCoreElement = nodeCoreElements.get(node.id)

      if (!nodeCoreElement) {
        return
      }

      const connectedLinks = links.filter((link) => link.source.id === node.id || link.target.id === node.id)
      const connectedLinkElements = connectedLinks
        .filter((link) => {
          if (animatedLinkIds.has(link.id)) {
            return false
          }

          const otherNodeId = link.source.id === node.id ? link.target.id : link.source.id
          const otherTransitionIndex = transitionIndexByNodeId.get(otherNodeId)

          if (otherTransitionIndex === undefined) {
            return true
          }

          const otherPhase = nodeLifecyclePhase(otherNodeId)

          if (otherPhase !== phase) {
            return phase === 'leaving'
          }

          return phase === 'entering' ? index >= otherTransitionIndex : index <= otherTransitionIndex
        })
        .map((link) => {
          const element = linkElements.get(link.id)

          if (element) {
            animatedLinkIds.add(link.id)
          }

          return element
        })
        .filter(isDefined)

      if (phase === 'entering') {
        const start = index * NODE_JOIN_STAGGER_MS
        const nodeColor = nodeCoreElement.style.color || 'currentColor'

        enteringCoreRestingShadows.set(nodeCoreElement, nodeCoreElement.style.boxShadow)

        timeline.set(
          nodeCoreElement,
          {
            opacity: 0,
            scale: 0.54,
            boxShadow: `0 0 0 0 color-mix(in oklab, ${nodeColor} 0%, transparent)`
          },
          start
        )
        timeline.add(
          nodeCoreElement,
          {
            opacity: [0, 1, 0.98],
            scale: [0.6, 1.34, 1.08],
            boxShadow: [
              `0 0 0 0 color-mix(in oklab, ${nodeColor} 0%, transparent)`,
              `0 0 30px 3px color-mix(in oklab, ${nodeColor} 34%, transparent)`,
              `0 0 14px 1px color-mix(in oklab, ${nodeColor} 20%, transparent)`
            ],
            duration: NODE_JOIN_DURATION_MS
          },
          start
        )

        if (connectedLinkElements.length > 0) {
          for (const linkElement of connectedLinkElements) {
            enteringLinkElements.add(linkElement)
          }

          timeline.set(connectedLinkElements, { opacity: 0, strokeDashoffset: 1 }, start)
          timeline.add(
            connectedLinkElements,
            {
              opacity: [0, 0.62],
              strokeDashoffset: [1, 0],
              duration: LINK_JOIN_DURATION_MS
            },
            start + NODE_JOIN_DURATION_MS
          )
        }

        const connectedCoreElements = connectedLinks
          .map((link) => (link.source.id === node.id ? link.target.id : link.source.id))
          .filter((nodeId) => nodeLifecyclePhase(nodeId) === 'present')
          .map((nodeId) => nodeCoreElements.get(nodeId))
          .filter(isDefined)

        if (connectedCoreElements.length > 0) {
          for (const element of connectedCoreElements) {
            pulsedCoreElements.add(element)
          }

          timeline.add(
            connectedCoreElements,
            {
              opacity: [0.98, 1, 0.98],
              scale: [1, 1.12, 1],
              duration: CONNECTED_NODE_PULSE_DURATION_MS
            },
            start + CONNECTED_NODE_PULSE_DELAY_MS
          )
        }

        return
      }

      const start = index * NODE_LEAVE_STAGGER_MS
      const nodeColor = nodeCoreElement.style.color || 'currentColor'

      if (connectedLinkElements.length > 0) {
        for (const linkElement of connectedLinkElements) {
          timeline.add(
            linkElement,
            {
              opacity: [currentElementOpacity(linkElement, 0.62), 0],
              strokeDashoffset: [currentStrokeDashOffset(linkElement, 0), -1],
              duration: LINK_LEAVE_DURATION_MS,
              ease: 'inQuart'
            },
            start
          )
        }
      }

      timeline.add(
        nodeCoreElement,
        {
          opacity: [currentElementOpacity(nodeCoreElement, 0.98), 0],
          scale: [currentElementScale(nodeCoreElement, 1.08), 0.72],
          boxShadow: [
            nodeCoreElement.style.boxShadow || `0 0 14px 1px color-mix(in oklab, ${nodeColor} 20%, transparent)`,
            `0 0 0 0 color-mix(in oklab, ${nodeColor} 0%, transparent)`
          ],
          duration: NODE_LEAVE_DURATION_MS,
          ease: 'inQuart'
        },
        start + NODE_LEAVE_DELAY_MS
      )
    })

    meshLifecycleTimelineRecordsRef.current.set(timelineId, { keys: timelineKeys, timeline })

    return undefined
  }, [links, nodeLifecyclePhase, reduceMotion, renderNodes])

  useLayoutEffect(() => {
    const currentLinkIds = new Set(links.map((link) => link.id))
    const currentLinksById = new Map(links.map((link) => [link.id, link]))
    const previousLinkIds = previousLinkIdsRef.current

    previousLinkIdsRef.current = currentLinkIds

    for (const linkId of [...linkRestoreAnimationIdsRef.current]) {
      if (!currentLinkIds.has(linkId)) {
        linkRestoreAnimationIdsRef.current.delete(linkId)
      }
    }

    for (const [recordId, record] of linkRestoreTimelineRecordsRef.current) {
      const stillRestoring = [...record.linkIds].every((linkId) => {
        const link = currentLinksById.get(linkId)

        return link && linkLifecyclePhase(link.source.id, link.target.id) === 'present'
      })

      if (!stillRestoring) {
        record.timeline.pause()

        for (const linkId of record.linkIds) {
          linkRestoreAnimationIdsRef.current.delete(linkId)
        }

        linkRestoreTimelineRecordsRef.current.delete(recordId)
      }
    }

    if (reduceMotion) {
      for (const record of linkRestoreTimelineRecordsRef.current.values()) {
        record.timeline.revert()
      }

      linkRestoreTimelineRecordsRef.current.clear()
      linkRestoreAnimationIdsRef.current.clear()
      return undefined
    }

    if (previousLinkIds.size === 0) {
      return undefined
    }

    const restoredLinks = links.filter(
      (link) =>
        !previousLinkIds.has(link.id) &&
        !linkRestoreAnimationIdsRef.current.has(link.id) &&
        linkLifecyclePhase(link.source.id, link.target.id) === 'present'
    )

    if (restoredLinks.length === 0) {
      return undefined
    }

    const svgPanLayerElement = svgPanLayerRef.current

    if (!svgPanLayerElement) {
      return undefined
    }

    const linkElements = new Map<string, SVGLineElement>()
    svgPanLayerElement.querySelectorAll<SVGLineElement>('[data-mesh-link-id]').forEach((element) => {
      const linkId = element.dataset.meshLinkId

      if (linkId) {
        linkElements.set(linkId, element)
      }
    })
    const restoredLinkElements = restoredLinks.map((link) => linkElements.get(link.id)).filter(isDefined)

    if (restoredLinkElements.length === 0) {
      return undefined
    }

    const timelineId = linkRestoreTimelineIdRef.current
    linkRestoreTimelineIdRef.current += 1

    const restoredLinkIds = new Set(restoredLinks.map((link) => link.id))

    for (const linkId of restoredLinkIds) {
      linkRestoreAnimationIdsRef.current.add(linkId)
    }

    const timeline = createTimeline({
      defaults: { ease: 'outQuart' },
      onComplete: () => {
        for (const element of restoredLinkElements) {
          element.style.removeProperty('opacity')
          element.style.removeProperty('stroke-dashoffset')
        }

        for (const linkId of restoredLinkIds) {
          linkRestoreAnimationIdsRef.current.delete(linkId)
        }

        linkRestoreTimelineRecordsRef.current.delete(timelineId)
      }
    })

    timeline.set(restoredLinkElements, { opacity: 0, strokeDashoffset: 1 }, 0)
    timeline.add(
      restoredLinkElements,
      {
        opacity: [0, 0.62],
        strokeDashoffset: [1, 0],
        duration: LINK_JOIN_DURATION_MS
      },
      0
    )

    linkRestoreTimelineRecordsRef.current.set(timelineId, { linkIds: restoredLinkIds, timeline })

    return undefined
  }, [links, linkLifecyclePhase, reduceMotion])

  useEffect(
    () => () => {
      for (const timeout of nodeLifecycleTimeoutsRef.current.values()) {
        window.clearTimeout(timeout)
      }

      nodeLifecycleTimeoutsRef.current.clear()
      nodeLifecycleAnimationKeysRef.current.clear()
      previousLinkIdsRef.current.clear()
      linkRestoreAnimationIdsRef.current.clear()

      for (const record of meshLifecycleTimelineRecordsRef.current.values()) {
        record.timeline.revert()
      }

      meshLifecycleTimelineRecordsRef.current.clear()

      for (const record of linkRestoreTimelineRecordsRef.current.values()) {
        record.timeline.revert()
      }

      linkRestoreTimelineRecordsRef.current.clear()
    },
    []
  )

  useEffect(() => {
    if (openNodeId && !currentRenderNodeIds.has(openNodeId)) {
      setOpenNodeId(undefined)
    }

    if (hoveredNodeId && !currentRenderNodeIds.has(hoveredNodeId)) {
      setHoveredNodeId(undefined)
    }
  }, [currentRenderNodeIds, hoveredNodeId, openNodeId])

  const addDebugNode = useCallback(
    (shortcut: DebugNodeShortcut) => {
      const blueprint = getDebugNodeShortcutBlueprint(shortcut)

      setDebugNodes((current) => {
        const debugIndex = debugNodeCounterRef.current + 1
        const placementNodes: MeshNode[] = [...nodes, ...current]
        const position = chooseClusteredMeshNodePosition(meshSeed, debugIndex, blueprint, placementNodes)
        const debugNode = createDebugNode(debugIndex, blueprint, position)

        debugNodeCounterRef.current = debugIndex
        return [...current, debugNode]
      })
    },
    [meshSeed, nodes]
  )

  const removeDebugNode = useCallback((shortcut: DebugNodeShortcut) => {
    setDebugNodes((current) => {
      let removeIndex = -1

      for (let index = current.length - 1; index >= 0; index -= 1) {
        if (debugNodeMatchesShortcut(current[index], shortcut)) {
          removeIndex = index
          break
        }
      }

      if (removeIndex === -1) {
        return current
      }

      return current.filter((_, index) => index !== removeIndex)
    })
  }, [])

  const cycleDotColorScheme = useCallback(() => {
    setDotColorSchemeIndex(nextMeshVizDotColorSchemeIndex)
  }, [])

  const selectDotColorScheme = useCallback((index: number) => {
    setDotColorSchemeIndex(index)
  }, [])

  useImperativeHandle(ref, () => ({ playTraffic }), [playTraffic])

  useEffect(() => {
    if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') {
      return undefined
    }

    const root = document.documentElement
    const syncTheme = () => setDotColorSchemeTheme(themeFromDocument())

    syncTheme()

    const observer = new MutationObserver(syncTheme)
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] })

    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!debugShortcutsEnabled) {
      return undefined
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isTextEditingTarget(event.target)) {
        return
      }

      const key = event.key.toLowerCase()
      const debugNodeShortcut = debugNodeShortcutCount(event)

      if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
        if (key === 'z') {
          event.preventDefault()
          playRandomTraffic()
          return
        }

        if (key === 'x') {
          event.preventDefault()
          playSelfTraffic()
          return
        }
      }

      if (debugNodeShortcut !== undefined) {
        if (event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
          event.preventDefault()
          removeDebugNode(debugNodeShortcut)
          return
        }

        if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
          return
        }

        event.preventDefault()
        addDebugNode(debugNodeShortcut)
        return
      }

      if (key === 'b') {
        if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
          return
        }

        event.preventDefault()
        setShowPanBounds((current) => !current)
        return
      }

      if (key === 'g') {
        if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
          return
        }

        event.preventDefault()
        setGridMode((current) => (current === 'line' ? 'dot' : 'line'))
        return
      }

      if (key === 'c') {
        if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
          return
        }

        event.preventDefault()
        cycleDotColorScheme()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [addDebugNode, cycleDotColorScheme, debugShortcutsEnabled, playRandomTraffic, playSelfTraffic, removeDebugNode])

  useEffect(() => {
    const canvasElement = canvasRef.current

    if (!canvasElement) {
      return undefined
    }

    updateCanvasSize()

    const resizeObserver =
      typeof ResizeObserver === 'undefined'
        ? undefined
        : new ResizeObserver(() => {
            updateCanvasSize()
            placeTrafficPackets()
          })

    resizeObserver?.observe(canvasElement)

    return () => {
      resizeObserver?.disconnect()
      clearTrafficPackets()
      viewportAnimationRef.current?.revert()
      viewportAnimationRef.current = undefined
    }
  }, [clearTrafficPackets, placeTrafficPackets, updateCanvasSize])

  useEffect(() => {
    const radarPingElement = radarPingRef.current

    if (!radarPingElement || reduceMotion) {
      if (radarPingElement) {
        radarPingElement.style.opacity = '0'
        radarPingElement.style.transform = 'scale(1)'
      }

      return undefined
    }

    radarPingElement.style.opacity = '0.6'
    radarPingElement.style.transform = 'scale(1)'

    const animation = animate(radarPingElement, {
      opacity: { from: 0.6, to: 0 },
      scale: { from: 1, to: 2.6 },
      duration: RADAR_PULSE_DURATION,
      ease: RADAR_PULSE_EASE,
      loop: true,
      loopDelay: RADAR_PULSE_LOOP_DELAY
    })

    return () => {
      animation.revert()
      radarPingElement.style.opacity = '0.6'
      radarPingElement.style.transform = 'scale(1)'
    }
  }, [reduceMotion])

  useEffect(() => {
    if (canvasSize.width <= 0 || canvasSize.height <= 0) {
      return
    }

    const nodesChanged = fittedNodesSignatureRef.current !== nodesFitSignature
    const previousNodeIds = fittedNodeIdsRef.current
    const trackedPreviousNodes = previousNodeIds.size > 0
    const addedNodes = trackedPreviousNodes ? currentRenderNodes.filter((node) => !previousNodeIds.has(node.id)) : []
    const addedNodeOutsideViewport = addedNodes.some(
      (node) => !nodeFitsInsideViewport(node, canvasSize, viewportRef.current)
    )
    const trackCurrentTopology = () => {
      fittedNodesSignatureRef.current = nodesFitSignature
      fittedNodeIdsRef.current = new Set(currentRenderNodes.map((node) => node.id))
    }

    if (addedNodeOutsideViewport) {
      trackCurrentTopology()
      zoomFocusRef.current = undefined
      zoomAnchorRef.current = undefined
      hasUserControlledViewportRef.current = false
      transitionViewportTo(calculateFitViewport(canvasSize))
      return
    }

    if (hasUserControlledViewportRef.current) {
      if (nodesChanged) {
        trackCurrentTopology()
      }

      if (zoomFocusRef.current && !focusPointWithinNodeBounds(currentRenderNodes, canvasSize, zoomFocusRef.current)) {
        zoomFocusRef.current = undefined
        zoomAnchorRef.current = undefined
      }

      const clampedViewport = clampViewportToPanBounds(
        currentRenderNodes,
        canvasSize,
        viewportRef.current,
        zoomFocusRef.current,
        calculateMaxZoomOut(currentRenderNodes, canvasSize)
      )

      if (!viewportsMatch(clampedViewport, viewportRef.current)) {
        transitionViewportTo(clampedViewport)
      }

      return
    }

    trackCurrentTopology()

    const fitViewport = calculateFitViewport(canvasSize)

    if (trackedPreviousNodes && nodesChanged) {
      zoomFocusRef.current = undefined
      zoomAnchorRef.current = undefined
      transitionViewportTo(fitViewport)
      return
    }

    setViewport(fitViewport)
  }, [calculateFitViewport, canvasSize, currentRenderNodes, nodesFitSignature, setViewport, transitionViewportTo])

  useLayoutEffect(() => {
    renderedViewportRef.current = viewport

    if (pendingPanTransformResetRef.current && viewportsMatch(viewport, viewportRef.current)) {
      pendingPanTransformResetRef.current = false
      clearViewportLayerTransform()
    }

    placeTrafficPackets()
  }, [clearViewportLayerTransform, placeTrafficPackets, viewport])

  useEffect(
    () => () => {
      if (panTransformFrameRef.current !== null) {
        window.cancelAnimationFrame(panTransformFrameRef.current)
      }

      if (wheelZoomCommitTimeoutRef.current !== null) {
        window.clearTimeout(wheelZoomCommitTimeoutRef.current)
      }
    },
    []
  )

  useEffect(() => {
    if (!isPanning) {
      return undefined
    }

    const previousUserSelect = document.body.style.userSelect
    document.body.style.userSelect = 'none'

    return () => {
      document.body.style.userSelect = previousUserSelect
    }
  }, [isPanning])

  useEffect(() => {
    const syncFullscreenState = () => {
      const isCanvasFullscreen = document.fullscreenElement === canvasRef.current

      setIsFullscreen(isCanvasFullscreen)

      if (wasFullscreenRef.current && !isCanvasFullscreen) {
        updateCanvasSize()
        fitNodes()
      }

      wasFullscreenRef.current = isCanvasFullscreen
    }

    syncFullscreenState()
    document.addEventListener('fullscreenchange', syncFullscreenState)

    return () => {
      document.removeEventListener('fullscreenchange', syncFullscreenState)
    }
  }, [fitNodes, updateCanvasSize])

  const touchPoints = () => Array.from(touchPointersRef.current.values())

  const distanceBetweenTouchPoints = (first: TouchPointState, second: TouchPointState) =>
    Math.hypot(second.clientX - first.clientX, second.clientY - first.clientY)

  const midpointRelativeToCanvas = (element: HTMLDivElement, first: TouchPointState, second: TouchPointState) => {
    const rect = element.getBoundingClientRect()

    return {
      x: (first.clientX + second.clientX) * 0.5 - rect.left,
      y: (first.clientY + second.clientY) * 0.5 - rect.top
    }
  }

  const beginPinchZoom = () => {
    const [first, second] = touchPoints()

    if (!first || !second) {
      return
    }

    dragRef.current.active = false
    dragRef.current.pointerId = null
    setIsPanning(false)
    pinchZoomRef.current = {
      active: true,
      initialDistance: Math.max(1, distanceBetweenTouchPoints(first, second)),
      initialZoom: viewportRef.current.zoom
    }
  }

  const handleCanvasPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return
    }

    if (event.target instanceof Element && event.target.closest('button, a, input, label')) {
      return
    }

    event.preventDefault()
    viewportAnimationRef.current?.revert()
    viewportAnimationRef.current = undefined
    hasUserControlledViewportRef.current = true
    pendingPanTransformResetRef.current = false
    if (wheelZoomCommitTimeoutRef.current !== null) {
      window.clearTimeout(wheelZoomCommitTimeoutRef.current)
      wheelZoomCommitTimeoutRef.current = null
    }

    if (!liveLayerTransformActiveRef.current) {
      clearViewportLayerTransform()
      liveLayerBaseViewportRef.current = renderedViewportRef.current
      liveLayerTransformActiveRef.current = true
    }

    event.currentTarget.setPointerCapture(event.pointerId)

    if (event.pointerType === 'touch') {
      touchPointersRef.current.set(event.pointerId, {
        pointerId: event.pointerId,
        clientX: event.clientX,
        clientY: event.clientY
      })

      if (touchPointersRef.current.size >= 2) {
        beginPinchZoom()
        setOpenNodeId(undefined)
        setHoveredNodeId(undefined)
        return
      }
    }

    dragRef.current = {
      active: true,
      pointerId: event.pointerId,
      originX: event.clientX,
      originY: event.clientY,
      panX: viewportRef.current.panX,
      panY: viewportRef.current.panY
    }
    setIsPanning(true)
    setOpenNodeId(undefined)
    setHoveredNodeId(undefined)
  }

  const handleCanvasPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.pointerType === 'touch' && touchPointersRef.current.has(event.pointerId)) {
      touchPointersRef.current.set(event.pointerId, {
        pointerId: event.pointerId,
        clientX: event.clientX,
        clientY: event.clientY
      })

      const [first, second] = touchPoints()

      if (first && second) {
        event.preventDefault()

        if (!pinchZoomRef.current.active) {
          beginPinchZoom()
        }

        const currentDistance = Math.max(1, distanceBetweenTouchPoints(first, second))
        const anchor = midpointRelativeToCanvas(event.currentTarget, first, second)
        const nextZoom = pinchZoomRef.current.initialZoom * (currentDistance / pinchZoomRef.current.initialDistance)

        zoomAroundPoint(nextZoom, anchor.x, anchor.y, { live: true })
        return
      }
    }

    const drag = dragRef.current

    if (!drag.active || drag.pointerId !== event.pointerId) {
      return
    }

    event.preventDefault()
    const canvasSize = canvasSizeRef.current
    const minZoom = calculateMaxZoomOut(currentRenderNodes, canvasSize)
    const nextViewport = clampViewportToPanBounds(
      currentRenderNodes,
      canvasSize,
      {
        zoom: viewportRef.current.zoom,
        panX: drag.panX + event.clientX - drag.originX,
        panY: drag.panY + event.clientY - drag.originY
      },
      zoomFocusRef.current,
      minZoom
    )

    viewportRef.current = nextViewport
    scheduleViewportLayerTransform(nextViewport)
  }

  const stopPanning = (event: ReactPointerEvent<HTMLDivElement>) => {
    const shouldCommitViewport = dragRef.current.active && dragRef.current.pointerId === event.pointerId

    if (dragRef.current.pointerId === event.pointerId && event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }

    if (event.pointerType === 'touch') {
      touchPointersRef.current.delete(event.pointerId)

      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }
    }

    const shouldCommitPinchZoom = pinchZoomRef.current.active && touchPointersRef.current.size < 2

    dragRef.current.active = false
    dragRef.current.pointerId = null

    if (shouldCommitViewport || shouldCommitPinchZoom) {
      if (wheelZoomCommitTimeoutRef.current !== null) {
        window.clearTimeout(wheelZoomCommitTimeoutRef.current)
        wheelZoomCommitTimeoutRef.current = null
      }

      pinchZoomRef.current.active = false
      pendingPanTransformResetRef.current = true
      setViewport(viewportRef.current, { userControlled: true })
    }

    if (!shouldCommitPinchZoom && touchPointersRef.current.size < 2) {
      pinchZoomRef.current.active = false
    }

    setIsPanning(false)
  }

  const handleCanvasWheel = useCallback(
    (event: WheelEvent) => {
      if (event.deltaY === 0) {
        return
      }

      if (event.cancelable) {
        event.preventDefault()
      }
      event.stopPropagation()

      const canvasElement = canvasRef.current
      if (!canvasElement) {
        return
      }

      const rect = canvasElement.getBoundingClientRect()
      const anchorX = event.clientX - rect.left
      const anchorY = event.clientY - rect.top
      const factor = event.deltaY > 0 ? WHEEL_ZOOM_OUT : WHEEL_ZOOM_IN

      zoomAroundPoint(viewportRef.current.zoom * factor, anchorX, anchorY, { live: true })
    },
    [zoomAroundPoint]
  )

  useEffect(() => {
    const canvasElement = canvasRef.current
    if (!canvasElement) {
      return undefined
    }

    canvasElement.addEventListener('wheel', handleCanvasWheel, { passive: false })

    return () => canvasElement.removeEventListener('wheel', handleCanvasWheel)
  }, [handleCanvasWheel])

  const handleFullscreen = useCallback(() => {
    if (onFullscreen) {
      onFullscreen()
      return
    }

    const canvasElement = canvasRef.current

    if (!canvasElement || typeof canvasElement.requestFullscreen !== 'function') {
      return
    }

    void canvasElement.requestFullscreen().catch((error: unknown) => {
      console.warn('Unable to enter mesh fullscreen mode', error)
    })
  }, [onFullscreen])

  const screenLinks = links.map((link) => ({
    ...link,
    sourcePoint: pointToScreen(link.source, safeCanvasWidth, safeCanvasHeight, viewport),
    targetPoint: pointToScreen(link.target, safeCanvasWidth, safeCanvasHeight, viewport)
  }))
  const maxZoomOut = calculateMaxZoomOut(currentRenderNodes, canvasSize)
  const maxZoomOutLabel = maxZoomOut.toFixed(2)
  const viewportControlClassName = cn(
    'ui-control grid place-items-center rounded-[var(--radius)] border',
    isFullscreen ? 'size-[52px]' : 'size-[26px]'
  )
  const viewportControlIconClassName = isFullscreen ? 'size-6' : 'size-3'

  return (
    <section className="panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <header className="flex items-center justify-between border-b border-border-soft px-3.5 py-2.5">
        <h2 className="type-panel-title">Mesh overview</h2>
        {!compact && (
          <button
            onClick={handleFullscreen}
            type="button"
            className="ui-control inline-flex items-center gap-1.5 rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium"
          >
            <Maximize2 className="size-3" /> Fullscreen
          </button>
        )}
      </header>
      <div className="p-3.5">
        <div
          ref={canvasRef}
          data-testid="mesh-canvas"
          className={cn(
            'relative touch-none overflow-hidden rounded-[var(--radius-lg)] mesh-canvas',
            isPanning ? 'cursor-grabbing' : 'cursor-grab'
          )}
          style={{
            height,
            background:
              'radial-gradient(ellipse at 60% 40%, color-mix(in oklab, var(--color-accent) 10%, var(--color-panel-strong)) 0%, var(--color-panel-strong) 60%, var(--color-panel) 100%)'
          }}
          onPointerDown={handleCanvasPointerDown}
          onPointerMove={handleCanvasPointerMove}
          onPointerUp={stopPanning}
          onPointerCancel={stopPanning}
        >
          <div className="absolute left-3.5 top-3 z-10 flex flex-wrap items-center gap-2.5 font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.14em] text-muted-foreground">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-panel/90 px-2 py-px text-accent">
              <span className="size-[5px] rounded-full bg-current mesh-live-pulse" /> Live
            </span>
            <span>
              {nodes.length} nodes{debugNodes.length > 0 ? ` + ${debugNodes.length} debug` : ''} · {linkCount} links ·
              Nearest mesh
            </span>
          </div>

          {isDevelopment && (
            <div
              data-testid="mesh-max-zoom-label"
              className="pointer-events-none absolute right-3.5 top-3 z-10 rounded-full border border-border bg-panel/90 px-2 py-px font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.14em] text-muted-foreground"
            >
              Max Zoom: {maxZoomOutLabel}
            </div>
          )}

          <svg
            viewBox={`0 0 ${safeCanvasWidth} ${safeCanvasHeight}`}
            preserveAspectRatio="none"
            className="pointer-events-none absolute inset-0 h-full w-full overflow-hidden"
            role="img"
            aria-label="Nearest mesh topology"
          >
            <defs>
              <pattern
                ref={gridPatternRef}
                id="mesh-viz-grid"
                width={gridSize}
                height={gridSize}
                patternUnits="userSpaceOnUse"
                patternTransform={gridTransform}
              >
                {gridMode === 'line' ? (
                  <path
                    ref={gridPathRef}
                    data-testid="mesh-viz-line-grid"
                    d={`M ${gridSize} 0 L 0 0 0 ${gridSize}`}
                    fill="none"
                    stroke="color-mix(in oklab, var(--color-foreground) 7.2%, transparent)"
                    strokeWidth="1"
                  />
                ) : (
                  <>
                    <circle
                      ref={gridDotRef}
                      data-testid="mesh-viz-dot-grid"
                      cx="0"
                      cy="0"
                      r="1.35"
                      fill={dotColorScheme.colors[0]}
                    />
                    <circle
                      ref={gridAccentDotRef}
                      data-testid="mesh-viz-accent-dot-grid"
                      cx={gridSize / 2}
                      cy={gridSize / 2}
                      r="1.25"
                      fill={dotColorScheme.colors[1]}
                    />
                    <circle
                      ref={gridTertiaryDotRef}
                      data-testid="mesh-viz-tertiary-dot-grid"
                      cx="0"
                      cy={gridSize / 2}
                      r="0.85"
                      fill={dotColorScheme.colors[2]}
                    />
                  </>
                )}
              </pattern>
            </defs>
            <rect width={safeCanvasWidth} height={safeCanvasHeight} fill="url(#mesh-viz-grid)" />
            <g ref={svgPanLayerRef}>
              {screenLinks.map((link) => (
                <line
                  key={link.id}
                  className="mesh-link"
                  data-link-lifecycle={linkLifecyclePhase(link.source.id, link.target.id)}
                  data-mesh-link-id={link.id}
                  data-source-node-id={link.source.id}
                  data-target-node-id={link.target.id}
                  data-testid="mesh-link"
                  pathLength={1}
                  x1={link.sourcePoint.x}
                  y1={link.sourcePoint.y}
                  x2={link.targetPoint.x}
                  y2={link.targetPoint.y}
                  stroke="color-mix(in oklab, var(--color-accent) 48%, var(--color-border))"
                  strokeDasharray="0.0275 0.0275"
                  strokeLinecap="round"
                  strokeWidth="1"
                  opacity="0.62"
                  vectorEffect="non-scaling-stroke"
                />
              ))}
              {isDevelopment && showPanBounds && nodeBoundsRect && deadZoneRect && centeredBoundsRect && (
                <g aria-label="Mesh pan bounds debug overlay">
                  <rect
                    data-testid="mesh-pan-dead-zone-box"
                    x={deadZoneRect.x}
                    y={deadZoneRect.y}
                    width={deadZoneRect.width}
                    height={deadZoneRect.height}
                    fill="color-mix(in oklab, var(--color-accent) 7%, transparent)"
                    stroke="color-mix(in oklab, var(--color-accent) 72%, transparent)"
                    strokeDasharray="8 6"
                    strokeWidth="1.2"
                    vectorEffect="non-scaling-stroke"
                  />
                  <rect
                    data-testid="mesh-node-bounds-box"
                    x={nodeBoundsRect.x}
                    y={nodeBoundsRect.y}
                    width={nodeBoundsRect.width}
                    height={nodeBoundsRect.height}
                    fill="none"
                    stroke="color-mix(in oklab, var(--color-good) 78%, transparent)"
                    strokeWidth="1.4"
                    vectorEffect="non-scaling-stroke"
                  />
                  <rect
                    data-testid="mesh-centered-bounds-box"
                    x={centeredBoundsRect.x}
                    y={centeredBoundsRect.y}
                    width={centeredBoundsRect.width}
                    height={centeredBoundsRect.height}
                    fill="none"
                    stroke="color-mix(in oklab, var(--color-warn) 82%, transparent)"
                    strokeDasharray="6 5"
                    strokeWidth="1.2"
                    vectorEffect="non-scaling-stroke"
                  />
                </g>
              )}
            </g>
          </svg>

          <div
            ref={packetLayerRef}
            className="pointer-events-none absolute inset-0 z-[5]"
            data-testid="mesh-packet-layer"
            style={{ transformOrigin: '0 0', willChange: 'transform' }}
            aria-hidden="true"
          />

          <div
            ref={nodeLayerRef}
            className="absolute inset-0"
            style={{ transformOrigin: '0 0', willChange: 'transform' }}
          >
            {renderNodes.map((node) => {
              const peer = getNodePeer?.(node)

              return (
                <MeshVizNode
                  key={node.id}
                  node={node}
                  peer={peer}
                  selfId={selfId}
                  selectedNodeId={selectedNodeId}
                  openNodeId={openNodeId}
                  hoveredNodeId={hoveredNodeId}
                  shouldFadeNodeLabels={shouldFadeNodeLabels}
                  reduceMotion={reduceMotion}
                  canvasWidth={safeCanvasWidth}
                  canvasHeight={safeCanvasHeight}
                  viewport={viewport}
                  nodeColors={dotColorScheme.nodeColors}
                  lifecycle={nodeLifecyclePhase(node.id)}
                  radarPingRef={radarPingRef}
                  onHoverStart={setHoveredNodeId}
                  onHoverEnd={(nodeId) => setHoveredNodeId((current) => (current === nodeId ? undefined : current))}
                  onToggleOpen={(nodeId) => {
                    setOpenNodeId((current) => (current === nodeId ? undefined : nodeId))
                  }}
                  onCloseOpen={() => setOpenNodeId(undefined)}
                />
              )
            })}
          </div>

          <div
            ref={labelLayerRef}
            className="pointer-events-none absolute inset-0 z-[40]"
            data-testid="mesh-node-label-layer"
            style={{ transformOrigin: '0 0', willChange: 'transform' }}
            aria-hidden="true"
          >
            {renderNodes.map((node) => {
              const peer = getNodePeer?.(node)

              return (
                <MeshVizNodeLabel
                  key={node.id}
                  node={node}
                  peer={peer}
                  selfId={selfId}
                  selectedNodeId={selectedNodeId}
                  openNodeId={openNodeId}
                  hoveredNodeId={hoveredNodeId}
                  shouldFadeNodeLabels={shouldFadeNodeLabels}
                  reduceMotion={reduceMotion}
                  canvasWidth={safeCanvasWidth}
                  canvasHeight={safeCanvasHeight}
                  viewport={viewport}
                  nodeColors={dotColorScheme.nodeColors}
                  lifecycle={nodeLifecyclePhase(node.id)}
                />
              )
            })}
          </div>

          {isDevelopment && (
            <MeshVizDebugControls
              debugNodeCount={debugNodes.length}
              dotColorSchemeIndex={dotColorSchemeIndex}
              dotColorSchemes={dotColorSchemes}
              gridMode={gridMode}
              isFullscreen={isFullscreen}
              onAddDebugNode={addDebugNode}
              onDotColorSchemeChange={selectDotColorScheme}
              onDotColorSchemeNext={cycleDotColorScheme}
              onGridModeChange={setGridMode}
              onPlayRandomTraffic={playRandomTraffic}
              onPlaySelfTraffic={playSelfTraffic}
              onRemoveDebugNode={removeDebugNode}
              onShowPanBoundsChange={setShowPanBounds}
              showPanBounds={showPanBounds}
            />
          )}

          <div className="absolute bottom-3 right-3 flex flex-col gap-1.5">
            <button
              type="button"
              className={viewportControlClassName}
              aria-label="Zoom in"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => zoomAtCenter(BUTTON_ZOOM_IN)}
            >
              <Plus className={viewportControlIconClassName} />
            </button>
            <button
              type="button"
              className={viewportControlClassName}
              aria-label="Zoom out"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => zoomAtCenter(BUTTON_ZOOM_OUT)}
            >
              <Minus className={viewportControlIconClassName} />
            </button>
            <button
              type="button"
              className={viewportControlClassName}
              aria-label="Reset view"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={fitNodes}
            >
              <RotateCcw className={viewportControlIconClassName} />
            </button>
          </div>
        </div>
      </div>
    </section>
  )
})

MeshViz.displayName = 'MeshViz'
