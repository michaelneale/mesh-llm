import type { MeshNode } from '@/features/app-tabs/types'

export type Point = { x: number; y: number }
export type Viewport = { zoom: number; panX: number; panY: number }
export type NodeBounds = {
  minX: number
  maxX: number
  minY: number
  maxY: number
  width: number
  height: number
  centerX: number
  centerY: number
}
export type PanBounds = {
  minPanX: number
  maxPanX: number
  minPanY: number
  maxPanY: number
}
export type ScreenRect = { x: number; y: number; width: number; height: number }
export type CanvasSize = { width: number; height: number }
export type WorldPoint = { x: number; y: number }
export type LayerTransform = { x: number; y: number; scale: number }

type VisualAxisBounds = {
  min: number
  max: number
  size: number
}

export const DEFAULT_VIEWPORT: Viewport = { zoom: 1, panX: 0, panY: 0 }
export const MIN_ZOOM = 0.7
export const MAX_ZOOM = 2.4
export const FIT_PADDING_PX = 48
export const PAN_DEAD_ZONE_PX = 48
export const NODE_VISUAL_BOUNDS_PADDING_PX = 24
export const GRID_SIZE_PX = 32

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function viewportsMatch(first: Viewport, second: Viewport) {
  return first.zoom === second.zoom && first.panX === second.panX && first.panY === second.panY
}

export function pointToScreen(point: Point, width: number, height: number, viewport: Viewport) {
  return {
    x: (point.x / 100) * width * viewport.zoom + viewport.panX,
    y: (point.y / 100) * height * viewport.zoom + viewport.panY
  }
}

function gridOffset(pan: number, gridSize: number) {
  return ((pan % gridSize) + gridSize) % gridSize
}

export function gridPatternTransform(viewport: Viewport, gridSize: number) {
  return `translate(${gridOffset(viewport.panX, gridSize)} ${gridOffset(viewport.panY, gridSize)})`
}

export function viewportLayerTransform(fromViewport: Viewport, toViewport: Viewport): LayerTransform {
  const scale = toViewport.zoom / fromViewport.zoom

  return {
    x: toViewport.panX - fromViewport.panX * scale,
    y: toViewport.panY - fromViewport.panY * scale,
    scale
  }
}

export function isIdentityLayerTransform(transform: LayerTransform) {
  return transform.x === 0 && transform.y === 0 && transform.scale === 1
}

export function nodeFitsInsideViewport(node: MeshNode, size: CanvasSize, viewport: Viewport) {
  const screenPoint = pointToScreen(node, size.width, size.height, viewport)
  const horizontalPadding = Math.min(NODE_VISUAL_BOUNDS_PADDING_PX, size.width * 0.25)
  const verticalPadding = Math.min(NODE_VISUAL_BOUNDS_PADDING_PX, size.height * 0.25)

  return (
    screenPoint.x >= horizontalPadding &&
    screenPoint.x <= size.width - horizontalPadding &&
    screenPoint.y >= verticalPadding &&
    screenPoint.y <= size.height - verticalPadding
  )
}

export function calculateNodeBounds(nodes: MeshNode[], size: CanvasSize): NodeBounds | undefined {
  if (nodes.length === 0 || size.width <= 0 || size.height <= 0) {
    return undefined
  }

  const minX = (Math.min(...nodes.map((node) => node.x)) / 100) * size.width
  const maxX = (Math.max(...nodes.map((node) => node.x)) / 100) * size.width
  const minY = (Math.min(...nodes.map((node) => node.y)) / 100) * size.height
  const maxY = (Math.max(...nodes.map((node) => node.y)) / 100) * size.height

  return {
    minX,
    maxX,
    minY,
    maxY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY),
    centerX: (minX + maxX) * 0.5,
    centerY: (minY + maxY) * 0.5
  }
}

export function focusPointWithinNodeBounds(nodes: MeshNode[], size: CanvasSize, focusPoint: WorldPoint | undefined) {
  if (!focusPoint) {
    return false
  }

  const bounds = calculateNodeBounds(nodes, size)

  if (!bounds) {
    return false
  }

  return (
    focusPoint.x >= bounds.minX &&
    focusPoint.x <= bounds.maxX &&
    focusPoint.y >= bounds.minY &&
    focusPoint.y <= bounds.maxY
  )
}

function activeZoomFocus(nodes: MeshNode[], size: CanvasSize, focusPoint: WorldPoint | undefined) {
  return focusPointWithinNodeBounds(nodes, size, focusPoint) ? focusPoint : undefined
}

export function calculateMaxZoomOut(nodes: MeshNode[], size: CanvasSize) {
  const bounds = calculateNodeBounds(nodes, size)

  if (!bounds) {
    return MIN_ZOOM
  }

  const availableWidth = Math.max(1, size.width - FIT_PADDING_PX * 2)
  const availableHeight = Math.max(1, size.height - FIT_PADDING_PX * 2)
  const boundaryZoom = Math.min(availableWidth / bounds.width, availableHeight / bounds.height)

  return Math.min(MIN_ZOOM, boundaryZoom)
}

export function calculatePanBounds(
  nodes: MeshNode[],
  size: CanvasSize,
  zoom: number,
  focusPoint?: WorldPoint
): PanBounds | undefined {
  const bounds = calculateNodeBounds(nodes, size)

  if (!bounds) {
    return undefined
  }

  const xVisualBounds = calculateVisualAxisBounds(bounds.minX, bounds.maxX, zoom)
  const yVisualBounds = calculateVisualAxisBounds(bounds.minY, bounds.maxY, zoom)
  const xCenteredVisualBounds = calculateCenteredVisualAxisBounds(xVisualBounds)
  const yCenteredVisualBounds = calculateCenteredVisualAxisBounds(yVisualBounds)
  const useIntersectionBounds = xVisualBounds.size > size.width || yVisualBounds.size > size.height
  const xBounds = calculateAxisPanBounds(
    xVisualBounds,
    xCenteredVisualBounds,
    size.width,
    zoom,
    useIntersectionBounds,
    focusPoint?.x
  )
  const yBounds = calculateAxisPanBounds(
    yVisualBounds,
    yCenteredVisualBounds,
    size.height,
    zoom,
    useIntersectionBounds,
    focusPoint?.y
  )

  return {
    minPanX: xBounds.min,
    maxPanX: xBounds.max,
    minPanY: yBounds.min,
    maxPanY: yBounds.max
  }
}

function constrainAxisToFocus(
  bounds: { min: number; max: number },
  focusPosition: number | undefined,
  size: number,
  zoom: number
) {
  if (focusPosition === undefined) {
    return bounds
  }

  const focusBounds = {
    min: -focusPosition * zoom,
    max: size - focusPosition * zoom
  }
  const min = Math.max(bounds.min, focusBounds.min)
  const max = Math.min(bounds.max, focusBounds.max)

  return min <= max ? { min, max } : focusBounds
}

function calculateVisualAxisBounds(min: number, max: number, zoom: number): VisualAxisBounds {
  const visualMin = min * zoom - NODE_VISUAL_BOUNDS_PADDING_PX
  const visualMax = max * zoom + NODE_VISUAL_BOUNDS_PADDING_PX

  return {
    min: visualMin,
    max: visualMax,
    size: visualMax - visualMin
  }
}

function calculateCenteredVisualAxisBounds(visualBounds: VisualAxisBounds): VisualAxisBounds {
  const inset = visualBounds.size / 4
  const visualMin = visualBounds.min + inset
  const visualMax = visualBounds.max - inset

  return {
    min: visualMin,
    max: visualMax,
    size: visualMax - visualMin
  }
}

export function centerScreenRect(rect: ScreenRect): ScreenRect {
  return {
    x: rect.x + rect.width / 4,
    y: rect.y + rect.height / 4,
    width: rect.width / 2,
    height: rect.height / 2
  }
}

function calculateAxisPanBounds(
  visualBounds: VisualAxisBounds,
  centeredVisualBounds: VisualAxisBounds,
  size: number,
  zoom: number,
  useIntersectionBounds: boolean,
  focusPosition?: number
) {
  const deadZoneSize = Math.max(0, size - PAN_DEAD_ZONE_PX * 2)

  if (useIntersectionBounds) {
    return {
      min: -centeredVisualBounds.max,
      max: size - centeredVisualBounds.min
    }
  }

  if (visualBounds.size > deadZoneSize) {
    return constrainAxisToFocus(
      {
        min: -visualBounds.min,
        max: size - visualBounds.max
      },
      focusPosition,
      size,
      zoom
    )
  }

  return constrainAxisToFocus(
    {
      min: PAN_DEAD_ZONE_PX - visualBounds.min,
      max: size - PAN_DEAD_ZONE_PX - visualBounds.max
    },
    focusPosition,
    size,
    zoom
  )
}

export function clampViewportToPanBounds(
  nodes: MeshNode[],
  size: CanvasSize,
  viewport: Viewport,
  focusPoint?: WorldPoint,
  minZoom = calculateMaxZoomOut(nodes, size)
): Viewport {
  const zoom = clamp(viewport.zoom, minZoom, MAX_ZOOM)
  const bounds = calculatePanBounds(nodes, size, zoom, activeZoomFocus(nodes, size, focusPoint))

  if (!bounds) {
    return { ...viewport, zoom }
  }

  return {
    zoom,
    panX: clamp(viewport.panX, bounds.minPanX, bounds.maxPanX),
    panY: clamp(viewport.panY, bounds.minPanY, bounds.maxPanY)
  }
}

export function nodeBoundsToScreenRect(bounds: NodeBounds | undefined, viewport: Viewport): ScreenRect | undefined {
  if (!bounds) {
    return undefined
  }

  return {
    x: bounds.minX * viewport.zoom + viewport.panX - NODE_VISUAL_BOUNDS_PADDING_PX,
    y: bounds.minY * viewport.zoom + viewport.panY - NODE_VISUAL_BOUNDS_PADDING_PX,
    width: bounds.width * viewport.zoom + NODE_VISUAL_BOUNDS_PADDING_PX * 2,
    height: bounds.height * viewport.zoom + NODE_VISUAL_BOUNDS_PADDING_PX * 2
  }
}

export function packetTransform(point: Point, width: number, height: number, viewport: Viewport) {
  const screenPoint = pointToScreen(point, width, height, viewport)

  return `translate3d(${screenPoint.x}px, ${screenPoint.y}px, 0) translate(-50%, -50%)`
}
