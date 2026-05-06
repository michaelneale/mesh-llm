import { describe, expect, it } from 'vitest'
import {
  calculateMaxZoomOut,
  clampViewportToPanBounds,
  DEFAULT_VIEWPORT,
  gridPatternTransform,
  pointToScreen
} from '@/features/network/lib/mesh-viewport'
import type { MeshNode } from '@/features/app-tabs/types'

const nodes: MeshNode[] = [
  { id: 'left', label: 'Left', x: 10, y: 20, status: 'online' },
  { id: 'right', label: 'Right', x: 90, y: 80, status: 'online' }
]

describe('mesh viewport math', () => {
  it('projects world coordinates into screen coordinates', () => {
    expect(pointToScreen({ x: 50, y: 25 }, 800, 400, { zoom: 1.5, panX: 10, panY: -20 })).toEqual({
      x: 610,
      y: 130
    })
  })

  it('keeps grid offsets positive while panning negatively', () => {
    expect(gridPatternTransform({ zoom: 1, panX: -10, panY: -34 }, 32)).toBe('translate(22 30)')
  })

  it('clamps zoom and pan to keep nodes reachable', () => {
    const size = { width: 320, height: 180 }
    const minZoom = calculateMaxZoomOut(nodes, size)
    const viewport = clampViewportToPanBounds(nodes, size, { zoom: 10, panX: -999, panY: 999 })

    expect(viewport.zoom).toBeLessThanOrEqual(2.4)
    expect(viewport.zoom).toBeGreaterThanOrEqual(minZoom)
    expect(viewport).not.toEqual(DEFAULT_VIEWPORT)
  })
})
