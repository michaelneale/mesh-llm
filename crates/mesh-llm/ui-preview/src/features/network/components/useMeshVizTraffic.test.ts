import { describe, expect, it } from 'vitest'
import { packetColor } from './useMeshVizTraffic'

describe('MeshViz traffic packets', () => {
  it('interpolates packet colour from source node colour to target node colour', () => {
    expect(packetColor('source-node', 'target-node', 0)).toBe('source-node')
    expect(packetColor('source-node', 'target-node', 0.5)).toBe('color-mix(in oklab, target-node 50.00%, source-node)')
    expect(packetColor('source-node', 'target-node', 1)).toBe('target-node')
  })

  it('clamps invalid progress while preserving identical node colours', () => {
    expect(packetColor('source-node', 'target-node', Number.NaN)).toBe('source-node')
    expect(packetColor('source-node', 'target-node', -1)).toBe('source-node')
    expect(packetColor('source-node', 'target-node', 2)).toBe('target-node')
    expect(packetColor('same-node', 'same-node', 0.5)).toBe('same-node')
  })
})
