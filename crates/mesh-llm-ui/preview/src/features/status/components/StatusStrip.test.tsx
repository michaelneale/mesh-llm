import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StatusStrip } from '@/features/status/components/StatusStrip'

describe('StatusStrip', () => {
  it('renders a single-point sparkline without invalid SVG coordinates', () => {
    const { container } = render(
      <StatusStrip
        metrics={[
          {
            id: 'single-point',
            label: 'Single point',
            value: 1,
            sparkline: [42]
          }
        ]}
      />
    )
    const points = Array.from(container.querySelectorAll('polyline'))
      .map((polyline) => polyline.getAttribute('points') ?? '')
      .join(' ')

    expect(points).not.toMatch(/NaN|Infinity/)
    expect(points).toContain('0,')
    expect(points).toContain('0,9')
  })

  it('renders default icons and metric histories for live metrics without backing sparkline data', () => {
    const { container } = render(
      <StatusStrip
        metrics={[
          { id: 'node-id', label: 'Node ID', value: 'abc123' },
          { id: 'owner', label: 'Owner', value: 'Unsigned' },
          { id: 'active-models', label: 'Active models', value: 1 },
          { id: 'mesh-vram', label: 'Mesh VRAM', value: '32.5', unit: 'GB' },
          { id: 'nodes', label: 'Nodes', value: 2 },
          { id: 'inflight', label: 'Inflight', value: 3 }
        ]}
      />
    )

    expect(container.querySelectorAll('svg')).toHaveLength(8)
    expect(container.querySelectorAll('polyline')).toHaveLength(4)
  })

  it('pre-seeds generated metric histories with a configurable zero window', () => {
    const { container } = render(
      <StatusStrip
        historyPointCount={5}
        metrics={[
          { id: 'mesh-vram', label: 'Mesh VRAM', value: '32.5', unit: 'GB' },
          { id: 'inflight', label: 'Inflight', value: 3 }
        ]}
      />
    )
    const linePoints = Array.from(container.querySelectorAll('polyline'))
      .map((polyline) => polyline.getAttribute('points') ?? '')
      .filter((points) => points.split(' ').length === 5)

    expect(linePoints).toHaveLength(2)
    for (const points of linePoints) {
      expect(points.split(' ')).toHaveLength(5)
      expect(points).not.toMatch(/NaN|Infinity/)
      expect(points.split(' ').every((point) => point.endsWith(',9'))).toBe(true)
    }
  })
})
