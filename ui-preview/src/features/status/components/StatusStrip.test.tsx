import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StatusStrip } from '@/features/status/components/StatusStrip'

describe('StatusStrip', () => {
  it('renders a single-point sparkline without invalid SVG coordinates', () => {
    const { container } = render(<StatusStrip metrics={[{
      id: 'single-point',
      label: 'Single point',
      value: 1,
      sparkline: [42],
    }]} />)
    const points = Array.from(container.querySelectorAll('polyline'))
      .map((polyline) => polyline.getAttribute('points') ?? '')
      .join(' ')

    expect(points).not.toMatch(/NaN|Infinity/)
    expect(points).toContain('0,')
  })
})
