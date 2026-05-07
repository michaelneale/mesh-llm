import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Sparkline } from '@/components/ui/Sparkline'

describe('Sparkline', () => {
  it('centers a flat series on the component baseline', () => {
    const { container } = render(<Sparkline values={[42]} />)
    const points = Array.from(container.querySelectorAll('polyline'))
      .map((polyline) => polyline.getAttribute('points') ?? '')
      .join(' ')

    expect(points).not.toMatch(/NaN|Infinity/)
    expect(points).toContain('0,9')
    expect(points).toContain('72,9')
  })

  it('can expose an accessible label when the sparkline is meaningful', () => {
    render(<Sparkline ariaLabel="Inflight request history" values={[1, 2, 3]} />)

    expect(screen.getByRole('img', { name: 'Inflight request history' })).toBeInTheDocument()
  })

  it('left-pads short series to a requested point count', () => {
    const { container } = render(<Sparkline pointCount={5} values={[3]} />)
    const linePoints = container.querySelector('polyline')?.getAttribute('points') ?? ''

    expect(linePoints.split(' ')).toHaveLength(5)
    expect(linePoints).toMatch(/^0,9 /)
  })
})
