import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Slider } from '@/components/ui/Slider'

describe('Slider', () => {
  it('emits string values from the native range input', () => {
    const handleValueChange = vi.fn()

    render(<Slider ariaLabel="Memory margin" max={4} min={0} name="memory-margin" onValueChange={handleValueChange} step={0.5} unit="GB" value="2" />)

    fireEvent.change(screen.getByRole('slider', { name: 'Memory margin' }), { target: { value: '2.5' } })

    expect(handleValueChange).toHaveBeenCalledWith('2.5')
  })

  it('renders caller-provided value labels including zero', () => {
    render(<Slider ariaLabel="Draft tokens" max={128} min={0} name="draft-max-tokens" onValueChange={vi.fn()} value="0" valueLabel={0} />)

    expect(screen.getByText('0')).toBeInTheDocument()
  })

  it('supports labels, units, bottom value placement, alignment, and open or closed boundary labels', () => {
    render(
      <Slider
        ariaLabel="Acceptance threshold"
        formatValue={(value) => Number(value).toFixed(2)}
        label="Draft acceptance"
        lowerBound={{ inclusive: false, value: '0.00' }}
        max={1}
        min={0}
        name="draft-acceptance-threshold"
        onValueChange={vi.fn()}
        step={0.05}
        unit="ratio"
        upperBound={{ inclusive: true, value: '1.00' }}
        value="0.7"
        valueLabelAlign="center"
        valueLabelPlacement="bottom"
      />,
    )

    expect(screen.getByText('Draft acceptance')).toBeInTheDocument()
    expect(screen.getByText('0.70')).toHaveClass('font-mono')
    expect(screen.getByText('ratio')).not.toHaveClass('font-mono')
    expect(screen.getByText('0.70').parentElement).toHaveClass('justify-self-center')
    expect(screen.getByText('0.00')).toBeInTheDocument()
    expect(screen.getByText('1.00')).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'Acceptance threshold' })).toHaveAttribute('aria-valuetext', '0.70 ratio')
  })
})
