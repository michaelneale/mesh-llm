import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ResponseStatsBar } from '@/features/chat/components/ResponseStatsBar'

describe('ResponseStatsBar', () => {
  it('renders the complete generation stats line', () => {
    const { container } = render(<ResponseStatsBar tokens="27 tok" tokPerSec="15.3 tok/s" ttft="1116ms" />)

    const stats = container.firstElementChild

    expect(stats).toHaveClass('select-none')
    expect(stats).toHaveTextContent('27 tok')
    expect(stats).toHaveTextContent('15.3 tok/s')
    expect(stats).toHaveTextContent('TTFT 1116ms')
  })

  it('keeps the stat layout complete when only one metric has arrived', () => {
    const { container } = render(<ResponseStatsBar tokens="17 tok" />)

    expect(container.firstElementChild).toHaveTextContent('17 tok·0.0 tok/s·TTFT 0ms')
  })

  it('hides the stats line until response metrics exist', () => {
    const { container } = render(<ResponseStatsBar />)

    expect(container).toBeEmptyDOMElement()
  })
})
