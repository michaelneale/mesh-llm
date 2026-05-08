import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ConnectBlock, type ConnectApiTargetLiveness } from '@/features/network/components/ConnectBlock'

function renderConnectBlock(apiTargetLiveness?: ConnectApiTargetLiveness) {
  render(
    <ConnectBlock
      installHref="https://docs.meshllm.cloud/#install"
      apiUrl="http://127.0.0.1:9337/v1"
      apiStatus="configured target"
      apiTargetLiveness={apiTargetLiveness}
      runCommand="mesh-llm --auto"
      description="join the mesh"
    />
  )
}

function getApiStatusDot() {
  const statusPill = screen.getByText('configured target')
  const dot = statusPill.querySelector('span')

  if (!dot) throw new Error('Expected the API status pill to render a dot')
  return dot
}

function getApiStatusPill() {
  return screen.getByText('configured target')
}

describe('ConnectBlock', () => {
  it('renders the configured API target dot as warning state', () => {
    renderConnectBlock('configured')

    expect(getApiStatusDot()).toHaveClass('bg-current')
    expect(getApiStatusPill()).toHaveAttribute('style', expect.stringContaining('var(--color-warn)'))
  })

  it('renders the live API target dot as healthy state', () => {
    renderConnectBlock('live')

    expect(getApiStatusDot()).toHaveClass('bg-current')
    expect(getApiStatusPill()).toHaveAttribute('style', expect.stringContaining('var(--color-good)'))
  })

  it('renders the unavailable API target dot as error state', () => {
    renderConnectBlock('unavailable')

    expect(getApiStatusDot()).toHaveClass('bg-current')
    expect(getApiStatusPill()).toHaveAttribute('style', expect.stringContaining('var(--color-bad)'))
  })

  it('keeps the API target dot muted while liveness is still checking', () => {
    renderConnectBlock('checking')

    expect(getApiStatusDot()).toHaveClass('bg-current')
    expect(getApiStatusPill()).toHaveAttribute('style', expect.stringContaining('var(--color-fg-dim)'))
  })
})
