import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProviders } from '@/app/providers/AppProviders'
import { DashboardPage } from './DashboardPage'
import { useLlamaRuntime } from '@/features/network/api/use-llama-runtime'
import type { LlamaRuntimePayload } from '@/lib/api/types'

vi.mock('@/features/network/api/use-llama-runtime', () => ({
  useLlamaRuntime: vi.fn()
}))

vi.mock('@/features/network/components/MeshViz', () => ({
  MeshViz: ({
    nodes,
    hoveredNodeId,
    dimmedNodeIds,
    onPick
  }: {
    nodes: Array<{ id: string; role?: string; label: string }>
    hoveredNodeId?: string
    dimmedNodeIds?: Set<string>
    onPick?: (node: unknown) => void
  }) => {
    const selfNode = nodes.find((node) => node.role === 'self') ?? nodes[0]

    return (
      <div>
        <button type="button" onClick={() => onPick?.(selfNode)}>
          View CARRACK node
        </button>
        <output data-testid="mesh-viz-hover-target">{hoveredNodeId ?? 'none'}</output>
        <output data-testid="mesh-viz-dimmed-target">
          {[...(dimmedNodeIds ?? new Set())].sort().join(',') || 'none'}
        </output>
      </div>
    )
  }
}))

const mockedUseLlamaRuntime = vi.mocked(useLlamaRuntime)

const RUNTIME: LlamaRuntimePayload = {
  metrics: {
    status: 'ready',
    samples: [{ name: 'llamacpp:requests_processing', value: 1 }]
  },
  slots: {
    status: 'ready',
    slots: [{ index: 0, is_processing: true }]
  },
  items: {
    metrics: [{ name: 'llamacpp:requests_processing', value: 1 }],
    slots: [{ index: 0, is_processing: true }],
    slots_total: 1,
    slots_busy: 1
  }
}

describe('DashboardPage self-node runtime wiring', () => {
  beforeEach(() => {
    mockedUseLlamaRuntime.mockReturnValue({ data: RUNTIME, loading: false, error: null })
  })

  afterEach(() => {
    mockedUseLlamaRuntime.mockReset()
  })

  it('opens the self-node drawer with the runtime section from the dashboard flow', async () => {
    const user = userEvent.setup()

    render(
      <AppProviders initialDataMode="harness" persistDataMode={false}>
        <DashboardPage />
      </AppProviders>
    )

    await user.click(screen.getByRole('button', { name: 'View CARRACK node' }))

    expect(await screen.findByRole('heading', { name: 'Runtime' })).toBeInTheDocument()
    expect(screen.getByText('Metrics • Live')).toBeInTheDocument()
    expect(screen.getByText('requests processing')).toBeInTheDocument()
    expect(screen.getByText('1.00')).toBeInTheDocument()
  })

  it('marks the connect pane API target as configured in harness mode', () => {
    render(
      <AppProviders initialDataMode="harness" persistDataMode={false}>
        <DashboardPage />
      </AppProviders>
    )

    const statusPill = screen.getByText('configured target')
    const dot = statusPill.querySelector('span')

    expect(dot).toHaveClass('bg-current')
    expect(statusPill).toHaveAttribute('style', expect.stringContaining('var(--color-warn)'))
  })

  it('passes hovered peer rows to the matching mesh node', async () => {
    const user = userEvent.setup()

    render(
      <AppProviders initialDataMode="harness" persistDataMode={false}>
        <DashboardPage />
      </AppProviders>
    )

    const peerRow = screen.getByRole('button', { name: 'View lemony-28 node, peer ID p2' })

    await user.hover(peerRow)

    expect(screen.getByTestId('mesh-viz-hover-target')).toHaveTextContent('lemony')

    await user.unhover(peerRow)

    expect(screen.getByTestId('mesh-viz-hover-target')).toHaveTextContent('none')
  })

  it('dims non-matching mesh nodes when peer filters are active', async () => {
    const user = userEvent.setup()

    render(
      <AppProviders initialDataMode="harness" persistDataMode={false}>
        <DashboardPage />
      </AppProviders>
    )

    expect(screen.getByTestId('mesh-viz-dimmed-target')).toHaveTextContent('none')

    await user.click(screen.getByRole('button', { name: 'Filter peers' }))
    await user.click(screen.getByRole('checkbox', { name: 'Host, 2 peers' }))

    expect(screen.getByTestId('mesh-viz-dimmed-target')).toHaveTextContent('lemony,lemony-29')
  })
})
