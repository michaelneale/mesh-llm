import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PeersTable } from '@/features/network/components/PeersTable'
import type { Peer } from '@/features/app-tabs/types'

function buildPeer(index: number): Peer {
  return {
    id: `peer-${index}`,
    hostname: `worker-${index}`,
    region: '',
    status: 'online',
    hostedModels: [`Model-${index}`],
    sharePct: index,
    latencyMs: index,
    loadPct: 0,
    shortId: `p${index}`,
    role: 'worker',
    nodeState: 'serving',
    version: '0.65.0',
    vramGB: 12
  }
}

describe('PeersTable', () => {
  it('jumps to the page containing an externally selected peer', () => {
    const peers = Array.from({ length: 12 }, (_, index) => buildPeer(index + 1))

    render(
      <PeersTable
        peers={peers}
        summary={{ total: peers.length, online: peers.length, capacity: '144 GB' }}
        selectedPeerId="peer-12"
      />
    )

    expect(screen.getByRole('button', { name: 'View worker-12 node, peer ID peer-12 (selected)' })).toHaveAttribute(
      'data-active',
      'true'
    )
    expect(screen.queryByRole('button', { name: 'View worker-1 node, peer ID peer-1' })).not.toBeInTheDocument()
    expect(screen.getByText('11-12')).toBeInTheDocument()
    expect(screen.getByText('12 total')).toBeInTheDocument()
    expect(screen.queryByText('● 144 GB')).not.toBeInTheDocument()
  })

  it('sorts the full peer list by clicked column before rendering rows', async () => {
    const user = userEvent.setup()
    const peers = [
      { ...buildPeer(1), hostname: 'middle-vram', shortId: 'middle', vramGB: 24, sharePct: 24 },
      { ...buildPeer(2), hostname: 'high-vram', shortId: 'high', vramGB: 96, sharePct: 96 },
      { ...buildPeer(3), hostname: 'low-vram', shortId: 'low', vramGB: 8, sharePct: 8 }
    ]

    render(<PeersTable peers={peers} summary={{ total: peers.length, online: peers.length, capacity: '128 GB' }} />)

    await user.click(screen.getByRole('button', { name: 'Sort peers by VRAM ascending' }))
    expect(
      screen.getAllByRole('button', { name: /View .* node/ }).map((row) => row.getAttribute('aria-label'))
    ).toEqual([
      'View low-vram node, peer ID peer-3',
      'View middle-vram node, peer ID peer-1',
      'View high-vram node, peer ID peer-2'
    ])

    await user.click(screen.getByRole('button', { name: 'Sort peers by VRAM descending' }))
    expect(
      screen.getAllByRole('button', { name: /View .* node/ }).map((row) => row.getAttribute('aria-label'))
    ).toEqual([
      'View high-vram node, peer ID peer-2',
      'View middle-vram node, peer ID peer-1',
      'View low-vram node, peer ID peer-3'
    ])
  })

  it('filters peers by deselected column values and updates the table total', async () => {
    const user = userEvent.setup()
    const peers = [
      { ...buildPeer(1), role: 'host' as const, nodeState: 'serving' as const, hostedModels: ['Qwen-8B'] },
      { ...buildPeer(2), role: 'client' as const, nodeState: 'client' as const, hostedModels: [] },
      { ...buildPeer(3), role: 'worker' as const, nodeState: 'serving' as const, hostedModels: ['Qwen-27B'] }
    ]
    const onFilteredPeerIdsChange = vi.fn()

    render(
      <PeersTable
        peers={peers}
        summary={{ total: peers.length, online: peers.length, capacity: '36 GB' }}
        onFilteredPeerIdsChange={onFilteredPeerIdsChange}
      />
    )

    expect(onFilteredPeerIdsChange).toHaveBeenLastCalledWith(['peer-1', 'peer-2', 'peer-3'])

    await user.click(screen.getByRole('button', { name: 'Filter peers' }))
    await user.click(screen.getByRole('checkbox', { name: 'Client, 1 peers' }))

    expect(onFilteredPeerIdsChange).toHaveBeenLastCalledWith(['peer-1', 'peer-3'])

    expect(screen.queryByRole('button', { name: 'View worker-2 node, peer ID peer-2' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View worker-1 node, peer ID peer-1' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View worker-3 node, peer ID peer-3' })).toBeInTheDocument()
    expect(screen.getByText('2 of 3 total')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Filter peers, 1 active' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Reset' }))

    expect(onFilteredPeerIdsChange).toHaveBeenLastCalledWith(['peer-1', 'peer-2', 'peer-3'])

    expect(screen.getByRole('button', { name: 'View worker-2 node, peer ID peer-2' })).toBeInTheDocument()
    expect(screen.getByText('3 total')).toBeInTheDocument()
  })

  it('keeps the filter popover click-toggleable and provides a clear no-results state', async () => {
    const user = userEvent.setup()
    const peers = [
      { ...buildPeer(1), role: 'host' as const, nodeState: 'serving' as const },
      { ...buildPeer(2), role: 'client' as const, nodeState: 'client' as const, hostedModels: [] }
    ]

    render(<PeersTable peers={peers} summary={{ total: peers.length, online: peers.length, capacity: '24 GB' }} />)

    const filterButton = screen.getByRole('button', { name: 'Filter peers' })
    await user.click(filterButton)

    expect(screen.getByRole('dialog', { name: 'Filter connected peers' })).toBeInTheDocument()

    await user.click(filterButton)

    expect(screen.queryByRole('dialog', { name: 'Filter connected peers' })).not.toBeInTheDocument()

    await user.click(filterButton)
    await user.click(screen.getByRole('checkbox', { name: 'Host, 1 peers' }))
    await user.click(screen.getByRole('checkbox', { name: 'Client, 1 peers' }))

    expect(screen.getByText('No peers match these filters.')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Clear filters' }))

    expect(screen.getByRole('button', { name: 'View worker-1 node, peer ID peer-1' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View worker-2 node, peer ID peer-2' })).toBeInTheDocument()
  })

  it('repages to keep a selected peer visible after sorting', async () => {
    const user = userEvent.setup()
    const peers = Array.from({ length: 12 }, (_, index) => ({
      ...buildPeer(index + 1),
      vramGB: index === 1 ? 999 : index + 1
    }))

    render(
      <PeersTable
        peers={peers}
        summary={{ total: peers.length, online: peers.length, capacity: '1,076 GB' }}
        selectedPeerId="peer-2"
      />
    )

    expect(screen.getByRole('button', { name: 'View worker-2 node, peer ID peer-2 (selected)' })).toHaveAttribute(
      'data-active',
      'true'
    )

    await user.click(screen.getByRole('button', { name: 'Sort peers by VRAM ascending' }))

    expect(
      await screen.findByRole('button', { name: 'View worker-2 node, peer ID peer-2 (selected)' })
    ).toHaveAttribute('data-active', 'true')
    expect(screen.queryByRole('button', { name: 'View worker-1 node, peer ID peer-1' })).not.toBeInTheDocument()
    expect(screen.getByText('11-12')).toBeInTheDocument()
  })

  it('renders status pills from each peer operational state', () => {
    const peers = [
      { ...buildPeer(1), status: 'online' as const, nodeState: 'serving' as const },
      { ...buildPeer(2), status: 'degraded' as const, nodeState: 'loading' as const },
      { ...buildPeer(3), status: 'online' as const, nodeState: 'standby' as const, hostedModels: [] },
      {
        ...buildPeer(4),
        status: 'online' as const,
        nodeState: 'client' as const,
        role: 'client' as const,
        hostedModels: []
      },
      { ...buildPeer(5), status: 'offline' as const, nodeState: undefined, hostedModels: [] }
    ]

    render(<PeersTable peers={peers} summary={{ total: peers.length, online: 1, capacity: '36 GB' }} />)

    expect(screen.getAllByText('Serving').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Loading').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Standby').length).toBeGreaterThan(0)
    expect(screen.getAllByText('API-only').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Offline').length).toBeGreaterThan(0)
  })

  it('keeps long hosted model values available in a styled tooltip when truncated', async () => {
    const user = userEvent.setup()
    const longModelName = 'unusually-long-model-name-that-should-stay-readable-on-hover-Q8_0'
    const peers = [{ ...buildPeer(1), hostedModels: [longModelName] }]

    render(<PeersTable peers={peers} summary={{ total: peers.length, online: peers.length, capacity: '12 GB' }} />)

    const hostedModel = screen.getAllByText(longModelName)[0]
    expect(hostedModel).not.toHaveAttribute('title')

    await user.hover(hostedModel)

    expect(await screen.findByRole('tooltip')).toHaveTextContent(longModelName)
  })

  it('truncates long peer IDs and keeps the full ID available in a styled tooltip', async () => {
    const user = userEvent.setup()
    const longPeerId = 'peer-id-without-short-id-that-is-long-enough-to-overflow-the-id-column-1234567890'
    const peers = [{ ...buildPeer(1), id: longPeerId, shortId: undefined }]

    render(<PeersTable peers={peers} summary={{ total: peers.length, online: peers.length, capacity: '12 GB' }} />)

    expect(screen.getByText(longPeerId)).toHaveClass('truncate')
    expect(screen.getByText(longPeerId)).not.toHaveAttribute('title')

    await user.hover(screen.getByText(longPeerId))

    expect(await screen.findByRole('tooltip')).toHaveTextContent(longPeerId)
  })

  it('does not repeat ID-derived hostnames in the ID column', async () => {
    const user = userEvent.setup()
    const fullPeerId = '006c9b7a70'
    const peers = [{ ...buildPeer(1), id: fullPeerId, shortId: '006c9b7a', hostname: fullPeerId }]

    render(<PeersTable peers={peers} summary={{ total: peers.length, online: peers.length, capacity: '12 GB' }} />)

    expect(screen.getByText('006c9b7a')).toBeInTheDocument()
    expect(screen.getByText('—')).toBeInTheDocument()
    expect(screen.queryByText(fullPeerId)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'View peer 006c9b7a, peer ID 006c9b7a70' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /006c9b7a70 node/ })).not.toBeInTheDocument()

    await user.hover(screen.getByText('006c9b7a'))

    expect(await screen.findByRole('tooltip')).toHaveTextContent(fullPeerId)
  })

  it('keeps real hostnames visible under the short ID', () => {
    const peers = [{ ...buildPeer(1), id: '02ad2a5100', shortId: '02ad2a51', hostname: 'Mac' }]

    render(<PeersTable peers={peers} summary={{ total: peers.length, online: peers.length, capacity: '12 GB' }} />)

    expect(screen.getByText('02ad2a51')).toBeInTheDocument()
    expect(screen.getByText('Mac')).toBeInTheDocument()
  })

  it('rounds latency to the nearest millisecond without hiding positive sub-millisecond values', () => {
    const peers = [
      { ...buildPeer(1), latencyMs: 241.4 },
      { ...buildPeer(2), latencyMs: 0.4 },
      { ...buildPeer(3), latencyMs: 0 }
    ]

    render(<PeersTable peers={peers} summary={{ total: peers.length, online: peers.length, capacity: '12 GB' }} />)

    expect(screen.getAllByText('241 ms').length).toBeGreaterThan(0)
    expect(screen.getAllByText('<1 ms').length).toBeGreaterThan(0)
    expect(screen.getAllByText('0 ms').length).toBeGreaterThan(0)
    expect(screen.queryByText('241.4 ms')).not.toBeInTheDocument()
  })
})
