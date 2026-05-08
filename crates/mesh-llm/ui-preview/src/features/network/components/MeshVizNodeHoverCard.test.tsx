import * as HoverCardPrimitive from '@radix-ui/react-hover-card'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import { MeshVizNodeHoverCard } from './MeshVizNodeHoverCard'

const node: MeshNode = {
  id: 'node-1',
  label: 'CARRACK',
  x: 24,
  y: 40,
  status: 'online',
  role: 'self',
  subLabel: 'Local node',
  servingModels: ['llama-local-q4'],
  latencyMs: 0.6,
  vramGB: 16
}

const peer: Peer = {
  id: 'peer-1',
  hostname: 'carrack.mesh',
  region: 'local',
  status: 'online',
  hostedModels: ['llama-local-q4'],
  sharePct: 52,
  latencyMs: 0.8,
  loadPct: 33,
  shortId: 'abcd1234',
  role: 'you',
  nodeState: 'serving',
  vramGB: 16
}

function renderOpenHoverCard(cardNode: MeshNode, cardPeer?: Peer) {
  render(
    <div data-testid="mesh-hover-card-inline-host">
      <HoverCardPrimitive.Root open>
        <HoverCardPrimitive.Trigger asChild>
          <button type="button">Open</button>
        </HoverCardPrimitive.Trigger>
        <MeshVizNodeHoverCard node={cardNode} peer={cardPeer} />
      </HoverCardPrimitive.Root>
    </div>
  )
}

describe('MeshVizNodeHoverCard', () => {
  it('renders peer-backed hover details without changing the current labels', () => {
    renderOpenHoverCard(node, peer)

    expect(screen.getByRole('tooltip')).toHaveAttribute('id', 'mesh-node-popover-node-1')
    expect(screen.getByText('carrack.mesh')).toBeInTheDocument()
    expect(screen.getByText('abcd1234')).toBeInTheDocument()
    expect(screen.getByText('Serving')).toBeInTheDocument()
    expect(screen.getByText('You')).toBeInTheDocument()
    expect(screen.getByText('local')).toBeInTheDocument()
    expect(screen.getByText('llama-local-q4')).toBeInTheDocument()
    expect(screen.getByText('52% mesh share · 1 models')).toBeInTheDocument()
  })

  it('falls back to classification text when a peer region is blank', () => {
    renderOpenHoverCard(
      { ...node, role: 'peer', client: true, renderKind: 'client', meshState: 'client' },
      { ...peer, role: 'client', nodeState: 'client', region: '', hostedModels: [], vramGB: 0 }
    )

    expect(screen.getAllByText('Client').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('API-only')).toBeInTheDocument()
    expect(screen.getByText('CLIENT')).toBeInTheDocument()
  })

  it('portals the hover card outside the inline mesh host to avoid viewport clipping', () => {
    renderOpenHoverCard(node, peer)

    expect(screen.getByTestId('mesh-hover-card-inline-host')).not.toContainElement(screen.getByRole('tooltip'))
  })

  it('falls back to node fields when peer data is unavailable', () => {
    renderOpenHoverCard(node)

    expect(screen.getByText('CARRACK')).toBeInTheDocument()
    expect(screen.getByText('node-1')).toBeInTheDocument()
    expect(screen.getByText('Local node')).toBeInTheDocument()
    expect(screen.getByText('Local')).toBeInTheDocument()
    expect(screen.getByText('<1 ms')).toBeInTheDocument()
    expect(screen.getByText('16.0 GB')).toBeInTheDocument()
  })
})
