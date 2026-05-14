import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import type { StatusPayload } from '@/lib/api/types'

const featureFlagMocks = vi.hoisted(() => ({
  newReservesPageEnabled: true,
  wakePolicyConfigurationEnabled: true
}))

const dataModeMocks = vi.hoisted(() => ({
  mode: 'harness'
}))

const statusQueryMocks = vi.hoisted(() => ({
  data: undefined as StatusPayload | undefined
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

vi.mock('@tanstack/react-router', () => ({
  useNavigate: vi.fn(() => routerMocks.navigate)
}))

vi.mock('@/lib/feature-flags', () => ({
  useBooleanFeatureFlag: vi.fn((path: string) => {
    if (path === 'configuration/wakePolicyConfiguration') return featureFlagMocks.wakePolicyConfigurationEnabled
    return featureFlagMocks.newReservesPageEnabled
  })
}))

vi.mock('@/lib/data-mode', () => ({
  useDataMode: vi.fn(() => ({ mode: dataModeMocks.mode }))
}))

vi.mock('@/features/network/api/use-status-query', () => ({
  useStatusQuery: vi.fn(() => ({ data: statusQueryMocks.data }))
}))

import { ReservesPageContent } from '@/features/reserves/pages/ReservesPage'

describe('ReservesPageContent', () => {
  beforeEach(() => {
    featureFlagMocks.newReservesPageEnabled = true
    featureFlagMocks.wakePolicyConfigurationEnabled = true
    dataModeMocks.mode = 'harness'
    statusQueryMocks.data = undefined
    routerMocks.navigate.mockReset()
  })

  it('shows a gated message when the new reserves page flag is disabled', () => {
    featureFlagMocks.newReservesPageEnabled = false

    render(<ReservesPageContent data={{ ...DASHBOARD_HARNESS, wakeableNodes: [] }} />)

    expect(screen.getByRole('heading', { name: 'Reserves is gated' })).toBeInTheDocument()
    expect(screen.getByText(/global\/newReservesPage/i)).toBeInTheDocument()
    expect(screen.queryByTestId('reserves-section')).not.toBeInTheDocument()
  })

  it('uses the full-width reserves layout when the page flag is enabled', async () => {
    const user = userEvent.setup()
    const { container } = render(<ReservesPageContent data={{ ...DASHBOARD_HARNESS, wakeableNodes: [] }} />)

    expect(container.firstElementChild).toHaveClass('flex', 'min-w-0', 'flex-col', 'gap-[14px]')
    expect(screen.getByRole('heading', { name: 'Reserves' })).toBeInTheDocument()
    expect(screen.getByTestId('reserves-section')).toBeInTheDocument()
    expect(screen.getByTestId('reserve-policy-panel')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /add provider/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /add provider/i }))

    expect(screen.getByRole('dialog', { name: 'Add reserve provider' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /bare metal/i })).toBeChecked()
    expect(screen.getByRole('radio', { name: /digital ocean/i })).toBeDisabled()
    expect(screen.getByRole('radio', { name: /gcp/i })).toBeDisabled()
    expect(screen.getByRole('radio', { name: /aws/i })).toBeDisabled()
    expect(screen.getByLabelText('Provider name')).toBeInTheDocument()
    expect(screen.getByLabelText('Location or rack')).toBeInTheDocument()
    expect(screen.getByText('Provider type')).toBeInTheDocument()
  })

  it('keeps all provider groups collapsed on first render even when fixtures mark some as expanded', () => {
    const { container } = render(<ReservesPageContent data={{ ...DASHBOARD_HARNESS, wakeableNodes: undefined }} />)

    const toggles = [...container.querySelectorAll('button[aria-expanded="false"]')]
    expect(toggles.length).toBeGreaterThan(0)

    for (const toggle of toggles) {
      expect(toggle).toHaveAttribute('aria-expanded', 'false')
    }
  })

  it('opens the reserve policy configuration tab from the Reserves policy panel', async () => {
    const user = userEvent.setup()

    render(<ReservesPageContent data={{ ...DASHBOARD_HARNESS, wakeableNodes: [] }} />)

    await user.click(screen.getByRole('button', { name: /edit policy/i }))

    expect(routerMocks.navigate).toHaveBeenCalledWith({
      to: '/configuration/$configurationTab',
      params: { configurationTab: 'wake-policy' }
    })
  })

  it('disables wake actions for providers with no standby nodes', async () => {
    const user = userEvent.setup()

    render(<ReservesPageContent data={{ ...DASHBOARD_HARNESS, wakeableNodes: undefined }} />)

    await user.click(screen.getByRole('button', { name: /lambda labs/i }))

    expect(screen.getByRole('button', { name: 'Wake provider' })).toBeDisabled()
  })

  it('retries failed reserve nodes through local preview state only', async () => {
    const user = userEvent.setup()

    render(<ReservesPageContent data={{ ...DASHBOARD_HARNESS, wakeableNodes: undefined }} />)

    await user.click(screen.getByRole('button', { name: /vast.ai/i }))
    expect(screen.getByText(/2 nodes need attention/i)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Retry all' }))
    await user.click(screen.getByRole('button', { name: 'Retry all' }))

    expect(screen.queryByText(/2 nodes need attention/i)).not.toBeInTheDocument()
    expect(screen.getAllByRole('progressbar').length).toBeGreaterThan(0)
  })

  it('uses live status VRAM for the mesh comparison in live mode', () => {
    dataModeMocks.mode = 'live'
    statusQueryMocks.data = {
      node_id: 'self',
      node_state: 'serving',
      model_name: 'qwen',
      peers: [
        {
          node_id: 'peer-vram',
          vram_gb: 22
        },
        {
          node_id: 'peer-gpu',
          gpus: [{ idx: 0, name: 'GPU', total_vram_gb: 12 }]
        }
      ],
      models: [],
      my_vram_gb: 10,
      gpus: [],
      serving_models: [],
      wakeable_nodes: []
    }

    render(
      <ReservesPageContent
        data={{ ...DASHBOARD_HARNESS, wakeableNodes: [], peers: [{ ...DASHBOARD_HARNESS.peers[0], vramGB: 99 }] }}
      />
    )

    expect(screen.getByText('vs 44 GB on the live mesh')).toBeInTheDocument()
  })

  it('does not show preview reserve providers in live mode when wakeable nodes are unavailable', () => {
    dataModeMocks.mode = 'live'
    statusQueryMocks.data = {
      node_id: 'self',
      node_state: 'standby',
      model_name: '',
      my_vram_gb: 0,
      peers: [],
      models: [],
      gpus: [],
      serving_models: []
    }

    render(<ReservesPageContent data={{ ...DASHBOARD_HARNESS, wakeableNodes: undefined }} />)

    expect(screen.getByText('No reserve providers are configured yet.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /vast.ai/i })).not.toBeInTheDocument()
  })
})
