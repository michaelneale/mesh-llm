import '@testing-library/jest-dom/vitest'

import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { WakeableNode } from '@/features/app-tabs/types'
import { ReservesPanel } from '@/features/reserves/components/ReservesPanel'

afterEach(() => {
  cleanup()
})

describe('ReservesPanel', () => {
  it('renders sleeping and waking reserve inventory', () => {
    render(<ReservesPanel wakeableNodes={WAKEABLE_NODES} />)

    const section = screen.getByTestId('reserves-section')
    expect(within(section).getByText('Reserves')).toBeInTheDocument()
    expect(within(section).getByText('Sleeping')).toBeInTheDocument()
    expect(within(section).getByText('Waking')).toBeInTheDocument()
    expect(within(section).getByText('vast-a100-1')).toBeInTheDocument()
    expect(within(section).getByText('runpod-h100-2')).toBeInTheDocument()
    expect(within(section).getByText('Qwen2.5-72B-Instruct')).toBeInTheDocument()
    expect(within(section).getByText('DeepSeek-R1')).toBeInTheDocument()
    expect(within(section).getByText('Qwen3-32B')).toBeInTheDocument()
    expect(within(section).getByText('80.0 GB')).toBeInTheDocument()
    expect(within(section).getByText('94.0 GB')).toBeInTheDocument()
    expect(within(section).getByText('Vast')).toBeInTheDocument()
    expect(within(section).getByText('7 min')).toBeInTheDocument()
  })

  it('shows an empty state when wakeableNodes is an empty array', () => {
    render(<ReservesPanel wakeableNodes={[]} />)

    const section = screen.getByTestId('reserves-section')
    expect(within(section).getByText('Reserves')).toBeInTheDocument()
    expect(within(section).getByText('0 nodes')).toBeInTheDocument()
    expect(within(section).getByText(/No reserve nodes are advertised yet/i)).toBeInTheDocument()
  })

  it('shows an empty state when wakeableNodes is absent', () => {
    render(<ReservesPanel />)

    expect(screen.getByTestId('reserves-section')).toBeInTheDocument()
    expect(screen.getByText(/No reserve nodes are advertised yet/i)).toBeInTheDocument()
  })
})

const WAKEABLE_NODES: WakeableNode[] = [
  {
    logical_id: 'vast-a100-1',
    state: 'sleeping',
    models: ['Qwen2.5-72B-Instruct'],
    vram_gb: 80,
    provider: 'Vast'
  },
  {
    logical_id: 'runpod-h100-2',
    state: 'waking',
    models: ['DeepSeek-R1', 'Qwen3-32B'],
    vram_gb: 94,
    wake_eta_secs: 420
  }
]
