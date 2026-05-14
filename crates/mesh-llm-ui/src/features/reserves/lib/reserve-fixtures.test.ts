import { describe, expect, it } from 'vitest'
import type { WakeableNode } from '@/features/app-tabs/types'
import { RESERVE_PROVIDER_FIXTURES, reserveProvidersFromWakeableNodes } from '@/features/reserves/lib/reserve-fixtures'

describe('reserveProvidersFromWakeableNodes', () => {
  it('keeps fixture providers available when no live wakeable data has loaded', () => {
    expect(reserveProvidersFromWakeableNodes(undefined)).toBeUndefined()
    expect(RESERVE_PROVIDER_FIXTURES.length).toBeGreaterThan(0)
  })

  it('groups wakeable nodes by provider and maps live states into reserve states', () => {
    const wakeableNodes: WakeableNode[] = [
      {
        logical_id: 'cloud-1',
        models: ['Qwen3-32B'],
        provider: 'Cloud Burst',
        state: 'sleeping',
        vram_gb: 24
      },
      {
        logical_id: 'cloud-2',
        models: ['DeepSeek-R1'],
        provider: 'Cloud Burst',
        state: 'waking',
        vram_gb: 80,
        wake_eta_secs: 120
      },
      {
        logical_id: 'lan-1',
        models: ['Mistral-Large'],
        state: 'sleeping',
        vram_gb: 48
      }
    ]

    const providers = reserveProvidersFromWakeableNodes(wakeableNodes)

    expect(providers).toHaveLength(2)
    expect(providers?.[0]).toMatchObject({ id: 'cloud-burst', name: 'Cloud Burst', kind: 'Provider' })
    expect(providers?.[0]?.nodes.map((node) => node.state)).toEqual(['standby', 'waking'])
    expect(providers?.[0]?.nodes[1]).toMatchObject({ eta: 120, vram: 80 })
    expect(providers?.[1]).toMatchObject({ id: 'unassigned-reserves', name: 'Unassigned reserves' })
    expect(providers?.[1]?.nodes[0]).toMatchObject({ id: 'lan-1', state: 'standby', vram: 48 })
  })
})
