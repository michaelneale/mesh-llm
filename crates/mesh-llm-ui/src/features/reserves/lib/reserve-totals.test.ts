import { describe, expect, it } from 'vitest'
import type { ReserveProvider } from '@/features/reserves/lib/reserve-types'
import { getReserveFleetTotals, getReserveProviderTotals } from '@/features/reserves/lib/reserve-totals'

const provider: ReserveProvider = {
  id: 'test-provider',
  name: 'Test provider',
  kind: 'Cloud VM',
  region: 'test-region',
  billing: 'test meter',
  nodes: [
    { id: 'standby', hw: 'RTX 4090', vram: 24, models: ['Qwen3'], state: 'standby' },
    { id: 'waking', hw: 'H100', vram: 80, models: ['DeepSeek'], state: 'waking', eta: 180 },
    { id: 'joining', hw: 'A100', vram: 80, models: ['Mistral'], state: 'joining', eta: 90 },
    { id: 'online', hw: 'M2 Ultra', vram: 64, models: ['GLM'], state: 'online' },
    { id: 'failed', hw: 'L40S', vram: 48, models: ['Llama'], state: 'failed' },
    { id: 'unreachable', hw: 'A6000', vram: 48, models: ['Mixtral'], state: 'unreachable' }
  ]
}

describe('reserve totals', () => {
  it('summarizes provider node counts, VRAM, active wakes, errors, and ETA', () => {
    expect(getReserveProviderTotals(provider)).toEqual({
      counts: {
        standby: 1,
        waking: 1,
        joining: 1,
        online: 1,
        failed: 1,
        unreachable: 1
      },
      vramByState: {
        standby: 24,
        waking: 80,
        joining: 80,
        online: 64,
        failed: 48,
        unreachable: 48
      },
      totalNodes: 6,
      totalVram: 344,
      activeNodes: 2,
      errorNodes: 2,
      longestEta: 180
    })
  })

  it('separates online nodes from reserve nodes for fleet totals', () => {
    const totals = getReserveFleetTotals([provider])

    expect(totals.onlineNodes).toBe(1)
    expect(totals.reserveNodes).toBe(5)
  })
})
