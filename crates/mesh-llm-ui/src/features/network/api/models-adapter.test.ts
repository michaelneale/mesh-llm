import { describe, expect, it } from 'vitest'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'
import type { MeshModelRaw } from '@/lib/api/types'

describe('adaptModelsToSummary', () => {
  it('accepts public API model rows without a nested capabilities object', () => {
    const models: MeshModelRaw[] = [
      {
        name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
        status: 'warm',
        size_gb: 4.4,
        node_count: 1,
        quantization: 'Q4_K_M',
        moe: false,
        vision: false
      }
    ]

    expect(adaptModelsToSummary(models)).toEqual([
      expect.objectContaining({
        name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
        status: 'warm',
        size: '4.4B',
        context: 'Unknown',
        ctxMaxK: undefined,
        moe: false,
        vision: false
      })
    ])
  })

  it('prefers nested capabilities when available', () => {
    const models: MeshModelRaw[] = [
      {
        name: 'Qwen3-VL-8B-Q4_K_M',
        status: 'cold',
        size_gb: 5,
        node_count: 0,
        capabilities: { moe: true, vision: true },
        quantization: 'Q4_K_M',
        context_length: 128_000,
        moe: false,
        vision: false
      }
    ]

    expect(adaptModelsToSummary(models)[0]).toEqual(
      expect.objectContaining({
        status: 'offline',
        context: '128K',
        ctxMaxK: 128,
        moe: true,
        vision: true
      })
    )
  })
})
