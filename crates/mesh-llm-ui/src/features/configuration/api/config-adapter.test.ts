import { describe, expect, it } from 'vitest'
import { adaptStatusToConfiguration } from '@/features/configuration/api/config-adapter'
import type { MeshModelRaw, StatusPayload } from '@/lib/api/types'

const STATUS_PAYLOAD: StatusPayload = {
  node_id: 'self',
  node_state: 'serving',
  model_name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
  peers: [],
  models: [],
  my_vram_gb: 0,
  gpus: [],
  serving_models: []
}

describe('adaptStatusToConfiguration', () => {
  it('accepts public status peers without node_id', () => {
    const configuration = adaptStatusToConfiguration(
      {
        ...STATUS_PAYLOAD,
        peers: [
          {
            id: 'aeac0d8e53',
            state: 'client',
            role: 'Client',
            hostname: '1266a345aeb9',
            serving_models: [],
            vram_gb: 0
          }
        ]
      },
      []
    )

    expect(configuration.nodes[0]).toEqual(
      expect.objectContaining({
        id: 'aeac0d8e53',
        hostname: '1266a345aeb9',
        status: 'offline'
      })
    )
  })

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

    const configuration = adaptStatusToConfiguration(STATUS_PAYLOAD, models)

    expect(configuration.catalog[0]).toEqual(
      expect.objectContaining({
        id: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
        sizeGB: 4.4,
        ctxMaxK: 0,
        moe: false,
        vision: false
      })
    )
  })
})
