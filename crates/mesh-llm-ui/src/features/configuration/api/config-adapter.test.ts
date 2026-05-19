import { describe, expect, it } from 'vitest'
import {
  adaptStatusToConfiguration,
  createConfigurationDefaultsValuesFromMeshConfig,
  mergeConfigurationDefaultsIntoMeshConfig,
  runtimeControlApplyErrorMessage,
  type RuntimeControlMeshConfig
} from '@/features/configuration/api/config-adapter'
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

  it('overlays hydrated runtime-control defaults onto the harness settings', () => {
    const defaultsValues = createConfigurationDefaultsValuesFromMeshConfig({
      defaults: {
        throughput: {
          threads: 12
        },
        hardware: {
          mlock: true
        },
        model_fit: {
          kv_cache_policy: 'quality'
        },
        speculative: {
          draft_max_tokens: 24
        },
        request_defaults: {
          temperature: 0.8,
          top_p: 0.95,
          reasoning_format: 'qwen',
          top_k: 55,
          repeat_penalty: 1.35,
          repeat_last_n: 384
        },
        skippy: {
          activation_wire_dtype: 'q8'
        },
        multimodal: {
          image_min_tokens: 64
        },
        advanced: {
          server: {
            alias: 'carrack-mesh'
          }
        }
      }
    })

    const configuration = adaptStatusToConfiguration(STATUS_PAYLOAD, [], defaultsValues)
    const values = Object.fromEntries(
      configuration.defaults.settings.map((setting) => [setting.id, setting.control.value])
    )

    expect(values['threads']).toBe('12')
    expect(values['mlock']).toBe('on')
    expect(values['kv-cache']).toBe('quality')
    expect(values['draft-max-tokens']).toBe('24')
    expect(values['temperature']).toBe('0.8')
    expect(values['top-p']).toBe('0.95')
    expect(values['reasoning-format']).toBe('qwen')
    expect(values['top-k']).toBe('55')
    expect(values['repeat-penalty']).toBe('1.35')
    expect(values['repeat-last-n']).toBe('384')
    expect(values['activation-wire-dtype']).toBe('q8')
    expect(values['image-min-tokens']).toBe('64')
    expect(values['server-alias']).toBe('carrack-mesh')
  })

  it('merges only modified defaults back into the full mesh config without dropping unrelated fields', () => {
    const meshConfig: RuntimeControlMeshConfig = {
      version: 1,
      owner_control: {
        bind: '127.0.0.1:7447'
      },
      telemetry: {
        enabled: true
      },
      models: [{ model: 'hf://meshllm/base@main:Q4_K_M', ctx_size: 8192 }],
      plugin: [{ name: 'telemetry', enabled: true }],
      defaults: {
        throughput: {
          threads: 6
        },
        hardware: {
          mlock: false
        },
        request_defaults: {
          temperature: 0.8,
          reasoning_format: 'deepseek'
        },
        speculative: {
          draft_max_tokens: 16
        },
        advanced: {
          server: {
            alias: 'existing-alias'
          }
        }
      }
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(meshConfig, {
      threads: '0',
      mlock: 'off',
      'reasoning-format': 'qwen',
      temperature: '1.0',
      'top-p': '1.0',
      'top-k': '55',
      'draft-max-tokens': '16',
      'repeat-last-n': '64',
      'activation-wire-dtype': 'auto',
      'image-min-tokens': '0',
      'server-alias': 'carrack-mesh'
    })

    expect(merged).toEqual({
      version: 1,
      owner_control: {
        bind: '127.0.0.1:7447'
      },
      telemetry: {
        enabled: true
      },
      models: [{ model: 'hf://meshllm/base@main:Q4_K_M', ctx_size: 8192 }],
      plugin: [{ name: 'telemetry', enabled: true }],
      defaults: {
        request_defaults: {
          reasoning_format: 'qwen',
          temperature: 1,
          top_p: 1,
          top_k: 55,
          repeat_last_n: 64
        },
        advanced: {
          server: {
            alias: 'carrack-mesh'
          }
        }
      }
    })
    expect(meshConfig.defaults).toEqual({
      throughput: {
        threads: 6
      },
      hardware: {
        mlock: false
      },
      request_defaults: {
        temperature: 0.8,
        reasoning_format: 'deepseek'
      },
      speculative: {
        draft_max_tokens: 16
      },
      advanced: {
        server: {
          alias: 'existing-alias'
        }
      }
    })
  })

  it('removes known defaults when saved UI values return to canonical defaults', () => {
    const meshConfig: RuntimeControlMeshConfig = {
      version: 1,
      defaults: {
        request_defaults: {
          temperature: 0.7,
          top_p: 0.82,
          reasoning_format: 'qwen'
        },
        custom_extension: {
          keep: true
        }
      }
    }

    const merged = mergeConfigurationDefaultsIntoMeshConfig(meshConfig, {
      temperature: '0.70',
      'top-p': '0.95',
      'reasoning-format': 'auto'
    })

    expect(merged).toEqual({
      version: 1,
      defaults: {
        custom_extension: {
          keep: true
        }
      }
    })
  })

  it('extracts runtime-control apply error messages from structured payloads', () => {
    expect(
      runtimeControlApplyErrorMessage({
        success: false,
        current_revision: 7,
        config_hash: 'abc123',
        apply_mode: 'unspecified',
        error: { code: 'revision_conflict', message: 'config revision changed on disk' }
      })
    ).toBe('config revision changed on disk')

    expect(
      runtimeControlApplyErrorMessage({
        success: false,
        current_revision: 7,
        config_hash: 'abc123',
        apply_mode: 'unspecified',
        error: { code: 'control_unavailable' }
      })
    ).toBe('control unavailable')
  })
})
