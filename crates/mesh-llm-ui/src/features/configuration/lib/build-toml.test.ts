import { describe, expect, it } from 'vitest'
import { CONFIGURATION_DEFAULTS, CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { ConfigAssign, ConfigNode, ConfigurationDefaultsHarnessData } from '@/features/app-tabs/types'
import { buildTOML } from '@/features/configuration/lib/build-toml'

describe('buildTOML', () => {
  it('escapes generated string values and omits legacy model keys', () => {
    const node: ConfigNode = {
      id: 'node "quoted"',
      hostname: 'mesh\\host\nalpha',
      region: 'iad "1"',
      status: 'online',
      cpu: 'test cpu',
      ramGB: 64,
      gpus: [],
      placement: 'pooled'
    }
    const assign: ConfigAssign = {
      id: 'assign "quoted"',
      modelId: 'custom\\model\nname',
      nodeId: node.id,
      containerIdx: 0,
      ctx: 4096
    }

    const toml = buildTOML([node], [assign])

    expect(toml).toContain('version = 1')
    expect(toml).toContain(`model = ${JSON.stringify(assign.modelId)}`)
    expect(toml).toContain('ctx_size = 4096')
    expect(toml).not.toContain('ctx = 4096')
    expect(toml).not.toContain('gpu_index =')
    expect(toml).not.toContain('[node]')
    expect(toml).not.toContain(`id = ${JSON.stringify(assign.id)}`)
  })

  it('emits changed canonical nested defaults and model override sections', () => {
    const toml = buildTOML(CONFIGURATION_HARNESS.nodes, CONFIGURATION_HARNESS.assigns, CONFIGURATION_HARNESS.catalog, {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: {
        'flash-attention': 'enabled',
        'llamacpp-flavor': 'metal',
        'memory-margin': '3.5',
        'speculation-mode': 'draft',
        'incompatible-pairing-behavior': 'fail_closed',
        'draft-max-tokens': '32',
        temperature: '0.55',
        'reasoning-format': 'deepseek-legacy'
      }
    })

    expect(toml).toContain('version = 1')
    expect(toml).toContain('[defaults.model_fit]')
    expect(toml).toContain('flash_attention = "enabled"')
    expect(toml).toContain('[defaults.hardware]')
    expect(toml).toContain('model_runtime = "metal"')
    expect(toml).toContain('safety_margin_gb = 3.5')
    expect(toml).toContain('[defaults.speculative]')
    expect(toml).toContain('mode = "draft"')
    expect(toml).toContain('pairing_fault = "fail_closed"')
    expect(toml).toContain('draft_max_tokens = 32')
    expect(toml).toContain('[defaults.request_defaults]')
    expect(toml).toContain('temperature = 0.55')
    expect(toml).toContain('reasoning_format = "deepseek-legacy"')
    expect(toml).toContain('[[models]]')
    expect(toml).toContain('[models.model_fit]')
    expect(toml).toContain('ctx_size = 16384')
    expect(toml).toContain('[models.hardware]')
    expect(toml).toContain('device = "cuda:0"')
    expect(toml).toContain('gpu_layers = -1')
    expect(toml).not.toContain('[defaults.speculative_decoding]')
    expect(toml).not.toContain('draft_selection_policy = "auto"')
    expect(toml).not.toContain('top_p = 0.95')
    expect(toml).not.toContain('reasoning_budget = 0')
    expect(toml).not.toContain('ctx = ')
    expect(toml).not.toContain('gpu_index =')
    expect(toml).not.toContain('[node]')
    expect(toml).not.toContain('fail_launch')
    expect(toml).not.toContain('chat_template =')
    expect(toml).not.toContain(`id = ${JSON.stringify(CONFIGURATION_HARNESS.assigns[0]?.id)}`)
  })

  it('omits disabled draft speculative settings unless draft mode is selected and changed', () => {
    const draftToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'speculation-mode': 'draft', 'draft-max-tokens': '32' }
    })
    const autoToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'speculation-mode': 'auto', 'incompatible-pairing-behavior': 'fail_closed' }
    })
    const ngramToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'speculation-mode': 'ngram', 'incompatible-pairing-behavior': 'fail_closed' }
    })

    expect(draftToml).toContain('[defaults.speculative]')
    expect(draftToml).toContain('mode = "draft"')
    expect(draftToml).toContain('draft_max_tokens = 32')
    expect(draftToml).not.toContain('pairing_fault = "warn_disable"')
    expect(autoToml).not.toContain('[defaults.speculative]')
    expect(autoToml).not.toContain('pairing_fault')
    expect(ngramToml).toContain('mode = "ngram"')
    expect(ngramToml).not.toContain('pairing_fault')
    expect(ngramToml).not.toContain('draft_selection_policy')
  })

  it('uses canonical inventory defaults when live defaults are hydrated into control values', () => {
    const liveHydratedDefaults = {
      ...CONFIGURATION_DEFAULTS,
      settings: CONFIGURATION_DEFAULTS.settings.map((setting) => {
        if (setting.id === 'activation-wire-dtype') {
          return {
            ...setting,
            control: {
              ...setting.control,
              value: 'q8'
            }
          }
        }

        if (setting.id === 'image-min-tokens') {
          return {
            ...setting,
            control: {
              ...setting.control,
              value: '64'
            }
          }
        }

        if (setting.id === 'server-alias') {
          return {
            ...setting,
            control: {
              ...setting.control,
              value: 'carrack-mesh'
            }
          }
        }

        return setting
      })
    } satisfies ConfigurationDefaultsHarnessData

    const toml = buildTOML([], [], [], {
      defaults: liveHydratedDefaults,
      defaultsValues: {}
    })

    expect(toml).toContain('[defaults.skippy]')
    expect(toml).toContain('activation_wire_dtype = "q8"')
    expect(toml).toContain('[defaults.multimodal]')
    expect(toml).toContain('image_min_tokens = 64')
    expect(toml).toContain('[defaults.advanced.server]')
    expect(toml).toContain('alias = "carrack-mesh"')
    expect(toml).not.toContain('[defaults.request_defaults]')
  })
})
