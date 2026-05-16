import { describe, expect, it } from 'vitest'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import type { ConfigAssign, ConfigNode, ConfigurationDefaultsHarnessData } from '@/features/app-tabs/types'

describe('buildTOML', () => {
  it('escapes generated string values', () => {
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

    expect(toml).toContain(`id = ${JSON.stringify(node.id)}`)
    expect(toml).toContain(`hostname = ${JSON.stringify(node.hostname)}`)
    expect(toml).toContain(`region = ${JSON.stringify(node.region)}`)
    expect(toml).toContain(`  id = ${JSON.stringify(assign.id)}`)
    expect(toml).toContain(`  model = ${JSON.stringify(assign.modelId)}`)
  })

  it('emits incompatible pairing behavior only for draft model defaults', () => {
    const draftToml = buildTOML([], [], [], { defaults: CONFIGURATION_DEFAULTS })
    const ngramToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'speculation-mode': 'ngram', 'incompatible-pairing-behavior': 'fail_launch' }
    })

    expect(draftToml).not.toContain('pairing_fault = "warn_disable"')
    expect(draftToml).not.toContain('[defaults.speculative]')
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
