import { describe, expect, it } from 'vitest'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import type { ConfigAssign, ConfigNode } from '@/features/app-tabs/types'

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
      placement: 'pooled',
    }
    const assign: ConfigAssign = {
      id: 'assign "quoted"',
      modelId: 'custom\\model\nname',
      nodeId: node.id,
      containerIdx: 0,
      ctx: 4096,
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
      defaultsValues: { 'speculation-mode': 'ngram', 'incompatible-pairing-behavior': 'fail_launch' },
    })

    expect(draftToml).toContain('mode = "draft_model"')
    expect(draftToml).toContain('pairing_fault = "warn_disable"')
    expect(ngramToml).toContain('mode = "ngram"')
    expect(ngramToml).not.toContain('pairing_fault')
  })
})
