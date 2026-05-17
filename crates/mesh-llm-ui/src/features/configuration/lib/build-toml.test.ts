import { describe, expect, it } from 'vitest'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { CONFIGURATION_DEFAULTS, CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { ConfigAssign, ConfigNode } from '@/features/app-tabs/types'

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

  it('emits canonical nested defaults and model override sections', () => {
    const toml = buildTOML(CONFIGURATION_HARNESS.nodes, CONFIGURATION_HARNESS.assigns, CONFIGURATION_HARNESS.catalog, {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: {
        'flash-attention': 'enabled',
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
    expect(toml).toContain('kv_cache_policy = "auto"')
    expect(toml).toContain('[defaults.hardware]')
    expect(toml).toContain('model_runtime = "cuda"')
    expect(toml).toContain('safety_margin_gb = 2.0')
    expect(toml).toContain('[defaults.throughput]')
    expect(toml).toContain('parallel = 4')
    expect(toml).toContain('tuning_profile = "balanced"')
    expect(toml).toContain('[defaults.speculative]')
    expect(toml).toContain('mode = "draft"')
    expect(toml).toContain('draft_selection_policy = "auto"')
    expect(toml).toContain('pairing_fault = "fail_closed"')
    expect(toml).toContain('draft_max_tokens = 32')
    expect(toml).toContain('[defaults.request_defaults]')
    expect(toml).toContain('temperature = 0.55')
    expect(toml).toContain('top_p = 0.95')
    expect(toml).toContain('reasoning_format = "deepseek-legacy"')
    expect(toml).toContain('reasoning_budget = 0')
    expect(toml).toContain('repeat_penalty = 1.10')
    expect(toml).toContain('repeat_last_n = 256')
    expect(toml).toContain('[[models]]')
    expect(toml).toContain('[models.model_fit]')
    expect(toml).toContain('ctx_size = 16384')
    expect(toml).toContain('[models.hardware]')
    expect(toml).toContain('device = "cuda:0"')
    expect(toml).toContain('gpu_layers = -1')
    expect(toml).not.toContain('[defaults.speculative_decoding]')
    expect(toml).not.toContain('ctx = ')
    expect(toml).not.toContain('gpu_index =')
    expect(toml).not.toContain('[node]')
    expect(toml).not.toContain('fail_launch')
    expect(toml).not.toContain('draft_min_tokens')
    expect(toml).not.toContain('draft_acceptance_threshold')
    expect(toml).not.toContain('chat_template =')
    expect(toml).not.toContain(`id = ${JSON.stringify(CONFIGURATION_HARNESS.assigns[0]?.id)}`)
  })

  it('emits draft-only speculative settings only when draft mode is selected', () => {
    const draftToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'speculation-mode': 'draft' }
    })
    const autoToml = buildTOML([], [], [], {
      defaults: CONFIGURATION_DEFAULTS,
      defaultsValues: { 'speculation-mode': 'auto', 'incompatible-pairing-behavior': 'fail_closed' }
    })

    expect(draftToml).toContain('mode = "draft"')
    expect(draftToml).toContain('draft_selection_policy = "auto"')
    expect(draftToml).toContain('pairing_fault = "warn_disable"')
    expect(draftToml).toContain('draft_max_tokens = 16')
    expect(autoToml).toContain('mode = "auto"')
    expect(autoToml).not.toContain('draft_selection_policy')
    expect(autoToml).not.toContain('pairing_fault')
    expect(autoToml).not.toContain('draft_max_tokens')
  })
})
