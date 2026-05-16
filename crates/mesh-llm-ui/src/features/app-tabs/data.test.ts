import { describe, expect, it } from 'vitest'
import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import type {
  ConfigurationDefaultsCategory,
  ConfigurationDefaultsControl,
  ConfigurationDefaultsSetting
} from '@/features/app-tabs/types'

function settingById(id: string): ConfigurationDefaultsSetting {
  const setting = CONFIGURATION_DEFAULTS.settings.find((entry) => entry.id === id)

  if (!setting) {
    throw new Error(`Expected setting ${id}`)
  }

  return setting
}

function categoryById(id: string): ConfigurationDefaultsCategory {
  const category = CONFIGURATION_DEFAULTS.categories.find((entry) => entry.id === id)

  if (!category) {
    throw new Error(`Expected category ${id}`)
  }

  return category
}

function settingIdsForCategory(categoryId: ConfigurationDefaultsCategory['id']) {
  return CONFIGURATION_DEFAULTS.settings
    .filter((setting) => setting.categoryId === categoryId)
    .map((setting) => setting.id)
}

function expectChoice(id: string, name: string, value: string, presentation?: 'segmented' | 'select' | 'toggle') {
  const control = settingById(id).control as Extract<ConfigurationDefaultsControl, { kind: 'choice' }>

  expect(control.kind).toBe('choice')
  expect(control.name).toBe(name)
  expect(control.value).toBe(value)
  if (presentation) expect(control.presentation).toBe(presentation)
}

function choiceValues(id: string) {
  const control = settingById(id).control
  if (control.kind !== 'choice') return []
  return control.options.map((option) => option.value)
}

describe('CONFIGURATION_DEFAULTS', () => {
  it('declares merged defaults categories in deterministic UI order', () => {
    expect(CONFIGURATION_DEFAULTS.categories).toEqual([
      {
        id: 'runtime',
        label: 'Runtime',
        summary: 'Model fit, hardware, and throughput defaults.',
        help: 'Load-time runtime behavior and concurrency defaults'
      },
      {
        id: 'memory',
        label: 'Memory',
        summary: 'KV cache policy and fit headroom.',
        help: 'VRAM accounting and fit headroom'
      },
      {
        id: 'speculative-decoding',
        label: 'Speculative Decoding',
        summary: 'Draft acceleration defaults.',
        help: 'Speculative draft policy defaults',
        tomlSection: 'defaults.speculative'
      },
      {
        id: 'advanced',
        label: 'Reasoning',
        summary: 'Reasoning and repetition controls.',
        help: 'Reasoning and sampling defaults'
      },
      {
        id: 'request-defaults',
        label: 'Request Defaults',
        summary: 'Sampling, reasoning, and request-time fallback defaults.',
        help: 'Request-time sampling and reasoning defaults'
      },
      {
        id: 'skippy-transport',
        label: 'Skippy Transport',
        summary: 'Activation wire dtype, prefill chunking, and lifecycle timing.',
        help: 'Stage transport, chunking, and lifecycle defaults',
        tomlSection: 'defaults.skippy'
      },
      {
        id: 'multimodal',
        label: 'Multimodal',
        summary: 'Projector and image token defaults.',
        help: 'Vision projector and image token defaults',
        tomlSection: 'defaults.multimodal'
      },
      {
        id: 'advanced-server',
        label: 'Advanced Server',
        summary: 'Server identity and operator overrides.',
        help: 'Advanced server defaults and identity overrides',
        tomlSection: 'defaults.advanced.server'
      }
    ])
  })

  it('keeps current request-default semantics while adding advanced sampling controls', () => {
    expect(categoryById('request-defaults').label).toBe('Request Defaults')
    expect(settingIdsForCategory('request-defaults')).toEqual(
      expect.arrayContaining([
        'temperature',
        'top-p',
        'reasoning-format',
        'reasoning-budget',
        'repeat-penalty',
        'repeat-last-n',
        'top-k',
        'min-p',
        'presence-penalty',
        'frequency-penalty',
        'max-tokens',
        'seed',
        'ignore-eos',
        'mirostat-mode',
        'mirostat-entropy',
        'mirostat-learning-rate',
        'samplers',
        'sampler-sequence',
        'stop'
      ])
    )
    expect(settingById('temperature').control).toMatchObject({ kind: 'range', name: 'temperature', value: '0.70' })
    expect(settingById('top-p').control).toMatchObject({ kind: 'range', name: 'top_p', value: '0.95' })
    expectChoice('reasoning-format', 'reasoning_format', 'auto')
    expect(choiceValues('reasoning-format')).toEqual(['auto', 'none', 'deepseek', 'deepseek-legacy'])

    for (const id of settingIdsForCategory('request-defaults')) {
      expect(settingById(id).tomlSection).toBe('defaults.request_defaults')
    }
  })

  it('uses current speculation values and includes incoming draft tuning controls', () => {
    expectChoice('speculation-mode', 'mode', 'auto', 'segmented')
    expect(choiceValues('speculation-mode')).toEqual(['auto', 'disabled', 'draft', 'ngram'])
    expectChoice('incompatible-pairing-behavior', 'pairing_fault', 'warn_disable', 'toggle')
    expect(choiceValues('incompatible-pairing-behavior')).toEqual(['warn_disable', 'fail_closed'])

    for (const id of [
      'draft-selection-policy',
      'incompatible-pairing-behavior',
      'draft-max-tokens',
      'draft-min-tokens',
      'draft-acceptance-threshold'
    ]) {
      const setting = settingById(id)
      expect(setting.tomlSection).toBe('defaults.speculative')
      expect(setting.dependsOn?.settingId).toBe('speculation-mode')
      expect(setting.dependsOn?.condition('draft')).toBe(true)
      expect(setting.dependsOn?.condition('disabled')).toBe(false)
    }

    expect(settingById('draft-min-tokens').control).toMatchObject({
      kind: 'range',
      name: 'draft_min_tokens',
      value: '0',
      min: 0,
      max: 32,
      step: 1,
      unit: 'tokens'
    })
    expect(settingById('draft-acceptance-threshold').visibility).toBe('advanced')
    expect(settingById('draft-acceptance-threshold').control).toMatchObject({
      kind: 'range',
      name: 'draft_acceptance_threshold',
      value: '0.70',
      min: 0,
      max: 1,
      step: 0.05
    })
  })

  it('includes skippy transport, multimodal, and advanced server inventories', () => {
    expect(settingIdsForCategory('skippy-transport')).toEqual(
      expect.arrayContaining([
        'activation-wire-dtype',
        'stage-model-path',
        'stage-role',
        'stage-topology',
        'prefill-chunking',
        'prefill-chunk-size',
        'prefill-chunk-schedule',
        'binary-stage-transport',
        'lifecycle-startup-timeout-ms',
        'lifecycle-readiness-interval-ms',
        'lifecycle-health-interval-ms'
      ])
    )
    expect(settingById('activation-wire-dtype').tomlSection).toBe('defaults.skippy')
    expect(settingById('prefill-chunk-size').dependsOn?.settingId).toBe('prefill-chunking')
    expect(settingById('prefill-chunk-size').dependsOn?.condition('fixed')).toBe(true)
    expect(settingById('prefill-chunk-schedule').dependsOn?.condition('schedule')).toBe(true)

    expect(settingIdsForCategory('multimodal')).toEqual([
      'mmproj-offload',
      'image-min-tokens',
      'image-max-tokens',
      'mmproj',
      'mmproj-url'
    ])
    for (const id of settingIdsForCategory('multimodal')) {
      expect(settingById(id).tomlSection).toBe('defaults.multimodal')
    }

    expect(settingIdsForCategory('advanced-server')).toEqual(['server-alias'])
    expect(settingById('server-alias')).toMatchObject({
      tomlSection: 'defaults.advanced.server',
      mutability: 'restart-required',
      visibility: 'advanced'
    })
    expect(settingById('server-alias').control).toMatchObject({ kind: 'text', name: 'alias', value: '' })
  })

  it('keeps all settings keyed to canonical TOML sections without duplicate ids', () => {
    const settingIds = CONFIGURATION_DEFAULTS.settings.map((setting) => setting.id)

    expect(CONFIGURATION_DEFAULTS.settings.length).toBeGreaterThanOrEqual(75)
    expect(new Set(settingIds).size).toBe(settingIds.length)
    expect(CONFIGURATION_DEFAULTS.settings.every((setting) => setting.tomlSection.startsWith('defaults.'))).toBe(true)
  })
})
