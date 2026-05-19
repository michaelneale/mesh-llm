import { describe, expect, it } from 'vitest'

import type {
  ConfigurationDefaultsCategory,
  ConfigurationDefaultsCategoryId,
  ConfigurationDefaultsControl,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsSettingIcon
} from '@/features/app-tabs/types'

// eslint-disable-next-line @typescript-eslint/no-empty-object-type -- Type-level check for optional keys
type OptionalKey<T, K extends keyof T> = {} extends Pick<T, K> ? true : false

const VISIBILITY_OPTIONAL = true satisfies OptionalKey<ConfigurationDefaultsSetting, 'visibility'>
const MUTABILITY_OPTIONAL = true satisfies OptionalKey<ConfigurationDefaultsSetting, 'mutability'>
const DEPENDS_ON_OPTIONAL = true satisfies OptionalKey<ConfigurationDefaultsSetting, 'dependsOn'>

describe('configuration defaults type contracts', () => {
  it('accepts the expanded category and icon unions without changing control kinds', () => {
    const categoryIds = [
      'runtime',
      'memory',
      'speculative-decoding',
      'advanced',
      'request-defaults',
      'skippy-transport',
      'multimodal',
      'advanced-server'
    ] as const satisfies readonly ConfigurationDefaultsCategoryId[]

    const iconNames = [
      'brain',
      'cpu',
      'layers',
      'memory',
      'binary',
      'folder',
      'gauge',
      'shield',
      'cog',
      'filter',
      'zap',
      'image',
      'server'
    ] as const satisfies readonly ConfigurationDefaultsSettingIcon[]

    const controlKinds = [
      'choice',
      'text',
      'range',
      'metric'
    ] as const satisfies readonly ConfigurationDefaultsControl['kind'][]

    expect(categoryIds).toContain('request-defaults')
    expect(iconNames).toContain('server')
    expect(controlKinds).toEqual(['choice', 'text', 'range', 'metric'])
    expect(VISIBILITY_OPTIONAL).toBe(true)
    expect(MUTABILITY_OPTIONAL).toBe(true)
    expect(DEPENDS_ON_OPTIONAL).toBe(true)
  })

  it('keeps the existing exported defaults inventory type-compatible', () => {
    const expandedSetting = {
      id: 'transport-buffering',
      categoryId: 'skippy-transport',
      icon: 'server',
      label: 'Transport buffering',
      description: 'Buffer stage transport messages before dispatch.',
      inheritedLabel: 'Inherited by stage transport children',
      visibility: 'advanced',
      mutability: 'restart-required',
      dependsOn: {
        settingId: 'speculation-mode',
        condition: (value: string) => value === 'draft_model'
      },
      control: {
        kind: 'choice',
        name: 'transport_buffering',
        value: 'enabled',
        options: [
          { value: 'enabled', label: 'enabled' },
          { value: 'disabled', label: 'disabled' }
        ]
      }
    } satisfies ConfigurationDefaultsSetting

    expect(expandedSetting.dependsOn?.condition('draft_model')).toBe(true)
  })

  it('allows tomlSection on categories and settings', () => {
    const category = {
      id: 'request-defaults',
      label: 'Request Defaults',
      summary: 'Sampling and response defaults.',
      help: 'Sampling, stopping, and repeat-penalty defaults',
      tomlSection: 'defaults.request_defaults'
    } satisfies ConfigurationDefaultsCategory

    const setting = {
      id: 'top-k',
      categoryId: 'request-defaults',
      icon: 'filter',
      label: 'Top-k',
      description: 'Limit sampling to the top-k tokens.',
      inheritedLabel: 'Applied when a request does not override top-k',
      tomlSection: 'defaults.request_defaults',
      mutability: 'runtime',
      control: {
        kind: 'range',
        name: 'top_k',
        value: '40',
        min: 0,
        max: 100,
        step: 1
      }
    } satisfies ConfigurationDefaultsSetting

    expect(category.tomlSection).toBe('defaults.request_defaults')
    expect(setting.tomlSection).toBe('defaults.request_defaults')
  })
})
