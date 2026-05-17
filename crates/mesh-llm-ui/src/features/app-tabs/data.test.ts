import { describe, expect, it } from 'vitest'
import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'

describe('CONFIGURATION_DEFAULTS', () => {
  it('uses Request Defaults for request-scoped sampling controls', () => {
    expect(CONFIGURATION_DEFAULTS.categories).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: 'request-defaults', label: 'Request Defaults' })])
    )

    const requestDefaultIds = CONFIGURATION_DEFAULTS.settings
      .filter((setting) => setting.categoryId === 'request-defaults')
      .map((setting) => setting.id)

    expect(requestDefaultIds).toEqual(
      expect.arrayContaining(['temperature', 'top-p', 'reasoning-format', 'reasoning-budget', 'repeat-penalty'])
    )
  })

  it('matches the canonical reasoning format options and omits stale speculative controls', () => {
    const reasoningFormat = CONFIGURATION_DEFAULTS.settings.find((setting) => setting.id === 'reasoning-format')
    const settingIds = CONFIGURATION_DEFAULTS.settings.map((setting) => setting.id)

    expect(reasoningFormat?.control.kind).toBe('choice')
    expect(
      reasoningFormat?.control.kind === 'choice' ? reasoningFormat.control.options.map((option) => option.value) : []
    ).toEqual(['auto', 'none', 'deepseek', 'deepseek-legacy'])

    expect(settingIds).not.toContain('draft-min-tokens')
    expect(settingIds).not.toContain('draft-acceptance-threshold')
  })
})
