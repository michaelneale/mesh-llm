import { describe, expect, it } from 'vitest'
import {
  getSettingDependencyStatus,
  getSettingDisabledReason,
  getSettingValue,
  isSettingDisabled
} from '@/features/configuration/lib/settings-utils'
import type { ConfigurationDefaultsSetting, ConfigurationDefaultsValues } from '@/features/app-tabs/types'

const parentSetting = {
  id: 'test-parent-mode',
  categoryId: 'runtime',
  icon: 'cpu',
  label: 'Parent mode',
  description: 'Controls the dependent setting.',
  inheritedLabel: 'Inherited by children',
  control: {
    kind: 'choice',
    name: 'test_parent_mode',
    value: 'off',
    options: [
      { value: 'off', label: 'off' },
      { value: 'on', label: 'on' }
    ]
  }
} satisfies ConfigurationDefaultsSetting

const multiChoiceParentSetting = {
  id: 'mirostat-mode',
  categoryId: 'request-defaults',
  icon: 'brain',
  label: 'Mirostat mode',
  description: 'Controls the dependent setting.',
  inheritedLabel: 'Inherited by children',
  control: {
    kind: 'choice',
    name: 'mirostat_mode',
    value: 'disabled',
    presentation: 'segmented',
    options: [
      { value: 'disabled', label: 'disabled' },
      { value: '1', label: '1' },
      { value: '2', label: '2' }
    ]
  }
} satisfies ConfigurationDefaultsSetting

const multiChoiceDependentSetting = {
  id: 'mirostat-entropy',
  categoryId: 'request-defaults',
  icon: 'gauge',
  label: 'Mirostat entropy',
  description: 'Only usable when Mirostat is on.',
  inheritedLabel: 'Inherited by dependent placements',
  dependsOn: {
    settingId: 'mirostat-mode',
    condition: (value: string) => value !== 'disabled'
  },
  control: {
    kind: 'range',
    name: 'mirostat_entropy',
    value: '5',
    min: 0.1,
    max: 10,
    step: 0.1
  }
} satisfies ConfigurationDefaultsSetting

const dependentSetting = {
  id: 'test-dependent-setting',
  categoryId: 'runtime',
  icon: 'filter',
  label: 'Dependent setting',
  description: 'Only usable when the parent is on.',
  inheritedLabel: 'Inherited by dependent placements',
  dependsOn: {
    settingId: 'test-parent-mode',
    condition: (value: string) => value === 'on'
  },
  control: {
    kind: 'text',
    name: 'test_dependent_setting',
    value: '',
    placeholder: 'Optional value'
  }
} satisfies ConfigurationDefaultsSetting

const settings = [parentSetting, dependentSetting] as const
const multiChoiceSettings = [multiChoiceParentSetting, multiChoiceDependentSetting] as const

describe('settings-utils', () => {
  it('disables dependent settings and explains the requirement', () => {
    const values: ConfigurationDefaultsValues = {}
    const disabled = getSettingDependencyStatus(dependentSetting, settings, values)

    expect(isSettingDisabled(dependentSetting, settings, values)).toBe(true)
    expect(disabled).toEqual({ disabled: true, reason: 'Requires test-parent-mode = on' })
    expect(getSettingDisabledReason(dependentSetting, settings, values)).toBe('Requires test-parent-mode = on')
  })

  it('uses the current value and clears the disabled state when satisfied', () => {
    const values = { 'test-parent-mode': 'on' } satisfies ConfigurationDefaultsValues

    expect(getSettingValue(parentSetting, values)).toBe('on')
    expect(isSettingDisabled(dependentSetting, settings, values)).toBe(false)
    expect(getSettingDependencyStatus(dependentSetting, settings, values)).toEqual({ disabled: false })
  })

  it('describes multi-option choice dependencies honestly', () => {
    const values: ConfigurationDefaultsValues = {}
    const disabled = getSettingDependencyStatus(multiChoiceDependentSetting, multiChoiceSettings, values)

    expect(isSettingDisabled(multiChoiceDependentSetting, multiChoiceSettings, values)).toBe(true)
    expect(disabled).toEqual({ disabled: true, reason: 'Requires mirostat-mode = 1 or 2' })
    expect(getSettingDisabledReason(multiChoiceDependentSetting, multiChoiceSettings, values)).toBe(
      'Requires mirostat-mode = 1 or 2'
    )
  })
})
