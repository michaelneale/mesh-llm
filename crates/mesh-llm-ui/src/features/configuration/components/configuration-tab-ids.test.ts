import { describe, expect, it } from 'vitest'
import {
  getEnabledConfigurationTabIds,
  isConfigurationTabId
} from '@/features/configuration/components/configuration-tab-ids'

describe('configuration-tab-ids', () => {
  it('omits gated configuration tabs when their flags are disabled', () => {
    expect(
      getEnabledConfigurationTabIds({
        integrationsEnabled: false,
        signingAttestationEnabled: false,
        wakePolicyEnabled: false
      })
    ).toEqual(['defaults', 'local-deployment', 'toml-review'])
  })

  it('treats the Reserves tab as invalid when its feature flag is disabled', () => {
    expect(isConfigurationTabId('wake-policy', { wakePolicyEnabled: false })).toBe(false)
    expect(isConfigurationTabId('wake-policy', { wakePolicyEnabled: true })).toBe(true)
  })
})
