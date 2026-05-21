export type ConfigurationTabId =
  | 'defaults'
  | 'local-deployment'
  | 'wake-policy'
  | 'signing'
  | 'integrations'
  | 'toml-review'

export const CONFIGURATION_TAB_IDS = [
  'defaults',
  'local-deployment',
  'wake-policy',
  'signing',
  'integrations',
  'toml-review'
] as const satisfies readonly ConfigurationTabId[]

type ConfigurationTabAvailability = {
  integrationsEnabled?: boolean
  signingAttestationEnabled?: boolean
  wakePolicyEnabled?: boolean
}

export function getEnabledConfigurationTabIds({
  integrationsEnabled = true,
  signingAttestationEnabled = true,
  wakePolicyEnabled = true
}: ConfigurationTabAvailability = {}): ConfigurationTabId[] {
  return CONFIGURATION_TAB_IDS.filter((tabId) => {
    if (tabId === 'wake-policy') return wakePolicyEnabled
    if (tabId === 'signing') return signingAttestationEnabled
    if (tabId === 'integrations') return integrationsEnabled
    return true
  })
}

export function isConfigurationTabId(
  value: string | undefined,
  availability?: ConfigurationTabAvailability
): value is ConfigurationTabId {
  return value !== undefined && getEnabledConfigurationTabIds(availability).some((tabId) => tabId === value)
}
