import { createContext } from 'react'
import type {
  FeatureFlagGroup,
  FeatureFlagOverrides,
  FeatureFlagPath,
  FeatureFlagSectionDefinition
} from './definitions'

export type FeatureFlagSettingsContextValue = {
  flags: FeatureFlagGroup
  overrides: FeatureFlagOverrides
  sections: readonly FeatureFlagSectionDefinition[]
  storageKey: string
  getBaseValue: (path: FeatureFlagPath) => boolean
  getEffectiveValue: (path: FeatureFlagPath) => boolean
  isOverridden: (path: FeatureFlagPath) => boolean
  resetAllOverrides: () => void
  resetOverride: (path: FeatureFlagPath) => void
  setOverride: (path: FeatureFlagPath, enabled: boolean) => void
}

export const FeatureFlagSettingsContext = createContext<FeatureFlagSettingsContextValue | null>(null)
