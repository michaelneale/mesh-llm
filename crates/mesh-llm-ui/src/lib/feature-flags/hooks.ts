import { useContext } from 'react'
import { useFeature } from 'flagged'
import { FeatureFlagSettingsContext } from '@/lib/feature-flags/context'
import type { FeatureFlagPath } from '@/lib/feature-flags/definitions'

export function useFeatureFlagSettings() {
  const context = useContext(FeatureFlagSettingsContext)
  if (!context) throw new Error('useFeatureFlagSettings must be used within FeatureFlagProvider')
  return context
}

export function useBooleanFeatureFlag(path: FeatureFlagPath) {
  return useFeature(path) === true
}
