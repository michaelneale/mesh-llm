import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { FlagsProvider } from 'flagged'
import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'
import { FeatureFlagSettingsContext, type FeatureFlagSettingsContextValue } from '@/lib/feature-flags/context'
import {
  FEATURE_FLAG_SECTIONS,
  findFeatureFlagDefinition,
  getBaseFeatureFlagValue,
  normalizeFeatureFlagOverrides,
  readStoredFeatureFlagOverrides,
  removeFeatureFlagOverride,
  resolveFeatureFlags,
  writeStoredFeatureFlagOverrides,
  type FeatureFlagOverrides,
  type FeatureFlagPath
} from '@/lib/feature-flags/definitions'

export function FeatureFlagProvider({
  children,
  storageKey = APP_STORAGE_KEYS.featureFlagOverrides
}: {
  children: ReactNode
  storageKey?: string
}) {
  const [overrides, setOverrides] = useState<FeatureFlagOverrides>(() => readStoredFeatureFlagOverrides(storageKey))
  const flags = useMemo(() => resolveFeatureFlags(overrides), [overrides])

  useEffect(() => {
    writeStoredFeatureFlagOverrides(overrides, storageKey)
  }, [overrides, storageKey])

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key !== storageKey || event.storageArea !== window.localStorage) return
      try {
        setOverrides(event.newValue ? normalizeFeatureFlagOverrides(JSON.parse(event.newValue)) : {})
      } catch {
        setOverrides({})
      }
    }

    window.addEventListener('storage', handleStorage)
    return () => window.removeEventListener('storage', handleStorage)
  }, [storageKey])

  const getEffectiveValue = useCallback(
    (path: FeatureFlagPath) => {
      const definition = findFeatureFlagDefinition(path)
      if (!definition) return false
      return overrides[definition.sectionId]?.[definition.key] ?? getBaseFeatureFlagValue(path)
    },
    [overrides]
  )

  const isOverridden = useCallback(
    (path: FeatureFlagPath) => {
      const definition = findFeatureFlagDefinition(path)
      if (!definition) return false
      return overrides[definition.sectionId]?.[definition.key] !== undefined
    },
    [overrides]
  )

  const resetOverride = useCallback((path: FeatureFlagPath) => {
    const definition = findFeatureFlagDefinition(path)
    if (!definition) return
    setOverrides((currentOverrides) =>
      removeFeatureFlagOverride(currentOverrides, definition.sectionId, definition.key)
    )
  }, [])

  const setOverride = useCallback((path: FeatureFlagPath, enabled: boolean) => {
    const definition = findFeatureFlagDefinition(path)
    if (!definition) return

    setOverrides((currentOverrides) => {
      if (enabled === getBaseFeatureFlagValue(path))
        return removeFeatureFlagOverride(currentOverrides, definition.sectionId, definition.key)

      return {
        ...currentOverrides,
        [definition.sectionId]: {
          ...(currentOverrides[definition.sectionId] ?? {}),
          [definition.key]: enabled
        }
      }
    })
  }, [])

  const value = useMemo<FeatureFlagSettingsContextValue>(
    () => ({
      flags,
      overrides,
      sections: FEATURE_FLAG_SECTIONS,
      storageKey,
      getBaseValue: getBaseFeatureFlagValue,
      getEffectiveValue,
      isOverridden,
      resetAllOverrides: () => setOverrides({}),
      resetOverride,
      setOverride
    }),
    [flags, getEffectiveValue, isOverridden, overrides, resetOverride, setOverride, storageKey]
  )

  return (
    <FeatureFlagSettingsContext.Provider value={value}>
      <FlagsProvider features={flags}>{children}</FlagsProvider>
    </FeatureFlagSettingsContext.Provider>
  )
}
